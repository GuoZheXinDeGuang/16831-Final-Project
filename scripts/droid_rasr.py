#!/usr/bin/env python
"""DROID: limited, OFFLINE check of reachability-aware subgoal reranking (P2 only).

DROID in H-JEPA is open-loop planning on recorded real-robot clips: nothing can be executed, so
the closed-loop label of P1 ("the low-level MPC reached the subgoal") does not exist. Here the
label comes from the expert video instead:

  y(g_j) = 1  iff  min_{1 <= k <= 3j} || g_j - enc2(o_k) ||^2 <= eps

i.e. the j-th level-2 subgoal candidate (level-2 stride 3 frames) lies, in the level-2 latent
space, on the path the expert actually took by the time the plan wants to be there. eps = median
squared latent distance between expert frames one level-2 step (3 frames, 0.6 s) apart, TRAIN clips.

Subcommands
  collect  run only the level-2 planner on clips (train/val: random windows from DROID training
           shard 0; test: the 16 official evaluation clips) and save every GD candidate plan,
           its progress cost and the level-2 latents of all real frames.
  train    fit the reachability MLP (train), temperature (val), report AUROC/ECE on test.
  eval     full open-loop planning; level-2 candidate chosen by cost - lam * c_min * mean_j
           log(R_j + delta) (lam = 0 is the unmodified planner), scored with the official
           Fréchet fidelity / ATE metrics.

Model weights stay frozen; upstream code is imported, never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr.common import CKPTS, HJEPA_HOME, HJEPA_PKG, load_eval_cfg  # noqa: E402

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import droid_eval as de  # noqa: E402
from droid_data import DROIDClipReader  # noqa: E402
from eval_config_utils import _build_policy_plan_config  # noqa: E402
from planning_eval import _load_policy_normalizer_artifact, build_solver, load_model  # noqa: E402
from rasr import reachability as R  # noqa: E402
from rasr.rasr_solver import RASRConfig, RASRHierarchicalSolver  # noqa: E402

EVAL_CLIPS = HJEPA_PKG / "droid_assets/droid_clips_waypoint_curated16v2_5fps_gw36.json"
STRIDE = 3  # level-2 temporal stride (frames)


class _StopAfterTop(Exception):
    pass


class DistLogistic(torch.nn.Module):
    """Planner-visible distance baseline as a reachability predictor:
    R = sigmoid(w . [d, log H, d * log H, 1]) with d = ||g - z||^2 / d_scale, fit on TRAIN rows only.
    Same call signature as ReachabilityMLP (c[:, 0] = log H)."""

    def __init__(self, w, d_scale):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))
        self.d_scale = float(d_scale)

    def forward(self, z, g, c):
        d = ((g - z) ** 2).sum(-1) / self.d_scale
        lh = c[:, 0]
        x = torch.stack([d, lh, d * lh, torch.ones_like(d)], -1)
        return torch.sigmoid(x @ self.w)


def _dist_feats(rows, d_scale):
    d = ((rows["g"] - rows["z"]) ** 2).sum(-1) / d_scale
    lh = np.log(rows["H"].astype(np.float64))
    return np.stack([d, lh, d * lh], -1)


def load_any_predictor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("type") == "dist_logistic":
        return DistLogistic(ck["w"], ck["d_scale"]).cuda().eval()
    from rasr.rasr_solver import load_predictor
    return load_predictor(path)


class DroidRerankSolver(RASRHierarchicalSolver):
    """Top-level hook: log the full candidate pool; optionally rerank it with R over all subgoals."""

    def __init__(self, *a, lam=0.0, delta=0.05, stop_after_top=False, **k):
        super().__init__(*a, **k)
        self.lam, self.delta, self.stop_after_top = lam, delta, stop_after_top
        self.last = None

    def _rasr_select(self, level, level_info, planning_horizon, steps_taken, eval_budget,
                     obs_embeddings, subgoal_latents, subgoal_actions):
        P, A, C = self._candidate_pool(level, level_info, self.level_solvers[level])
        base = int(torch.argmin(C))
        z = obs_embeddings["embed_2"][0, 0].float()
        rec = {"P": P.cpu().numpy(), "C": C.cpu().numpy(), "base_idx": base, "z0": z.cpu().numpy()}
        sel = base
        if self.predictor is not None:
            S, T1, D = P.shape
            js = torch.arange(1, T1 - 1, device=P.device)  # subgoals 1..T-1 (T is replaced by the goal)
            g = P[:, 1:T1 - 1].reshape(-1, D)
            H = (js * STRIDE).float().log().repeat(S)
            c = torch.stack([H, torch.zeros_like(H), torch.zeros_like(H)], -1)
            r = self.predictor(z.expand(g.shape[0], -1), g, c).view(S, -1)
            score = C - self.lam * C.min().clamp_min(1e-6) * torch.log(r + self.delta).mean(-1)
            sel = int(torch.argmin(score)) if self.lam > 0 else base
            rec.update({"R": r.cpu().numpy(), "score": score.cpu().numpy()})
        rec["sel_idx"] = sel
        self.last = rec
        if self.stop_after_top:
            raise _StopAfterTop
        return P[sel][None].cpu(), A[sel][None].cpu()


def build(model_seed: int, planner_seed: int, predictor=None, lam=0.0, stop_after_top=False, overrides=None):
    ckpt = str(CKPTS["droid_hjepa_l2"]).format(seed=model_seed)
    cfg = load_eval_cfg("droid_l2", {"policy": ckpt, "seed": planner_seed, "output.dir": "/dev/null",
                                     "cache_dir": str(HJEPA_HOME), **(overrides or {})})
    model = load_model(cfg, None)
    up = build_solver(cfg, model)
    solver = DroidRerankSolver(up.level_solvers, up.level_models, RASRConfig(mode="rerank"), predictor,
                               lam=lam, stop_after_top=stop_after_top)
    stats = _load_policy_normalizer_artifact(cfg)["stats"]["action"]
    mean, std = (np.asarray(stats[k]).reshape(-1) for k in ("mean", "std"))
    solver.configure(action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(1, mean.shape[0])), n_envs=1,
                     config=_build_policy_plan_config(cfg))
    return cfg, solver, mean, std


def shard_episodes(n_max: int):
    """'success' episodes of the extracted training shard 0 (relative to droid/) that are long enough
    for a 37-frame 5 fps window (>= 150 frames at 15 Hz), excluding the 16 evaluation episodes."""
    root = HJEPA_HOME / "droid"
    eval_ids = {c["episode_id"] for c in json.loads(EVAL_CLIPS.read_text())}
    eps = []
    for p in sorted((root / "droid_256p").glob("*/*/success/*/*/trajectory.npz")):
        if p.parent.name in eval_ids:
            continue
        with np.load(p) as tr:
            if len(tr["cartesian_position"]) >= 150:
                eps.append(str(p.parent.relative_to(root)))
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    return eps[:n_max]


def split_of(ep: str) -> str:  # deterministic 80/20 split by episode path
    return "val" if int(hashlib.md5(ep.encode()).hexdigest(), 16) % 5 == 0 else "train"


def make_reader(split: str, n_clips: int):
    if split == "test":
        return DROIDClipReader(None, ["left_mp4_path"], 37, 5, frozen_clips=str(EVAL_CLIPS), deterministic_getitem=True)
    eps = [e for e in shard_episodes(10 ** 9) if split_of(e) == split][:n_clips]
    lst = HJEPA_HOME / "droid" / f"rasr_{split}_n{len(eps)}.txt"
    lst.write_text("".join(f"{e} {i}\n" for i, e in enumerate(eps)))
    return DROIDClipReader(lst.name, ["left_mp4_path", "right_mp4_path"], 37, 5, deterministic_getitem=True)


@torch.no_grad()
def encode_frames(solver, tf, visual):
    px = torch.stack([tf(v) for v in visual])[:, None].cuda()  # [T, 1, C, H, W]
    out = []
    for i in range(0, px.shape[0], 8):
        e = solver._encode_all_levels(info_dict={"pixels": px[i:i + 8]}, key="pixels")
        out.append(e["embed_2"][:, 0].float().cpu())
    return torch.cat(out).numpy()  # [T, D]


def plan_info(tf, visual, action_dim):
    return {"pixels": tf(visual[0])[None, None].cuda(), "goal": tf(visual[-1])[None, None].cuda(),
            "action": torch.zeros(1, 1, action_dim, device="cuda")}


def cmd_collect(a):
    cfg, solver, mean, std = build(a.model_seed, a.planner_seed, stop_after_top=True,
                                   overrides={"solver.solvers.level2.num_samples": a.num_samples})
    tf = de.img_transform(cfg.img_size)
    ds = make_reader(a.split, a.n_clips)
    out = Path(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in solver.level_solvers.values()]
    t0 = time.time()
    for k in range(len(ds)):
        dst = out / f"clip_{k:04d}.npz"
        if dst.exists():
            continue
        for gen, seed in gens:
            gen.manual_seed(seed + k)
        try:
            obs, actions, *_ = ds[k]
        except Exception as e:  # unreadable episode: skip, record
            print(f"clip {k}: unreadable ({e})", flush=True)
            continue
        visual = obs["visual"]
        zf = encode_frames(solver, tf, visual)
        try:
            solver(plan_info(tf, visual, mean.shape[0]), steps_taken=0, eval_budget=int(cfg.eval_budget))
        except _StopAfterTop:
            pass
        r = solver.last
        np.savez_compressed(dst, z_frames=zf, P=r["P"], C=r["C"], base_idx=r["base_idx"], z0=r["z0"],
                            episode=ds.samples[k], actions=actions.numpy())
        if k % 10 == 0:
            print(f"clip {k}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
    print("done", out)


LABEL_MODE = "any"   # "any": match any expert frame 1..3j (original); "aligned": frames within 3j +- ALIGN_W only
ALIGN_W = STRIDE     # +- one level-2 step (3 frames, 0.6 s); fixed a priori, not tuned


def label_rows(files, eps):
    rows = {k: [] for k in ("z", "g", "H", "n_steps", "num_samples", "y", "min_d", "clip", "cand", "j")}
    for ci, f in enumerate(files):
        d = np.load(f)
        zf, P = d["z_frames"], d["P"]  # [37, D], [S, 13, D]
        S, T1, _ = P.shape
        for s in range(S):
            for j in range(1, T1 - 1):
                H = j * STRIDE
                if LABEL_MODE == "aligned":
                    lo, hi = max(1, H - ALIGN_W), min(len(zf) - 1, H + ALIGN_W)
                    dist = ((zf[lo:hi + 1] - P[s, j]) ** 2).sum(-1)
                else:
                    dist = ((zf[1:H + 1] - P[s, j]) ** 2).sum(-1)
                rows["z"].append(d["z0"]); rows["g"].append(P[s, j]); rows["H"].append(H)
                rows["n_steps"].append(1); rows["num_samples"].append(1)
                rows["min_d"].append(float(dist.min())); rows["y"].append(float(dist.min() <= eps))
                rows["clip"].append(ci); rows["cand"].append(s); rows["j"].append(j)
    return {k: np.asarray(v, dtype=np.float32 if k in ("z", "g", "y", "min_d") else np.int64) for k, v in rows.items()}


def cmd_train(a):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    src = Path(a.data)
    files = {s: sorted((src / s).glob("clip_*.npz")) for s in ("train", "val", "test")}
    zfs = [np.load(f)["z_frames"] for f in files["train"]]
    eps_by_gap = {g: float(np.median(np.concatenate([((z[g:] - z[:-g]) ** 2).sum(-1) for z in zfs]))) for g in (1, 3, 6)}
    eps = eps_by_gap[STRIDE]
    rows = {s: label_rows(fs, eps) for s, fs in files.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"label_mode": LABEL_MODE, "align_w_frames": ALIGN_W, "eps": eps, "eps_def": "median squared level-2 distance between expert frames one level-2 step "
           "(3 frames, 0.6 s) apart, train clips", "eps_by_frame_gap": eps_by_gap,
           "pos_rate_by_gap": {g: {s: float(label_rows(fs, e)["y"].mean()) for s, fs in files.items()}
                               for g, e in eps_by_gap.items()},
           "splits": {s: {"clips": len(files[s]), "rows": int(len(r["y"])), "pos_rate": float(r["y"].mean())}
                      for s, r in rows.items()}}
    print(json.dumps(res, indent=1))
    models = []
    for seed in range(a.mlp_seeds):
        m, hist = R.train_mlp(rows["train"], rows["val"], seed=seed, log=lambda *x: None)
        m.temperature.fill_(1.0)
        t = R.fit_temperature(m, rows["val"])
        models.append((m, t))

    def ece(p, y, n=10):
        b = np.clip((p * n).astype(int), 0, n - 1)
        return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(n) if (b == i).any()))

    per = []
    for seed, (m, t) in enumerate(models):
        r = {}
        for s in ("val", "test"):
            p = R.mlp_predict(m, rows[s]); y = rows[s]["y"]
            r[s] = {"auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                    "brier": float(brier_score_loss(y, p)), "ece": ece(p, y)}
            # distance-only reference (logistic on -min latent distance to the start frame is not
            # available; use ||g - z0||^2 as the trivial score)
            dref = -((rows[s]["g"] - rows[s]["z"]) ** 2).sum(-1)
            r[s]["auroc_dist_only"] = float(roc_auc_score(y, dref))
        per.append({"seed": seed, "temperature": t, **r})
    res["mlp"] = per
    best = int(np.argmax([p["val"]["auroc"] for p in per]))
    res["selected_seed"] = best
    m, t = models[best]
    torch.save({"state_dict": m.state_dict(), "d_latent": int(rows["train"]["z"].shape[1]), "hidden": 256,
                "dropout": 0.1, "eps": eps}, out / "mlp_model.pt")
    np.savez_compressed(out / "test_predictions.npz", p=R.mlp_predict(m, rows["test"]), **{k: rows["test"][k] for k in ("y", "H", "clip", "cand", "j", "min_d")})
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(per, indent=1))


def cmd_train_dist(a):
    """Fit DistLogistic on train (no regularisation tuning), report val/test like the MLP."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    src = Path(a.data)
    files = {s: sorted((src / s).glob("clip_*.npz")) for s in ("train", "val", "test")}
    eps = json.loads((Path(a.out) / "results.json").read_text())["eps"]  # same labels as the MLP
    rows = {s: label_rows(fs, eps) for s, fs in files.items()}
    d_scale = float(eps)
    lr = LogisticRegression(C=1e3, max_iter=5000).fit(_dist_feats(rows["train"], d_scale), rows["train"]["y"])
    w = np.concatenate([lr.coef_[0], lr.intercept_])

    def ece(p, y, n=10):
        b = np.clip((p * n).astype(int), 0, n - 1)
        return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(n) if (b == i).any()))

    res = {"label_mode": LABEL_MODE, "w": w.tolist(), "d_scale": d_scale, "features": "d/eps, log H, d/eps*log H, 1 (d = ||g - z0||^2)"}
    for s in ("val", "test"):
        p = lr.predict_proba(_dist_feats(rows[s], d_scale))[:, 1]; y = rows[s]["y"]
        res[s] = {"auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                  "brier": float(brier_score_loss(y, p)), "ece": ece(p, y)}
    out = Path(a.out)
    torch.save({"type": "dist_logistic", "w": w, "d_scale": d_scale, "eps": eps}, out / "dist_model.pt")
    (out / "dist_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_eval(a):
    pred = load_any_predictor(a.predictor) if a.predictor else None
    cfg, solver, mean, std = build(a.model_seed, a.planner_seed, predictor=pred, lam=a.lam)
    tf = de.img_transform(cfg.img_size)
    ds = make_reader(a.split, a.n_clips)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in solver.level_solvers.values()]
    log = out / "clips.jsonl"
    done = {json.loads(l)["clip"] for l in log.read_text().splitlines()} if log.exists() else set()
    for k in range(len(ds)):
        if k in done:
            continue
        for gen, seed in gens:
            gen.manual_seed(seed + k)  # identical to droid_eval
        obs, actions, _s, _r, env_info = ds[k]
        t0 = time.time()
        planned = solver(plan_info(tf, obs["visual"], mean.shape[0]), steps_taken=0,
                         eval_budget=int(cfg.eval_budget))["actions"][0].cpu().numpy()
        ep = {"planned": torch.tensor(planned * std + mean, dtype=torch.float32),
              "gt": actions[: planned.shape[0]].to(torch.float32),
              "grasp_pos": env_info["grasp_pos"] if env_info else None}
        m = de.clip_metrics(**ep)
        r = solver.last
        rec = {"clip": k, "frechet_skill": float(m["frechet/skill"]), "ate_xyz": float(m["ate/end_distance_xyz"]),
               "base_idx": r["base_idx"], "sel_idx": r["sel_idx"], "changed": r["sel_idx"] != r["base_idx"],
               "C": r["C"].tolist(), "R_mean": r["R"].mean(-1).tolist() if "R" in r else None,
               "wall_clock": time.time() - t0}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"clip {k}: frechet={rec['frechet_skill']:.4f} changed={rec['changed']}", flush=True)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    summ = {"n": len(recs), "frechet_skill_mean": float(np.mean([r["frechet_skill"] for r in recs])),
            "ate_xyz_mean": float(np.mean([r["ate_xyz"] for r in recs])),
            "n_changed": int(sum(r["changed"] for r in recs)), "lam": a.lam, "model_seed": a.model_seed,
            "planner_seed": a.planner_seed, "split": a.split}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--split", choices=("train", "val", "test"), required=True)
    c.add_argument("--n-clips", type=int, default=160)
    c.add_argument("--num-samples", type=int, default=16, help="level-2 GD samples for label collection")
    c.add_argument("--model-seed", type=int, default=1)
    c.add_argument("--planner-seed", type=int, default=1)
    c.add_argument("--out", default="outputs/rasr/droid/p1_candidates")
    t = sub.add_parser("train")
    t.add_argument("--data", default="outputs/rasr/droid/p1_candidates")
    t.add_argument("--out", default="outputs/rasr/droid/p1_reachability")
    t.add_argument("--mlp-seeds", type=int, default=3)
    t.add_argument("--label", choices=("any", "aligned"), default="any")
    td = sub.add_parser("train-dist")
    td.add_argument("--data", default="outputs/rasr/droid/p1_candidates")
    td.add_argument("--out", default="outputs/rasr/droid/p1_reachability")
    td.add_argument("--label", choices=("any", "aligned"), default="any")
    e = sub.add_parser("eval")
    e.add_argument("--split", choices=("val", "test"), default="test")
    e.add_argument("--n-clips", type=int, default=24)
    e.add_argument("--lam", type=float, default=0.0)
    e.add_argument("--predictor", default=None)
    e.add_argument("--model-seed", type=int, default=1)
    e.add_argument("--planner-seed", type=int, default=1)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    global LABEL_MODE
    LABEL_MODE = getattr(a, "label", "any")
    torch.use_deterministic_algorithms(True, warn_only=True)
    {"collect": cmd_collect, "train": cmd_train, "train-dist": cmd_train_dist, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
