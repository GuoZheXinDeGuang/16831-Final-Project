"""Reachability dataset, kNN baseline and MLP predictor R(z_t, g, H, B) -> P(reachable).

Label: Y(H, B) = 1  iff  min_{1<=k<=H} ||enc2(o_{t+k}) - g||^2 <= eps,
where the controller is the frozen level-1 H-JEPA MPC with budget B steering towards fixed g,
and eps is calibrated on the TRAIN split only (see calibrate_eps).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

H_VALUES = (5, 10, 20, 30)


# ----------------------------------------------------------------------------- data
def load_attempts(dirs: list[Path]) -> dict[str, np.ndarray]:
    files = sorted(f for d in dirs for f in Path(d).glob("attempts_task*.npz"))
    if not files:
        raise FileNotFoundError(f"no attempts in {dirs}")
    parts = [dict(np.load(f)) for f in files]
    for p in parts:  # per-task goal xy, broadcast to attempts
        p["goal_xy_task"] = np.repeat(np.asarray(p["goal_xy"])[None], len(p["task"]), 0)
    keys = [k for k in parts[0] if k != "goal_xy"]
    out = {}
    for k in keys:
        arrs = [p[k] for p in parts]
        if arrs[0].ndim == 2 and k.startswith("solve_"):
            m = max(a.shape[1] for a in arrs)
            arrs = [np.pad(a, ((0, 0), (0, m - a.shape[1])), constant_values=np.nan) for a in arrs]
        out[k] = np.concatenate(arrs)
    return out


def calibrate_eps(baseline_dirs: list[Path], radius_px: float = 8.0, max_pairs: int = 2_000_000, seed: int = 0) -> dict:
    """eps = median squared level-2 latent distance between real observations (same episode)
    whose agent positions are within radius_px. Uses only the given (train) baseline episodes."""
    rng = np.random.default_rng(seed)
    ds = []
    for d in baseline_dirs:
        for f in sorted(Path(d).glob("**/ep_*.npz")):
            ep = np.load(f)
            e2, xy = ep["step_embed_2"], ep["xy"]
            D = ((e2[:, None] - e2[None]) ** 2).sum(-1)
            X = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
            iu = np.triu_indices(len(xy), 1)
            m = X[iu] < radius_px
            ds.append(D[iu][m])
    ds = np.concatenate(ds)
    if len(ds) > max_pairs:
        ds = rng.choice(ds, max_pairs, replace=False)
    return {"eps": float(np.median(ds)), "radius_px": radius_px, "n_pairs": int(len(ds)),
            "pct10": float(np.percentile(ds, 10)), "pct90": float(np.percentile(ds, 90))}


def expand_rows(att: dict, eps: float, h_values=None) -> dict[str, np.ndarray]:
    """One row per (attempt, H). Features use only planner-visible quantities."""
    h_values = H_VALUES if h_values is None else h_values
    n = len(att["task"])
    rows = {k: [] for k in ("task", "call_idx", "attempt_id", "H", "n_steps", "num_samples", "y", "min_d",
                            "steps_to_reach", "cost_pred_rows", "cost_wall_clock", "env_goal")}
    feats_z, feats_g = [], []
    receding = 10
    for i in range(n):
        d = att["dists"][i]
        for H in h_values:
            md = float(d[:H].min())
            hit = np.nonzero(d[:H] <= eps)[0]
            rows["task"].append(att["task"][i])
            rows["call_idx"].append(att["call_idx"][i])
            rows["attempt_id"].append(i)
            rows["H"].append(H)
            rows["n_steps"].append(att["budget_n_steps"][i])
            rows["num_samples"].append(att["budget_num_samples"][i])
            rows["y"].append(int(len(hit) > 0))
            rows["min_d"].append(md)
            rows["steps_to_reach"].append(int(hit[0]) + 1 if len(hit) else -1)
            n_solves = int(np.ceil(H / receding))
            rows["cost_pred_rows"].append(float(np.nansum(att["solve_pred_rows_l1"][i][:n_solves])))
            rows["cost_wall_clock"].append(float(np.nansum(att["solve_wall_clock"][i][:n_solves])))
            rows["env_goal"].append(bool(att["env_goal_reached"][i][:H].any()))
            feats_z.append(att["z2"][i])
            feats_g.append(att["subgoal"][i])
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["z"] = np.stack(feats_z).astype(np.float32)
    out["g"] = np.stack(feats_g).astype(np.float32)
    return out


def controller_features(rows) -> np.ndarray:
    return np.stack([np.log(rows["H"]), np.log(rows["n_steps"]), np.log(rows["num_samples"])], -1).astype(np.float32)


# ----------------------------------------------------------------------------- metrics
def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def all_metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": roc_auc(y, p), "pr_auc": pr_auc(y, p), "brier": brier(y, p), "ece": ece(y, p),
            "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "n": int(len(y)),
            "pos_rate": float(np.mean(y))}


def bootstrap_ci_by_task(y, p, task, fn, n_boot=1000, seed=0):
    """95% CI resampling start-goal tasks (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    ut = np.unique(task)
    idx_by_t = {t: np.nonzero(task == t)[0] for t in ut}
    vals = []
    for _ in range(n_boot):
        ts = rng.choice(ut, len(ut), replace=True)
        ii = np.concatenate([idx_by_t[t] for t in ts])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(fn(y[ii], np.clip(p[ii], 1e-6, 1 - 1e-6)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ----------------------------------------------------------------------------- kNN
@dataclass
class KNNReachability:
    """Controller-conditioned kNN: neighbours are searched among TRAIN attempts with the same
    (H, B) cell, in standardized [z_t, g] space. P = (sum y + alpha) / (k + 2 alpha)."""

    k: int = 25
    alpha: float = 1.0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    cells: dict = field(default_factory=dict)

    def _x(self, rows):
        x = np.concatenate([rows["z"], rows["g"]], -1)
        return (x - self.mean_) / self.std_

    def fit(self, rows):
        x = np.concatenate([rows["z"], rows["g"]], -1)
        self.mean_, self.std_ = x.mean(0), x.std(0) + 1e-6
        xs = self._x(rows)
        self.cells = {}
        for key in set(zip(rows["H"], rows["n_steps"], rows["num_samples"])):
            m = (rows["H"] == key[0]) & (rows["n_steps"] == key[1]) & (rows["num_samples"] == key[2])
            self.cells[tuple(int(v) for v in key)] = (torch.as_tensor(xs[m]), rows["y"][m].astype(np.float32))
        return self

    def predict_proba(self, rows, k: int | None = None) -> np.ndarray:
        k = k or self.k
        xs = torch.as_tensor(self._x(rows))
        p = np.zeros(len(xs), dtype=np.float32)
        for key, (xtr, ytr) in self.cells.items():
            m = (rows["H"] == key[0]) & (rows["n_steps"] == key[1]) & (rows["num_samples"] == key[2])
            if not m.any():
                continue
            d = torch.cdist(xs[m], xtr)
            kk = min(k, len(xtr))
            nn_idx = d.topk(kk, largest=False).indices.numpy()
            p[m] = (ytr[nn_idx].sum(1) + self.alpha) / (kk + 2 * self.alpha)
        return p


# ----------------------------------------------------------------------------- distance-logistic
class DistLogRegReachability(nn.Module):
    """Torch (differentiable) version of train_reachability.py's distance-logistic reference:
    x = [d, c, d * c] with d = log1p(||g - z||^2), c = [log H, log n_steps, log num_samples]; R = sigmoid(w.x + b)."""

    def __init__(self, coef, intercept):
        super().__init__()
        self.register_buffer("coef", torch.as_tensor(np.asarray(coef), dtype=torch.float32).reshape(-1))
        self.register_buffer("intercept", torch.as_tensor(float(np.asarray(intercept).reshape(-1)[0])))

    def forward(self, z, g, c):
        d = torch.log1p(((g - z) ** 2).sum(-1, keepdim=True))
        x = torch.cat([d, c, d * c], -1)
        return torch.sigmoid(x @ self.coef + self.intercept)


# ----------------------------------------------------------------------------- MLP
class ReachabilityMLP(nn.Module):
    """R(z_t, g, H, B): input [z, g, g - z, ||g - z||^2, log H, log n_steps, log num_samples]."""

    def __init__(self, d_latent: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        d_in = 3 * d_latent + 1 + 3
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("z_mean", torch.zeros(d_latent))
        self.register_buffer("z_std", torch.ones(d_latent))
        self.register_buffer("c_mean", torch.zeros(3))
        self.register_buffer("c_std", torch.ones(3))
        self.register_buffer("dist_scale", torch.ones(1))
        self.register_buffer("temperature", torch.ones(1))

    def set_normalization(self, z, g, c):
        zz = np.concatenate([z, g])
        self.z_mean.copy_(torch.as_tensor(zz.mean(0)))
        self.z_std.copy_(torch.as_tensor(zz.std(0) + 1e-6))
        self.c_mean.copy_(torch.as_tensor(c.mean(0)))
        self.c_std.copy_(torch.as_tensor(c.std(0) + 1e-6))
        self.dist_scale.fill_(float(np.mean(((g - z) ** 2).sum(-1))))

    def logits(self, z, g, c):
        dist = ((g - z) ** 2).sum(-1, keepdim=True) / self.dist_scale
        zn = (z - self.z_mean) / self.z_std
        gn = (g - self.z_mean) / self.z_std
        cn = (c - self.c_mean) / self.c_std
        return self.net(torch.cat([zn, gn, gn - zn, dist, cn], -1)).squeeze(-1)

    def forward(self, z, g, c):
        return torch.sigmoid(self.logits(z, g, c) / self.temperature)


def train_mlp(train_rows, val_rows, *, epochs=200, lr=1e-3, wd=1e-4, batch=512, hidden=256, dropout=0.1,
              patience=20, seed=0, device="cuda", log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ctr, cva = controller_features(train_rows), controller_features(val_rows)
    model = ReachabilityMLP(train_rows["z"].shape[1], hidden=hidden, dropout=dropout)
    model.set_normalization(train_rows["z"], train_rows["g"], ctr)
    model.to(device)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    ztr, gtr, ctr_t, ytr = T(train_rows["z"]), T(train_rows["g"]), T(ctr), T(train_rows["y"])
    zva, gva, cva_t, yva = T(val_rows["z"]), T(val_rows["g"]), T(cva), T(val_rows["y"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g).to(device)
        tl = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = bce(model.logits(ztr[b], gtr[b], ctr_t[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            vl = float(bce(model.logits(zva, gva, cva_t), yva))
        hist.append({"epoch": ep, "train_bce": tl / len(ytr), "val_bce": vl})
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0:
            log(f"epoch {ep}: train_bce={tl / len(ytr):.4f} val_bce={vl:.4f}")
    model.load_state_dict(best_state)
    return model, hist


def fit_temperature(model, rows, device="cuda") -> float:
    """Temperature scaling on the VALIDATION split (post-hoc calibration)."""
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    with torch.no_grad():
        lg = model.logits(T(rows["z"]), T(rows["g"]), T(controller_features(rows)))
    y = T(rows["y"])
    log_t = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp())
    model.temperature.fill_(t)
    return t


def mlp_predict(model, rows, device="cuda") -> np.ndarray:
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    model.eval()
    with torch.no_grad():
        return model(T(rows["z"]), T(rows["g"]), T(controller_features(rows))).cpu().numpy()


def save_json(path: Path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
