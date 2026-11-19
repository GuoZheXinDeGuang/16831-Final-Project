#!/usr/bin/env python
"""P1: collect controller-conditioned subgoal execution attempts by branching from P0 snapshots.

For every planning call of every baseline episode in --baseline-run (or a stride of them), and
for every low-level budget B, restore the env and run the frozen level-1 MPC towards the fixed
level-2 subgoal the baseline produced at that call, for --h-max env steps.

Writes <out>/attempts_task<XXX>.npz (one per task) + <out>/collect_config.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr.common import CKPTS, load_eval_cfg, prepare_eval  # noqa: E402

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import planning_eval as pe  # noqa: E402
from rasr.branch import DEFAULT_BUDGETS, Level1Controller, run_attempt, setup_task_world  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run", required=True, help="P0 run dir (contains shard_*/ep_XXX_snap.pkl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--call-stride", type=int, default=1)
    ap.add_argument("--h-max", type=int, default=30)
    ap.add_argument("--budgets", default="B0,B1,B2")
    ap.add_argument("--budget-spec", default=None,
                    help="override budgets, e.g. 'B0:2x4,B1:10x20,B2:90x25' (name:n_steps x num_samples)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alt-from", default=None,
                    help="rasr_calls.jsonl with alt_subgoals (dump_alternatives run of the SAME trajectory): "
                         "execute the baseline subgoal (alt=-1) and each logged non-selected candidate")
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    base = Path(args.baseline_run)
    rcfg = json.loads(next(base.glob("**/run_config.json")).read_text())
    episodes = {}
    for p in base.glob("**/episodes.jsonl"):
        for l in p.read_text().splitlines():
            if l.strip():
                e = json.loads(l)
                e["_dir"] = p.parent
                episodes[e["task"]] = e
    tasks = sorted(episodes) if args.tasks == "all" else [int(t) for t in args.tasks.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [b for b in DEFAULT_BUDGETS if b.name in args.budgets.split(",")]
    if args.budget_spec:
        from rasr.branch import Budget
        budgets = [Budget(n, int(v.split("x")[0]), int(v.split("x")[1]))
                   for n, v in (kv.split(":") for kv in args.budget_spec.split(","))]
    (out / "collect_config.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "baseline_run": str(base), "h_max": args.h_max,
        "budgets": [b.__dict__ for b in budgets], "call_stride": args.call_stride, "seed": args.seed,
        "tasks_file": rcfg["tasks_file"], "checkpoint": rcfg["checkpoint"],
    }, indent=2))

    cfg = load_eval_cfg(rcfg["config_name"], {"seed": rcfg["planner_seed"], "policy": rcfg["checkpoint"]})
    payload = torch.load(rcfg["tasks_file"], map_location="cpu", weights_only=False)
    cfg.eval.num_eval = len(payload["data"])
    cfg, world_cfg, transform, process, image_shape = prepare_eval(cfg)
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    policy = pe._build_policy(cfg, process, transform, model=None)
    world = swm.World(**{**world_cfg, "num_envs": 1}, image_shape=image_shape)
    world.set_policy(policy)
    ctrl = Level1Controller(policy, cfg, budgets=budgets, seed=args.seed)

    alts = {}
    if args.alt_from:
        for l in Path(args.alt_from).read_text().splitlines():
            r = json.loads(l)
            if "alt_subgoals" in r:
                alts[(r["task"], r["call_idx"])] = r
    for task in tasks:
        dst = out / f"attempts_task{task:03d}.npz"
        if dst.exists():
            print(f"task {task}: exists, skip")
            continue
        e = episodes[task]
        ep = np.load(e["_dir"] / f"ep_{task:03d}.npz")
        snaps = pickle.loads((e["_dir"] / f"ep_{task:03d}_snap.pkl").read_bytes())
        n_calls = len(ep["call_env_step"])
        assert len(snaps) == n_calls, (len(snaps), n_calls)
        goal_hwc = setup_task_world(world, policy, cfg, payload, task, e.get("env_seed"), callables, process)[0, -1]
        recs = []
        t_task = time.time()
        with sdpa_kernel(SDPBackend.MATH):
            for c in range(0, n_calls, args.call_stride):
                goals = [(ep["subgoal"][c], -1, np.nan, -1)]
                if args.alt_from:
                    a = alts.get((task, c))
                    if a is None:
                        continue
                    goals += [(np.asarray(g, np.float32), j, a["alt_R"][j], a["alt_cost_rank"][j])
                              for j, g in enumerate(a["alt_subgoals"])]
                for g, alt_j, alt_r, alt_rank in goals:
                    for bi, b in enumerate(budgets):
                        gen_seed = args.seed * 1_000_003 + task * 10_007 + c * 101 + bi + 17 * (alt_j + 1)
                        r = run_attempt(world, ctrl, snaps[c], goal_hwc, g, b.name, args.h_max, gen_seed)
                        recs.append({"call_idx": c, "budget": bi, "r": r, "g": g, "alt": alt_j,
                                     "alt_R": alt_r, "alt_rank": alt_rank})
        n = len(recs)
        solves_per = [len(x["r"]["solves"]) for x in recs]
        np.savez_compressed(
            dst,
            task=np.full(n, task),
            call_idx=np.array([x["call_idx"] for x in recs]),
            env_step=np.array([int(ep["call_env_step"][x["call_idx"]]) for x in recs]),
            budget_idx=np.array([x["budget"] for x in recs]),
            budget_n_steps=np.array([budgets[x["budget"]].n_steps for x in recs]),
            budget_num_samples=np.array([budgets[x["budget"]].num_samples for x in recs]),
            z1=np.stack([ep["call_obs_embed_1"][x["call_idx"]] for x in recs]),
            z2=np.stack([ep["call_obs_embed_2"][x["call_idx"]] for x in recs]),
            goal_z2=np.stack([ep["call_goal_embed_2"][x["call_idx"]] for x in recs]),
            subgoal=np.stack([x["g"] for x in recs]),
            alt=np.array([x["alt"] for x in recs]),
            alt_R_logged=np.array([x["alt_R"] for x in recs], dtype=np.float32),
            alt_cost_rank=np.array([x["alt_rank"] for x in recs]),
            subgoal_from_upper_plan=np.array([bool(ep["subgoal_from_upper_plan"][x["call_idx"]]) for x in recs]),
            d0=np.array([x["r"]["d0"] for x in recs], dtype=np.float32),
            dists=np.stack([x["r"]["dists"] for x in recs]),
            xy0=np.stack([x["r"]["xy0"] for x in recs]),
            xys=np.stack([x["r"]["xys"] for x in recs]),
            env_goal_reached=np.stack([x["r"]["env_goal_reached"] for x in recs]),
            wall_clock=np.array([x["r"]["wall_clock"] for x in recs]),
            render_err=np.array([x["r"]["render_err"] for x in recs]),
            n_solves=np.array(solves_per),
            # per-solve compute, padded to max solves (h_max / receding)
            solve_wall_clock=_pad([[s["wall_clock"] for s in x["r"]["solves"]] for x in recs]),
            solve_gd_iters=_pad([[s["gd_iters"] for s in x["r"]["solves"]] for x in recs]),
            solve_pred_calls_l1=_pad([[s["pred_calls"][1] for s in x["r"]["solves"]] for x in recs]),
            solve_pred_rows_l1=_pad([[s["pred_rows"][1] for s in x["r"]["solves"]] for x in recs]),
            goal_xy=np.asarray(ep["goal_xy"]),
            episode_success=np.full(n, bool(e["success"])),
        )
        print(f"task {task}: {n} attempts in {time.time() - t_task:.1f}s, max render err "
              f"{max(x['r']['render_err'] for x in recs):.0f}", flush=True)
    world.close()


def _pad(rows):
    m = max(len(r) for r in rows)
    return np.array([list(r) + [np.nan] * (m - len(r)) for r in rows], dtype=np.float64)


if __name__ == "__main__":
    main()
