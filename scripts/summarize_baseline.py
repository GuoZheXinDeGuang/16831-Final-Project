#!/usr/bin/env python
"""Aggregate P0 baseline shards -> summary.json, episodes.csv, and trajectory plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_jsonl(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    ep_files = sorted(run.glob("**/episodes.jsonl"))
    eps = sorted(load_jsonl(ep_files), key=lambda r: r["task"])
    calls = load_jsonl(sorted(run.glob("**/calls.jsonl")))
    tasks = [e["task"] for e in eps]
    assert len(tasks) == len(set(tasks)), "duplicate tasks across shards"

    n = len(eps)
    k = sum(e["success"] for e in eps)
    lo, hi = wilson(k, n)
    succ_steps = [e["steps_to_success"] for e in eps if e["success"]]
    pc1 = np.array([e["predictor_calls"]["1"] for e in eps])
    pc2 = np.array([e["predictor_calls"].get("2", 0) for e in eps])
    pr1 = np.array([e["predictor_rows"]["1"] for e in eps])
    pr2 = np.array([e["predictor_rows"].get("2", 0) for e in eps])
    summary = {
        "run": str(run),
        "n_episodes": n,
        "n_success": int(k),
        "success_rate": 100.0 * k / max(n, 1),
        "success_rate_wilson95": [100 * lo, 100 * hi],
        "failed_tasks": [e["task"] for e in eps if not e["success"]],
        "steps_to_success_mean": float(np.mean(succ_steps)) if succ_steps else None,
        "planning_wall_clock_per_episode_mean_s": float(np.mean([e["planning_wall_clock"] for e in eps])),
        "episode_wall_clock_mean_s": float(np.mean([e["episode_wall_clock"] for e in eps])),
        "planning_calls_per_episode": float(np.mean([e["n_planning_calls"] for e in eps])),
        "predictor_calls_per_episode": {"l1": float(pc1.mean()), "l2": float(pc2.mean())},
        "predictor_rows_per_episode": {"l1": float(pr1.mean()), "l2": float(pr2.mean())},
        "gd_iters_per_episode": {
            "l1": float(np.mean([e["gd_iters"]["1"] for e in eps])),
            "l2": float(np.mean([e["gd_iters"].get("2", 0) for e in eps])),
        },
        "per_call_wall_clock_mean_s": {
            "total": float(np.mean([c["wall_clock"] for c in calls])),
            "l1": float(np.mean([c["l1_wall_clock"] for c in calls])),
            "l2_when_solved": float(np.mean([c["l2_wall_clock"] for c in calls if c["l2_horizon"] > 0] or [0])),
        },
        "by_closed_door": {},
    }
    for door in sorted({str(e["closed_door"]) for e in eps}):
        sub = [e for e in eps if str(e["closed_door"]) == door]
        summary["by_closed_door"][door] = {"n": len(sub), "success_rate": 100.0 * sum(e["success"] for e in sub) / len(sub)}
    run_cfgs = [json.loads(p.read_text()) for p in sorted(run.glob("**/run_config.json"))]
    if run_cfgs:
        rc = run_cfgs[0]
        summary["provenance"] = {k: rc.get(k) for k in ("checkpoint", "planner_seed", "model_seed", "tasks_file", "tasks_file_seed", "env_seed_base", "deterministic", "hjepa_git_commit", "torch", "cuda_device")}
        summary["provenance"]["commands"] = [r["command"] for r in run_cfgs]
    (run / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(run / "episodes.csv", "w", newline="") as f:
        cols = ["task", "closed_door", "success", "steps_to_success", "episode_wall_clock", "planning_wall_clock",
                "n_planning_calls", "final_dist_to_target", "min_dist_to_target"]
        w = csv.writer(f)
        w.writerow(cols + ["pred_calls_l1", "pred_calls_l2", "pred_rows_l1", "pred_rows_l2"])
        for e in eps:
            w.writerow([e[c] for c in cols] + [e["predictor_calls"]["1"], e["predictor_calls"].get("2", 0),
                                               e["predictor_rows"]["1"], e["predictor_rows"].get("2", 0)])
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))

    if not args.no_plots:
        plot_trajectories(run, eps)


ANT_MEDIUM_MAP = [  # ogbench/locomaze/maze.py, maze_type="medium"
    [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]]


def plot_trajectories(run: Path, eps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz = {int(p.stem.split("_")[1]): p for p in run.glob("**/ep_*.npz")}
    n = len(eps)
    cols = 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    cfg_p = next(run.glob("**/run_config.json"), None)
    is_ant = cfg_p is not None and json.loads(cfg_p.read_text()).get("config_name", "").startswith("ant")
    for ax, e in zip(np.ravel(axes), eps):
        d = np.load(npz[e["task"]])
        xy = d["xy"]
        if is_ant:  # OGBench medium maze: cell (i, j) centred at (4j - 4, 4i - 4), unit 4
            for i, row in enumerate(ANT_MEDIUM_MAP):
                for j, w in enumerate(row):
                    if w:
                        ax.add_patch(plt.Rectangle((4 * j - 6, 4 * i - 6), 4, 4, color="0.85", lw=0))
        else:
            ax.plot([112, 112], [0, 224], color="0.7", lw=0.8)
            ax.plot([0, 224], [112, 112], color="0.7", lw=0.8)
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue" if e["success"] else "tab:red", lw=1)
        ax.scatter(*xy[0], s=8, color="k")
        ax.scatter(*d["goal_xy"], s=20, marker="*", color="tab:green")
        ce = d["call_env_step"]
        ax.scatter(xy[ce, 0], xy[ce, 1], s=2, color="0.3")
        if is_ant:
            ax.set_xlim(-6, 26)
            ax.set_ylim(-6, 26)
            ax.set_aspect("equal")
            ax.set_title(f"t{e['task']} {'ok' if e['success'] else 'FAIL'}", fontsize=5)
        else:
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)
            ax.set_title(f"t{e['task']} {'ok' if e['success'] else 'FAIL'} door={e['closed_door']}", fontsize=5)
    env_name = "Visual AntMaze medium" if is_ant else "FourRoom"
    fig.suptitle(f"H-JEPA L2 baseline trajectories ({env_name}, blue=success, red=fail; dots=planning calls)", fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "trajectories.png", dpi=150)
    print("wrote", run / "trajectories.png")


if __name__ == "__main__":
    main()
