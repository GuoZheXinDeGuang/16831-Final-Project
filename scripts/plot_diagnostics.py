#!/usr/bin/env python
"""Figures for the report/website from real logs:
  fig_failure_mode.png   agent path + decoded subgoals (xy probe, analysis only) for a success and a failure
  fig_goal_distance.png  real latent goal distance per planning call, successful vs failed P0 episodes
  fig_subgoal_reach.png  subgoal reach rate within the 10-step window, successful vs failed episodes
  fig_budget_cells.png   P1 positive rate per (H, B) cell (test split)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rasr.xy_probe import XYProbe  # noqa: E402

P0 = Path("outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42")
OUT = Path("outputs/rasr/figures")
INK, MUTED, OK, BAD, SUB = "#1d2733", "#8a94a3", "#2f7fd1", "#d1495b", "#b03fb5"


def walls(ax):
    # schematic FourRoom layout: walls at x=112 / y=112, 16 px doors centred at 56 and 168
    k = dict(color="#222", lw=3, solid_capstyle="butt")
    for a, b in ((0, 48), (64, 160), (176, 224)):
        ax.plot([112, 112], [a, b], **k)
        ax.plot([a, b], [112, 112], **k)
    ax.set_xlim(0, 224); ax.set_ylim(224, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def eps():
    out = {}
    for p in P0.glob("shard_*/episodes.jsonl"):
        for l in p.read_text().splitlines():
            e = json.loads(l)
            out[e["task"]] = e
    return out


def npz(task):
    return np.load(next(P0.glob(f"shard_*/ep_{task:03d}.npz")))


def fig_failure_mode(E):
    probe = XYProbe(["outputs/rasr/p1_source/train", "outputs/rasr/p1_source/val"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, task, title in ((axes[0], 5, "Success (task 5)"), (axes[1], 9, "Failure (task 9): corner attractor")):
        ep = npz(task)
        walls(ax)
        xy = ep["xy"]
        ax.plot(xy[:, 0], xy[:, 1], color=OK if E[task]["success"] else BAD, lw=1.6, label="agent path")
        sx, _ = probe(ep["subgoal"])
        ce = ep["call_env_step"]
        sc = ax.scatter(sx[:, 0], sx[:, 1], c=np.arange(len(sx)), cmap="viridis", s=28, marker="x",
                        label="decoded subgoal per call", zorder=3)
        for c in range(0, len(ce), 3):
            ax.annotate("", xy=sx[c], xytext=xy[ce[c]], arrowprops=dict(arrowstyle="->", color=SUB, lw=0.8, alpha=0.7))
        ax.add_patch(plt.Circle(ep["goal_xy"], 16, fill=False, color="#2a9d4b", lw=1.8))
        ax.scatter(*xy[0], color=INK, s=30, zorder=4)
        ax.set_title(title, fontsize=11)
    fig.colorbar(sc, ax=axes, shrink=0.7, label="planning call index")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.savefig(OUT / "fig_failure_mode.png", dpi=160, bbox_inches="tight")


def fig_goal_distance(E):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for t, e in sorted(E.items()):
        ep = npz(t)
        d = ((ep["call_obs_embed_2"] - ep["call_goal_embed_2"]) ** 2).sum(-1)
        ax.plot(np.arange(len(d)), d, color=OK if e["success"] else BAD, alpha=0.35 if e["success"] else 0.9,
                lw=1 if e["success"] else 1.8)
    ax.plot([], [], color=OK, label=f"successful episodes (n={sum(e['success'] for e in E.values())})")
    ax.plot([], [], color=BAD, lw=1.8, label=f"failed episodes (n={sum(not e['success'] for e in E.values())})")
    ax.set_xlabel("planning call (every 10 env steps)")
    ax.set_ylabel(r"$\|enc_2(o_t) - enc_2(goal)\|^2$")
    ax.set_title("Real latent distance to the goal, P0 baseline (50 test tasks)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "fig_goal_distance.png", dpi=160, bbox_inches="tight")


def fig_subgoal_reach():
    s = json.loads((P0 / "subgoal_outcomes_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    vals = [100 * s["reach_rate_successful_episodes"], 100 * s["reach_rate_failed_episodes"]]
    bars = ax.bar(["successful\nepisodes", "failed\nepisodes"], vals, color=[OK, BAD], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("subgoals reached within 10 steps (%)")
    ax.set_title("Subgoal execution in P0 episodes", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig_subgoal_reach.png", dpi=160, bbox_inches="tight")


def fig_budget_cells():
    r = json.loads(Path("outputs/rasr/p1_reachability/results.json").read_text())
    cells = r["label_balance_by_cell"]["test"]
    H = [5, 10, 20, 30]
    B = {"2": "B0 (2 it × 4)", "10": "B1 (10 it × 20)", "90": "B2 (90 it × 100, default)"}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (it, lab), col in zip(B.items(), ["#c9a227", "#5b8def", "#1d2733"]):
        ax.plot(H, [100 * cells[f"H{h}_iters{it}"]["pos_rate"] for h in H], marker="o", label=lab, color=col)
    ax.set_xlabel("execution horizon H (env steps)")
    ax.set_ylabel("subgoal reached (%)")
    ax.set_xticks(H)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="low-level MPC budget B", title_fontsize=8)
    ax.set_title("Controller-conditioned reachability (P1 test, 1 650 attempts)", fontsize=10)
    fig.savefig(OUT / "fig_budget_cells.png", dpi=160, bbox_inches="tight")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E = eps()
    fig_failure_mode(E)
    fig_goal_distance(E)
    fig_subgoal_reach()
    fig_budget_cells()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
