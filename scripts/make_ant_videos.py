#!/usr/bin/env python
"""Top-down AntMaze comparison videos from logged episodes (no re-simulation).

Each panel is one method on the same task / planner seed: OGBench medium maze, the ant's logged xy trail, the goal
(green ring = env success radius), and the current level-2 subgoal decoded to xy with an analysis-only kNN probe
(latent -> xy over real frames of the RASR train/val source episodes; never used by the planner). The subgoal is drawn
red while the repair objective is active (stall trigger fired). Writes <out>/<name>.mp4 (+ a last-frame .png) and,
with --single, one mp4 per method.

  python scripts/make_ant_videos.py --task 7 --seed 42 \
      --method "H-JEPA baseline=test/b1_baseline" --method "Ours (final)=test/repair_noR_b50" --name task007
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "outputs/rasr/ant/v2"
MAZE = [  # ogbench/locomaze/maze.py, maze_type="medium"; cell (i, j) centred at (4j - 4, 4i - 4), unit 4
    [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
GOAL_TOL = 1.0
COL = {"ok": "#1a7f37", "fail": "#c62828", "trail": "#1f6feb", "sub": "#d633c4", "rep": "#e8590c"}


class Probe:
    """kNN latent -> xy over level-2 latents of real frames (analysis only)."""

    def __init__(self, dirs, k=5):
        Z, XY = [], []
        for d in dirs:
            for f in sorted(Path(d).glob("**/ep_*.npz")):
                e = np.load(f)
                if "step_embed_2" in e.files:
                    Z.append(e["step_embed_2"]); XY.append(e["xy"][: len(e["step_embed_2"])])
                else:
                    Z.append(e["call_obs_embed_2"]); XY.append(e["xy"][e["call_env_step"]])
        self.Z = np.concatenate(Z).astype(np.float32)
        self.XY = np.concatenate(XY)
        self.k = k

    def __call__(self, z):
        import torch
        z = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)
        d = torch.cdist(z, torch.as_tensor(self.Z))
        i = d.topk(self.k, largest=False).indices.numpy()
        return self.XY[i].mean(1)


def load_episode(run: Path, task: int):
    d = run / f"shard_{task}"
    rec = json.loads((d / "episodes.jsonl").read_text().splitlines()[0])
    ep = np.load(d / f"ep_{task:03d}.npz")
    calls = []
    if (d / "rasr_calls.jsonl").exists():
        calls = [json.loads(l) for l in (d / "rasr_calls.jsonl").read_text().splitlines() if l.strip()]
    return rec, ep, calls


def draw_maze(ax):
    import matplotlib.pyplot as plt
    for i, row in enumerate(MAZE):
        for j, w in enumerate(row):
            if w:
                ax.add_patch(plt.Rectangle((4 * j - 6, 4 * i - 6), 4, 4, color="#9aa0a6", lw=0))
    ax.set_xlim(-2.2, 22.2); ax.set_ylim(-2.2, 22.2); ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", action="append", required=True, help="'Label=<split>/<method>' under outputs/rasr/ant/v2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=str(V2 / "videos"))
    ap.add_argument("--stride", type=int, default=2, help="env steps per frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--single", action="store_true", help="also write one mp4 per method")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probe = Probe([V2 / "train/src/seed42", V2 / "val/src/seed42"])
    runs = []
    for m in a.method:
        label, rel = m.rsplit("=", 1)
        rec, ep, calls = load_episode(V2 / rel / f"seed{a.seed}", a.task)
        sub_xy = probe(ep["subgoal"])
        rep = [bool(c.get("repair")) for c in calls] if calls else [False] * len(ep["call_env_step"])
        trig = [c.get("trigger") for c in calls] if calls else [None] * len(ep["call_env_step"])
        runs.append(dict(label=label, rec=rec, ep=ep, sub=sub_xy, rep=rep, trig=trig))

    T = max(len(r["ep"]["xy"]) for r in runs)
    steps = list(range(0, T, a.stride)) + [T - 1] * a.fps  # hold the last frame for one second

    def render(panel_runs, t, fig, axes):
        for ax, r in zip(axes, panel_runs):
            ax.clear(); draw_maze(ax)
            ep, rec = r["ep"], r["rec"]
            tt = min(t, len(ep["xy"]) - 1)
            ce = ep["call_env_step"]
            ci = int(np.searchsorted(ce, tt, side="right") - 1)
            succ = rec["success"] and tt >= rec["steps_to_success"]
            done_fail = (not rec["success"]) and tt >= len(ep["xy"]) - 1
            g = ep["goal_xy"]
            ax.add_patch(plt.Circle(g, GOAL_TOL, fill=False, color=COL["ok"], lw=2))
            ax.plot(*g, marker="*", ms=12, color=COL["ok"])
            ax.plot(ep["xy"][: tt + 1, 0], ep["xy"][: tt + 1, 1], color=COL["trail"], lw=1.6, alpha=0.9)
            ax.plot(*ep["xy"][0], "o", ms=5, color="k")
            ax.plot(*ep["xy"][tt], "o", ms=9, color=COL["trail"], mec="white", mew=1.5)
            repair = ci < len(r["rep"]) and r["rep"][ci]
            sx, sy = r["sub"][ci]
            ax.plot(sx, sy, marker="X", ms=11, color=COL["rep"] if repair else COL["sub"], mec="white", mew=1)
            ax.plot([ep["xy"][tt, 0], sx], [ep["xy"][tt, 1], sy], ls=":", lw=1, color=COL["rep"] if repair else COL["sub"])
            status = "GOAL REACHED" if succ else ("FAILED (time out)" if done_fail else ("REPAIR: stuck → explore" if repair else ""))
            sc = COL["ok"] if succ else (COL["fail"] if done_fail else (COL["rep"] if repair else "k"))
            ax.set_title(r["label"], fontsize=12, fontweight="bold", pad=4)
            ax.text(0.5, -0.04, f"step {tt:3d}/350   {status}", transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color=sc, fontweight="bold" if status else "normal")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def write(panel_runs, path):
        n = len(panel_runs)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.1), dpi=110)
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.08, wspace=0.05)
        fig.suptitle(f"Visual AntMaze — task {a.task}, planner seed {a.seed}   (★ goal, × current subgoal, "
                     f"orange × = repair active)", fontsize=9)
        frames = [render(panel_runs, t, fig, axes) for t in steps]
        h, w = frames[0].shape[:2]
        h2, w2 = h - h % 2, w - w % 2
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}",
               "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
               "-movflags", "+faststart", str(path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
        p.stdin.close(); p.wait()
        plt.imsave(str(path.with_suffix(".png")), frames[-1])
        plt.close(fig)
        print("wrote", path, f"({len(frames)} frames)")

    write(runs, out / f"{a.name}.mp4")
    if a.single:
        for r in runs:
            slug = "".join(ch if ch.isalnum() else "_" for ch in r["label"]).strip("_").lower()
            write([r], out / f"{a.name}_{slug}.mp4")


if __name__ == "__main__":
    main()
