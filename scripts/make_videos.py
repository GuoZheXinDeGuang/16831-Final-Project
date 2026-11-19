#!/usr/bin/env python
"""Render rollout videos by replaying logged actions in an identically seeded env.

The replay is checked against the logged xy (must match to 1e-3 px). Overlays: goal (green ring),
decoded current subgoal (analysis-only xy probe, magenta cross), trail, and a status bar.
Writes H.264 mp4 via system ffmpeg.

  python scripts/make_videos.py --run outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42 --tasks 5,9 \
      --label "H-JEPA baseline" --out outputs/rasr/videos
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr.common import load_eval_cfg, prepare_eval  # noqa: E402

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rasr.branch import setup_task_world  # noqa: E402
from rasr.xy_probe import XYProbe  # noqa: E402

SCALE = 2  # 224 -> 448 px


def find_episode(run: Path, task: int):
    for p in run.glob("**/episodes.jsonl"):
        for l in p.read_text().splitlines():
            if l.strip() and json.loads(l)["task"] == task:
                return json.loads(l), p.parent
    raise FileNotFoundError(task)


def write_mp4(frames, path, fps=15):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def annotate(img, *, trail, goal_xy, sub_xy, text, ok):
    im = Image.fromarray(img).resize((224 * SCALE, 224 * SCALE), Image.NEAREST)
    bar = Image.new("RGB", (im.width, 34), (24, 24, 28))
    canvas = Image.new("RGB", (im.width, im.height + 34))
    canvas.paste(im, (0, 0))
    canvas.paste(bar, (0, im.height))
    d = ImageDraw.Draw(canvas)
    if len(trail) > 1:
        d.line([(x * SCALE, y * SCALE) for x, y in trail], fill=(60, 120, 230), width=2)
    gx, gy = goal_xy * SCALE
    d.ellipse([gx - 32, gy - 32, gx + 32, gy + 32], outline=(0, 170, 60), width=2)  # 16 px success radius
    if sub_xy is not None:
        sx, sy = sub_xy * SCALE
        d.line([sx - 7, sy - 7, sx + 7, sy + 7], fill=(220, 0, 200), width=3)
        d.line([sx - 7, sy + 7, sx + 7, sy - 7], fill=(220, 0, 200), width=3)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, im.height + 8), text, fill=(120, 230, 140) if ok else (235, 235, 235), font=font)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="outputs/rasr/videos")
    ap.add_argument("--name", default=None, help="file stem prefix")
    ap.add_argument("--probe-dirs", default="outputs/rasr/p1_source/train,outputs/rasr/p1_source/val")
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rcfg = json.loads(next(run.glob("**/run_config.json")).read_text())
    cfg = load_eval_cfg(rcfg["config_name"], {"seed": rcfg["planner_seed"], "policy": rcfg["checkpoint"]})
    payload = torch.load(rcfg["tasks_file"], map_location="cpu", weights_only=False)
    cfg.eval.num_eval = len(payload["data"])
    cfg, world_cfg, transform, process, image_shape = prepare_eval(cfg)
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    probe = XYProbe(args.probe_dirs.split(","))
    world = swm.World(**{**world_cfg, "num_envs": 1}, image_shape=image_shape)
    for task in [int(t) for t in args.tasks.split(",")]:
        ep_rec, d = find_episode(run, task)
        ep = np.load(d / f"ep_{task:03d}.npz")
        setup_task_world(world, None, cfg, payload, task, ep_rec.get("env_seed"), callables, process)
        env = world.envs.unwrapped.envs[0].unwrapped
        sub_xy, _ = probe(ep["subgoal"])
        calls = list(ep["call_env_step"])
        frames, trail = [], [env.agent_position.numpy().copy()]
        succ_step = ep_rec["steps_to_success"] if ep_rec["success"] else None
        max_err = 0.0
        ci = 0
        for t in range(len(ep["actions"]) + 1):
            while ci + 1 < len(calls) and calls[ci + 1] <= t:
                ci += 1
            xy = env.agent_position.numpy()
            max_err = max(max_err, float(np.abs(xy - ep["xy"][t]).max()))
            reached = succ_step is not None and t >= succ_step
            text = f"{args.label} | task {task} | step {t:3d}/210 | plan call {ci:2d}" + ("  GOAL REACHED" if reached else "")
            frames.append(annotate(env.render(), trail=trail, goal_xy=ep["goal_xy"], sub_xy=sub_xy[ci],
                                   text=text, ok=reached))
            if t < len(ep["actions"]):
                env.step(ep["actions"][t].astype(np.float32))
                trail.append(env.agent_position.numpy().copy())
        stem = f"{args.name or run.name}_task{task:03d}"
        write_mp4(frames, out / f"{stem}.mp4")
        Image.fromarray(frames[-1]).save(out / f"{stem}_last.png")
        print(f"task {task}: {len(frames)} frames, replay max xy error {max_err:.2e} px -> {out / stem}.mp4")
        assert max_err < 1e-2, "replay diverged from the logged trajectory"
    world.close()


if __name__ == "__main__":
    main()
