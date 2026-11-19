#!/usr/bin/env python
"""Side-by-side video: the same restored state and the same fixed level-2 subgoal, executed by the
frozen level-1 MPC under budget B0 (2 GD iters x 4 samples) vs B2 (baseline 90 x 100), 30 steps."""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr.common import load_eval_cfg, prepare_eval  # noqa: E402

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import planning_eval as pe  # noqa: E402
from rasr.branch import Level1Controller, run_attempt, setup_task_world  # noqa: E402
from rasr.xy_probe import XYProbe  # noqa: E402
from scripts.make_videos import write_mp4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42")
ap.add_argument("--task", type=int, default=5)
ap.add_argument("--call", type=int, default=7)
ap.add_argument("--out", default="outputs/rasr/videos/p1_budget_B0_vs_B2.mp4")
args = ap.parse_args()
torch.use_deterministic_algorithms(True, warn_only=True)
run = Path(args.run)
rcfg = json.loads(next(run.glob("**/run_config.json")).read_text())
ep_rec = None
for p in run.glob("**/episodes.jsonl"):
    for l in p.read_text().splitlines():
        if json.loads(l)["task"] == args.task:
            ep_rec, d = json.loads(l), p.parent
cfg = load_eval_cfg(rcfg["config_name"], {"seed": rcfg["planner_seed"], "policy": rcfg["checkpoint"]})
payload = torch.load(rcfg["tasks_file"], map_location="cpu", weights_only=False)
cfg.eval.num_eval = len(payload["data"])
cfg, world_cfg, transform, process, image_shape = prepare_eval(cfg)
callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
policy = pe._build_policy(cfg, process, transform, model=None)
world = swm.World(**{**world_cfg, "num_envs": 1}, image_shape=image_shape)
world.set_policy(policy)
ctrl = Level1Controller(policy, cfg, seed=7)
ep = np.load(d / f"ep_{args.task:03d}.npz")
snaps = pickle.loads((d / f"ep_{args.task:03d}_snap.pkl").read_bytes())
goal = setup_task_world(world, policy, cfg, payload, args.task, ep_rec["env_seed"], callables, process)[0, -1]
g = ep["subgoal"][args.call]
sub_xy = XYProbe(["outputs/rasr/p1_source/train"])(g)[0][0]
eps = json.loads(Path("outputs/rasr/p1_reachability/results.json").read_text())["eps"]
panels = {}
with sdpa_kernel(SDPBackend.MATH):
    for b in ("B0", "B2"):
        fr = []
        r = run_attempt(world, ctrl, snaps[args.call], goal, g, b, 30, 12345, frames=fr)
        panels[b] = (fr, r)
        print(b, "min latent dist", r["dists"].min(), "reached at", int(np.argmax(r["dists"] <= eps)) + 1 if (r["dists"] <= eps).any() else None)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    font = ImageFont.load_default()
frames = []
for k in range(31):
    row = []
    for b, lab in (("B0", "B0 (2 it x 4)"), ("B2", "B2 (90 it x 100)")):
        fr, r = panels[b]
        im = Image.fromarray(fr[k]).resize((448, 448), Image.NEAREST)
        c = Image.new("RGB", (448, 482), (24, 24, 28)); c.paste(im, (0, 0))
        dr = ImageDraw.Draw(c)
        sx, sy = sub_xy * 2
        dr.line([sx - 7, sy - 7, sx + 7, sy + 7], fill=(220, 0, 200), width=3)
        dr.line([sx - 7, sy + 7, sx + 7, sy - 7], fill=(220, 0, 200), width=3)
        dist = r["d0"] if k == 0 else float(r["dists"][k - 1])
        hit = (r["dists"][:max(k, 1)] <= eps).any() and k > 0
        dr.text((8, 456), f"{lab} | step {k:2d} | d={dist:5.1f}" + ("  REACHED" if hit else ""),
                fill=(120, 230, 140) if hit else (235, 235, 235), font=font)
        row.append(np.asarray(c))
    frames.append(np.concatenate([row[0], np.full((482, 8, 3), 255, np.uint8), row[1]], 1))
frames = [frames[0]] * 8 + frames + [frames[-1]] * 15
write_mp4(frames, Path(args.out), fps=6)
Image.fromarray(frames[-1]).save(Path(args.out).with_suffix(".png"))
print("wrote", args.out)
