#!/usr/bin/env python
"""Check that env snapshots restore exactly: restore the snapshot before planning call i, replay the
logged env actions up to call i+1, compare xy and pixels with the logged episode."""
import argparse, json, pickle, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr.common import load_eval_cfg, prepare_eval  # noqa
import numpy as np, stable_worldmodel as swm, torch  # noqa
from omegaconf import OmegaConf  # noqa
import planning_eval as pe  # noqa
from rasr import envs as E  # noqa
from rasr.branch import restore_env, setup_task_world  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--task", type=int, default=0)
ap.add_argument("--calls", default="0,3,7"); a = ap.parse_args()
run = Path(a.run); rc = json.loads((run / "run_config.json").read_text())
cfg = load_eval_cfg(rc["config_name"], {"seed": rc["planner_seed"], "policy": rc["checkpoint"]})
payload = torch.load(rc["tasks_file"], map_location="cpu", weights_only=False); cfg.eval.num_eval = len(payload["data"])
cfg, wcfg, tf, proc, ish = prepare_eval(cfg)
cal = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
policy = pe._build_policy(cfg, proc, tf, model=None)
world = swm.World(**{**wcfg, "num_envs": 1}, image_shape=ish); world.set_policy(policy)
seed = None if rc["env_seed_base"] < 0 else rc["env_seed_base"] + a.task
setup_task_world(world, policy, cfg, payload, a.task, seed, cal, proc)
ep = np.load(run / f"ep_{a.task:03d}.npz"); snaps = pickle.load(open(run / f"ep_{a.task:03d}_snap.pkl", "rb"))
cs = ep["call_env_step"]; res = []
for i in [int(x) for x in a.calls.split(",")]:
    snap, pix = restore_env(world, snaps[i]); env = E.inner_env(world)
    perr = int(np.abs(pix.astype(int) - snap["pixels"].astype(int)).max())
    t0, t1 = int(cs[i]), int(cs[i + 1]) if i + 1 < len(cs) else int(cs[i]) + 10
    xe = [float(np.abs(E.xy(env) - ep["xy"][t0]).max())]
    for t in range(t0, t1):
        E.step(env, ep["actions"][t]); xe.append(float(np.abs(E.xy(env) - ep["xy"][t + 1]).max()))
    res.append({"call": i, "render_err": perr, "xy_err_max": max(xe)})
print(json.dumps(res)); import os; os._exit(0)
