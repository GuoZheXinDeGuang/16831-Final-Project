#!/usr/bin/env python
"""P0: run the *unmodified* H-JEPA hierarchical MPC with structured logging.

Mirrors upstream planning_eval._run_chunked_eval with chunk_size=1 (one World per task, planner
generators reseeded to seed + task_index), so success numbers are comparable with eval.py.

Outputs (in --out):
  run_config.json            config, checkpoint, git commits, command
  episodes.jsonl             one line per task: success, steps, wall-clock, compute
  calls.jsonl                one line per planning call: horizons, GD iters, predictor calls, time
  ep_XXX.npz / ep_XXX.pkl    tensors (trajectory, latents, subgoals, plans) / env snapshots
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rasr.common import (  # noqa: E402  (sets sys.path for upstream imports)
    CKPTS,
    HJEPA_PKG,
    HJEPA_REPO,
    OFFICIAL_TASKS,
    PROJECT_ROOT,
    git_commit,
    git_dirty,
    load_eval_cfg,
    prepare_eval,
)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import planning_eval as pe  # noqa: E402
from rasr.instrument import EpisodeRecorder  # noqa: E402


def parse_tasks(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fourroom_l2")
    ap.add_argument("--model", default="fourroom_hjepa_l2")
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42, help="planner seed (paper: = model seed)")
    ap.add_argument("--tasks-file", default=str(OFFICIAL_TASKS))
    ap.add_argument("--tasks", default="all", help="e.g. 'all', '0-9', '0,3,7'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-snapshots", action="store_true")
    ap.add_argument("--no-step-encode", action="store_true")
    ap.add_argument("--env-seed-base", type=int, default=100000,
                    help="env reset seed = base + task (upstream leaves FourRoom resets unseeded; -1 = unseeded)")
    ap.add_argument("--nondeterministic", action="store_true", help="skip torch deterministic algorithms")
    ap.add_argument("--rasr-config", default=None, help="yaml with RASRConfig fields (P2/P3); omit = upstream")
    ap.add_argument("--predictor", default="outputs/rasr/p1_reachability/mlp_model.pt")
    ap.add_argument("--override", nargs="*", default=[], help="cfg overrides key=value (yaml-parsed)")
    args = ap.parse_args()

    if not args.nondeterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = str(CKPTS[args.model]).format(seed=args.model_seed)
    overrides = {"seed": args.seed, "policy": ckpt}
    for kv in args.override:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    cfg = load_eval_cfg(args.config, overrides)
    payload = torch.load(args.tasks_file, map_location="cpu", weights_only=False)
    n_tasks = len(payload["data"])
    tasks = parse_tasks(args.tasks, n_tasks)
    cfg.eval.num_eval = n_tasks

    cfg, world_cfg, transform, process, image_shape = prepare_eval(cfg)
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    ep_idx = payload.get("episodes_idx")
    ep_idx = list(ep_idx) if ep_idx is not None and len(ep_idx) else list(range(n_tasks))

    run_cfg = {
        "command": " ".join(sys.argv),
        "config_name": args.config,
        "eval_config": OmegaConf.to_container(cfg, resolve=True),
        "checkpoint": ckpt,
        "model_seed": args.model_seed,
        "planner_seed": args.seed,
        "tasks_file": str(args.tasks_file),
        "tasks_file_seed": payload.get("task_info", {}).get("seed"),
        "tasks": tasks,
        "env_seed_base": args.env_seed_base,
        "deterministic": not args.nondeterministic,
        "hjepa_git_commit": git_commit(HJEPA_REPO),
        "hjepa_git_dirty": git_dirty(HJEPA_REPO),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2, default=str))

    policy = pe._build_policy(cfg, process, transform, model=None)
    rasr_cfg = None
    if args.rasr_config:
        from rasr.rasr_solver import RASRConfig, RASRHierarchicalSolver, load_predictor
        rasr_cfg = RASRConfig(**(yaml.safe_load(Path(args.rasr_config).read_text()) or {}))
        pred = load_predictor(args.predictor) if rasr_cfg.mode in ("rerank", "rasr") else None
        up = policy.solver
        policy.solver = RASRHierarchicalSolver(up.level_solvers, up.level_models, rasr_cfg, pred)
        run_cfg["rasr_config"] = rasr_cfg.__dict__
        run_cfg["predictor"] = args.predictor if pred is not None else None
        (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2, default=str))
    solver = policy.solver
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in solver.level_solvers.values()]
    n_levels = len(solver.level_models)

    done = set()
    ep_log = out / "episodes.jsonl"
    if ep_log.exists():
        done = {json.loads(l)["task"] for l in ep_log.read_text().splitlines() if l.strip()}

    for task in tasks:
        if task in done:
            print(f"task {task}: already done, skipping")
            continue
        for gen, seed in gens:
            gen.manual_seed(seed + task)  # identical to upstream chunked eval with chunk_size=1
        ep_data = dict(payload["data"][task])
        columns = list(payload["columns"])
        env_seed = None
        if args.env_seed_base >= 0:
            env_seed = args.env_seed_base + task
            ep_data["seed"] = torch.full((ep_data["xy"].shape[0],), env_seed, dtype=torch.int64)
            columns = columns + ["seed"]
        chunk = {**payload, "data": [ep_data], "columns": columns, "episodes_idx": [ep_idx[task]]}
        world = swm.World(**{**world_cfg, "num_envs": 1}, image_shape=image_shape)
        try:
            world.set_policy(policy)
            from rasr.envs import wrap_seeded_reset
            wrap_seeded_reset(world)
            # policy/solver wrappers are re-attached per episode on fresh closures
            rec = EpisodeRecorder(world, policy, encode_every_step=not args.no_step_encode, snapshot_env=not args.no_snapshots)
            t0 = time.time()
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                metrics = world.evaluate_from_dataset(
                    None, episodes_idx=None, start_steps=None,
                    goal_offset_steps=cfg.eval.goal_offset_steps,
                    eval_budget=cfg.eval.eval_budget,
                    callables=callables, load_eval_trajs_path=chunk, process=process,
                    start_state_mode=cfg.eval.get("start_state_mode", "dataset_full"),
                    goal_state_mode=cfg.eval.get("goal_state_mode", "dataset_full"),
                )
            ep_time = time.time() - t0
        finally:
            world.close()
            # un-wrap so the next episode wraps fresh
            for attr in ("solve", "_encode_all_levels"):  # instance-level wrappers only
                solver.__dict__.pop(attr, None)
            for s in solver.level_solvers.values():
                s.__dict__.pop("solve", None)
                s.solve_records = []
            for m in solver.level_models:
                m.__dict__.pop("predict", None)
            policy.__dict__.pop("get_action", None)

        data = rec.export()
        rasr_log = solver.pop_episode_log() if hasattr(solver, "pop_episode_log") else None
        if rasr_log is not None:
            with open(out / "rasr_calls.jsonl", "a") as f:
                for r in rasr_log:
                    f.write(json.dumps({"task": task, **r}, default=float) + "\n")
        task_info = payload.get("task_info", {}).get("tasks", [{}] * n_tasks)[task]
        td = payload["data"][task]
        goal_xy = np.asarray(td["goal_xy"][-1] if "goal_xy" in td else td["goal_state"][-1], dtype=np.float32)
        np.savez_compressed(
            out / f"ep_{task:03d}.npz",
            **{k: v for k, v in data.items() if not isinstance(v, list)},
            goal_xy=goal_xy,
        )
        with open(out / f"ep_{task:03d}_plans.pkl", "wb") as f:
            pickle.dump({"l1_pred_plans": data["l1_pred_plans"], "l2_pred_plans": data["l2_pred_plans"]}, f)
        if rec.snapshots:
            with open(out / f"ep_{task:03d}_snap.pkl", "wb") as f:
                pickle.dump(rec.snapshots, f)

        success = bool(np.asarray(metrics["episode_successes"])[0])
        steps_to_success = float(np.asarray(metrics["steps_to_success"])[0])
        pred_calls = {l: int(data[f"call_pred_calls_l{l}"].sum()) for l in range(1, n_levels + 1)}
        pred_rows = {l: int(data[f"call_pred_rows_l{l}"].sum()) for l in range(1, n_levels + 1)}
        # planning compute up to success (calls whose env_step < steps_to_success); upstream keeps
        # planning after success, so we report both.
        n_calls = len(data["call_env_step"])
        upto = data["call_env_step"] < (steps_to_success if np.isfinite(steps_to_success) else 1e9)
        ep_rec = {
            "task": task,
            "dataset_episode_idx": int(ep_idx[task]),
            "env_seed": env_seed,
            "closed_door": task_info.get("closed_door"),
            "start_xy": data["xy"][0].tolist(),
            "goal_xy": goal_xy.tolist(),
            "success": success,
            "steps_to_success": steps_to_success,
            "wall_clock_to_success": float(np.asarray(metrics["wall_clock_to_success"])[0]),
            "episode_wall_clock": ep_time,
            "planning_wall_clock": float(data["call_wall_clock"].sum()),
            "planning_wall_clock_to_success": float(data["call_wall_clock"][upto].sum()),
            "n_planning_calls": n_calls,
            "n_planning_calls_to_success": int(upto.sum()),
            "predictor_calls": pred_calls,
            "predictor_rows": pred_rows,
            "predictor_calls_to_success": {l: int(data[f"call_pred_calls_l{l}"][upto].sum()) for l in range(1, n_levels + 1)},
            "gd_iters": {l: int(data[f"l{l}_gd_iters"].sum()) for l in range(1, n_levels + 1)},
            "final_dist_to_target": float(data["dist_to_target"][-1]),
            "min_dist_to_target": float(data["dist_to_target"].min()),
        }
        with open(ep_log, "a") as f:
            f.write(json.dumps(ep_rec) + "\n")
        with open(out / "calls.jsonl", "a") as f:
            for i in range(n_calls):
                f.write(json.dumps({
                    "task": task, "call_idx": i, "env_step": int(data["call_env_step"][i]),
                    "wall_clock": float(data["call_wall_clock"][i]),
                    **{f"l{l}_horizon": int(data[f"l{l}_horizon"][i]) for l in range(1, n_levels + 1)},
                    **{f"l{l}_gd_iters": int(data[f"l{l}_gd_iters"][i]) for l in range(1, n_levels + 1)},
                    **{f"l{l}_num_samples": int(data[f"l{l}_num_samples"][i]) for l in range(1, n_levels + 1)},
                    **{f"l{l}_wall_clock": float(data[f"l{l}_wall_clock"][i]) for l in range(1, n_levels + 1)},
                    **{f"pred_calls_l{l}": int(data[f"call_pred_calls_l{l}"][i]) for l in range(1, n_levels + 1)},
                    "subgoal_from_upper_plan": bool(data["subgoal_from_upper_plan"][i]),
                    "xy": data["xy"][int(data["call_env_step"][i])].tolist(),
                }) + "\n")
        print(f"task {task}: success={success} steps={steps_to_success} time={ep_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
