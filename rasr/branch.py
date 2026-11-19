"""Controller-conditioned subgoal execution attempts (P1 label collection).

An *attempt* = (snapshot of env at a baseline planning call t, fixed latent subgoal g that the
baseline's level-1 MPC was given at t, low-level controller budget B, execution horizon H).
We restore the env, then run ONLY the level-1 H-JEPA MPC (frozen upstream GradientSolver +
LevelCostModel, same plan config as the baseline) towards the fixed g, replanning every
receding_horizon * action_block env steps, for H_max steps. Every step we encode the real
observation into the level-2 latent and record d_k = ||enc2(o_{t+k}) - g||^2 (the same squared
L2 the planner's cost uses). Labels for H <= H_max are prefixes of the same rollout.

Ground-truth simulator state (xy) is recorded for analysis/labelling only; the controller sees
only pixels + latents.
"""

from __future__ import annotations

import pickle
import time
import zlib
from copy import deepcopy
from dataclasses import dataclass

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from hierarchical_solver import LevelCostModel
from rasr import envs as E
from rasr.instrument import PredictCounter


class _SetupDone(Exception):
    pass


@dataclass(frozen=True)
class Budget:
    name: str
    n_steps: int
    num_samples: int

    @property
    def max_sample_forwards(self) -> int:
        return self.n_steps * self.num_samples


DEFAULT_BUDGETS = (
    Budget("B0", 2, 4),
    Budget("B1", 10, 20),
    Budget("B2", 90, 100),  # = baseline level-1 solver (fourroom_l2.yaml)
)


def setup_task_world(world, policy, cfg, payload, task, env_seed, callables, process):
    """Run upstream evaluate_from_dataset up to (not including) the first step, so the env has
    exactly the baseline episode's reset seed, closed door, goal image and callables applied."""
    ep_data = dict(payload["data"][task])
    columns = list(payload["columns"])
    if env_seed is not None:
        ep_data["seed"] = torch.full((ep_data["xy"].shape[0],), env_seed, dtype=torch.int64)
        columns = columns + ["seed"]
    chunk = {**payload, "data": [ep_data], "columns": columns, "episodes_idx": [task]}
    orig_reset = E.wrap_seeded_reset(world)

    def _stop():
        raise _SetupDone

    world.step = _stop
    try:
        world.evaluate_from_dataset(
            None, episodes_idx=None, start_steps=None,
            goal_offset_steps=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
            callables=callables, load_eval_trajs_path=chunk, process=process,
            start_state_mode=cfg.eval.get("start_state_mode", "dataset_full"),
            goal_state_mode=cfg.eval.get("goal_state_mode", "dataset_full"),
        )
    except _SetupDone:
        pass
    finally:
        del world.step
        world.reset = orig_reset
    return world.infos["goal"].copy()


def restore_env(world, snapshot_blob: bytes):
    snap = pickle.loads(zlib.decompress(snapshot_blob))
    env = E.inner_env(world)
    E.load_state(env, snap["env_state"])
    pixels = E.render(env)
    return snap, pixels


class Level1Controller:
    """Level-1 H-JEPA MPC with a fixed level-2 latent goal; one GradientSolver per budget."""

    def __init__(self, policy, cfg, budgets=DEFAULT_BUDGETS, seed: int = 0):
        hsolver = policy.solver
        self.policy = policy
        self.level_models = hsolver.level_models
        self.plan_cfg = hsolver._level_plan_config(1)
        self.n_envs = 1
        self.receding = int(hsolver._config.receding_horizon) * int(self.plan_cfg.action_block)
        self.horizon = int(self.plan_cfg.pre_decay_horizon or self.plan_cfg.horizon)
        base = OmegaConf.to_container(cfg.solver.solvers.level1, resolve=True)
        self.solvers = {}
        for b in budgets:
            scfg = {**base, "n_steps": b.n_steps, "num_samples": b.num_samples, "seed": seed}
            adapted = LevelCostModel(model=self.level_models[0], upper_models=self.level_models[1:])
            s = hydra.utils.instantiate(scfg, model=adapted)
            s.configure(action_space=policy.env.action_space, n_envs=1, config=self.plan_cfg, level=1)
            self.solvers[b.name] = s
        self.budgets = {b.name: b for b in budgets}
        self.counters = {lvl: PredictCounter(m, lvl) for lvl, m in enumerate(self.level_models, start=1)}

    @torch.no_grad()
    def encode(self, pixels_hwc: np.ndarray, goal_hwc: np.ndarray, proprio=None):
        info = {"pixels": pixels_hwc[None, None], "goal": goal_hwc[None, None]}
        if proprio is not None:
            info["proprio"] = np.asarray(proprio, dtype=np.float32)[None, None]
        info = self.policy._prepare_info(info)
        emb = self.policy.solver._encode_all_levels(info_dict=info, key="pixels")
        return info, emb

    def solve(self, budget: str, info: dict, emb: dict, subgoal: np.ndarray, gen_seed: int):
        s = self.solvers[budget]
        s.torch_gen.manual_seed(gen_seed)
        level_info = dict(info)
        level_info["embed_0"] = emb["embed_1"]
        level_info["action_cost_weight"] = float(self.plan_cfg.action_cost_weight)
        level_info["goal_embed_0"] = torch.as_tensor(subgoal, dtype=torch.float32, device=emb["embed_1"].device).view(1, 1, -1)
        level_info["goal_embed_0_in_upper_space"] = True
        c0 = {l: c.snapshot() for l, c in self.counters.items()}
        t0 = time.perf_counter()
        with torch.enable_grad():
            out = s.solve(info_dict=level_info, planning_horizon=self.horizon)
        dt = time.perf_counter() - t0
        rec = s.solve_records.pop()
        calls = {l: self.counters[l].calls - c0[l][0] for l in self.counters}
        rows = {l: self.counters[l].rows - c0[l][1] for l in self.counters}
        return out["actions"], {"wall_clock": dt, "gd_iters": int(rec["n_iters"][0]), "pred_calls": calls, "pred_rows": rows}

    def to_env_actions(self, actions: torch.Tensor) -> np.ndarray:
        keep = min(int(self.policy.cfg.receding_horizon), actions.shape[1])
        plan = actions[:, :keep].reshape(1, keep * int(self.plan_cfg.action_block), -1)[0].numpy()
        if "action" in self.policy.process:
            plan = self.policy.process["action"].inverse_transform(plan)
        return plan  # (receding steps, 2)


def run_attempt(world, ctrl: Level1Controller, snapshot_blob: bytes, goal_hwc: np.ndarray, subgoal: np.ndarray,
                budget: str, h_max: int, gen_seed: int, frames: list | None = None):
    """Restore snapshot, run level-1 MPC towards fixed `subgoal` for h_max env steps."""
    snap, pixels = restore_env(world, snapshot_blob)
    render_err = float(np.abs(pixels.astype(np.int32) - snap["pixels"].astype(np.int32)).max())
    # Start from the exact observation the baseline planner saw (identical to the render on FourRoom;
    # on AntMaze the upstream xy_neutral start frame is rendered from pre-settle kinematics).
    pixels = snap["pixels"].copy()
    env = E.inner_env(world)
    g = np.asarray(subgoal, dtype=np.float32)
    dists, xys, env_goal = [], [], []
    solve_infos = []
    t0 = time.perf_counter()
    if frames is not None:
        frames.append(pixels.copy())
    info, emb = ctrl.encode(pixels, goal_hwc, E.proprio(env))
    d0 = float(((emb["embed_2"][0, 0].cpu().numpy() - g) ** 2).sum())
    xy0 = E.xy(env)
    buf = []
    k = 0
    while k < h_max:
        if not buf:
            actions, sinfo = ctrl.solve(budget, info, emb, g, gen_seed + k)
            solve_infos.append(sinfo)
            buf = list(ctrl.to_env_actions(actions))
        a = buf.pop(0)
        pixels, terminated = E.step(env, a)
        k += 1
        if frames is not None:
            frames.append(pixels.copy())
        info, emb = ctrl.encode(pixels, goal_hwc, E.proprio(env))
        dists.append(float(((emb["embed_2"][0, 0].cpu().numpy() - g) ** 2).sum()))
        xys.append(E.xy(env))
        env_goal.append(bool(terminated))
    return {
        "d0": d0,
        "dists": np.array(dists, dtype=np.float32),
        "xy0": xy0,
        "xys": np.stack(xys).astype(np.float32),
        "env_goal_reached": np.array(env_goal),
        "solves": solve_infos,
        "wall_clock": time.perf_counter() - t0,
        "render_err": render_err,
    }
