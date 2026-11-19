"""Non-invasive instrumentation of the upstream H-JEPA hierarchical MPC.

`EpisodeRecorder` wraps (by instance attribute assignment, never by editing upstream code):
  * HierarchicalSolver.solve / _encode_all_levels  -> per-planning-call obs/goal latents
  * GradientSolver.solve (one per level)            -> per-level horizon, target, predicted plan,
                                                       chosen actions, GD iterations, wall-clock
  * JEPA.predict (one per level)                    -> world-model predictor forward counts
  * World.step                                       -> per-step xy, action, success flag, and an
                                                       env snapshot right before each planning call

The wrappers only read tensors; they do not consume planner RNG, so the planner behaves exactly
as upstream (verified by tests/test_p0_reproducibility.py).
"""

from __future__ import annotations

import pickle
import time
import zlib
from copy import deepcopy
from typing import Any

import numpy as np
import torch


def _np(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


class PredictCounter:
    """Counts batched predictor forward calls and the number of rows (samples x envs) they carry."""

    def __init__(self, jepa, level: int):
        self.level = level
        self.calls = 0
        self.rows = 0
        self._orig = jepa.predict

        def predict(emb, act_emb, *a, **k):
            self.calls += 1
            self.rows += int(emb.shape[0])
            return self._orig(emb, act_emb, *a, **k)

        jepa.predict = predict

    def snapshot(self):
        return self.calls, self.rows


class EpisodeRecorder:
    def __init__(self, world, policy, *, encode_every_step: bool = True, snapshot_env: bool = True):
        self.world = world
        self.policy = policy
        self.solver = policy.solver
        self.encode_every_step = encode_every_step
        self.snapshot_env = snapshot_env
        self.n_levels = len(self.solver.level_models)

        self.calls: list[dict] = []
        self.steps: list[dict] = []
        self.snapshots: list[bytes] = []
        self._cur: dict | None = None
        self._capture_encode = True
        self._encode_buf: list[dict] = []

        self.counters = {
            lvl: PredictCounter(self.solver.level_models[lvl - 1], lvl) for lvl in range(1, self.n_levels + 1)
        }
        self._wrap_solver()
        self._wrap_world()

    # ------------------------------------------------------------------ solver wrappers
    def _wrap_solver(self):
        solver = self.solver
        orig_solve = solver.solve
        orig_encode = solver._encode_all_levels

        def encode_all_levels(info_dict, key="pixels"):
            out = orig_encode(info_dict=info_dict, key=key)
            if self._capture_encode:
                self._encode_buf.append({k: _np(v) for k, v in out.items()})
            return out

        solver._encode_all_levels = encode_all_levels

        def solve(info_dict, steps_taken, eval_budget):
            self._encode_buf = []
            counts0 = {l: c.snapshot() for l, c in self.counters.items()}
            self._cur = {
                "call_idx": len(self.calls),
                "env_step": int(steps_taken),
                "levels": {},
            }
            t0 = time.perf_counter()
            out = orig_solve(info_dict, steps_taken=steps_taken, eval_budget=eval_budget)
            dt = time.perf_counter() - t0
            obs_emb, goal_emb = self._encode_buf[0], self._encode_buf[1]
            rec = self._cur
            rec["wall_clock"] = dt
            rec["obs_embed"] = {k: v[0, 0] for k, v in obs_emb.items()}  # (D,) per level
            rec["goal_embed"] = {k: v[0, 0] for k, v in goal_emb.items()}
            rec["predictor_calls"] = {
                l: self.counters[l].calls - counts0[l][0] for l in self.counters
            }
            rec["predictor_rows"] = {
                l: self.counters[l].rows - counts0[l][1] for l in self.counters
            }
            rec["actions_l1"] = _np(out["actions"])[0]  # (H1, action_dim*action_block), normalized
            self.calls.append(rec)
            self._cur = None
            return out

        solver.solve = solve

        for level, lsolver in solver.level_solvers.items():
            self._wrap_level_solver(level, lsolver)

    def _wrap_level_solver(self, level: int, lsolver):
        orig = lsolver.solve

        def solve(info_dict, *a, planning_horizon=None, **k):
            goal = info_dict.get("goal_embed_0")
            n_rec0 = len(lsolver.solve_records)
            t0 = time.perf_counter()
            out = orig(info_dict, *a, planning_horizon=planning_horizon, **k)
            dt = time.perf_counter() - t0
            gd = lsolver.solve_records[n_rec0] if len(lsolver.solve_records) > n_rec0 else {}
            if self._cur is not None:
                self._cur["levels"][level] = {
                    "planning_horizon": int(planning_horizon),
                    "goal_target": _np(goal)[0].reshape(-1, _np(goal).shape[-1]),  # (n_subgoals, D_upper)
                    "goal_in_upper_space": bool(info_dict.get("goal_embed_0_in_upper_space", False)),
                    "goal_level": int(info_dict.get("goal_embed_0_level", level + 1)),
                    "dense_upper_goal": bool(info_dict.get("dense_upper_goal_projection", False)),
                    "pred_plan": _np(out["predictions"]["predicted_embed_0"])[0],  # (T+1, D_level)
                    "actions": _np(out["actions"])[0],
                    "gd_iters": int(gd.get("n_iters", [lsolver.n_steps])[0]),
                    "gd_max_iters": int(lsolver.n_steps),
                    "num_samples": int(lsolver.num_samples),
                    "wall_clock": dt,
                }
            return out

        lsolver.solve = solve

    # ------------------------------------------------------------------ world wrapper
    def _wrap_world(self):
        world = self.world
        orig_step = world.step
        policy = self.policy

        orig_get_action = policy.get_action
        last_action = {}

        def get_action(info_dict, **k):
            a = orig_get_action(info_dict, **k)
            last_action["a"] = np.array(a, copy=True)
            return a

        policy.get_action = get_action

        def step():
            will_plan = len(policy._action_buffer) == 0
            if will_plan and self.snapshot_env:
                self.snapshots.append(self.snapshot(world))
            if len(self.steps) == 0:
                self.initial = self._step_info(world, encode=True)
            orig_step()
            rec = self._step_info(world, encode=self.encode_every_step)
            rec["action"] = last_action["a"][0]
            rec["planned_before"] = will_plan
            self.steps.append(rec)

        world.step = step

    STATIC_ATTRS = {"variation_space", "grid_x", "grid_y", "observation_space", "action_space", "spec"}

    @classmethod
    def snapshot(cls, world) -> bytes:
        """Restorable copy of the inner env's dynamic state + current pixels (compressed pickle).

        Static attributes (variation space, pixel grids, spaces) are rebuilt by re-running the
        task set-up; see rasr.branch.restore_env.
        """
        from rasr.envs import dump_state, inner_env

        state = dump_state(inner_env(world))
        payload = {
            "env_state": state,
            "pixels": np.asarray(world.infos["pixels"])[0, -1].copy(),
            "terminated": bool(np.asarray(world.terminateds)[0]) if world.terminateds is not None else False,
        }
        return zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), 3)

    def _step_info(self, world, encode: bool) -> dict:
        infos = world.infos
        xy = np.asarray(infos["xy"])[0, -1].astype(np.float32)
        if "distance_to_target" in infos:  # FourRoom
            dist = float(np.asarray(infos["distance_to_target"])[0, -1])
        else:  # AntMaze: Euclidean distance (maze units) to the navigation goal
            dist = float(np.linalg.norm(xy[:2] - np.asarray(infos["goal_xy"])[0, -1][:2]))
        rec = {
            "xy": xy,
            "distance_to_target": dist,
            "terminated": bool(np.asarray(world.terminateds)[0]) if world.terminateds is not None else False,
        }
        if "distractor_xy" in infos:
            rec["distractor_xy"] = np.asarray(infos["distractor_xy"])[0, -1].astype(np.float32)
        if encode:
            rec["embed"] = self.encode_obs(infos)
        return rec

    @torch.no_grad()
    def encode_obs(self, infos) -> dict:
        """Encode the current observation through all levels (measurement only, not planner compute)."""
        info = {"pixels": np.asarray(infos["pixels"])}
        if "proprio" in infos:  # AntMaze FusionEncoder needs proprioception
            info["proprio"] = np.asarray(infos["proprio"])
        info = self.policy._prepare_info(deepcopy(info))
        self._capture_encode = False
        try:
            out = self.solver._encode_all_levels(info_dict=info, key="pixels")
        finally:
            self._capture_encode = True
        return {k: _np(v)[0, 0] for k, v in out.items()}

    # ------------------------------------------------------------------ export
    def export(self) -> dict[str, Any]:
        n_levels = self.n_levels
        calls = self.calls
        steps = self.steps
        out: dict[str, Any] = {
            "xy": np.stack([self.initial["xy"]] + [s["xy"] for s in steps]),  # (T+1, 2)
            "dist_to_target": np.array([self.initial["distance_to_target"]] + [s["distance_to_target"] for s in steps]),
            "terminated": np.array([False] + [s["terminated"] for s in steps]),
            "actions": np.stack([s["action"] for s in steps]),  # env-scale actions (T, 2)
            "call_env_step": np.array([c["env_step"] for c in calls]),
            "call_wall_clock": np.array([c["wall_clock"] for c in calls]),
        }
        if "distractor_xy" in self.initial:
            out["distractor_xy"] = np.stack([self.initial["distractor_xy"]] + [s["distractor_xy"] for s in steps])
        for lvl in range(1, n_levels + 1):
            key = f"embed_{lvl}"
            if self.encode_every_step:
                out[f"step_{key}"] = np.stack([self.initial["embed"][key]] + [s["embed"][key] for s in steps])
            out[f"call_obs_{key}"] = np.stack([c["obs_embed"][key] for c in calls])
            out[f"call_goal_{key}"] = np.stack([c["goal_embed"][key] for c in calls])
            out[f"call_pred_calls_l{lvl}"] = np.array([c["predictor_calls"][lvl] for c in calls])
            out[f"call_pred_rows_l{lvl}"] = np.array([c["predictor_rows"][lvl] for c in calls])
        # per-level solve info (level may be skipped at horizon 1 -> NaN / -1)
        for lvl in range(1, n_levels + 1):
            present = [lvl in c["levels"] for c in calls]
            out[f"l{lvl}_solved"] = np.array(present)
            out[f"l{lvl}_horizon"] = np.array([c["levels"][lvl]["planning_horizon"] if p else -1 for c, p in zip(calls, present)])
            out[f"l{lvl}_gd_iters"] = np.array([c["levels"][lvl]["gd_iters"] if p else 0 for c, p in zip(calls, present)])
            out[f"l{lvl}_num_samples"] = np.array([c["levels"][lvl]["num_samples"] if p else 0 for c, p in zip(calls, present)])
            out[f"l{lvl}_wall_clock"] = np.array([c["levels"][lvl]["wall_clock"] if p else 0.0 for c, p in zip(calls, present)])
        # Level-1 target ("subgoal") for every call: what the low-level MPC actually optimised towards.
        l1 = [c["levels"][1] for c in calls]
        out["subgoal"] = np.stack([x["goal_target"][-1] for x in l1])  # (n_calls, D_goal)
        out["subgoal_level"] = np.array([x["goal_level"] if x["goal_in_upper_space"] else 1 for x in l1])
        out["subgoal_dense"] = np.array([x["dense_upper_goal"] for x in l1])
        out["subgoal_from_upper_plan"] = np.array([2 in c["levels"] for c in calls])
        if 2 in calls[0]["levels"]:
            out["l2_pred_plan_first"] = calls[0]["levels"][2]["pred_plan"]
        out["l2_pred_plans"] = [c["levels"][2]["pred_plan"] if 2 in c["levels"] else None for c in calls]
        out["l1_pred_plans"] = [c["levels"][1]["pred_plan"] for c in calls]
        return out
