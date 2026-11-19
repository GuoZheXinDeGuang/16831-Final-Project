"""Thin per-environment adapters (FourRoom Distractors / OGBench Visual AntMaze).

Used only for snapshot/restore and for direct inner-env stepping during P1 branching.
FourRoom: pure-python state in env.__dict__. AntMaze: MuJoCo physics state + goal + RNG.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

FOURROOM_STATIC = {"variation_space", "grid_x", "grid_y", "observation_space", "action_space", "spec"}


def inner_env(world):
    return world.envs.unwrapped.envs[0].unwrapped


def is_mujoco(env) -> bool:
    return hasattr(env, "model") and hasattr(env, "data") and hasattr(env.data, "qpos")


def dump_state(env) -> dict:
    if is_mujoco(env):
        # Full MjData copy: after mj_step the kinematics (xpos/xquat, used by the renderer) are one
        # integration step stale w.r.t. qpos, so qpos/qvel + mj_forward would not reproduce pixels.
        return {
            "mj_data": deepcopy(env.data),
            "cur_goal_xy": deepcopy(env.cur_goal_xy),
            "_np_random": deepcopy(env._np_random),
        }
    return {k: v for k, v in env.__dict__.items() if k not in FOURROOM_STATIC}


def load_state(env, state: dict) -> None:
    if "mj_data" in state:
        import mujoco

        mujoco.mj_copyData(env.data, env.model, state["mj_data"])
        env.cur_goal_xy = deepcopy(state["cur_goal_xy"])
        env._np_random = deepcopy(state["_np_random"])
    else:
        env.__dict__.update(deepcopy(state))


def render(env) -> np.ndarray:
    return env.get_ob() if is_mujoco(env) else env.render()


def xy(env) -> np.ndarray:
    return np.asarray(env.get_xy(), dtype=np.float32).copy() if is_mujoco(env) else env.agent_position.numpy().copy()


def proprio(env):
    """Proprio as the model's processor expects it (AntMaze: qpos[2:] ++ qvel); None for FourRoom."""
    if not is_mujoco(env):
        return None
    return np.concatenate([env.data.qpos[2:], env.data.qvel]).astype(np.float32)


def step(env, action: np.ndarray):
    """Step the inner env; returns (pixels, terminated)."""
    obs, _, terminated, _, _ = env.step(action.astype(np.float32))
    pixels = obs if is_mujoco(env) else env.render()
    return pixels, bool(terminated)


def wrap_seeded_reset(world):
    """Make World.reset accept numpy-int seeds and, for MuJoCo envs, also seed every action_space
    in the wrapper chain. Upstream AntMaze 'xy_neutral' start states settle the ant with 5
    `action_space.sample()` steps from an unseeded action-space RNG, so the start pose otherwise
    differs run to run. FourRoom is left untouched (its behaviour is identical to P0)."""
    orig_reset = world.reset

    def reset(seed=None, options=None):
        seeds = None if seed is None else [int(x) for x in np.asarray(seed).reshape(-1)]
        out = orig_reset(seed=seeds, options=options)
        if seeds is not None:
            for i, e in enumerate(world.envs.unwrapped.envs):
                if not is_mujoco(e.unwrapped):
                    continue
                w = e
                while True:
                    w.action_space.seed(seeds[i % len(seeds)])
                    if not hasattr(w, "env"):
                        break
                    w = w.env
        return out

    world.reset = reset
    return orig_reset
