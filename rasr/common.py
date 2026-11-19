"""Shared setup: import paths into the upstream H-JEPA repo, config loading, policy building.

Nothing here modifies upstream code; we only import and wrap it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HJEPA_REPO = PROJECT_ROOT / "H-JEPA"
HJEPA_PKG = HJEPA_REPO / "h_jepa"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rasr"

os.environ.setdefault("HJEPA_HOME", str(PROJECT_ROOT / "hjepa_home"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before CUDA init)
for p in (HJEPA_PKG, HJEPA_REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

HJEPA_HOME = Path(os.environ["HJEPA_HOME"])

CKPTS = {
    "fourroom_hjepa_l2": HJEPA_HOME / "ckpts/fourroom/fourroom_hjepa_l2/seed{seed}/fourroom_hjepa_l2_object.ckpt",
    "fourroom_hjepa_l3": HJEPA_HOME / "ckpts/fourroom/fourroom_hjepa_l3/seed{seed}/fourroom_hjepa_l3_object.ckpt",
    "ant_hjepa_l2": HJEPA_HOME / "ckpts/ant/ant_hjepa_l2/seed{seed}/ant_hjepa_l2_object.ckpt",
    "droid_hjepa_l2": HJEPA_HOME / "ckpts/droid/droid_hjepa_l2/seed{seed}/droid_hjepa_l2_object.ckpt",
}
OFFICIAL_TASKS = HJEPA_PKG / "assets/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt"
ANT_OFFICIAL_TASKS = HJEPA_PKG / "assets/eval_trajs/ant/expert_grid_d3_n50.pt"


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def git_dirty(path: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True).stdout
        return out.splitlines()
    except Exception:  # pragma: no cover
        return []


def load_eval_cfg(config_name: str, overrides: dict | None = None):
    """Load an upstream eval yaml exactly as eval.py would (no hydra CLI), then apply overrides."""
    cfg = OmegaConf.load(HJEPA_PKG / "config" / "eval" / f"{config_name}.yaml")
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def prepare_eval(cfg):
    """Mirror planning_eval.run_planning_eval set-up (world cfg, transforms, process)."""
    import planning_eval as pe

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    transform = {"pixels": pe.img_transform(cfg), "goal": pe.img_transform(cfg)}
    process = pe._build_eval_process_from_policy_normalizer(cfg, model=None)
    image_shape = (int(cfg.eval.img_size),) * 2
    return cfg, world_cfg, transform, process, image_shape


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
