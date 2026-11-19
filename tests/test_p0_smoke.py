"""Integration smoke tests (GPU, ~3-4 min): P0 determinism + P1 snapshot restoration.

Run:  python -m pytest -q tests/test_p0_smoke.py   (from Project/, after `source activate_rasr.sh`)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    outs = []
    for name in ("a", "b"):
        out = tmp_path_factory.mktemp(f"p0_{name}")
        subprocess.run([PY, "scripts/run_baseline.py", "--tasks", "3", "--out", str(out)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outs.append(out)
    return outs


def test_baseline_is_bit_reproducible(two_runs):
    a, b = (np.load(o / "ep_003.npz") for o in two_runs)
    for k in ("xy", "subgoal", "actions", "call_obs_embed_2", "distractor_xy"):
        assert np.array_equal(np.nan_to_num(a[k]), np.nan_to_num(b[k])), k
    ea, eb = (json.loads((o / "episodes.jsonl").read_text().splitlines()[0]) for o in two_runs)
    for k in ("success", "steps_to_success", "predictor_calls", "gd_iters"):
        assert ea[k] == eb[k], k
    assert ea["n_planning_calls"] == 22


def test_snapshot_restore_reproduces_observation(two_runs, tmp_path):
    out = tmp_path / "att"
    subprocess.run([PY, "scripts/collect_attempts.py", "--baseline-run", str(two_runs[0]), "--out", str(out),
                    "--tasks", "3", "--call-stride", "10", "--budgets", "B2"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    att = np.load(out / "attempts_task003.npz")
    ep = np.load(two_runs[0] / "ep_003.npz")
    assert att["render_err"].max() <= 2  # uint8 re-render of restored env vs logged frame
    for i, c in enumerate(att["call_idx"]):
        t = ep["call_env_step"][c]
        assert np.allclose(att["xy0"][i], ep["xy"][t], atol=1e-4)
        d0_base = ((ep["step_embed_2"][t] - ep["subgoal"][c]) ** 2).sum()
        assert abs(att["d0"][i] - d0_base) < 0.05 * max(1.0, d0_base)
