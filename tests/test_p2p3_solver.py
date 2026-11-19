"""RASR solver regression (GPU, ~1.5 min): with mode=baseline the RASR solver must reproduce the
upstream planner bit-for-bit; with mode=rasr it must run and log triggers/repairs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(out, cfg=None, task="1"):
    cmd = [PY, "scripts/run_baseline.py", "--tasks", task, "--no-snapshots", "--out", str(out)]
    if cfg:
        cmd += ["--rasr-config", cfg]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.load(out / f"ep_{int(task):03d}.npz")


def test_baseline_mode_is_upstream(tmp_path):
    a = _run(tmp_path / "up")
    b = _run(tmp_path / "rasr_off", "configs/rasr/baseline_copy.yaml")
    for k in ("xy", "subgoal", "actions", "call_obs_embed_2"):
        assert np.array_equal(a[k], b[k]), k


def test_rasr_mode_runs_and_logs(tmp_path):
    _run(tmp_path / "rasr", "configs/rasr/rasr_b20_gd.yaml")
    rows = [json.loads(l) for l in (tmp_path / "rasr" / "rasr_calls.jsonl").read_text().splitlines()]
    assert len(rows) == 22
    assert all("trigger" in r for r in rows)
    assert any(r.get("R_sel") is not None for r in rows)
