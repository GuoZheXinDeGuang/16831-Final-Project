"""Fast unit tests for rasr.reachability (CPU, synthetic data)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr import reachability as R  # noqa: E402


def _fake_attempts(n=60, d=8, h=30, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d)).astype(np.float32)
    g = z + rng.normal(scale=1.0, size=(n, d)).astype(np.float32)
    n_steps = rng.choice([2, 10, 90], n)
    # distance decays faster with more budget
    d0 = ((g - z) ** 2).sum(-1)
    k = np.arange(1, h + 1)
    rate = {2: 0.02, 10: 0.1, 90: 0.2}
    dists = np.stack([d0[i] * np.exp(-rate[n_steps[i]] * k) for i in range(n)]).astype(np.float32)
    return {
        "task": np.repeat(np.arange(n // 6), 6), "call_idx": np.tile(np.arange(6), n // 6),
        "budget_n_steps": n_steps, "budget_num_samples": np.where(n_steps == 90, 100, 20),
        "dists": dists, "z2": z, "subgoal": g,
        "solve_pred_rows_l1": np.ones((n, 3)), "solve_wall_clock": np.ones((n, 3)),
        "env_goal_reached": np.zeros((n, h), bool),
    }


def test_labels_are_prefix_monotone():
    att = _fake_attempts()
    rows = R.expand_rows(att, eps=1.0)
    # for a fixed attempt, y(H) must be non-decreasing in H (prefix minimum)
    for a in np.unique(rows["attempt_id"]):
        m = rows["attempt_id"] == a
        ys = rows["y"][m][np.argsort(rows["H"][m])]
        assert np.all(np.diff(ys) >= 0)
    assert set(np.unique(rows["H"])) == set(R.H_VALUES)
    # steps_to_reach consistent with label
    assert np.all((rows["steps_to_reach"] > 0) == (rows["y"] == 1))


def test_knn_and_mlp_learn_budget_effect():
    tr = R.expand_rows(_fake_attempts(600, seed=1), eps=1.0)
    va = R.expand_rows(_fake_attempts(120, seed=2), eps=1.0)
    knn = R.KNNReachability(k=15).fit(tr)
    p = knn.predict_proba(va)
    assert p.min() >= 0 and p.max() <= 1
    assert R.roc_auc(va["y"], p) > 0.7
    model, hist = R.train_mlp(tr, va, epochs=40, device="cpu", log=lambda *_: None)
    t = R.fit_temperature(model, va, device="cpu")
    assert t > 0
    pm = R.mlp_predict(model, va, device="cpu")
    assert R.roc_auc(va["y"], pm) > 0.7


def test_metrics_sane():
    y = np.array([0, 0, 1, 1])
    assert R.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(R.brier(y, y.astype(float))) < 1e-12
    assert R.ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
