#!/usr/bin/env python
"""Refit train_reachability.py's distance-logistic reference (same features, C = 1.0, train rows only) for a run
that predates saving it, verify it reproduces that run's saved predictions, and save dist_model.pt for the planner."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rasr import reachability as R  # noqa: E402
from rasr.rasr_solver import load_predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", required=True, help="dir with train/val/test attempt subdirs")
    ap.add_argument("--run", required=True, help="train_reachability.py output dir (results.json, predictions.npz)")
    ap.add_argument("--h-values", default="5,10,20,30")
    a = ap.parse_args()
    R.H_VALUES = tuple(int(x) for x in a.h_values.split(","))
    run = Path(a.run)
    eps = json.loads((run / "results.json").read_text())["eps"]
    rows = {s: R.expand_rows(R.load_attempts([Path(a.attempts) / s]), eps) for s in ("train", "val", "test")}

    def feats(r):
        d = np.log1p(((r["g"] - r["z"]) ** 2).sum(-1))
        c = R.controller_features(r)
        return np.column_stack([d, c, d[:, None] * c])

    lr = LogisticRegression(C=1.0, max_iter=2000).fit(feats(rows["train"]), rows["train"]["y"])
    saved = np.load(run / "predictions.npz")
    for s in ("val", "test"):
        err = np.abs(lr.predict_proba(feats(rows[s]))[:, 1] - saved[f"dist_logreg_{s}"]).max()
        print(f"{s}: max |refit - saved| = {err:.2e}")
        assert err < 1e-4, "refit does not reproduce the saved distance-logistic predictions"
    torch.save({"type": "dist_logreg", "coef": lr.coef_[0], "intercept": lr.intercept_, "eps": eps,
                "features": "log1p||g-z||^2, log H, log n_steps, log num_samples, d * controller"}, run / "dist_model.pt")
    m = load_predictor(run / "dist_model.pt")
    r = rows["test"]
    dev = next(m.buffers()).device
    p = m(torch.tensor(r["z"], device=dev), torch.tensor(r["g"], device=dev),
          torch.tensor(R.controller_features(r), device=dev)).cpu().numpy()
    err = np.abs(p - saved["dist_logreg_test"]).max()
    print(f"torch module vs saved test predictions: {err:.2e}")
    assert err < 1e-4


if __name__ == "__main__":
    main()
