#!/usr/bin/env python
"""Distribution-shift check for P2: is the reachability model (trained only on subgoals the baseline
SELECTED) still accurate on the non-selected level-2 candidates that reranking scores?

Uses outputs/rasr/p4_shift/attempts (val tasks, B2 controller, same snapshots): alt = -1 is the
baseline's subgoal, alt = 0..3 are R-stratified non-selected candidates from the same call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rasr import reachability as R  # noqa: E402
from rasr.rasr_solver import load_predictor  # noqa: E402

OUT = Path("outputs/rasr/p4_shift")


def main():
    eps = json.loads(Path("outputs/rasr/p1_reachability/results.json").read_text())["eps"]
    att = R.load_attempts([OUT / "attempts"])
    alt = att["alt"]
    rows = R.expand_rows(att, eps)
    rows["alt"] = np.repeat(alt, len(R.H_VALUES))
    rows["alt_cost_rank"] = np.repeat(att["alt_cost_rank"], len(R.H_VALUES))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = load_predictor("outputs/rasr/p1_reachability/mlp_model.pt", dev)
    p_mlp = R.mlp_predict(mlp, rows, dev)
    k = np.load("outputs/rasr/p1_reachability/knn_model.npz")
    knn = R.KNNReachability(k=int(k["k"]), alpha=float(k["alpha"]))
    knn.fit({"z": k["train_z"], "g": k["train_g"], "y": k["train_y"], "H": k["train_H"],
             "n_steps": k["train_n_steps"], "num_samples": k["train_num_samples"]})
    p_knn = knn.predict_proba(rows)
    res = {"eps": eps, "n_attempts": int(len(alt)), "n_tasks": int(len(np.unique(att["task"])))}
    for grp, m in (("selected", rows["alt"] < 0), ("non_selected", rows["alt"] >= 0)):
        for H in (10, 30, "all"):
            mm = m & ((rows["H"] == H) if H != "all" else True)
            res[f"{grp}_H{H}"] = {"mlp": R.all_metrics(rows["y"][mm], p_mlp[mm]),
                                  "knn": R.all_metrics(rows["y"][mm], p_knn[mm])}
    # does R rank candidates of the same call correctly? (pairwise within call, H=10)
    m10 = rows["H"] == 10
    agree, n = 0, 0
    keys = list(zip(rows["task"][m10], rows["call_idx"][m10]))
    y10, p10 = rows["y"][m10], p_mlp[m10]
    from collections import defaultdict
    by = defaultdict(list)
    for i, kk in enumerate(keys):
        by[kk].append(i)
    for ii in by.values():
        for a in ii:
            for b in ii:
                if y10[a] == 1 and y10[b] == 0:
                    n += 1
                    agree += p10[a] > p10[b]
    res["within_call_pairwise_auc_H10"] = agree / max(n, 1)
    res["within_call_pairs"] = n
    # how good are the selected vs alternative subgoals actually
    res["reach_rate_H10"] = {"selected": float(y10[rows["alt"][m10] < 0].mean()),
                             "non_selected": float(y10[rows["alt"][m10] >= 0].mean())}
    (OUT / "shift_results.json").write_text(json.dumps(res, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    bins = np.linspace(0, 1, 11)
    for grp, m, col in (("selected (training distribution)", rows["alt"] < 0, "#1d2733"),
                        ("non-selected candidates", rows["alt"] >= 0, "#d1495b")):
        p, y = p_mlp[m], rows["y"][m]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ax.plot(xs, ys, marker="o", color=col,
                label=f"{grp}: AUROC {R.roc_auc(y, p):.3f}, ECE {R.ece(y, p):.3f} (n={m.sum()})")
    ax.set_xlabel("MLP predicted P(reachable)")
    ax.set_ylabel("empirical success frequency")
    ax.set_title("Reachability MLP under P2's distribution shift (val tasks, B2)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("outputs/rasr/figures/fig_shift_calibration.png", dpi=160)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith(("selected", "non"))}, indent=1))
    for kk in ("selected_Hall", "non_selected_Hall", "selected_H10", "non_selected_H10"):
        print(kk, {m: {x: round(res[kk][m][x], 3) for x in ("auroc", "brier", "ece", "pos_rate", "n")} for m in ("mlp", "knn")})


if __name__ == "__main__":
    main()
