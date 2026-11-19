#!/usr/bin/env python
"""P1: build labelled reachability dataset, fit kNN + MLP (+ distance-logistic reference), evaluate.

Splits are by start-goal task set (no task appears in two splits):
  train = RASR train tasks (generator seed 1001), val = RASR val tasks (seed 2002),
  test  = the official 50 H-JEPA FourRoom eval tasks (P0 baseline episodes).
Every hyper-parameter / threshold / temperature is chosen on train/val only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rasr import reachability as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-attempts", default="outputs/rasr/p1_attempts/train")
    ap.add_argument("--val-attempts", default="outputs/rasr/p1_attempts/val")
    ap.add_argument("--test-attempts", default="outputs/rasr/p1_attempts/test")
    ap.add_argument("--train-baseline", default="outputs/rasr/p1_source/train")
    ap.add_argument("--radius-px", type=float, default=8.0,
                    help="eps radius in env xy units (FourRoom px; AntMaze maze units)")
    ap.add_argument("--radius-list", default="4,8,16", help="radii for the eps sensitivity table")
    ap.add_argument("--h-values", default="5,10,20,30", help="execution horizons H (env steps)")
    ap.add_argument("--out", default="outputs/rasr/p1_reachability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-seeds", type=int, default=3)
    args = ap.parse_args()
    R.H_VALUES = tuple(int(x) for x in args.h_values.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()

    # ---------------------------------------------------------------- eps calibration (train only)
    cal = {r: R.calibrate_eps([Path(args.train_baseline)], radius_px=r) for r in [float(x) for x in args.radius_list.split(",")]}
    eps = cal[args.radius_px]["eps"]
    print("eps calibration:", json.dumps(cal, indent=1))

    att = {s: R.load_attempts([Path(p)]) for s, p in
           (("train", args.train_attempts), ("val", args.val_attempts), ("test", args.test_attempts))}
    rows = {s: R.expand_rows(a, eps) for s, a in att.items()}
    # leakage guard: splits come from different task files; assert no start-goal configuration is shared
    sg = {s: {tuple(np.round(np.r_[a["xy0"][i], a["goal_xy_task"][i]], 0)) for i in np.nonzero(a["call_idx"] == 0)[0]}
          for s, a in att.items()}
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sg[s1] & sg[s2]
        assert not shared, f"start-goal configs shared between {s1} and {s2}: {shared}"
    split_info = {s: {"n_tasks": int(len(np.unique(att[s]["task"]))), "n_attempts": int(len(att[s]["task"])),
                      "n_rows": int(len(rows[s]["y"])), "pos_rate": float(rows[s]["y"].mean()),
                      "n_pos": int(rows[s]["y"].sum()), "n_neg": int((1 - rows[s]["y"]).sum()),
                      "collection_wall_clock_s": float(att[s]["wall_clock"].sum())} for s in att}
    # label balance per (H, B) and eps sensitivity
    cells = {}
    for s in rows:
        r = rows[s]
        for H in R.H_VALUES:
            for ns in np.unique(r["n_steps"]):
                m = (r["H"] == H) & (r["n_steps"] == ns)
                cells.setdefault(s, {})[f"H{H}_iters{ns}"] = {"n": int(m.sum()), "pos_rate": float(r["y"][m].mean())}
    sens = {f"r{rr:g}_eps{c['eps']:.2f}": {s: float(R.expand_rows(att[s], c["eps"])["y"].mean()) for s in att}
            for rr, c in cal.items()}
    print(json.dumps(split_info, indent=1))
    print("eps sensitivity (pos rate):", json.dumps(sens, indent=1))

    results = {"eps": eps, "eps_calibration": cal, "eps_sensitivity_pos_rate": sens,
               "splits": split_info, "label_balance_by_cell": cells, "models": {}}
    preds = {}

    # ---------------------------------------------------------------- distance-only logistic reference
    from sklearn.linear_model import LogisticRegression

    def dist_feats(r):
        d = np.log1p(((r["g"] - r["z"]) ** 2).sum(-1))
        c = R.controller_features(r)
        return np.column_stack([d, c, d[:, None] * c])

    lr = LogisticRegression(C=1.0, max_iter=2000).fit(dist_feats(rows["train"]), rows["train"]["y"])
    preds["dist_logreg"] = {s: lr.predict_proba(dist_feats(rows[s]))[:, 1] for s in rows}
    torch.save({"type": "dist_logreg", "coef": lr.coef_[0], "intercept": lr.intercept_, "eps": eps,
                "features": "log1p||g-z||^2, log H, log n_steps, log num_samples, d * controller"}, out / "dist_model.pt")

    # ---------------------------------------------------------------- kNN (k chosen on val NLL)
    t0 = time.time()
    knn = R.KNNReachability().fit(rows["train"])
    k_scores = {}
    for k in (5, 10, 25, 50, 100, 200):
        p = knn.predict_proba(rows["val"], k=k)
        k_scores[k] = R.all_metrics(rows["val"]["y"], p)
    best_k = min(k_scores, key=lambda k: k_scores[k]["nll"])
    knn.k = best_k
    preds["knn"] = {s: knn.predict_proba(rows[s]) for s in rows}
    knn_time = time.time() - t0
    results["knn_k_search_val"] = {str(k): v for k, v in k_scores.items()}
    results["knn_best_k"] = best_k
    np.savez_compressed(out / "knn_model.npz", k=best_k, alpha=knn.alpha, mean=knn.mean_, std=knn.std_,
                        train_z=rows["train"]["z"], train_g=rows["train"]["g"], train_y=rows["train"]["y"],
                        train_H=rows["train"]["H"], train_n_steps=rows["train"]["n_steps"],
                        train_num_samples=rows["train"]["num_samples"], eps=eps)
    print(f"kNN best k={best_k} (val NLL) fit+search {knn_time:.1f}s")

    # ---------------------------------------------------------------- MLP (frozen backbone features)
    mlp_runs = []
    for s in range(args.mlp_seeds):
        t0 = time.time()
        model, hist = R.train_mlp(rows["train"], rows["val"], seed=args.seed + s, device=device)
        raw_val = R.mlp_predict(model, rows["val"], device)
        temp = R.fit_temperature(model, rows["val"], device)
        mlp_runs.append({"seed": args.seed + s, "model": model, "hist": hist, "temperature": temp,
                         "val_nll_raw": R.all_metrics(rows["val"]["y"], raw_val)["nll"],
                         "train_time_s": time.time() - t0})
        print(f"MLP seed {args.seed + s}: epochs={len(hist)} best val_bce={min(h['val_bce'] for h in hist):.4f} T={temp:.3f}")
    best = min(mlp_runs, key=lambda r: min(h["val_bce"] for h in r["hist"]))  # selected on val
    model = best["model"]
    torch.save({"state_dict": model.state_dict(), "d_latent": int(rows["train"]["z"].shape[1]),
                "hidden": 256, "dropout": 0.1, "eps": eps, "temperature": best["temperature"],
                "inputs": "z2_t, subgoal (level-2 latent), log H, log n_steps, log num_samples",
                "seed": best["seed"]}, out / "mlp_model.pt")
    T = model.temperature.item()
    model.temperature.fill_(1.0)
    preds["mlp_raw"] = {s: R.mlp_predict(model, rows[s], device) for s in rows}
    model.temperature.fill_(T)
    preds["mlp"] = {s: R.mlp_predict(model, rows[s], device) for s in rows}
    # per-seed test metrics (variance across MLP seeds)
    results["mlp_seed_runs"] = []
    for r in mlp_runs:
        p = R.mlp_predict(r["model"], rows["test"], device)
        results["mlp_seed_runs"].append({"seed": r["seed"], "temperature": r["temperature"],
                                         "epochs": len(r["hist"]), "train_time_s": r["train_time_s"],
                                         "test": R.all_metrics(rows["test"]["y"], p)})
    (out / "mlp_train_history.json").write_text(json.dumps(best["hist"]))

    # ---------------------------------------------------------------- evaluation
    for name, pr in preds.items():
        res = {s: R.all_metrics(rows[s]["y"], pr[s]) for s in rows}
        yt, pt, tt = rows["test"]["y"], np.clip(pr["test"], 1e-6, 1 - 1e-6), rows["test"]["task"]
        res["test_ci95"] = {
            "auroc": R.bootstrap_ci_by_task(yt, pt, tt, R.roc_auc),
            "pr_auc": R.bootstrap_ci_by_task(yt, pt, tt, R.pr_auc),
            "brier": R.bootstrap_ci_by_task(yt, pt, tt, R.brier),
        }
        res["test_by_cell"] = {}
        for H in R.H_VALUES:
            for ns in np.unique(rows["test"]["n_steps"]):
                m = (rows["test"]["H"] == H) & (rows["test"]["n_steps"] == ns)
                res["test_by_cell"][f"H{H}_iters{ns}"] = R.all_metrics(yt[m], pt[m])
        cell_aucs = [(v["auroc"], v["n"]) for v in res["test_by_cell"].values() if np.isfinite(v["auroc"])]
        res["test_within_cell_auroc_mean"] = float(np.mean([a for a, _ in cell_aucs])) if cell_aucs else float("nan")
        res["test_within_cell_auroc_weighted"] = (float(np.average([a for a, _ in cell_aucs], weights=[n for _, n in cell_aucs]))
                                                  if cell_aucs else float("nan"))
        # decision threshold chosen on VAL (max F1) for later P2/P3 use
        yv, pv = rows["val"]["y"], pr["val"]
        ths = np.linspace(0.05, 0.95, 91)
        f1 = [_f1(yv, pv >= t) for t in ths]
        th = float(ths[int(np.argmax(f1))])
        res["val_threshold_maxF1"] = th
        res["test_at_val_threshold"] = {"f1": _f1(yt, pt >= th), "accuracy": float(np.mean((pt >= th) == yt)),
                                        "tpr": float(np.mean(pt[yt == 1] >= th)), "fpr": float(np.mean(pt[yt == 0] >= th))}
        results["models"][name] = res
    np.savez_compressed(out / "predictions.npz", **{f"{n}_{s}": preds[n][s] for n in preds for s in rows},
                        **{f"y_{s}": rows[s]["y"] for s in rows}, **{f"task_{s}": rows[s]["task"] for s in rows},
                        **{f"H_{s}": rows[s]["H"] for s in rows}, **{f"n_steps_{s}": rows[s]["n_steps"] for s in rows})
    results["total_time_s"] = time.time() - t_start
    R.save_json(out / "results.json", results)
    write_table(out, results)
    plots(out, rows, preds)
    print((out / "comparison.md").read_text())


def _f1(y, yhat):
    tp = float(np.sum((yhat == 1) & (y == 1)))
    fp = float(np.sum((yhat == 1) & (y == 0)))
    fn = float(np.sum((yhat == 0) & (y == 1)))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def write_table(out, results):
    lines = ["| model | split | AUROC | PR-AUC | Brier | ECE | NLL |", "|---|---|---|---|---|---|---|"]
    for name in ("dist_logreg", "knn", "mlp_raw", "mlp"):
        r = results["models"][name]
        for s in ("val", "test"):
            m = r[s]
            lines.append(f"| {name} | {s} | {m['auroc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.4f} | {m['ece']:.4f} | {m['nll']:.4f} |")
    lines.append("")
    lines.append("Within-(H,B)-cell test AUROC (discrimination beyond the controller condition; mean over cells with both labels):")
    for name in ("dist_logreg", "knn", "mlp"):
        r = results["models"][name]
        lines.append(f"- {name}: mean {r['test_within_cell_auroc_mean']:.3f}, n-weighted {r['test_within_cell_auroc_weighted']:.3f}")
    lines.append("")
    lines.append("Test 95% CI (cluster bootstrap over start-goal tasks):")
    for name in ("dist_logreg", "knn", "mlp"):
        c = results["models"][name]["test_ci95"]
        lines.append(f"- {name}: AUROC {c['auroc'][0]:.3f}–{c['auroc'][1]:.3f}, PR-AUC {c['pr_auc'][0]:.3f}–{c['pr_auc'][1]:.3f}, Brier {c['brier'][0]:.4f}–{c['brier'][1]:.4f}")
    (Path(out) / "comparison.md").write_text("\n".join(lines) + "\n")


def plots(out, rows, preds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    names = {"dist_logreg": "distance-logistic (ref.)", "knn": "kNN", "mlp_raw": "MLP (raw)", "mlp": "MLP (+val temp. scaling)"}
    colors = {"dist_logreg": "#888888", "knn": "#1f77b4", "mlp_raw": "#ff9f4a", "mlp": "#d62728"}
    y = rows["test"]["y"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    bins = np.linspace(0, 1, 11)
    for n in names:
        p = preds[n]["test"]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys, cs = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 10:
                xs.append(p[m].mean()); ys.append(y[m].mean()); cs.append(m.sum())
        ax.plot(xs, ys, marker="o", ms=4, color=colors[n], label=f"{names[n]} (ECE {R.ece(y, p):.3f})")
    ax.set_xlabel("predicted P(reachable)")
    ax.set_ylabel("empirical success frequency")
    ax.set_title(f"Reliability diagram, test (n={len(y)} rows, 50 tasks)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for n in names:
        fpr, tpr, _ = roc_curve(y, preds[n]["test"])
        ax.plot(fpr, tpr, color=colors[n], label=f"{names[n]} AUROC {R.roc_auc(y, preds[n]['test']):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.set_title("ROC, test")
    ax.legend(fontsize=7)
    ax = axes[2]
    for n in names:
        pr, rc, _ = precision_recall_curve(y, preds[n]["test"])
        ax.plot(rc, pr, color=colors[n], label=f"{names[n]} AP {R.pr_auc(y, preds[n]['test']):.3f}")
    ax.axhline(y.mean(), ls="--", color="0.6", lw=1, label=f"base rate {y.mean():.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision-recall, test")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out) / "calibration_roc_pr_test.png", dpi=150)

    # per-(H,B) reliability: predicted vs empirical per cell
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    r = rows["test"]
    for n in ("knn", "mlp"):
        xs, ys = [], []
        for H in R.H_VALUES:
            for ns in np.unique(r["n_steps"]):
                m = (r["H"] == H) & (r["n_steps"] == ns)
                xs.append(preds[n]["test"][m].mean()); ys.append(y[m].mean())
        ax.scatter(xs, ys, color=colors[n], label=names[n], s=18)
    ax.set_xlabel("mean predicted P(reachable) in (H,B) cell")
    ax.set_ylabel("empirical success rate in cell")
    ax.set_title("Controller-condition calibration (12 (H,B) cells, test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "calibration_by_HB_cell_test.png", dpi=150)
    print("wrote plots to", out)


if __name__ == "__main__":
    main()
