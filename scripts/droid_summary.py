#!/usr/bin/env python
"""DROID summary -> outputs/rasr/droid/summary.json (baseline reproduction, P1 predictors, P2 val selection + test).
Offline open-loop metric only (Fréchet skill fidelity, ATE xyz); no closed-loop success exists for DROID."""

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/rasr/droid"


def mse(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, "n": len(v)}


def clips(t, ps, split="test", sub="p2"):
    f = D / f"{sub}/{split}/{t}_ms1_ps{ps}/clips.jsonl"
    if not f.exists() and t == "lam0":  # lam 0 does not use the predictor -> shared with p2/
        f = D / f"p2/{split}/{t}_ms1_ps{ps}/clips.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def p2_block(sub):
    from scipy.stats import wilcoxon
    vs = json.loads((D / f"{sub}/val_selection.json").read_text())
    p, l = vs["selected"].split(":")
    tsel = "lam0" if l == "0" else (f"lam{l}" if p == "mlp" and sub == "p2" else f"{p}_lam{l}")
    rows = {}
    for t in dict.fromkeys(("lam0", tsel)):
        per = {ps: clips(t, ps, sub=sub) for ps in (1, 2, 3)}
        rows[t] = {"per_ps_frechet": {ps: float(np.mean([c["frechet_skill"] for c in r])) for ps, r in per.items() if r},
                   "per_ps_ate": {ps: float(np.mean([c["ate_xyz"] for c in r])) for ps, r in per.items() if r},
                   "changed": int(sum(c["changed"] for r in per.values() for c in r)),
                   "n": int(sum(len(r) for r in per.values()))}
        rows[t]["frechet_skill"] = mse(list(rows[t]["per_ps_frechet"].values()))
        rows[t]["ate_xyz"] = mse(list(rows[t]["per_ps_ate"].values()))
    a = {(ps, c["clip"]): c["frechet_skill"] for ps in (1, 2, 3) for c in clips("lam0", ps, sub=sub)}
    b = {(ps, c["clip"]): c["frechet_skill"] for ps in (1, 2, 3) for c in clips(tsel, ps, sub=sub)}
    k = sorted(set(a) & set(b))
    d = np.array([b[x] - a[x] for x in k])
    nz = d[d != 0]
    return {"val_selection": vs, "test_selected_tag": tsel, "test": rows,
            "paired": {"n": len(k), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
                       "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                       "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) else None}}


def main():
    out = {}
    # ---- baseline: official eval.py, mean over planner seeds per model seed, then mean ± SE over model seeds
    base = {}
    for ms in (1, 1000, 10000):
        v = [float(next(csv.DictReader(open(f)))["frechet/skill"])
             for f in sorted(glob.glob(str(D / f"baseline/hjepa_l2_ms{ms}/plan_seed*/eval.csv")))]
        a = [float(next(csv.DictReader(open(f)))["ate/end_distance_xyz"])
             for f in sorted(glob.glob(str(D / f"baseline/hjepa_l2_ms{ms}/plan_seed*/eval.csv")))]
        base[str(ms)] = {"frechet_skill_per_ps": v, "frechet_skill": float(np.mean(v)), "ate_xyz": float(np.mean(a))}
    out["baseline"] = {"per_model_seed": base, "frechet_skill": mse([b["frechet_skill"] for b in base.values()]),
                       "paper_frechet_pct": "38.42 ± 0.94"}
    # ---- P1
    p1 = json.loads((D / "p1_reachability/results.json").read_text())
    sel = p1["mlp"][p1["selected_seed"]]
    out["p1"] = {"eps": p1["eps"], "splits": p1["splits"], "mlp_selected": sel,
                 "mlp_test_auroc_all_seeds": [m["test"]["auroc"] for m in p1["mlp"]],
                 "dist": json.loads((D / "p1_reachability/dist_results.json").read_text())}
    # ---- P2 (original "any" labels in p2/, time-aligned labels in p2_aligned/)
    out["p2"] = p2_block("p2")
    if (D / "p2_aligned/val_selection.json").exists():
        pa = json.loads((D / "p1_reachability_aligned/results.json").read_text())
        out["p1_aligned"] = {"eps": pa["eps"], "splits": pa["splits"], "mlp_selected": pa["mlp"][pa["selected_seed"]],
                             "mlp_test_auroc_all_seeds": [m["test"]["auroc"] for m in pa["mlp"]],
                             "dist": json.loads((D / "p1_reachability_aligned/dist_results.json").read_text())}
        out["p2_aligned"] = p2_block("p2_aligned")
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"baseline": out["baseline"]["frechet_skill"],
                      **{k: {"paired": out[k]["paired"], "test": {t: r["frechet_skill"] for t, r in out[k]["test"].items()}}
                         for k in ("p2", "p2_aligned") if k in out}}, indent=1))


if __name__ == "__main__":
    main()
