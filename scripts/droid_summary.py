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


def clips(t, ps, split="test"):
    f = D / f"p2/{split}/{t}_ms1_ps{ps}/clips.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


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
    # ---- P2
    vs = json.loads((D / "p2/val_selection.json").read_text())
    p, l = vs["selected"].split(":")
    tsel = f"lam{l}" if p == "mlp" else f"{p}_lam{l}"
    rows = {}
    for t in ("lam0", tsel):
        per = {ps: clips(t, ps) for ps in (1, 2, 3)}
        rows[t] = {"per_ps_frechet": {ps: float(np.mean([c["frechet_skill"] for c in r])) for ps, r in per.items() if r},
                   "per_ps_ate": {ps: float(np.mean([c["ate_xyz"] for c in r])) for ps, r in per.items() if r},
                   "changed": int(sum(c["changed"] for r in per.values() for c in r)),
                   "n": int(sum(len(r) for r in per.values()))}
        rows[t]["frechet_skill"] = mse(list(rows[t]["per_ps_frechet"].values()))
        rows[t]["ate_xyz"] = mse(list(rows[t]["per_ps_ate"].values()))
    a = {(ps, c["clip"]): c["frechet_skill"] for ps in (1, 2, 3) for c in clips("lam0", ps)}
    b = {(ps, c["clip"]): c["frechet_skill"] for ps in (1, 2, 3) for c in clips(tsel, ps)}
    k = sorted(set(a) & set(b))
    d = np.array([b[x] - a[x] for x in k])
    from scipy.stats import wilcoxon
    nz = d[d != 0]
    out["p2"] = {"val_selection": vs, "test_selected_tag": tsel, "test": rows,
                 "paired": {"n": len(k), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
                            "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                            "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) else None}}
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"baseline": out["baseline"]["frechet_skill"], "p2": out["p2"]["paired"],
                      "test": {t: r["frechet_skill"] for t, r in rows.items()}}, indent=1))


if __name__ == "__main__":
    main()
