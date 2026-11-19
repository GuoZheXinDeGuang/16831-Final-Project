#!/usr/bin/env python
"""Aggregate P2/P3 methods over planner seeds 42/43/44 (model seed 42, same 50 tasks, same env seeds).

Seed 42 runs live in outputs/rasr/p0_baseline (B1) and outputs/rasr/p2p3/test/*; seeds 43/44 in
outputs/rasr/p4_seeds/<method>/seed<S>. Reports per-seed success, mean ± SE over seeds, a McNemar
test pooled over (task, seed) pairs vs B1 of the same seed, compute and repair statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from scripts.compare_methods import load_eps, mcnemar_exact, summarize, wilson  # noqa: E402

METHODS = [
    ("B1", "b1_baseline", "outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42"),
    ("B2", "b2_freq_replan", "outputs/rasr/p2p3/test/b2_freq_replan"),
    ("B3", "rerank_l0.25", "outputs/rasr/p2p3/test/rerank_l0.25"),
    ("B4", "full_replan", "outputs/rasr/p2p3/test/full_replan"),
    ("Ours", "rasr_b20", "outputs/rasr/p2p3/test/rasr_b20"),
    ("Ours−R", "repair_noR", "outputs/rasr/p2p3/test/repair_noR"),
]
SEEDS = (42, 43, 44)


def run_dir(key, s42, seed):
    return Path(s42) if seed == 42 else Path(f"outputs/rasr/p4_seeds/{key}/seed{seed}")


def main():
    base = {s: load_eps(run_dir("b1_baseline", METHODS[0][2], s)) for s in SEEDS}
    out = []
    for name, key, s42 in METHODS:
        per = {}
        b = c = 0
        allk = alln = 0
        summ = []
        for s in SEEDS:
            d = run_dir(key, s42, s)
            e = load_eps(d)
            if len(e) < 50:
                per[s] = None
                continue
            k = sum(x["success"] for x in e.values())
            per[s] = 100 * k / len(e)
            allk += k
            alln += len(e)
            for t, x in e.items():
                bb = base[s].get(t)
                if bb is None:
                    continue
                b += bb["success"] and not x["success"]
                c += (not bb["success"]) and x["success"]
            summ.append(summarize(name, d, base[s]))
        vals = [v for v in per.values() if v is not None]
        lo, hi = wilson(allk, alln) if alln else (np.nan, np.nan)
        out.append({
            "method": name, "per_seed": per, "n_seeds": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None,
            "pooled": f"{allk}/{alln}", "pooled_wilson95": [lo, hi],
            "gained": int(c), "lost": int(b), "mcnemar_p_pooled": mcnemar_exact(b, c),
            "pred_calls_l2": float(np.mean([r["pred_calls_l2"] for r in summ])) if summ else None,
            "extra_toplevel_solves_per_ep": float(np.mean([r["extra_toplevel_solves_per_ep"] for r in summ])) if summ else None,
            "triggers_per_ep": float(np.mean([r["triggers_per_ep"] for r in summ])) if summ else None,
            "recovery_rate": (float(np.mean([r["recovery_rate"] for r in summ if r["recovery_rate"] is not None]))
                              if any(r["recovery_rate"] is not None for r in summ) else None),
            "steps_to_success": float(np.mean([r["steps_to_success"] for r in summ])) if summ else None,
        })
    Path("outputs/rasr/p4_seeds").mkdir(parents=True, exist_ok=True)
    Path("outputs/rasr/p4_seeds/aggregate.json").write_text(json.dumps(out, indent=2))
    lines = ["| method | seed 42 | seed 43 | seed 44 | mean ± SE | pooled (150 eps) | +gained / −lost vs B1 | McNemar p | L2 calls/ep | recovery |",
             "|" + "---|" * 10]
    for r in out:
        ps = [("—" if r["per_seed"][s] is None else f"{r['per_seed'][s]:.0f}%") for s in SEEDS]
        ms = "—" if r["mean"] is None else (f"{r['mean']:.1f}" + (f" ± {r['se']:.1f}" if r["se"] is not None else ""))
        rec = "—" if r["recovery_rate"] is None else f"{100 * r['recovery_rate']:.0f}%"
        lines.append(f"| {r['method']} | {ps[0]} | {ps[1]} | {ps[2]} | {ms} | {r['pooled']} | +{r['gained']} / −{r['lost']} | "
                     f"{r['mcnemar_p_pooled']:.2f} | {r['pred_calls_l2'] or 0:.0f} | {rec} |")
    md = "\n".join(lines) + "\n"
    Path("outputs/rasr/p4_seeds/aggregate.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
