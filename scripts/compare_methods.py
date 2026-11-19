#!/usr/bin/env python
"""P2/P3 comparison table: success (Wilson CI), paired exact McNemar vs the baseline,
compute (predictor calls, extra top-level solves, wall clock) and repair statistics.

  python scripts/compare_methods.py --split test \
      --method B1=outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42 --method B3=outputs/rasr/p2p3/test/rerank_l0.5 ...
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def load_eps(run: Path):
    eps = {}
    for p in run.glob("**/episodes.jsonl"):
        for l in p.read_text().splitlines():
            if l.strip():
                e = json.loads(l)
                eps[e["task"]] = e
    return eps


def load_calls(run: Path):
    rows = []
    for p in run.glob("**/rasr_calls.jsonl"):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - r) / d, 100 * (c + r) / d


def mcnemar_exact(b, c):
    """two-sided exact binomial test on discordant pairs (b: base ok / method fail, c: base fail / method ok)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def goal_dist_series(run: Path, task: int):
    f = next(run.glob(f"**/ep_{task:03d}.npz"))
    ep = np.load(f)
    return ((ep["call_obs_embed_2"] - ep["call_goal_embed_2"]) ** 2).sum(-1)


def summarize(name, run: Path, base_eps, frac=0.05, horizon=4):
    eps = load_eps(run)
    tasks = sorted(eps)
    n = len(tasks)
    k = sum(eps[t]["success"] for t in tasks)
    lo, hi = wilson(k, n)
    common = [t for t in tasks if t in base_eps]
    b = sum(base_eps[t]["success"] and not eps[t]["success"] for t in common)
    c = sum((not base_eps[t]["success"]) and eps[t]["success"] for t in common)
    calls = load_calls(run)
    trig = [r for r in calls if r.get("trigger")]
    repairs = [r for r in calls if r.get("repair")]
    # recovery: after a trigger, real latent goal distance improves by >= frac within `horizon` calls
    rec = []
    for r in trig:
        d = goal_dist_series(run, r["task"])
        i = r["call_idx"]
        if i + 1 >= len(d):
            continue
        before = d[: i + 1].min()
        after = d[i + 1: i + 1 + horizon]
        rec.append(bool(len(after) and after.min() < (1 - frac) * before))
    pc1 = np.mean([eps[t]["predictor_calls"]["1"] for t in tasks])
    pc2 = np.mean([eps[t]["predictor_calls"]["2"] for t in tasks])
    pr2 = np.mean([eps[t]["predictor_rows"]["2"] for t in tasks])
    pr1 = np.mean([eps[t]["predictor_rows"]["1"] for t in tasks])
    succ_steps = [eps[t]["steps_to_success"] for t in tasks if eps[t]["success"]]
    return {
        "method": name, "run": str(run), "n": n, "success": int(k), "success_rate": 100 * k / n,
        "wilson95": [lo, hi],
        "vs_base_lost": int(b), "vs_base_gained": int(c), "mcnemar_p": mcnemar_exact(b, c),
        "gained_tasks": [t for t in common if (not base_eps[t]["success"]) and eps[t]["success"]],
        "lost_tasks": [t for t in common if base_eps[t]["success"] and not eps[t]["success"]],
        "steps_to_success": float(np.mean(succ_steps)) if succ_steps else None,
        "planning_calls": float(np.mean([eps[t]["n_planning_calls"] for t in tasks])),
        "pred_calls_l1": float(pc1), "pred_calls_l2": float(pc2),
        "pred_rows_total_M": float((pr1 + pr2) / 1e6),
        "planning_wall_clock_s": float(np.mean([eps[t]["planning_wall_clock"] for t in tasks])),
        "triggers_per_ep": len(trig) / n,
        "trigger_kinds": {k2: sum(r["trigger"] == k2 for r in trig) for k2 in sorted({r["trigger"] for r in trig})},
        "repair_calls_per_ep": len(repairs) / n,
        "repair_kinds": {k2: sum(r["repair"] == k2 for r in repairs) for k2 in sorted({r["repair"] for r in repairs})},
        "fallback_calls_per_ep": sum(bool(r.get("fallback")) for r in calls) / n,
        "extra_toplevel_solves_per_ep": sum(r.get("extra_toplevel_solves", 0) for r in calls) / n,
        "rerank_changed_frac": (float(np.mean([r["rerank_changed"] for r in calls if "rerank_changed" in r]))
                                if any("rerank_changed" in r for r in calls) else 0.0),
        "recovery_rate": float(np.mean(rec)) if rec else None, "n_triggers": len(trig),
        "episodes_with_trigger": len({r["task"] for r in trig}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", required=True, help="NAME=run_dir (first = reference baseline)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    methods = [m.rsplit("=", 1) for m in args.method]
    base_eps = load_eps(Path(methods[0][1]))
    rows = [summarize(n, Path(r), base_eps) for n, r in methods]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    hdr = "| method | success | 95% CI | +gained / -lost vs B1 | McNemar p | L2 pred. calls/ep | extra top solves/ep | triggers/ep | recovery | plan time/ep (s) |"
    lines = [hdr, "|" + "---|" * 10]
    for r in rows:
        rec = "—" if r["recovery_rate"] is None else f"{100 * r['recovery_rate']:.0f}% (n={r['n_triggers']})"
        lines.append(f"| {r['method']} | {r['success']}/{r['n']} ({r['success_rate']:.0f}%) | {r['wilson95'][0]:.0f}–{r['wilson95'][1]:.0f} | "
                     f"+{r['vs_base_gained']} / −{r['vs_base_lost']} | {r['mcnemar_p']:.2f} | {r['pred_calls_l2']:.0f} | "
                     f"{r['extra_toplevel_solves_per_ep']:.1f} | {r['triggers_per_ep']:.2f} | {rec} | {r['planning_wall_clock_s']:.0f} |")
    md = "\n".join(lines) + "\n"
    out.with_suffix(".md").write_text(md)
    print(md)
    for r in rows:
        print(r["method"], "gained", r["gained_tasks"], "lost", r["lost_tasks"], "repairs", r["repair_kinds"], "triggers", r["trigger_kinds"])


if __name__ == "__main__":
    main()
