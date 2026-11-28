#!/usr/bin/env python3
"""Aggregate the v2 (A6000 rerun) AntMaze runs: outputs/rasr/ant/v2/<split>/<method>/seed<S>/shard_*/episodes.jsonl.

Per method: success per seed, pooled success, paired exact McNemar vs the reference method on the episodes both have
(same task + planner seed), L2 predictor calls / episode, and repair diagnostics from rasr_calls.jsonl when present.
Usage: python3 scripts/aggregate_v2.py <split> [--ref b1_baseline] [--out file.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs/rasr/ant/v2"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(split: str):
    runs = {}
    for d in sorted(glob.glob(str(ROOT / split / "*" / "seed*"))):
        method, seed = Path(d).parent.name, int(Path(d).name[4:])
        eps, calls = {}, []
        for f in glob.glob(f"{d}/shard_*/episodes.jsonl"):
            for line in open(f):
                if line.strip():
                    e = json.loads(line)
                    eps[e["task"]] = e
        for f in glob.glob(f"{d}/shard_*/rasr_calls.jsonl"):
            for line in open(f):
                if line.strip():
                    calls.append(json.loads(line))
        if eps:
            runs.setdefault(method, {})[seed] = (eps, calls)
    return runs


def repair_stats(calls):
    flat = []
    for c in calls:
        flat.extend(c.get("calls", [c]) if isinstance(c, dict) else c)
    trig = [c for c in flat if c.get("trigger")]
    rep = [c for c in flat if c.get("repair")]
    fb = [c for c in rep if c.get("fallback")]
    return {"triggers": len(trig), "repairs": len(rep), "fallback_frac": (len(fb) / len(rep)) if rep else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--ref", default="b1_baseline")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = load(a.split)
    ref = runs.get(a.ref, {})
    seeds = sorted({s for m in runs.values() for s in m})
    lines = [f"| method | {' | '.join(f'seed {s}' for s in seeds)} | pooled | +gained / −lost vs {a.ref} | McNemar p | L2 calls/ep | fallback frac |",
             "|---|" + "---|" * (len(seeds) + 5)]
    order = sorted(runs, key=lambda m: (m != a.ref, m))
    for m in order:
        cells, ns, nk, gain, lost, l2, calls_all = [], 0, 0, 0, 0, [], []
        for s in seeds:
            if s not in runs[m]:
                cells.append("—")
                continue
            eps, calls = runs[m][s]
            k = sum(e["success"] for e in eps.values())
            cells.append(f"{k}/{len(eps)}")
            ns += len(eps); nk += k
            l2 += [e["predictor_calls"]["2"] for e in eps.values()]
            calls_all += calls
            if s in ref:
                for t, e in eps.items():
                    if t in ref[s][0]:
                        r = ref[s][0][t]["success"]
                        gain += e["success"] and not r
                        lost += r and not e["success"]
        p = mcnemar_exact(gain, lost)
        rs = repair_stats(calls_all) if calls_all else {"fallback_frac": float("nan")}
        lines.append(f"| {m} | {' | '.join(cells)} | {nk}/{ns} ({100 * nk / ns:.1f}%) | +{gain} / −{lost} | {p:.3f} | "
                     f"{sum(l2) / len(l2):.0f} | {rs['fallback_frac']:.2f} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
