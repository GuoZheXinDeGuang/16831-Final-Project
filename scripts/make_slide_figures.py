#!/usr/bin/env python3
"""Slide figures for the AntMaze v2/v3 study -> outputs/rasr/ant/v2/figures/*.png (also copied to results/ant/v2/figures).

fig1_main_test.png        test success (official 50 tasks x 3 planner seeds, model seed 42) per method, Wilson 95% CI
fig2_model_seeds.png      baseline vs final method for every (world-model seed, planner seed) pair
fig3_mechanism.png        repaired-subgoal similarity to visited states vs test success, per repair variant
fig4_val_R_attempts.png   val success of every reachability variant vs the final method (selection data)
Palette: reference instance (dataviz skill), light mode; final method = series blue, everything else neutral gray.
"""

from __future__ import annotations

import glob
import json
import shutil
import statistics as st
from math import comb, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs/rasr/ant/v2"
OUT = V2 / "figures"
BLUE, GRAY, GRAY_D, TEXT, TEXT2, GRID = "#257de0", "#b4b4b4", "#7a7a7a", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                     "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def episodes(split, method, seeds=(42, 43, 44)):
    out = {}
    for s in seeds:
        for f in glob.glob(str(V2 / split / method / f"seed{s}" / "shard_*/episodes.jsonl")):
            for line in open(f):
                if line.strip():
                    e = json.loads(line)
                    out[(s, e["task"])] = int(e["success"])
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b):
    keys = a.keys() & b.keys()
    g = sum(1 for k in keys if a[k] and not b[k]); l_ = sum(1 for k in keys if b[k] and not a[k])
    return g, l_, mcnemar(g, l_)


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig1():
    rows = [("H-JEPA baseline", "b1_baseline"), ("Replan every 5 steps (B2, 2× compute)", "b2_freq_replan"),
            ("Full replan on stall (B4)", "full_replan"), ("Original Ours: rerank + repair with R", "rasr_b20_dist"),
            ("Best R variant on val (repair, λ_R = 1)", "v2_lr1_dist"), ("Repair, no novelty (β = 0)", "repair_noR_b0"),
            ("Novelty always on (no stall trigger)", "v3_always"), ("Repair, stall trigger only", "v3_stallonly"),
            ("Repair, miss trigger only", "v3_missonly"), ("Ours: novelty repair (final, β = 50)", "repair_noR_b50")]
    base = episodes("test", "b1_baseline")
    data = []
    for label, m in rows:
        e = episodes("test", m)
        if not e:
            continue
        k, n = sum(e.values()), len(e)
        p, lo, hi = wilson(k, n)
        g, l_, pv = paired(e, base)
        data.append((label, m, p, lo, hi, n, g, l_, pv))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(data) + 1.4), dpi=160)
    for i, (label, m, p, lo, hi, n, g, l_, pv) in enumerate(data):
        y = len(data) - 1 - i
        final = m == "repair_noR_b50"
        ax.barh(y, p * 100, height=0.6, color=BLUE if final else GRAY, zorder=2)
        ax.plot([lo * 100, hi * 100], [y, y], color=TEXT if final else GRAY_D, lw=1.5, zorder=3)
        note = "" if m == "b1_baseline" else (f"   +{g}/−{l_}, p = {pv:.3f}" if pv >= 0.001 else f"   +{g}/−{l_}, p < 0.001")
        ax.text(hi * 100 + 1, y, f"{p * 100:.1f}%" + ("" if n == 150 else f" (n={n})") + note, va="center",
                fontsize=9.5, color=TEXT, fontweight="bold" if final else "normal")
    ax.axvline(data[0][2] * 100, color=GRAY_D, lw=1, ls="--", zorder=1)
    ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data][::-1])
    for t in ax.get_yticklabels():
        if t.get_text().startswith("Ours: novelty"):
            t.set_fontweight("bold")
    ax.set_xlim(0, 115); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("Test success rate, % (50 official tasks × 3 planner seeds; bars: 95% CI)")
    ax.set_title("Visual AntMaze: only novelty-driven repair beats the baseline", loc="left", fontsize=13, color=TEXT,
                 fontweight="bold")
    style(ax); fig.tight_layout(); fig.savefig(OUT / "fig1_main_test.png"); plt.close(fig)


def fig2():
    pairs = []
    for ms, split in ((42, "test"), (43, "test_ms43"), (44, "test_ms44")):
        b, o = episodes(split, "b1_baseline"), episodes(split, "repair_noR_b50")
        for s in (42, 43, 44):
            bs = [v for (ss, _), v in b.items() if ss == s]; os_ = [v for (ss, _), v in o.items() if ss == s]
            if len(bs) == 50 and len(os_) == 50:
                pairs.append((f"model {ms} · planner {s}", 2 * sum(bs), 2 * sum(os_)))
    allb = {(ms, k): v for ms, sp in ((42, "test"), (43, "test_ms43"), (44, "test_ms44")) for k, v in episodes(sp, "b1_baseline").items()}
    allo = {(ms, k): v for ms, sp in ((42, "test"), (43, "test_ms43"), (44, "test_ms44")) for k, v in episodes(sp, "repair_noR_b50").items()}
    g, l_, pv = paired(allo, allb)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(pairs) + 1.8), dpi=160)
    for i, (lab, bv, ov) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ax.plot([bv, ov], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(bv, y, s=70, color=GRAY_D, zorder=2, edgecolor="white", linewidth=2)
        ax.scatter(ov, y, s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=2)
        ax.text(ov + 1.2, y, f"+{ov - bv}", va="center", fontsize=9, color=TEXT2)
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[0] for p in pairs][::-1])
    ax.set_xlim(20, 75); ax.set_xlabel("Test success rate, % (50 tasks per row)")
    ax.scatter([], [], s=70, color=GRAY_D, label="H-JEPA baseline"); ax.scatter([], [], s=70, color=BLUE, label="Ours (novelty repair)")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    nb = sum(allb.values()) / len(allb) * 100; no = sum(allo.values()) / len(allo) * 100
    ax.set_title(f"Ours beats the baseline in {sum(1 for p in pairs if p[2] > p[1])}/{len(pairs)} world-model × planner seeds\n"
                 f"{no:.1f}% vs {nb:.1f}% over {len(allo)} paired episodes (McNemar p = {pv:.0e})", loc="left", fontsize=11.5,
                 color=TEXT, fontweight="bold")
    style(ax); fig.tight_layout(); fig.savefig(OUT / "fig2_model_seeds.png"); plt.close(fig)


def fig3():
    rows = [("baseline", "b1_baseline"), ("no novelty (β=0)", "repair_noR_b0"), ("Original Ours (R, λ_R=10)", "rasr_b20_dist"),
            ("R, λ_R=1", "v2_lr1_dist"), ("β = 20", "repair_noR"), ("Ours, β = 50", "repair_noR_b50")]
    pts = []
    for lab, m in rows:
        e = episodes("test", m)
        calls = [json.loads(l) for f in glob.glob(str(V2 / "test" / m / "seed*/shard_*/rasr_calls.jsonl")) for l in open(f) if l.strip()]
        nov = [c["repair_novelty"] for c in calls if c.get("repair") and c.get("repair_novelty") is not None]
        if e and nov:
            pts.append((lab, m, st.median(nov), 100 * sum(e.values()) / len(e)))
    base = episodes("test", "b1_baseline"); bp = 100 * sum(base.values()) / len(base)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.axhline(bp, color=GRAY_D, lw=1, ls="--"); ax.text(0.37, bp + 0.8, "baseline", color=TEXT2, fontsize=9, ha="right")
    for lab, m, x, y in pts:
        fin = m == "repair_noR_b50"
        ax.scatter(x, y, s=110, color=BLUE if fin else GRAY_D, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(lab, (x, y), xytext=(8, 6 if y > bp else -14), textcoords="offset points", fontsize=9.5, color=TEXT,
                    fontweight="bold" if fin else "normal")
    ax.set_xlim(-0.02, 0.4); ax.set_ylim(15, 65)
    ax.set_xlabel("Repaired subgoal: median similarity to already-visited states (0 = new region, 1 = visited)")
    ax.set_ylabel("Test success rate, %")
    ax.set_title("Repairs only help when they leave the visited region;\nthe R term pulls them back", loc="left",
                 fontsize=12, color=TEXT, fontweight="bold")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(OUT / "fig3_mechanism.png"); plt.close(fig)


def fig4():
    rows = [("H-JEPA baseline", "b1_baseline"), ("Original Ours (rerank + repair, log R)", "rasr_b20_dist"),
            ("Rerank only, expected cost", "v2_exp_rerank_dist"), ("Rerank + repair, expected cost", "v2_exp_full_dist"),
            ("Repair + R, expected cost", "v2_expected_dist"), ("Repair + R, miss memory", "v2_memory_dist"),
            ("Repair + R, hinge (R < 0.5 only)", "v2_hinge_dist"), ("Repair + R, λ_R = 10, no gate", "v2_lr10_nogate_dist"),
            ("Repair + R, λ_R = 10", "v2_lr10_dist"), ("Repair + R, λ_R = 3", "v2_lr3_dist"), ("Repair + R, λ_R = 1", "v2_lr1_dist"),
            ("Ours: repair, no R (β = 50)", "repair_noR_b50")]
    data = []
    for lab, m in rows:
        e = episodes("val", m, seeds=(42, 43))
        if len(e) >= 40:
            data.append((lab, m, 100 * sum(e.values()) / len(e)))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(data) + 1.5), dpi=160)
    for i, (lab, m, p) in enumerate(data):
        y = len(data) - 1 - i
        fin = m == "repair_noR_b50"
        ax.barh(y, p, height=0.6, color=BLUE if fin else GRAY, zorder=2)
        ax.text(p + 0.8, y, f"{p:.1f}%", va="center", fontsize=9.5, color=TEXT, fontweight="bold" if fin else "normal")
    ax.axvline(data[0][2], color=GRAY_D, lw=1, ls="--", zorder=1)
    ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data][::-1])
    ax.set_xlim(0, 50); ax.set_xlabel("Validation success rate, % (20 val tasks × 2 planner seeds)")
    ax.set_title("Every way of using the reachability predictor R\nscored at or below the baseline (val)", loc="left",
                 fontsize=12, color=TEXT, fontweight="bold")
    style(ax); fig.tight_layout(); fig.savefig(OUT / "fig4_val_R_attempts.png"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
        print("ok", f.__name__)
    dst = ROOT / "results/ant/v2/figures"
    dst.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.png"):
        shutil.copy(p, dst / p.name)
