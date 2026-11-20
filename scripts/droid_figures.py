#!/usr/bin/env python
"""DROID figures for the website (outputs/rasr/droid/figures/):
  droid_labels.png  - why the original label is a "stay near the start" label: positive rate per planned subgoal time and
                      which expert frame the positives match, original ("any earlier frame") vs time-aligned label
  droid_paired.png  - per clip x planner seed Fréchet fidelity, unmodified planner vs reranked, for both labels."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/rasr/droid"
OUT = D / "figures"
STRIDE, W, FPS = 3, 3, 5  # level-2 stride (frames), aligned window, clip frame rate
C_ANY, C_AL, C_BASE, INK, GRID = "#2a78d6", "#eb6834", "#8a8984", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def labels(split="train"):
    eps = json.loads((D / "p1_reachability/results.json").read_text())["eps"]
    by_j = {"any": {}, "aligned": {}}
    match_any = []  # (plan frame H, matched expert frame) for positives under the original label
    for f in sorted(glob.glob(str(D / f"p1_candidates/{split}/clip_*.npz"))):
        d = np.load(f)
        zf, P = d["z_frames"], d["P"]
        for s in range(P.shape[0]):
            for j in range(1, P.shape[1] - 1):
                H = j * STRIDE
                da = ((zf[1:H + 1] - P[s, j]) ** 2).sum(-1)
                lo, hi = max(1, H - W), min(len(zf) - 1, H + W)
                dl = ((zf[lo:hi + 1] - P[s, j]) ** 2).sum(-1)
                by_j["any"].setdefault(j, []).append(da.min() <= eps)
                by_j["aligned"].setdefault(j, []).append(dl.min() <= eps)
                if da.min() <= eps:
                    match_any.append((H, 1 + int(np.flatnonzero(da <= eps)[0])))
    return eps, by_j, np.array(match_any)


def fig_labels():
    eps, by_j, m = labels()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    js = sorted(by_j["any"])
    t = np.array(js) * STRIDE / FPS
    for k, c, name in (("any", C_ANY, "original label (any earlier expert frame)"), ("aligned", C_AL, "time-aligned label (frame 3j ± 3)")):
        y = [100 * np.mean(by_j[k][j]) for j in js]
        ax[0].plot(t, y, color=c, lw=2, marker="o", ms=5, label=name)
    ax[0].set_xlabel("planned time of subgoal j (s)"); ax[0].set_ylabel("positive labels (%)")
    ax[0].set_title("Positive rate per subgoal", color=INK, fontsize=10, loc="left")
    ax[0].grid(axis="y", color=GRID, lw=.8); ax[0].legend(frameon=False, fontsize=8.5)
    first = m[:, 1] / FPS
    bins = np.arange(0.5, 37.5, 1) / FPS
    ax[1].hist(first, bins=bins, color=C_ANY, edgecolor="white", linewidth=1)
    early = 100 * np.mean(m[:, 1] <= 3)
    ax[1].axvspan(0, 3.5 / FPS, color=C_AL, alpha=.12, lw=0)
    ax[1].text(3.7 / FPS, ax[1].get_ylim()[1] * .9, f"{early:.0f}% of positives match\none of the first 3 frames (0.6 s)", color=INK, fontsize=8.5, va="top")
    ax[1].set_xlabel("first matching expert frame (s)"); ax[1].set_ylabel("positive labels")
    ax[1].set_title("Original label: which expert frame a positive matches", color=INK, fontsize=10, loc="left")
    ax[1].grid(axis="y", color=GRID, lw=.8)
    fig.tight_layout(); fig.savefig(OUT / "droid_labels.png", facecolor="white"); plt.close(fig)
    return {"early_pct": early, "pos_first_any": 100 * np.mean(by_j["any"][1]), "pos_last_any": 100 * np.mean(by_j["any"][js[-1]])}


def clips(sub, tag):
    r = {}
    for ps in (1, 2, 3):
        f = D / f"{sub}/test/{tag}_ms1_ps{ps}/clips.jsonl"
        if not f.exists():
            f = D / f"p2/test/{tag}_ms1_ps{ps}/clips.jsonl"
        for l in f.read_text().splitlines():
            c = json.loads(l); r[(ps, c["clip"])] = c
    return r


def fig_paired():
    S = json.loads((D / "summary.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    base = clips("p2", "lam0")
    for a, key, c, title in ((ax[0], "p2", C_ANY, "original label"), (ax[1], "p2_aligned", C_AL, "time-aligned label")):
        tag = S[key]["test_selected_tag"]; sel = S[key]["val_selection"]["selected"].replace(":", ", λ = ")
        r = clips(key, tag)
        k = sorted(set(base) & set(r))
        x = np.array([100 * base[i]["frechet_skill"] for i in k]); y = np.array([100 * r[i]["frechet_skill"] for i in k])
        ch = np.array([r[i]["changed"] for i in k], bool)
        lo, hi = min(x.min(), y.min()) - 3, max(x.max(), y.max()) + 3
        a.plot([lo, hi], [lo, hi], color=GRID, lw=1.2, zorder=0)
        a.scatter(x[~ch], y[~ch], s=36, color=C_BASE, alpha=.55, edgecolor="white", lw=1, label=f"same plan ({(~ch).sum()})")
        a.scatter(x[ch], y[ch], s=44, color=c, edgecolor="white", lw=1, label=f"plan changed ({ch.sum()})")
        p = S[key]["paired"]
        a.set_title(f"{title} ({sel})\n{p['wins']} better, {p['losses']} worse, {p['ties']} identical; Wilcoxon p = {p['wilcoxon_p']:.2f}", color=INK, fontsize=10, loc="left")
        a.set_xlabel("Fréchet fidelity, unmodified planner (%)"); a.grid(color=GRID, lw=.6)
        a.legend(frameon=False, fontsize=8.5, loc="upper left"); a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    ax[0].set_ylabel("Fréchet fidelity, reranked (%)")
    fig.tight_layout(); fig.savefig(OUT / "droid_paired.png", facecolor="white"); plt.close(fig)


def fig_clips():
    """Film strip of three test clips (frames every 1.2 s; first = start, last = goal). Frames saved by
    test_clip_frames.npz (droid_rasr.make_reader('test'), clips 0 / 5 / 11, every 6th of 37 frames)."""
    f = OUT / "test_clip_frames.npz"
    if not f.exists():
        return
    d = np.load(f)
    ks = list(d.keys())
    n = d[ks[0]].shape[0]
    fig, ax = plt.subplots(len(ks), n, figsize=(1.55 * n, 1.0 * len(ks) + .5))
    for r, k in enumerate(ks):
        for c in range(n):
            a = ax[r, c]
            a.imshow(d[k][c].transpose(1, 2, 0)); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(c in (0, n - 1)); sp.set_color(C_AL if c == n - 1 else C_ANY); sp.set_linewidth(2)
            if r == 0:
                a.set_title("start" if c == 0 else ("goal" if c == n - 1 else f"{c * 6 / FPS:.1f} s"), fontsize=9, color=INK)
        ax[r, 0].set_ylabel(f"test clip {k[4:]}", fontsize=8.5, color=INK)
    fig.tight_layout(pad=.4, w_pad=.3, h_pad=.3); fig.savefig(OUT / "droid_clips.png", facecolor="white", dpi=110); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(fig_labels()); fig_paired(); fig_clips(); print("wrote", OUT)
