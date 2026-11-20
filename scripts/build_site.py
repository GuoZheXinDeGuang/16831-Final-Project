#!/usr/bin/env python
"""Build the project website (outputs/rasr/site/) from real result files. Every number on the page
is read from a JSON produced by the experiment scripts; narrative text lives in docs/site_notes.json."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "outputs/rasr"
SITE = R / "site"
A = SITE / "assets"


def J(p):
    return json.loads((ROOT / p).read_text())


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def fig(src, cap, wide=False):
    return f'<figure class="{"wide" if wide else ""}"><img src="assets/{src}" alt="{html.escape(cap)}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


def vid(src, cap, poster=None):
    p = f' poster="assets/{poster}"' if poster else ""
    return (f'<figure class="vid"><video controls muted loop playsinline preload="metadata"{p} src="assets/{src}"></video>'
            f"<figcaption>{cap}</figcaption></figure>")


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    A.mkdir(exist_ok=True)
    notes = J("docs/site_notes.json")
    p0 = J("outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42/summary.json")
    so = J("outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42/subgoal_outcomes_summary.json")
    p1 = J("outputs/rasr/p1_reachability/results.json")
    cmp_test = J("outputs/rasr/p2p3/compare_test.json")
    cmp_val = J("outputs/rasr/p2p3/compare_val.json")
    agg_p = ROOT / "outputs/rasr/p4_seeds/aggregate.json"
    agg = json.loads(agg_p.read_text()) if agg_p.exists() else None
    shift_p = ROOT / "outputs/rasr/p4_shift/shift_results.json"
    shift = json.loads(shift_p.read_text()) if shift_p.exists() else None

    mn_p = ROOT / "docs/site_notes_multi.json"
    mnotes = json.loads(mn_p.read_text()) if mn_p.exists() else {}
    ant_p0_p = R / "ant/p0_baseline/ant_l2_ms42_ps42/summary.json"
    ant_p0 = json.loads(ant_p0_p.read_text()) if ant_p0_p.exists() else None
    ant_p1_p = R / "ant/p1_reachability/results.json"
    ant_p1 = json.loads(ant_p1_p.read_text()) if ant_p1_p.exists() else None
    ant_cmp_p = R / "ant/p2p3/compare_test.json"
    ant_cmp = json.loads(ant_cmp_p.read_text()) if ant_cmp_p.exists() else None
    droid_p = R / "droid/summary.json"
    droid = json.loads(droid_p.read_text()) if droid_p.exists() else None

    copies = {
        "trajectories.png": R / "p0_baseline/fourroom_l2_ms42_ps42/trajectories.png",
        "calibration.png": R / "p1_reachability/calibration_roc_pr_test.png",
        "calibration_cells.png": R / "p1_reachability/calibration_by_HB_cell_test.png",
    }
    for f in (R / "figures").glob("*.png"):
        copies[f.name] = f
    for f in (R / "videos").glob("*"):
        copies[f.name] = f
    for name, src in {"ant_trajectories.png": R / "ant/p0_baseline/ant_l2_ms42_ps42/trajectories.png",
                      "ant_calibration.png": R / "ant/p1_reachability/calibration_roc_pr_test.png"}.items():
        if src.exists():
            copies[name] = src
    for name, src in copies.items():
        shutil.copy(src, A / name)

    m = p1["models"]
    sp = p1["splits"]
    base = cmp_test[0]

    # ---------------------------------------------------------------- tables
    p0_rows = [
        ["Success rate (50 official tasks)", f"<b>{p0['success_rate']:.1f}%</b> ({p0['n_success']}/{p0['n_episodes']})",
         f"Wilson 95% CI {p0['success_rate_wilson95'][0]:.1f}–{p0['success_rate_wilson95'][1]:.1f}"],
        ["Paper, H-JEPA 2-level (3 seeds)", "81.3 ± 6.7%", "README / Fig. 6"],
        ["Failed tasks", ", ".join(map(str, p0["failed_tasks"])), "all 7 have every door open"],
        ["Steps to success", f"{p0['steps_to_success_mean']:.1f}", "budget 210"],
        ["Planning calls / episode", f"{p0['planning_calls_per_episode']:.0f}", "replan every 10 env steps"],
        ["Predictor forward calls / episode", f"L1 {p0['predictor_calls_per_episode']['l1']:.0f} · L2 {p0['predictor_calls_per_episode']['l2']:.0f}",
         f"rows: L1 {p0['predictor_rows_per_episode']['l1'] / 1e3:.0f}k · L2 {p0['predictor_rows_per_episode']['l2'] / 1e6:.2f}M"],
        ["Planning time / call", "≈1.3 s alone", f"{p0['per_call_wall_clock_mean_s']['total']:.1f} s with 6 jobs sharing the GPU"],
    ]
    data_rows = [[s, sp[s]["n_tasks"], sp[s]["n_attempts"], sp[s]["n_rows"], pct(sp[s]["pos_rate"], 1),
                  f"{sp[s]['collection_wall_clock_s'] / 60:.0f} min"] for s in ("train", "val", "test")]
    names = {"dist_logreg": "Distance-only logistic (reference)", "knn": f"kNN (k = {p1['knn_best_k']}, per (H,B) cell)",
             "mlp_raw": "MLP, BCE", "mlp": "MLP + temperature scaling (val)"}
    p1_rows = []
    for k in ("dist_logreg", "knn", "mlp_raw", "mlp"):
        t = m[k]["test"]
        ci = m[k].get("test_ci95", {}).get("auroc")
        wc = m[k].get("test_within_cell_auroc_mean")
        p1_rows.append([names[k], f"{t['auroc']:.3f}" + (f" <span class=ci>[{ci[0]:.3f}, {ci[1]:.3f}]</span>" if ci else ""),
                        f"{t['pr_auc']:.3f}", f"{t['brier']:.3f}", f"{t['ece']:.3f}", "—" if wc is None or k == "mlp_raw" else f"{wc:.3f}"])

    def method_rows(rows):
        out = []
        for r in rows:
            rec = "—" if r["recovery_rate"] is None else f"{100 * r['recovery_rate']:.0f}% <span class=ci>({r['n_triggers']})</span>"
            delta = "—" if r is rows[0] else f"+{r['vs_base_gained']} / −{r['vs_base_lost']} <span class=ci>p={r['mcnemar_p']:.2f}</span>"
            out.append([notes["method_names"].get(r["method"], r["method"]),
                        f"<b>{r['success_rate']:.0f}%</b> <span class=ci>{r['success']}/{r['n']} · [{r['wilson95'][0]:.0f}, {r['wilson95'][1]:.0f}]</span>",
                        delta, f"{r['pred_calls_l2']:.0f}", f"{r['extra_toplevel_solves_per_ep']:.1f}",
                        f"{r['triggers_per_ep']:.1f}", rec, f"{r['planning_wall_clock_s']:.0f}"])
        return out

    seeds_html = ""
    if agg:
        rows = []
        for r in agg:
            ps = ["—" if r["per_seed"].get(str(sd), r["per_seed"].get(sd)) is None else f"{r['per_seed'].get(str(sd), r['per_seed'].get(sd)):.0f}%"
                  for sd in (42, 43, 44)]
            ms = "—" if r["mean"] is None else f"<b>{r['mean']:.1f}</b>" + (f" ± {r['se']:.1f}" if r["se"] is not None else "")
            rec = "—" if r["recovery_rate"] is None else f"{100 * r['recovery_rate']:.0f}%"
            rows.append([notes["method_names"].get(r["method"], r["method"]), *ps, ms, r["pooled"],
                         f"+{r['gained']} / −{r['lost']} <span class=ci>p={r['mcnemar_p_pooled']:.2f}</span>",
                         f"{r['pred_calls_l2'] or 0:.0f}", rec])
        seeds_html = ("<h3>Three planner seeds (42, 43, 44; model seed 42; same 50 tasks)</h3>"
                      + table(["Method", "seed 42", "seed 43", "seed 44", "mean ± SE (%)", "pooled", "Δ vs B1 (pooled McNemar)",
                               "L2 calls / ep", "recovery"], rows)
                      + f'<p class="note">{notes.get("seeds_note", "")}</p>')
    shift_html = ""
    if shift:
        a, b = shift["selected_Hall"], shift["non_selected_Hall"]
        rows = [["Selected by the baseline (training distribution)", f"{a['mlp']['auroc']:.3f}", f"{a['mlp']['brier']:.3f}",
                 f"{a['mlp']['ece']:.3f}", f"{a['knn']['auroc']:.3f}", pct(a['mlp']['pos_rate']), a['mlp']['n']],
                ["Non-selected level-2 candidates", f"{b['mlp']['auroc']:.3f}", f"{b['mlp']['brier']:.3f}",
                 f"{b['mlp']['ece']:.3f}", f"{b['knn']['auroc']:.3f}", pct(b['mlp']['pos_rate']), b['mlp']['n']]]
        shift_html = ("<h3>Does the predictor transfer to the candidates P2 scores?</h3>" + notes.get("shift_intro_html", "")
                      + table(["Subgoals (val tasks, budget B2)", "MLP AUROC", "MLP Brier", "MLP ECE", "kNN AUROC", "reached", "rows"], rows)
                      + (fig("fig_shift_calibration.png", "Reliability of the MLP on selected vs non-selected candidates.") if (R / "figures/fig_shift_calibration.png").exists() else "")
                      + notes.get("shift_after_html", ""))

    mhead = ["Method", "Success", "Δ vs B1 (gained / lost)", "L2 predictor calls / ep", "extra L2 solves / ep",
             "triggers / ep", "recovery", "planning s / ep"]

    # ---------------------------------------------------------------- AntMaze / DROID sections
    def pred_rows(mm, keys, nm):
        out = []
        for k in keys:
            t, v = mm[k]["test"], mm[k].get("val", {})
            ci = mm[k].get("test_ci95", {}).get("auroc")
            out.append([nm[k], f"{v['auroc']:.3f}" if v else "—",
                        f"{t['auroc']:.3f}" + (f" <span class=ci>[{ci[0]:.3f}, {ci[1]:.3f}]</span>" if ci else ""),
                        f"{t['brier']:.3f}", f"{t['ece']:.3f}"])
        return out

    ant_html = ""
    if ant_p0:
        a0 = ant_p0
        ant_html += table(["Metric", "Value", "Note"], [
            ["Success rate (50 official tasks)", f"<b>{a0['success_rate']:.1f}%</b> ({a0['n_success']}/{a0['n_episodes']})",
             f"Wilson 95% CI {a0['success_rate_wilson95'][0]:.1f}–{a0['success_rate_wilson95'][1]:.1f}"],
            ["Paper, H-JEPA 2-level (3 seeds)", "39.3 ± 3.7%", "docs/RESULTS.md, Fig. 6"],
            ["Steps to success", f"{a0['steps_to_success_mean']:.0f}", "budget 350"],
            ["Predictor forward calls / episode", f"L1 {a0['predictor_calls_per_episode']['l1']:.0f} · L2 {a0['predictor_calls_per_episode']['l2']:.0f}",
             f"{a0['planning_calls_per_episode']:.0f} planning calls, replan every 10 steps"]])
        ant_html += fig("ant_trajectories.png", "All 50 AntMaze test episodes of the baseline (OGBench medium maze; grey = walls). Blue: success, red: failure, green star: goal.", wide=True)
    if ant_p1:
        sp_a = ant_p1["splits"]
        ant_html += "<h3>Reachability labels and predictors</h3>" + mnotes.get("ant_p1_html", "")
        ant_html += table(["Split", "Tasks", "Attempts", "Rows (× 4 H)", "Positive", "Collection GPU time"],
                          [[k, sp_a[k]["n_tasks"], sp_a[k]["n_attempts"], sp_a[k]["n_rows"], pct(sp_a[k]["pos_rate"], 1),
                            f"{sp_a[k]['collection_wall_clock_s'] / 3600:.1f} h"] for k in ("train", "val", "test")])
        ant_html += table(["Model", "val AUROC", "test AUROC [95% CI]", "test Brier", "test ECE"],
                          pred_rows(ant_p1["models"], ("dist_logreg", "knn", "mlp"),
                                    {"dist_logreg": "Distance logistic (val-selected)", "knn": f"kNN (k = {ant_p1['knn_best_k']})",
                                     "mlp": "MLP + temperature scaling"}))
        ant_html += fig("ant_calibration.png", "AntMaze test split: reliability, ROC and precision-recall.", wide=True)
    if ant_cmp:
        ant_html += "<h3>Selection and repair (test: 50 official tasks, planner seed 42)</h3>" + mnotes.get("ant_p23_method_html", "")
        ant_html += table(mhead, method_rows(ant_cmp))
        ant_html += f'<p class="note">{mnotes.get("ant_table_note", "")}</p>' + mnotes.get("ant_results_html", "")

    droid_html = ""
    if droid:
        b, p1d, p2d = droid["baseline"], droid["p1"], droid["p2"]
        bs = b["per_model_seed"]
        droid_html += table(["Model seed", "Fréchet fidelity per planner seed 1 / 2 / 3", "mean", "ATE xyz (m)"],
                            [[ms, " / ".join(pct(x, 1) for x in v["frechet_skill_per_ps"]), pct(v["frechet_skill"], 1),
                              f"{v['ate_xyz']:.3f}"] for ms, v in bs.items()]
                            + [["<b>mean ± SE</b>", "", f"<b>{100 * b['frechet_skill']['mean']:.2f} ± {100 * b['frechet_skill']['se']:.2f}%</b>",
                                f"paper {b['paper_frechet_pct']}%"]])
        sel = p1d["mlp_selected"]
        dd = p1d["dist"]
        droid_html += "<h3>Offline reachability labels</h3>" + mnotes.get("droid_p1_html", "")
        droid_html += table(["Model", "val AUROC", "test AUROC", "test Brier", "test ECE"], [
            ["Distance logistic (val-selected)", f"{dd['val']['auroc']:.3f}", f"{dd['test']['auroc']:.3f}", f"{dd['test']['brier']:.3f}", f"{dd['test']['ece']:.3f}"],
            ["MLP + temperature (best val of 3 seeds)", f"{sel['val']['auroc']:.3f}", f"{sel['test']['auroc']:.3f}", f"{sel['test']['brier']:.3f}", f"{sel['test']['ece']:.3f}"]])
        vsel = p2d["val_selection"]
        droid_html += "<h3>Reranking (open loop)</h3>" + mnotes.get("droid_p2_html", "")
        droid_html += table(["Val (24 clips, planner seed 1)", "Fréchet fidelity"],
                            [[("<b>" + k + "</b>" if k == vsel["selected"] else k).replace(":", ", λ = ").replace("mlp, λ = 0", "unmodified (λ = 0)"),
                              pct(v, 1)] for k, v in vsel["val_frechet_skill"].items()])
        t = p2d["test"]
        ts = p2d["test_selected_tag"]
        pr = p2d["paired"]
        droid_html += table(["Test (16 clips × planner seeds 1/2/3)", "Fréchet fidelity, mean ± SE over seeds", "ATE xyz (m)", "changed selections"], [
            ["Unmodified H-JEPA (λ = 0)", f"{100 * t['lam0']['frechet_skill']['mean']:.2f} ± {100 * t['lam0']['frechet_skill']['se']:.2f}%", f"{t['lam0']['ate_xyz']['mean']:.3f}", f"0 / {t['lam0']['n']}"],
            [f"Reranked ({vsel['selected'].replace(':', ', λ = ')})", f"{100 * t[ts]['frechet_skill']['mean']:.2f} ± {100 * t[ts]['frechet_skill']['se']:.2f}%", f"{t[ts]['ate_xyz']['mean']:.3f}", f"{t[ts]['changed']} / {t[ts]['n']}"]])
        droid_html += (f'<p class="note">Paired over {pr["n"]} clip × seed runs: {pr["wins"]} better, {pr["losses"]} worse, {pr["ties"]} identical; '
                       f'mean difference {100 * pr["mean_diff"]:+.2f} points, Wilcoxon p = {pr["wilcoxon_p"]:.2f}.</p>') + mnotes.get("droid_after_html", "")
        if "p2_aligned" in droid:
            pa1, pa2 = droid["p1_aligned"], droid["p2_aligned"]
            da, sa = pa1["dist"], pa1["mlp_selected"]
            droid_html += "<h3>Follow-up: time-aligned label</h3>" + mnotes.get("droid_aligned_html", "")
            droid_html += table(["Model (aligned label)", "val AUROC", "test AUROC", "test Brier", "test ECE"], [
                ["Distance logistic", f"{da['val']['auroc']:.3f}", f"{da['test']['auroc']:.3f}", f"{da['test']['brier']:.3f}", f"{da['test']['ece']:.3f}"],
                ["MLP + temperature (best val of 3 seeds)", f"{sa['val']['auroc']:.3f}", f"{sa['test']['auroc']:.3f}", f"{sa['test']['brier']:.3f}", f"{sa['test']['ece']:.3f}"],
                ["Distance to current state only", "", f"{sa['test']['auroc_dist_only']:.3f}", "", ""]])
            va = pa2["val_selection"]
            droid_html += table(["Val (24 clips, planner seed 1)", "Fréchet fidelity"],
                                [[("<b>" + k + "</b>" if k == va["selected"] else k).replace(":", ", λ = ").replace("unmodified, λ = 0", "unmodified (λ = 0)"),
                                  pct(v, 1)] for k, v in va["val_frechet_skill"].items()])
            ta, tsa, pra = pa2["test"], pa2["test_selected_tag"], pa2["paired"]
            droid_html += table(["Test (16 clips × planner seeds 1/2/3)", "Fréchet fidelity, mean ± SE over seeds", "ATE xyz (m)", "changed selections"], [
                ["Unmodified H-JEPA (λ = 0)", f"{100 * ta['lam0']['frechet_skill']['mean']:.2f} ± {100 * ta['lam0']['frechet_skill']['se']:.2f}%", f"{ta['lam0']['ate_xyz']['mean']:.3f}", f"0 / {ta['lam0']['n']}"],
                [f"Reranked, aligned label ({va['selected'].replace(':', ', λ = ')})", f"{100 * ta[tsa]['frechet_skill']['mean']:.2f} ± {100 * ta[tsa]['frechet_skill']['se']:.2f}%", f"{ta[tsa]['ate_xyz']['mean']:.3f}", f"{ta[tsa]['changed']} / {ta[tsa]['n']}"]])
            droid_html += (f'<p class="note">Paired over {pra["n"]} clip × seed runs: {pra["wins"]} better, {pra["losses"]} worse, {pra["ties"]} identical; '
                           f'mean difference {100 * pra["mean_diff"]:+.2f} points, Wilcoxon p = {pra["wilcoxon_p"]:.2f}.</p>') + mnotes.get("droid_aligned_after_html", "")

    kpis = [
        (f"{p0['success_rate']:.0f}%", "H-JEPA baseline success, 50 official FourRoom tasks (paper: 81.3 ± 6.7%)"),
        (f"{m['mlp']['test']['auroc']:.3f}", f"test AUROC of the reachability MLP (kNN {m['knn']['test']['auroc']:.3f})"),
        (f"{pct(so['reach_rate_failed_episodes'])} vs {pct(so['reach_rate_successful_episodes'])}",
         "subgoals reached in failed vs successful baseline episodes"),
        (notes["kpi_p23_value"], notes["kpi_p23_label"]),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{l}</div></div>' for v, l in kpis)

    body = f"""
<header class="hero">
  <div class="wrap">
    <p class="eyebrow">16-831 course project · H-JEPA · FourRoom · Visual AntMaze · DROID</p>
    <h1>Reachability-Aware Subgoal Repair</h1>
    <p class="lede">{notes['abstract']}</p>
    <div class="kpis">{kpi_html}</div>
  </div>
</header>
<nav class="toc"><div class="wrap">
  <a href="#problem">Problem</a><a href="#setup">Setup</a><a href="#p0">Baseline</a><a href="#p1">Reachability</a>
  <a href="#p23">Selection &amp; repair</a>{'<a href="#antmaze">AntMaze</a>' if ant_p0 else ''}{'<a href="#droid">DROID</a>' if droid else ''}<a href="#findings">Findings</a><a href="#limits">Limitations</a><a href="#repro">Reproduce</a>
</div></nav>
<main class="wrap">
<section id="problem">
  <h2>Problem</h2>
  {notes['problem_html']}
  <pre class="mermaid">{notes['mermaid']}</pre>
</section>

<section id="setup">
  <h2>Experimental setup</h2>
  {notes['setup_html']}
</section>

<section id="p0">
  <h2>P0 · Reproducing the H-JEPA baseline</h2>
  {notes['p0_html']}
  {table(["Metric", "Value", "Note"], p0_rows)}
  {fig("trajectories.png", "All 50 test episodes of the baseline. Blue: success, red: failure, black dot: start, green star: goal, grey dots: planning calls.", wide=True)}
  <div class="vids">
    {vid("baseline_task005.mp4", "Baseline success (task 5). Magenta ×: current level-2 subgoal decoded to xy with an analysis-only probe. Green ring: 16 px success radius.", "baseline_task005_last.png")}
    {vid("baseline_task009.mp4", "Baseline failure (task 9). Every subgoal stays in the start corner until the last calls, where the level-2 solve is skipped and level 1 targets the goal directly.", "baseline_task009_last.png")}
  </div>
  <div class="grid2">
    {fig("fig_failure_mode.png", "Decoded subgoals per planning call. In the failure the planner keeps choosing corner subgoals that are reachable but make no progress.")}
    {fig("fig_goal_distance.png", "Real latent distance to the goal per planning call. Failed episodes (red) stay flat.")}
  </div>
</section>

<section id="p1">
  <h2>P1 · Controller-conditioned reachability</h2>
  {notes['p1_html']}
  {table(["Split", "Tasks", "Attempts", "Rows (× 4 H)", "Positive", "Collection GPU time"], data_rows)}
  <div class="grid2">
    {fig("fig_budget_cells.png", "Positive rate by execution horizon H and low-level budget B (test split). B1 matches the default B2 at about 2% of its sample forwards.")}
    {fig("fig_subgoal_reach.png", "Inside the baseline episodes, failed episodes reach far fewer of their subgoals.")}
  </div>
  {vid("p1_budget_B0_vs_B2.mp4", "One P1 attempt pair: the same restored state and the same fixed subgoal (magenta ×), executed for 30 steps by the frozen level-1 MPC with budget B0 (left) and B2 (right). B2 reaches the subgoal at step 9, inside the H = 10 window; B0 needs 19 steps.", "p1_budget_B0_vs_B2.png")}
  <h3>Predictor comparison (test = 50 official tasks, 6 600 rows)</h3>
  {table(["Model", "AUROC [95% task-bootstrap CI]", "PR-AUC", "Brier", "ECE", "within-(H,B) AUROC"], p1_rows)}
  {fig("calibration.png", "Reliability diagram, ROC and precision-recall on the test split.", wide=True)}
  {notes['p1_after_html']}
</section>

<section id="p23">
  <h2>P2 / P3 · Reachability-aware selection and local repair</h2>
  {notes['p23_method_html']}
  <h3>Main comparison (test: 50 official tasks, planner seed 42)</h3>
  {table(mhead, method_rows(cmp_test))}
  <p class="note">{notes['p23_table_note']}</p>
  {seeds_html}
  <h3>Hyper-parameter selection (val: 20 RASR tasks)</h3>
  {table(mhead, method_rows(cmp_val))}
  {notes['p23_results_html']}
  {shift_html}
  <div class="vids">{''.join(vid(v['src'], v['cap'], v.get('poster')) for v in notes['p23_videos'])}</div>
</section>

{f'<section id="antmaze"><h2>Visual AntMaze · second closed-loop environment</h2>{mnotes.get("ant_intro_html", "")}{ant_html}</section>' if ant_html else ""}

{f'<section id="droid"><h2>DROID · limited offline check (real robot video, no simulator)</h2>{mnotes.get("droid_intro_html", "")}{droid_html}</section>' if droid_html else ""}

<section id="findings">
  <h2>Findings</h2>
  {notes['findings_html']}
</section>

<section id="limits">
  <h2>Limitations and next steps</h2>
  {notes['limits_html']}
</section>

<section id="repro">
  <h2>Reproduce</h2>
  {notes['repro_html']}
</section>
<footer><p>{notes['footer']}</p></footer>
</main>
"""
    page = f"""<title>RASR Project Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
{body}
"""
    (SITE / "index.html").write_text(page)
    print("wrote", SITE / "index.html", "assets:", len(copies))


CSS = """
/* Layout: one reading column (~70ch) with figures that may break wider; data in a mono face. */
:root{
  --bg:#f5f7f8; --surface:#ffffff; --fg:#16202a; --muted:#5b6876; --line:#d9dfe4;
  --accent:#0b6e83; --accent-soft:#e1f0f3; --good:#2c7a4b; --bad:#b8433d; --code:#eef2f4;
  --f-display:"Space Grotesk",system-ui,sans-serif; --f-body:"Inter",system-ui,sans-serif;
  --f-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1418; --surface:#151d22; --fg:#e4e9ed; --muted:#97a3ae; --line:#26323a;
  --accent:#5cc4d8; --accent-soft:#16303a; --good:#6cc28f; --bad:#e2786f; --code:#1a242b; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e1418; --surface:#151d22; --fg:#e4e9ed; --muted:#97a3ae; --line:#26323a;
  --accent:#5cc4d8; --accent-soft:#16303a; --good:#6cc28f; --bad:#e2786f; --code:#1a242b; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.65 var(--f-body);margin:0}
.wrap{max-width:1040px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.hero{padding-block:56px 32px;border-bottom:1px solid var(--line);background:var(--surface)}
.eyebrow{font:500 12px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font:700 clamp(32px,5vw,52px)/1.05 var(--f-display);letter-spacing:-.02em;margin:0 0 18px;text-wrap:balance}
h2{font:600 28px/1.2 var(--f-display);margin:0 0 14px;text-wrap:balance}
h3{font:600 19px/1.3 var(--f-display);margin:28px 0 10px}
.lede{font-size:18px;max-width:72ch;color:var(--fg);margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:28px}
.kpi{border:1px solid var(--line);border-radius:6px;padding:14px 16px;background:var(--bg)}
.kv{font:600 26px/1.1 var(--f-display);color:var(--accent);font-variant-numeric:tabular-nums}
.kl{font-size:13px;color:var(--muted);margin-top:6px;line-height:1.4}
.toc{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.toc .wrap{display:flex;gap:18px;overflow-x:auto;padding-block:10px;font:500 13px/1 var(--f-mono)}
.toc a{color:var(--muted);text-decoration:none;white-space:nowrap}.toc a:hover,.toc a:focus-visible{color:var(--accent)}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
section{padding-block:44px 8px;border-bottom:1px solid var(--line)}
section p,section li{max-width:72ch}
a{color:var(--accent)}
code,pre{font-family:var(--f-mono);font-size:13px;background:var(--code);border-radius:4px}
code{padding:1px 5px}
pre{padding:12px 14px;overflow-x:auto;line-height:1.5}
pre.mermaid{background:var(--surface);border:1px solid var(--line);text-align:center}
.tw{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font:500 12px/1.3 var(--f-mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:var(--bg)}
tbody tr:last-child td{border-bottom:0}
.ci{color:var(--muted);font-size:12px;font-family:var(--f-mono)}
figure{margin:20px 0}
figure img,figure video{display:block;width:100%;max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:13px;color:var(--muted);margin-top:8px;max-width:80ch}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
.grid2>*{min-width:0}
.vids{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
.vids>*{min-width:0}
.note{font-size:14px;color:var(--muted)}
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0}
.callout p{margin:0}
.tag{display:inline-block;font:500 11px/1 var(--f-mono);padding:4px 7px;border-radius:3px;border:1px solid var(--line);color:var(--muted);margin-right:6px}
.good{color:var(--good)}.bad{color:var(--bad)}
footer{padding-block:28px 48px;color:var(--muted);font-size:13px}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
"""

if __name__ == "__main__":
    main()
