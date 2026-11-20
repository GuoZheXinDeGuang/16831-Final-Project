"""Results-dependent narrative for the website; merges with docs/site_notes_static.json -> docs/site_notes.json.
Numbers quoted in prose were read from outputs/rasr/p2p3/compare_{test,val}.json."""
import json
from pathlib import Path

S = json.loads(Path("docs/site_notes_static.json").read_text())
T = {r["method"]: r for r in json.loads(Path("outputs/rasr/p2p3/compare_test.json").read_text())}
AG = {r["method"]: r for r in json.loads(Path("outputs/rasr/p4_seeds/aggregate.json").read_text())}
o, onr, b4, b3, b2 = T["Ours"], T["Ours−R"], T["B4"], T["B3"], T["B2"]
N = dict(S)
N["abstract"] = (
    "We reproduce hierarchical H-JEPA planning on FourRoom Distractors (86% success), learn a controller-conditioned "
    "reachability model for its latent subgoals (test AUROC 0.903, well calibrated), and use it for subgoal reranking "
    "and local repair. Over three planner seeds on the 50 official tasks, reachability-aware selection and repair score "
    f"{AG['Ours']['mean']:.1f}% against {AG['B1']['mean']:.1f}% for the baseline; the best variant is the repair without "
    f"reachability ({AG['Ours−R']['mean']:.1f}%, not significant). Diagnosing why, we find the baseline's failures are not unreachable "
    "subgoals. The level-2 planner keeps choosing reachable subgoals that make no progress, a corner attractor that "
    "the world model scores as close to the goal. On Visual AntMaze, where failures are stalls, the repair without the reachability "
    "term is the only significant gain in the project (54.0% vs 38.0% over three planner seeds, +33 / −9 of 150 episodes, p &lt; 0.001), while adding the reachability term hurts (26.7%, p = 0.01). "
    "On DROID (offline, open loop) reachability-aware selection has no effect with either label.")
N["kpi_p23_value"] = f"{AG['Ours']['mean']:.1f}% vs {AG['B1']['mean']:.1f}%"
N["kpi_p23_label"] = "RASR vs baseline success, mean over 3 planner seeds (150 episodes each); no significant change"
N["method_names"] = {
    "B1": "B1 · H-JEPA baseline", "B2": "B2 · replan every 5 steps", "B3": "B3 · reachability rerank (λ = 0.25)",
    "B4": "B4 · full replan on trigger", "Ours": "Ours · rerank + repair (R + novelty)", "Ours−R": "Ablation · repair without R",
    "B3 λ=0.25": "B3 · λ = 0.25", "B3 λ=0.5": "B3 · λ = 0.5", "B3 λ=1.0": "B3 · λ = 1.0",
    "Ours β=5": "Ours · β = 5", "Ours β=20": "Ours · β = 20 (selected)",
    "B3-mlp": "B3 with MLP predictor (ablation)", "Ours-mlp": "Ours with MLP predictor (ablation)"}
N["p23_method_html"] = """
<p><b>Candidate pool (P2).</b> After the unchanged level-2 GD solve, all 800 optimised candidates are rolled out once
(one extra batched forward). Each candidate gives a progress cost (the planner's own objective) and a first predicted
latent, the subgoal level 1 would receive. <b>B3</b> picks
argmin<sub>c</sub> cost<sub>c</sub> − λ·c<sub>min</sub>·log(R(z<sub>t</sub>, g<sub>c</sub>, H = 10, B2) + 0.05).
λ is relative to the call's best cost, so λ = 0.25 accepts a candidate up to 25% more costly per unit of −log R.</p>
<p><b>Monitor (P3).</b> Planner-visible and debounced: a trigger fires after 3 consecutive missed subgoals
(the real frame's latent is not within ε of the previous subgoal) or when the real latent goal distance has not improved
by 5% in 5 calls. After a trigger the repair stays active for 3 calls, followed by a 1-call cooldown.</p>
<p><b>Repair (Ours).</b> Re-run the level-2 GD with the cost augmented by −λ<sub>R</sub>·c<sub>min</sub>·log R(z, ĝ<sub>1</sub>)
(differentiable through the frozen MLP) and +β·c<sub>min</sub>·visit(ĝ<sub>1:2</sub>), a kernel penalty on plan
waypoints near latents already observed in this episode. If the repaired subgoal still has R &lt; 0.3, fall back to a
plain full replan. <b>B4</b> uses the same trigger but only re-solves (fresh restart, min cost). <b>Ablation</b>: the same
repair with λ<sub>R</sub> = 0 and no rerank. <b>B2</b> replans every 5 instead of 10 steps (twice the planning calls).
λ ∈ {0.25, 0.5, 1.0} and β ∈ {5, 20} were chosen on the val tasks; the test set was run once per method.</p>
<p>Before tuning anything on val, the first prototypes were tried on three failed test tasks (9, 16, 30). That is
where the failure mode below was found. None of those prototypes solved them, and the configurations reported here
were fixed on val.</p>
"""
N["p23_table_note"] = (
    "Δ: tasks gained and lost relative to B1 on the same 50 tasks, with an exact two-sided McNemar test. "
    "Recovery: share of triggers after which the real latent goal distance improves by ≥ 5% within 4 calls. "
    "L2 predictor calls include the extra pool rollouts and re-solves. Planning seconds were measured with 6–7 jobs "
    "sharing one GPU.")
N["seeds_note"] = (
    "Planner seeds 43 and 44 re-run every method with the same model, tasks and env seeds. B2 (twice the planning calls, worst at "
    "seed 42) was run at seed 42 only to save compute. Pooled McNemar: exact test over the 150 (task, seed) pairs against B1 of "
    "the same seed. The paper's 81.3 ± 6.7% varies the model seed as well; ours varies the planner seed only.")
N["p23_results_html"] = f"""
<div class="callout"><p><b>Result over three seeds.</b> Rerank (B3), full replanning (B4) and Ours all score
{AG['Ours']['mean']:.1f}% against {AG['B1']['mean']:.1f} ± {AG['B1']['se']:.1f}% for the baseline. The repair without reachability
scores {AG['Ours−R']['mean']:.1f} ± {AG['Ours−R']['se']:.1f}% and is above the baseline at every seed, but the pooled difference
(+{AG['Ours−R']['gained']} / −{AG['Ours−R']['lost']} tasks, p = {AG['Ours−R']['mcnemar_p_pooled']:.2f}) is not significant. At seed 42 alone, reranking changes {100 * b3['rerank_changed_frac']:.0f}% of subgoal choices (B3) without
changing a single outcome. Ours loses task 45; the ablation without reachability gains task 33; B2 loses tasks 2 and 31
while doubling planner compute.</p></div>
<ul>
<li><b>The reachability term makes repairs more conservative.</b> Over the three seeds, the real goal distance improves after
{100 * AG['Ours−R']['recovery_rate']:.0f}% of triggers without R, {100 * AG['Ours']['recovery_rate']:.0f}% with R, and
{100 * AG['B4']['recovery_rate']:.0f}% for plain full replanning. R favours short, safe subgoals, which helps when the subgoal is unexecutable and hurts when the
problem is lack of progress.</li>
<li><b>False triggers cost time.</b> The goal-stall trigger fires in {o['episodes_with_trigger']} of 50 episodes, including most
successful ones, because latent goal distance is not monotone even on good routes. Ours needs
{AG['Ours']['steps_to_success']:.0f} steps per success vs {AG['B1']['steps_to_success']:.0f} for B1 (3 seeds), and task 45 (solved by B1 in 179 steps) runs out of budget.</li>
<li><b>Compute.</b> Ours adds about {o['extra_toplevel_solves_per_ep']:.0f} level-2 re-solves per episode
({o['pred_calls_l2'] / T['B1']['pred_calls_l2'] - 1:+.0%} level-2 predictor calls); B2 doubles them ({b2['pred_calls_l2'] / T['B1']['pred_calls_l2'] - 1:+.0%}).
The MLP itself costs one batched forward per call.</li>
</ul>
"""
N["p23_videos"] = [
    {"src": "baseline_task033.mp4", "poster": "baseline_task033_last.png", "cap": "Task 33, B1 baseline: fails."},
    {"src": "oursnoR_task033.mp4", "poster": "oursnoR_task033_last.png", "cap": "Task 33, repair without R: repairs fire at calls 3, 8 and 15 and the agent reaches the goal at step 168. This is the only task any method gained over the baseline."},
    {"src": "ours_task009.mp4", "poster": "ours_task009_last.png", "cap": "Task 9, Ours: repairs fire at calls 6, 11 and 16 and produce subgoals rated reachable (R 0.54–0.76), but the episode still fails."},
    {"src": "ours_task045.mp4", "poster": "ours_task045_last.png", "cap": "Task 45, Ours: a false trigger. The baseline solves this task in 179 steps; repeated conservative repairs slow the agent and it runs out of budget."},
]
N["findings_html"] = """
<ol>
<li><b>Baseline reproduced and made deterministic.</b> 86% on the official tasks (paper 81.3 ± 6.7%). Upstream
evaluation is not reproducible because env resets are unseeded; with seeded resets, runs are bit-identical.</li>
<li><b>Subgoal reachability is predictable.</b> Real MPC executions give balanced labels. An MLP on frozen latents plus
(H, B) reaches 0.903 AUROC and 0.022 ECE on held-out tasks, and beats kNN and a distance-only model mainly
<em>within</em> each controller condition. Low-level budget matters only at the extreme: 10 GD iterations × 20 samples
(B1) reach subgoals as often as the default 90 × 100.</li>
<li><b>Reachability correlates with failure but does not cause it here.</b> Failed episodes reach 16% of their subgoals
against 77% in successful ones. Inspecting them shows why: the level-2 planner keeps proposing subgoals in or near the
start corner. They are often reachable, but the world model rates the corner as near-goal (level-2 cost ≈ 1.2–1.7 at
the start of task 9). In some calls every one of the top-64 candidates has R ≈ 0.02. When the repair is pushed away
from visited states, gradient descent returns off-manifold latents that the reachability model correctly scores as
unreachable (R ≈ 0.02).</li>
<li><b>Hence no success gain.</b> Selecting or repairing by executability cannot fix a high-level model error about
<em>which</em> subgoal makes progress. The one consistent positive signal (88.0% vs 85.3% over three seeds, not significant)
comes from exploration pressure, the visited-region term without R, not from reachability. The predictor itself transfers
reasonably to the candidates it scores (AUROC 0.85 on non-selected candidates), so the null result is not a predictor failure.</li>
<li><b>AntMaze confirms the pattern with a significant result.</b> Over three planner seeds, repair without R: 54.0% vs 38.0% (+33 / −9 of 150, p &lt; 0.001), better than the baseline at every seed; full replan
on the same trigger 48.0% (p = 0.01). With R the repair drops to 26.7%, below the baseline (p = 0.01). The val-selected predictor is close to a distance prior, so the R term pulls repaired
subgoals back towards the stuck state and cancels the exploration term.</li>
<li><b>DROID (offline):</b> no effect with either the original or a time-aligned label (+0.5 points, p = 0.78). The original
label mostly rewarded not moving.</li>
</ol>
"""
N["limits_html"] = """
<ul>
<li>One model seed (the reachability model lives in its latent space) and three planner seeds; 150 episodes per method still
leave pooled Wilson intervals of about ±6 points. B2 has one seed.</li>
<li>The reachability model was trained only on selected subgoals; on non-selected candidates its AUROC drops from 0.91 to 0.85
(measured on val tasks). Training on candidate attempts would close part of that gap.</li>
<li>Labels use a latent threshold calibrated on train real-frame pairs; an xy-probe label was not used.</li>
<li>The goal-stall trigger is noisy because latent goal distance is not a progress metric across walls. A better
trigger (for example model-predicted vs realised progress) is likely the most useful fix.</li>
<li>A variant that bypasses the level-2 planner on a failed repair (level 1 aims at the real goal) was explored on
3 val tasks only, then dropped for compute.</li>
<li>Next: a progress-aware repair objective (for example route diversity across doors) and a better stall trigger.
AntMaze and DROID results are in their own sections above; their limitations are listed there.</li>
</ul>
"""
SH = json.loads(Path("outputs/rasr/p4_shift/shift_results.json").read_text()) if Path("outputs/rasr/p4_shift/shift_results.json").exists() else None
if SH:
    a, b = SH["selected_Hall"], SH["non_selected_Hall"]
    N["shift_intro_html"] = (
        f"<p>The predictor was trained only on subgoals the baseline <em>selected</em>, but reranking scores all 800 candidates. "
        f"To measure the shift we replayed the 20 val episodes with logging only (bit-identical trajectory) and, at every second "
        f"planning call, executed the baseline's subgoal plus four non-selected candidates spread over predicted R, from the same "
        f"restored state with the B2 controller ({SH['n_attempts']} attempts).</p>")
    N["shift_after_html"] = (
        f"<p>The MLP loses some accuracy off-distribution (AUROC {a['mlp']['auroc']:.3f} → {b['mlp']['auroc']:.3f}, "
        f"ECE {a['mlp']['ece']:.3f} → {b['mlp']['ece']:.3f}), and kNN loses more ({a['knn']['auroc']:.3f} → {b['knn']['auroc']:.3f}). "
        f"Among candidates of the same planning call, the MLP ranks a reached candidate above an unreached one "
        f"{100 * SH['within_call_pairwise_auc_H10']:.0f}% of the time ({SH['within_call_pairs']} pairs, H = 10). "
        f"So the reranker gets a meaningful signal, and the lack of a success gain is not explained by a broken predictor.</p>")
Path("docs/site_notes.json").write_text(json.dumps(N, indent=1))
print("ok")
