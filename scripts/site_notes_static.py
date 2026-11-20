"""Narrative text for the website (static parts). Results-dependent parts are added by site_notes_results.py."""
import json
from pathlib import Path

N = {}
N["problem_html"] = """
<p>H-JEPA plans top-down. A level-2 world model, optimised by gradient descent over 800 candidate latent-action
sequences, proposes the next <em>latent subgoal</em>. A level-1 model-predictive controller (MPC) then tries to reach
that subgoal within 10 environment steps before the whole hierarchy replans. Nothing in this loop checks whether the
subgoal is actually <em>executable</em> by the low-level controller under its horizon and search budget.</p>
<p>RASR adds three pieces on top of the frozen, pretrained H-JEPA:</p>
<ol>
<li><b>Reachability estimator</b> R(z<sub>t</sub>, g, H, B): probability that the level-1 MPC with budget B reaches latent
subgoal g from the current latent z<sub>t</sub> within H steps, trained on real MPC executions.</li>
<li><b>Reachability-aware selection (P2)</b>: rerank the unchanged level-2 candidate pool by
progress cost − λ·c<sub>min</sub>·log(R + δ).</li>
<li><b>Failure monitoring + local repair (P3)</b>: detect stalls from planner-visible signals, re-solve the subgoal with
a reachability term (and a visited-region penalty), and fall back to full replanning when the repaired subgoal is still
unreachable.</li>
</ol>
<p>Research question: under the same evaluation tasks and comparable compute, does reachability + local repair beat plain
H-JEPA, more frequent replanning, and full replanning on the same trigger?</p>
"""
N["mermaid"] = """flowchart TB
  O[current frame o_t] --> E[frozen H-JEPA encoders]
  G[goal frame] --> E
  E --> L2[level-2 GD planner 800 candidates]
  L2 -->|candidate subgoals + progress cost| P2[P2 rerank cost - λ log R]
  R[(reachability MLP R z,g,H,B)] --> P2
  P2 -->|subgoal g| L1[level-1 MPC 100 samples, 10 steps]
  L1 --> ENV[FourRoom env]
  ENV --> O
  ENV -.-> M{monitor subgoal misses / goal stall}
  M -->|trigger| RP[P3 repair GD with R + novelty terms]
  R --> RP
  RP -->|R below tau| FB[fallback: full replan]
  RP --> L1
  FB --> L1"""
N["setup_html"] = """
<ul>
<li><b>Code:</b> official <a href="https://github.com/kevinghst/H-JEPA" target="_blank" rel="noopener">kevinghst/H-JEPA</a>
at commit <code>97054ec</code>, unmodified. All RASR code wraps or subclasses upstream objects; with every RASR switch off
the planner is bit-identical to upstream (regression test).</li>
<li><b>Environment:</b> FourRoom Distractors (224×224 pixels, one moving distractor, agent speed 2.5 px/step, success =
within 16 px of the goal, 210-step budget). Chosen as the first environment because it needs no MuJoCo rendering, has an
env-native success signal and its state can be restored exactly for branching rollouts. Visual AntMaze and an offline
DROID check were added afterwards with the same code (sections <a href="#antmaze">AntMaze</a> and <a href="#droid">DROID</a>).</li>
<li><b>Model / planner:</b> pretrained 2-level H-JEPA (seed 42) with the official <code>fourroom_l2</code> planner:
level-2 GD with 800 samples and a decaying horizon of 12; level-1 GD MPC with 100 samples, a horizon of 2 × 5-step action
blocks, replanning every 10 steps.</li>
<li><b>Tasks and splits:</b> test = the official 50 start–goal tasks (two-room crossings). Train (40) and val (20) task
sets were generated with the official generator under different seeds (1001, 2002 vs. 3072), so no start–goal
configuration is shared. Train is used to fit the reachability models; every threshold, temperature and λ is chosen on val.</li>
<li><b>Determinism:</b> upstream resets FourRoom without a seed, and two runs of the official <code>eval.py</code> gave
different step counts (168 vs 141 on task 0). We seed each reset (100000 + task) and enable deterministic torch kernels;
runs are bit-reproducible across processes.</li>
<li><b>Privileged state:</b> simulator xy is used only for labels' calibration, evaluation and the analysis-only
latent→xy probe in figures. The planner sees pixels, latents, model predictions and the reachability model.</li>
<li><b>Hardware:</b> one RTX 4090 shared by up to 7 jobs. Wall-clock numbers are therefore inflated; predictor call
counts are the comparable compute measure.</li>
</ul>
"""
N["p0_html"] = """
<p>The instrumented runner mirrors the upstream chunked evaluation (one world per task, planner generators reseeded to
seed + task) and logs, per planning call: the latents of the current frame and goal, the level-1 subgoal, all predicted
level-1/level-2 plans, horizons, gradient-descent iterations, predictor forward calls and wall-clock. Per step it logs
xy, distractors, actions and the real-frame latents, plus an env snapshot before every planning call.</p>
<div class="callout"><p>The seed-42 baseline reaches 86% (43/50), inside the paper's 81.3 ± 6.7% range. All 7 failures occur
in tasks with every door open (7 of 14 such tasks fail; 36 of 36 tasks with a closed door succeed).</p></div>
"""
N["p1_html"] = """
<p><b>Attempt.</b> Restore the env at a baseline planning call, freeze the level-2 subgoal g the baseline gave its
level-1 MPC, and run only the frozen level-1 MPC towards g for 30 steps with budget
B ∈ {B0 = 2 GD iterations × 4 samples, B1 = 10 × 20, B2 = 90 × 100 (baseline)}. Labels for H ∈ {5, 10, 20, 30} are prefixes
of the same rollout. <b>Label:</b> Y = 1 iff min<sub>k≤H</sub> ‖enc<sub>2</sub>(o<sub>t+k</sub>) − g‖² ≤ ε, with
ε = 6.70: the median latent distance between real frames whose agents are within 8 px (about 3 agent steps), computed on
the train split only.</p>
<p>A restored branch reproduces the logged frame to within 1/255 per pixel. Branch-B2 labels agree with the baseline's own
execution of the same subgoal for 84% (H = 10) of 500 matched calls; the rest is solver noise, so labels are partly
stochastic.</p>
"""
N["p1_after_html"] = """
<p>The distance-only logistic model already reaches 0.877 AUROC: most of the reachability signal is "how far is the
subgoal, and how much budget and time does the controller get". The MLP adds a clear gain <em>within</em> each (H, B)
cell (0.853 vs 0.751 for kNN and 0.731 for the reference), and after temperature scaling on val it is the best-calibrated
model (ECE 0.022). Three MLP training seeds give test AUROC 0.870, 0.906 and 0.903.</p>
"""
N["repro_html"] = """
<pre>source activate_rasr.sh
scripts/launch_p0.sh                          # P0 baseline, 50 official tasks
scripts/launch_p1_baselines.sh                # baseline on RASR train/val tasks (P1 source)
scripts/launch_p1_collect.sh                  # branch attempts
python scripts/train_reachability.py          # eps, kNN, MLP, metrics, calibration plots
scripts/launch_p2p3.sh val  rerank_l0.25 rerank_l0.5 rerank_l1.0 rasr_b5 rasr_b20 full_replan
scripts/launch_p2p3.sh test rerank_l0.25 full_replan rasr_b20 repair_noR
scripts/launch_b2_shards.sh 0-27 28-49        # B2: replan every 5 steps
python scripts/compare_methods.py ...         # tables in outputs/rasr/p2p3/
xargs -P 6 -L 1 scripts/run_seed_job.sh &lt; outputs/rasr/p4_jobs.txt   # planner seeds 43, 44
python scripts/aggregate_seeds.py             # 3-seed table
scripts/launch_p4_shift.sh &amp;&amp; python scripts/eval_shift.py   # predictor on non-selected candidates
python -m pytest -q tests/                    # unit + GPU regression tests</pre>
<p>Full commands, outputs and status: <code>docs/RUN.md</code>, <code>docs/IMPLEMENTATION_STATUS.md</code>.</p>
"""
N["footer"] = "All numbers on this page are generated from the experiment logs by scripts/build_site.py. FourRoom: model seed 42, planner seeds 42/43/44; AntMaze: model and planner seed 42; DROID: model seed 1 (baseline: 3 model seeds), planner seeds 1/2/3. See Limitations."
Path("docs/site_notes_static.json").write_text(json.dumps(N, indent=1))
print("ok")
