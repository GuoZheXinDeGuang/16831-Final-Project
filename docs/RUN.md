# RASR — how to reproduce everything

All commands run from `Project/`.

## 0. Setup (once)

```bash
git clone https://github.com/kevinghst/H-JEPA.git          # pinned at 97054ec, unmodified
cd H-JEPA && uv venv --python 3.11.13 .venv && source .venv/bin/activate
uv pip install -e ".[train,env]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install pytest && cd ..
source ./activate_rasr.sh        # venv + HJEPA_HOME=Project/hjepa_home + PYTHONPATH + thread limits

# checkpoints (FourRoom, seed 42) and the official 50 eval tasks
uvx --from "huggingface_hub[cli]" hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts \
    --include "fourroom/fourroom_hjepa_l2/seed42/*" --include "fourroom/fourroom_hjepa_l3/seed42/*"
(cd H-JEPA/h_jepa && uvx --from "huggingface_hub[cli]" hf download jepa-world-models/h-jepa --repo-type dataset \
    --local-dir assets --include "eval_trajs/fourroom/*" --include "eval_trajs/SHA256SUMS" \
    && cd assets/eval_trajs && sha256sum -c --ignore-missing SHA256SUMS)

# RASR train / val start-goal sets (official generator, different seeds than the official eval set's 3072)
cd H-JEPA/h_jepa
python scripts/generate_fourroom_eval_tasks.py --data-config-path ../scripts/data/config/fourroom_tp35_d0to5.yaml \
  --output-dir $HJEPA_HOME/rasr_tasks/train --output-stem fourroom_tp35_train_s1001 --cross-n-rooms 2 --max-steps 75 --num-episodes 40 --seed 1001
python scripts/generate_fourroom_eval_tasks.py --data-config-path ../scripts/data/config/fourroom_tp35_d0to5.yaml \
  --output-dir $HJEPA_HOME/rasr_tasks/val --output-stem fourroom_tp35_val_s2002 --cross-n-rooms 2 --max-steps 75 --num-episodes 20 --seed 2002
cd ../..
```

## 1. P0 baseline (H-JEPA 2-level, official `fourroom_l2` planner, official 50 tasks)

```bash
scripts/launch_p0.sh          # 3 shards in parallel + summary; ~20-35 min on one RTX 4090
# single task / smoke:  python scripts/run_baseline.py --tasks 0 --out outputs/rasr/smoke/x
# summary only:         python scripts/summarize_baseline.py --run outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42
```
Outputs: `outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42/{summary.json,episodes.csv,trajectories.png,shard_*/...}`.

The official `eval.py` is also runnable unchanged:
```bash
cd H-JEPA/h_jepa && python eval.py --config-name fourroom_l2 seed=42 output.dir=eval_l2 \
   policy=$HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_l2/seed42/fourroom_hjepa_l2_object.ckpt
```

## 2. P1 reachability

```bash
scripts/launch_p1_baselines.sh     # baseline episodes on RASR train(40)/val(20) tasks -> snapshots
scripts/launch_p1_collect.sh       # branch attempts: train stride 1, val/test stride 2, budgets B0/B1/B2, H<=30
python scripts/train_reachability.py          # eps calibration, kNN, MLP, metrics, plots
python scripts/analyze_subgoal_outcomes.py    # baseline subgoal outcomes + branch/baseline consistency
```
Outputs: `outputs/rasr/p1_reachability/{results.json,comparison.md,calibration_*.png,knn_model.npz,mlp_model.pt}`.

## 3. FourRoom P2 / P3 / P4 (selection, repair, seeds, distribution shift)

```bash
scripts/launch_p2p3.sh val  rerank_l0.25 rerank_l0.5 rerank_l1.0 rasr_b5 rasr_b20 full_replan   # selection on val
scripts/launch_p2p3.sh test rerank_l0.25 full_replan rasr_b20 repair_noR                         # test, once per method
scripts/launch_b2_shards.sh 0-27 28-49        # B2: replan every 5 steps
python scripts/compare_methods.py --method B1=outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42 \
  --method B3=outputs/rasr/p2p3/test/rerank_l0.25 ... --out outputs/rasr/p2p3/compare_test.json
xargs -P 6 -L 1 scripts/run_seed_job.sh < outputs/rasr/p4_jobs.txt   # planner seeds 43, 44
python scripts/aggregate_seeds.py             # 3-seed table -> outputs/rasr/p4_seeds/aggregate.{json,md}
scripts/launch_p4_shift.sh && python scripts/eval_shift.py   # predictor on non-selected candidates
```
Method configs: `configs/rasr/*.yaml` (fields of `rasr.rasr_solver.RASRConfig`).

## 4. Visual AntMaze

```bash
# checkpoints, official 50 tasks, AntMaze stats/probing data, DROID eval clips
scripts/download_ant_droid.sh
# OGBench ant expert (needed by the official task generator), re-saved for current JAX
mkdir -p $HJEPA_HOME/ogbench_experts
wget -qO- https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz \
  | tar xz -C $HJEPA_HOME/ogbench_experts --strip-components=1 experts/ant
python scripts/convert_ogbench_expert.py $HJEPA_HOME/ogbench_experts/ant/params_400000.pkl
# RASR train/val tasks: official generator (seed 1001, 3-cell distance) -> split disjoint from the test (start, goal) cells
(cd H-JEPA/h_jepa && python scripts/generate_maze_expert_grid_eval_tasks.py --config-name ant_flat \
  --output-path $HJEPA_HOME/rasr_tasks/ant/pool_s1001_d3_n240.pt --d-low 3 --d-high 3 --num-episodes 240 \
  --rollout-budget 125 --seed 1001)
python scripts/split_ant_tasks.py H-JEPA/h_jepa/assets/eval_trajs/ant/expert_grid_d3_n50.pt \
  $HJEPA_HOME/rasr_tasks/ant/pool_s1001_d3_n240.pt $HJEPA_HOME/rasr_tasks/ant

scripts/launch_ant_p0.sh            # P0 test (3 shards) + P1 source episodes on train/val (MUJOCO_GL=egl)
python scripts/summarize_baseline.py --run outputs/rasr/ant/p0_baseline/ant_l2_ms42_ps42
scripts/launch_ant_p1_collect.sh    # attempts: budgets B0 2x4, B1 10x20, B2 90x25, H_max 60
scripts/launch_ant_after_p1.sh      # train_reachability (eps radius 0.5) -> make_ant_configs -> P2/P3 test, MLP predictor
python scripts/fit_dist_predictor.py --attempts outputs/rasr/ant/p1_attempts \
  --run outputs/rasr/ant/p1_reachability --h-values 10,20,40,60        # val-selected distance-logistic predictor
SHARDS="0-16 17-33 34-49" PRED=outputs/rasr/ant/p1_reachability/dist_model.pt TAG=_dist \
  scripts/launch_ant_p2p3.sh test 42 rerank_l0.25 rasr_b20
python scripts/compare_methods.py --method B1=outputs/rasr/ant/p0_baseline/ant_l2_ms42_ps42 \
  --method B3=outputs/rasr/ant/p2p3/test/ps42/rerank_l0.25_dist ... --out outputs/rasr/ant/p2p3/compare_test.json
```
AntMaze method configs (`configs/rasr/ant/`) are generated by `scripts/make_ant_configs.py`: FourRoom's val-selected
method hyper-parameters, with AntMaze eps / kernel width / controller condition.

## 5. DROID (offline, open loop)

```bash
# training clips for the reachability labels: three 256p training shards
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME \
  --include "droid/droid_256p/shard-00000.tar" --include "droid/droid_256p/shard-00003.tar" \
  --include "droid/droid_256p/shard-00023.tar"
(cd $HJEPA_HOME/droid/droid_256p && for f in shard-*.tar; do tar -xf $f; done)

scripts/run_droid_baseline.sh       # official eval.py, model seeds 1/1000/10000 x planner seeds 1/2/3
scripts/launch_droid_p1.sh          # candidate pools (160 train / 40 val / 16 test clips) + MLP
python scripts/droid_rasr.py train-dist                    # distance-logistic predictor
scripts/launch_droid_p2.sh          # original label: predictor x lam on val -> test
python scripts/droid_rasr.py train --label aligned --out outputs/rasr/droid/p1_reachability_aligned
python scripts/droid_rasr.py train-dist --label aligned --out outputs/rasr/droid/p1_reachability_aligned
scripts/launch_droid_p2_aligned.sh  # time-aligned label (post-hoc follow-up), same protocol
python scripts/droid_summary.py && python scripts/droid_figures.py   # summary.json + website figures (film strip needs test_clip_frames.npz, see script docstring)
python scripts/droid_summary.py     # -> outputs/rasr/droid/summary.json
```

## 6. Website and result snapshot

```bash
python3 scripts/site_notes_static.py && python3 scripts/site_notes_results.py && python scripts/build_site.py
python3 scripts/export_results.py   # small result files -> results/ (committed)
```

## 7. Tests

```bash
python -m pytest -q tests/test_reachability_unit.py   # seconds, CPU
python -m pytest -q tests/test_p0_smoke.py            # ~4 min, GPU: bit-reproducibility + snapshot restore
```
