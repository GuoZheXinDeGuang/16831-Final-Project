#!/usr/bin/env bash
# B2 (replan every 5 steps) on the official tasks, sharded; results merged via **/episodes.jsonl
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
TF=H-JEPA/h_jepa/assets/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt
for sh in "$@"; do
  python scripts/run_baseline.py --tasks-file $TF --tasks $sh --env-seed-base 100000 --no-snapshots \
    --override hierarchical_plan_config.receding_horizon=1 --out outputs/rasr/p2p3/test/b2_freq_replan/shard_$sh \
    > outputs/rasr/logs/p2p3_test_b2_$sh.log 2>&1 &
done
wait
