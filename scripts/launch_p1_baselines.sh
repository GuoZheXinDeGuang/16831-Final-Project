#!/usr/bin/env bash
# Baseline (P0 planner, unchanged) on the RASR train/val start-goal sets, used only as P1 data source.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
TR=$HJEPA_HOME/rasr_tasks/train/fourroom_tp35_train_s1001_cross2_goal75_n40_d1.pt
VA=$HJEPA_HOME/rasr_tasks/val/fourroom_tp35_val_s2002_cross2_goal75_n20_d1.pt
python scripts/run_baseline.py --tasks-file $TR --tasks 0-19  --env-seed-base 200000 --out outputs/rasr/p1_source/train/shard_0-19  > outputs/rasr/logs/p1src_train_0-19.log 2>&1 &
python scripts/run_baseline.py --tasks-file $TR --tasks 20-39 --env-seed-base 200000 --out outputs/rasr/p1_source/train/shard_20-39 > outputs/rasr/logs/p1src_train_20-39.log 2>&1 &
python scripts/run_baseline.py --tasks-file $VA --tasks all   --env-seed-base 300000 --out outputs/rasr/p1_source/val/shard_0-19    > outputs/rasr/logs/p1src_val.log 2>&1 &
wait
