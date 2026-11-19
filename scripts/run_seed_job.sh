#!/usr/bin/env bash
# one job: <method> <planner_seed> <tasks>   (test split, official 50 tasks, env seeds 100000+task)
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
M=$1; S=$2; TK=$3
# a job started by hand creates this marker so the queue skips it
if [ -f "outputs/rasr/p4_claimed_${M}_${S}_${TK}" ]; then echo "claimed $M $S $TK"; exit 0; fi
# compute budget: B2 (2x planning calls, worst at seed 42) is evaluated at seed 42 only
if [ "$M" = b2_freq_replan ] && [ -f outputs/rasr/p4_skip_b2 ]; then echo "skip $M $S $TK"; exit 0; fi
TF=H-JEPA/h_jepa/assets/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt
OUT=outputs/rasr/p4_seeds/$M/seed$S/shard_$TK
EXTRA=""
case $M in
  b1_baseline) ;;
  b2_freq_replan) EXTRA="--override hierarchical_plan_config.receding_horizon=1" ;;
  *) EXTRA="--rasr-config configs/rasr/$M.yaml" ;;
esac
python scripts/run_baseline.py --tasks-file $TF --tasks $TK --seed $S --env-seed-base 100000 --no-snapshots --no-step-encode $EXTRA \
  --out $OUT > outputs/rasr/logs/p4_${M}_s${S}_${TK}.log 2>&1
echo "done $M $S $TK $(grep -c . $OUT/episodes.jsonl 2>/dev/null)"
