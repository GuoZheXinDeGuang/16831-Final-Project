#!/usr/bin/env bash
# Usage: scripts/launch_p2p3.sh <split: val|test> <config names...>
#   val  = RASR val tasks (20, env seeds 300000+)  -> hyper-parameter selection only
#   test = official 50 tasks (env seeds 100000+, same as P0)
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
SPLIT=$1; shift
if [ "$SPLIT" = val ]; then
  TF=$HJEPA_HOME/rasr_tasks/val/fourroom_tp35_val_s2002_cross2_goal75_n20_d1.pt; ESB=300000; TASKS=all
else
  TF=H-JEPA/h_jepa/assets/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt; ESB=100000; TASKS=${TASKS:-all}
fi
for name in "$@"; do
  OUT=outputs/rasr/p2p3/$SPLIT/$name
  if [ "$name" = b2_freq_replan ]; then
    python scripts/run_baseline.py --tasks-file $TF --tasks $TASKS --env-seed-base $ESB --no-snapshots \
      --override hierarchical_plan_config.receding_horizon=1 --out $OUT > outputs/rasr/logs/p2p3_${SPLIT}_$name.log 2>&1 &
  else
    python scripts/run_baseline.py --tasks-file $TF --tasks $TASKS --env-seed-base $ESB --no-snapshots \
      --rasr-config configs/rasr/$name.yaml --out $OUT > outputs/rasr/logs/p2p3_${SPLIT}_$name.log 2>&1 &
  fi
done
wait
