#!/usr/bin/env bash
# AntMaze P2/P3. Usage: scripts/launch_ant_p2p3.sh <val|test> <planner_seed> <config names...>
#   configs from configs/rasr/ant/<name>.yaml (eps / kernel width / controller budget set for AntMaze);
#   baseline_copy = unmodified planner through the same code path. Max 7 GPU processes at once.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
export MUJOCO_GL=egl
SPLIT=$1; PS=$2; shift 2
if [ "$SPLIT" = val ]; then
  TF=$HJEPA_HOME/rasr_tasks/ant/ant_expert_grid_d3_val_s1001_n20.pt; ESB=300000
else
  TF=H-JEPA/h_jepa/assets/eval_trajs/ant/expert_grid_d3_n50.pt; ESB=100000
fi
# PRED / TAG: predictor file and output-name suffix (default MLP, no suffix; TAG=_dist for dist_model.pt)
PRED=${PRED:-outputs/rasr/ant/p1_reachability/mlp_model.pt}; TAG=${TAG:-}
for name in "$@"; do
  OUT=outputs/rasr/ant/p2p3/$SPLIT/ps$PS/$name$TAG
  for shard in ${SHARDS:-all}; do
    python scripts/run_baseline.py --config ant_l2 --model ant_hjepa_l2 --seed $PS --tasks-file $TF --tasks $shard \
      --env-seed-base $ESB --no-snapshots --no-step-encode --rasr-config configs/rasr/ant/$name.yaml --predictor $PRED \
      --out $OUT/shard_$shard > outputs/rasr/logs/ant_p2p3_${SPLIT}_ps${PS}_${name}${TAG}_$shard.log 2>&1 &
  done
done
wait
