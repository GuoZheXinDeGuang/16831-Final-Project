#!/usr/bin/env bash
# one AntMaze test job: <method> <planner_seed> <tasks>   (official 50 tasks, env seeds 100000+task, model seed 42)
# methods: b1_baseline, b2_freq_replan (replan every 5 instead of 10 steps), or a configs/rasr/ant/<name>.yaml
#   (*_dist = with the val-selected distance-logistic predictor, as in the seed-42 main table)
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
export MUJOCO_GL=egl
M=$1; S=$2; TK=$3
TF=H-JEPA/h_jepa/assets/eval_trajs/ant/expert_grid_d3_n50.pt
OUT=outputs/rasr/ant/p4_seeds/$M/seed$S/shard_$TK
[ "$(grep -c . $OUT/episodes.jsonl 2>/dev/null)" = "$(python3 -c "a,b='$TK'.split('-');print(int(b)-int(a)+1)")" ] && { echo "already done $M $S $TK"; exit 0; }
P1=outputs/rasr/ant/p1_reachability
case $M in
  b1_baseline) EXTRA="" ;;
  b2_freq_replan) EXTRA="--override hierarchical_plan_config.receding_horizon=1" ;;
  *_dist) EXTRA="--rasr-config configs/rasr/ant/${M%_dist}.yaml --predictor $P1/dist_model.pt" ;;
  *) EXTRA="--rasr-config configs/rasr/ant/$M.yaml --predictor $P1/mlp_model.pt" ;;
esac
python scripts/run_baseline.py --config ant_l2 --model ant_hjepa_l2 --seed $S --tasks-file $TF --tasks $TK \
  --env-seed-base 100000 --no-snapshots --no-step-encode $EXTRA --out $OUT > outputs/rasr/logs/ant_p4_${M}_s${S}_${TK}.log 2>&1
echo "$(date +%H:%M) done $M $S $TK $(grep -c . $OUT/episodes.jsonl 2>/dev/null)"
