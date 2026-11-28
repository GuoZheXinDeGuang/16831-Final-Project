#!/usr/bin/env bash
# One AntMaze job (v2 rerun, A6000): <split> <method> <planner_seed> <task(s)>
#   split: test (official 50, env seeds 100000+task) | val (RASR val 20, 300000+task) | train (RASR train 40, 200000+task)
#   method: b1_baseline | src (baseline with snapshots, P1 source) | b2_freq_replan |
#           <cfg>[_dist|_mlp] = configs/rasr/ant/<cfg>.yaml (+ predictor from outputs/rasr/ant/v2/p1_reachability)
# Output: outputs/rasr/ant/v2/<split>/<method>/seed<S>/shard_<tasks>; skips finished jobs.
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU:-1}
SPLIT=$1; M=$2; S=$3; TK=$4
case $SPLIT in
  test)  TF=H-JEPA/h_jepa/assets/eval_trajs/ant/expert_grid_d3_n50.pt; ESB=100000 ;;
  val)   TF=$HJEPA_HOME/rasr_tasks/ant/ant_expert_grid_d3_val_s1001_n20.pt; ESB=300000 ;;
  train) TF=$HJEPA_HOME/rasr_tasks/ant/ant_expert_grid_d3_train_s1001_n40.pt; ESB=200000 ;;
esac
OUT=outputs/rasr/ant/v2/$SPLIT/$M/seed$S/shard_$TK
N=$(python3 -c "t='$TK';a,b=(t.split('-')+[t])[:2];print(int(b)-int(a)+1)")
[ "$(grep -c . $OUT/episodes.jsonl 2>/dev/null)" = "$N" ] && { echo "already done $SPLIT $M $S $TK"; exit 0; }
# several queues may share a job list: skip a job another queue is running (lock dir, or a live process on $OUT)
ps -eo args | grep -q "[r]un_baseline.py.*--out $OUT\( \|$\)" && { echo "running elsewhere $SPLIT $M $S $TK"; exit 0; }
mkdir -p "$(dirname $OUT)"; mkdir "$OUT.lock" 2>/dev/null || { echo "locked $SPLIT $M $S $TK"; exit 0; }
trap 'rmdir "$OUT.lock" 2>/dev/null' EXIT
P1=outputs/rasr/ant/v2/p1_reachability
SNAP="--no-snapshots --no-step-encode"
case $M in
  b1_baseline) EXTRA="" ;;
  src) EXTRA=""; SNAP="" ;;
  b2_freq_replan) EXTRA="--override hierarchical_plan_config.receding_horizon=1" ;;
  *_dist) EXTRA="--rasr-config configs/rasr/ant/${M%_dist}.yaml --predictor $P1/dist_model.pt" ;;
  *_mlp) EXTRA="--rasr-config configs/rasr/ant/${M%_mlp}.yaml --predictor $P1/mlp_model.pt" ;;
  *) EXTRA="--rasr-config configs/rasr/ant/$M.yaml --predictor none" ;;
esac
mkdir -p outputs/rasr/logs/v2
LOG=outputs/rasr/logs/v2/${SPLIT}_${M}_s${S}_${TK}.log
for try in 1 2 3 4 5; do   # GPU is shared with other jobs: retry on CUDA OOM (episodes are deterministic per seed)
  python scripts/run_baseline.py --config ant_l2 --model ant_hjepa_l2 --seed $S --tasks-file $TF --tasks $TK \
    --env-seed-base $ESB $SNAP $EXTRA --out $OUT > $LOG 2>&1
  [ "$(grep -c . $OUT/episodes.jsonl 2>/dev/null)" = "$N" ] && break
  grep -q OutOfMemoryError $LOG || break
  cp $LOG $LOG.oom$try; rm -f $OUT/episodes.jsonl; sleep $((60 * try))
done
echo "$(date +%H:%M) done $SPLIT $M $S $TK $(grep -c . $OUT/episodes.jsonl 2>/dev/null)"
