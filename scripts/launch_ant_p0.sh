#!/usr/bin/env bash
# AntMaze P0 baseline (official ant_l2 planner, ant_hjepa_l2 seed 42) on the official 50 test tasks,
# plus the same planner on the RASR train/val task sets (P1 data source). 7 GPU processes.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
export MUJOCO_GL=egl
TE=H-JEPA/h_jepa/assets/eval_trajs/ant/expert_grid_d3_n50.pt
TR=$HJEPA_HOME/rasr_tasks/ant/ant_expert_grid_d3_train_s1001_n40.pt
VA=$HJEPA_HOME/rasr_tasks/ant/ant_expert_grid_d3_val_s1001_n20.pt
O=outputs/rasr/ant; L=outputs/rasr/logs
run () {  # tasks_file tasks seed_base out log
  python scripts/run_baseline.py --config ant_l2 --model ant_hjepa_l2 --tasks-file $1 --tasks $2 \
    --env-seed-base $3 --out $4 > $L/$5.log 2>&1
}
run $TE 0-16  100000 $O/p0_baseline/ant_l2_ms42_ps42/shard_0-16  ant_p0_0 &
run $TE 17-33 100000 $O/p0_baseline/ant_l2_ms42_ps42/shard_17-33 ant_p0_1 &
run $TE 34-49 100000 $O/p0_baseline/ant_l2_ms42_ps42/shard_34-49 ant_p0_2 &
run $TR 0-13  200000 $O/p1_source/train/shard_0-13  ant_src_tr0 &
run $TR 14-26 200000 $O/p1_source/train/shard_14-26 ant_src_tr1 &
run $TR 27-39 200000 $O/p1_source/train/shard_27-39 ant_src_tr2 &
run $VA all   300000 $O/p1_source/val/shard_0-19    ant_src_val &
wait
echo "ant p0 done"
