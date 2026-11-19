#!/usr/bin/env bash
# AntMaze chain after attempt collection: train reachability -> write configs -> P2/P3 on test, planner seed 42.
# Method hyperparameters (lam, repair_lam, novelty_beta, triggers) are the FourRoom val-selected values, NOT re-tuned
# on AntMaze (compute); only eps / kernel width / controller condition are AntMaze-specific (make_ant_configs.py).
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
A=outputs/rasr/ant/p1_attempts; L=outputs/rasr/logs
n_att () { ls $A/$1/attempts_task*.npz 2>/dev/null | wc -l; }
until [ "$(n_att train)" -ge 40 ] && [ "$(n_att val)" -ge 20 ] && [ "$(n_att test)" -ge 50 ]; do sleep 60; done
python scripts/train_reachability.py --train-attempts $A/train --val-attempts $A/val --test-attempts $A/test \
  --train-baseline outputs/rasr/ant/p1_source/train --radius-px 0.5 --radius-list 0.25,0.5,1.0 \
  --h-values 10,20,40,60 --out outputs/rasr/ant/p1_reachability > $L/ant_train_reach.log 2>&1
python scripts/make_ant_configs.py > $L/ant_make_configs.log 2>&1
export SHARDS="0-16 17-33 34-49"
bash scripts/launch_ant_p2p3.sh test 42 rerank_l0.25 rasr_b20
bash scripts/launch_ant_p2p3.sh test 42 full_replan repair_noR
echo "ant p2p3 ps42 done"
