#!/usr/bin/env bash
# AntMaze P1 attempt collection. Budgets: B0 2x4, B1 10x20, B2 90x25 (= ant_l2 level-1 solver).
# H_max 60 env steps (AntMaze subgoals take ~10-50 steps). train: every 2nd call; val/test: every 3rd (GPU time).
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
export MUJOCO_GL=egl
O=outputs/rasr/ant; L=outputs/rasr/logs; A=$O/p1_attempts
BS="B0:2x4,B1:10x20,B2:90x25"
wait_for () { until [ "$(cat $1/shard_*/episodes.jsonl 2>/dev/null | wc -l)" -ge "$2" ]; do sleep 30; done; }
collect () { python scripts/collect_attempts.py --baseline-run $1 --out $2 --tasks $3 --call-stride $4 \
  --h-max 60 --budget-spec $BS > $L/$5.log 2>&1; }
P0=$O/p0_baseline/ant_l2_ms42_ps42
( wait_for $O/p1_source/val 20;   collect $O/p1_source/val   $A/val   all 3 ant_collect_val ) &
( wait_for $O/p1_source/train 40; collect $O/p1_source/train $A/train $(seq -s, 0 13) 2 ant_collect_tr0 ) &
( wait_for $O/p1_source/train 40; collect $O/p1_source/train $A/train $(seq -s, 14 26) 2 ant_collect_tr1 ) &
( wait_for $O/p1_source/train 40; collect $O/p1_source/train $A/train $(seq -s, 27 39) 2 ant_collect_tr2 ) &
( wait_for $P0 50; collect $P0 $A/test $(seq -s, 0 16) 3 ant_collect_te0 ) &
( wait_for $P0 50; collect $P0 $A/test $(seq -s, 17 33) 3 ant_collect_te1 ) &
( wait_for $P0 50; collect $P0 $A/test $(seq -s, 34 49) 3 ant_collect_te2 ) &
wait
echo "ant collection done"
