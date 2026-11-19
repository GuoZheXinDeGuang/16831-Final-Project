#!/usr/bin/env bash
# P1 attempt collection (branch rollouts from baseline snapshots). Waits for each source run.
#   train: all planning calls (stride 1) of 40 train tasks; val/test: every 2nd call.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
wait_for () {  # dir n_expected
  until [ "$(cat $1/shard_*/episodes.jsonl 2>/dev/null | wc -l)" -ge "$2" ]; do sleep 20; done
}
collect () {  # src out tasks stride log
  python scripts/collect_attempts.py --baseline-run $1 --out $2 --tasks $3 --call-stride $4 > outputs/rasr/logs/$5.log 2>&1
}
P0=outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42
A=outputs/rasr/p1_attempts
( wait_for outputs/rasr/p1_source/val 20;   collect outputs/rasr/p1_source/val   $A/val   all 2 collect_val ) &
( wait_for outputs/rasr/p1_source/train 40; collect outputs/rasr/p1_source/train $A/train $(seq -s, 0 19) 1 collect_train_a ) &
( wait_for outputs/rasr/p1_source/train 40; collect outputs/rasr/p1_source/train $A/train $(seq -s, 20 39) 1 collect_train_b ) &
( wait_for $P0 50; collect $P0 $A/test $(seq -s, 0 24) 2 collect_test_a ) &
( wait_for $P0 50; collect $P0 $A/test $(seq -s, 25 49) 2 collect_test_b ) &
wait
echo "collection done"
