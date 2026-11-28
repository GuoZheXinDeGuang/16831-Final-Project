#!/usr/bin/env bash
# v2 (A6000 rerun) chain: wait for P1 source episodes -> collect attempts -> train predictors -> Phase C val sweep.
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU:-1}
V=outputs/rasr/ant/v2; A=$V/p1_attempts; L=outputs/rasr/logs/v2; mkdir -p $L $A
n_ep () { cat $V/$1/src/seed42/shard_*/episodes.jsonl 2>/dev/null | grep -c .; }
until [ "$(n_ep train)" -ge 40 ] && [ "$(n_ep val)" -ge 20 ]; do sleep 60; done
echo "$(date +%H:%M) sources done"
BS="B0:2x4,B1:10x20,B2:90x25"
{ for t in $(seq 0 39); do echo "train 2 $t"; done; for t in $(seq 0 19); do echo "val 3 $t"; done; } |
  xargs -P ${COLLECT_P:-8} -L 1 bash -c 'f='$A'/$0/attempts_task$(printf %03d $2).npz; for try in 1 2 3 4 5; do [ -f $f ] && break; \
    python scripts/collect_attempts.py --baseline-run '$V'/$0/src/seed42 --out '$A'/$0 --tasks $2 --call-stride $1 \
    --h-max 60 --budget-spec '$BS' > '$L'/collect_$0_$2.log 2>&1 || sleep $((60 * try)); done; \
    echo "$(date +%H:%M) collected $0 $2 $([ -f $f ] && echo ok || echo MISSING)"'
echo "$(date +%H:%M) collection done: $(ls $A/train | grep -c npz) train, $(ls $A/val | grep -c npz) val"
# no test attempts on this machine: val doubles as the 'test' slot of train_reachability (predictor metrics = val only)
python scripts/train_reachability.py --train-attempts $A/train --val-attempts $A/val --test-attempts $A/val --test-is-val \
  --train-baseline $V/train/src/seed42 --radius-px 0.5 --radius-list 0.25,0.5,1.0 \
  --h-values 10,20,40,60 --out $V/p1_reachability > $L/train_reach.log 2>&1 || { echo "train_reachability failed"; exit 1; }
python scripts/fit_dist_predictor.py --attempts $A --run $V/p1_reachability --h-values 10,20,40,60 > $L/fit_dist.log 2>&1 \
  || { echo "fit_dist failed"; exit 1; }
echo "$(date +%H:%M) predictors: $(ls $V/p1_reachability)"
J=$V/jobsC.txt; : > $J
for s in 42 43; do for c in rasr_b20 v2_lr1 v2_lr3 v2_lr10 v2_lr10_nogate v2_hinge v2_expected v2_memory v2_exp_full v2_exp_rerank; do
  for t in $(seq 0 19); do echo "val ${c}_dist $s $t" >> $J; done; done; done
xargs -P ${JOBS_P:-12} -L 1 scripts/run_ant_job.sh < $J
echo "$(date +%H:%M) phase C done"
