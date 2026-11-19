#!/usr/bin/env bash
# DROID offline baseline: official droid_l2 planner (config defaults = paper release cell) on the 16
# evaluation clips, H-JEPA model seeds 1/1000/10000 x planner seeds 1/2/3 (as scripts/eval_droid.sh).
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --include "droid/droid_hjepa_l2/*" > /dev/null
cd H-JEPA/h_jepa
O=../../outputs/rasr/droid/baseline
for ms in ${SEEDS:-1 1000 10000}; do
  for ps in 1 2 3; do
    out=$O/hjepa_l2_ms${ms}/plan_seed${ps}
    [ -f $out/eval.csv ] && continue
    python eval.py --config-name droid_l2 policy=$HJEPA_HOME/ckpts/droid/droid_hjepa_l2/seed${ms}/droid_hjepa_l2_object.ckpt \
      seed=$ps output.dir=$(realpath -m $out) hydra.run.dir=$(realpath -m $out)/hydra
  done
done
