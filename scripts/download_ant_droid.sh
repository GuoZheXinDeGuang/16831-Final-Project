#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source activate_rasr.sh
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --include "ant/ant_hjepa_l2/seed4*/*" --include "droid/droid_hjepa_l2/seed1/*"
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir H-JEPA/h_jepa/assets --include "eval_trajs/ant/*"
hf download jepa-world-models/h-jepa --repo-type dataset --local-dir $HJEPA_HOME --include visual_antmaze_medium_stitch_val.h5 --include visual_antmaze_medium_probing_eval.h5 --include droid/droid_raw_eval16.tar --include droid/SHA256SUMS --include SHA256SUMS
(cd $HJEPA_HOME/droid && tar -xf droid_raw_eval16.tar)
echo DONE
