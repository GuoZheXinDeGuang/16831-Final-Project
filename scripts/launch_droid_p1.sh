#!/usr/bin/env bash
# DROID P1 (offline): level-2 candidate pools (16 GD samples) on 160 train / 40 val clips drawn
# from DROID training shards 3 (AUTOLab success) and 23 (PennPAL success); test = 16 eval clips.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
python scripts/droid_rasr.py collect --split train --n-clips 160 > outputs/rasr/logs/droid_collect_train.log 2>&1
python scripts/droid_rasr.py collect --split val --n-clips 40 > outputs/rasr/logs/droid_collect_val.log 2>&1
python scripts/droid_rasr.py train > outputs/rasr/logs/droid_train.log 2>&1
echo "droid p1 done"
