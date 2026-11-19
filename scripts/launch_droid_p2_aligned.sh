#!/usr/bin/env bash
# DROID P2 with TIME-ALIGNED labels (post-hoc follow-up, added after the first DROID test result).
# Label: subgoal j counts as reachable only if it matches an expert frame within 3j +- 3 frames (droid_rasr.py --label aligned).
# Same protocol as launch_droid_p2.sh: predictor in {mlp, dist} x lam in {0.25, 0.5, 1.0} (+ lam 0) selected on 24 VAL clips,
# then the selection on the 16 test clips, planner seeds 1/2/3. lam 0 does not use the predictor -> reused from p2/.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
P1=outputs/rasr/droid/p1_reachability_aligned; O=outputs/rasr/droid/p2_aligned; L=outputs/rasr/logs
mkdir -p $O/val $O/test
# GPU memory gate: the GPU is shared with AntMaze jobs; start an eval only when > 2.5 GB is free, then let it allocate.
gate () { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 2500 ]; do sleep 30; done; }
ev () {  # split predictor lam planner_seed
  python scripts/droid_rasr.py eval --split $1 --lam $3 --planner-seed $4 --predictor $P1/${2}_model.pt \
    --out $O/$1/${2}_lam$3_ms1_ps$4 > $L/droid_aligned_eval_$1_${2}_lam$3_ps$4.log 2>&1
}
for lam in 0.25 0.5 1.0; do for p in mlp dist; do
  [ -f $O/val/${p}_lam${lam}_ms1_ps1/summary.json ] && continue
  gate; ev val $p $lam 1 & sleep 90
done; done
wait
SEL=$(python - <<'EOF'
import json
r = {"unmodified:0": json.load(open("outputs/rasr/droid/p2/val/lam0_ms1_ps1/summary.json"))["frechet_skill_mean"]}
for p in ("mlp", "dist"):
    for l in ("0.25", "0.5", "1.0"):
        r[f"{p}:{l}"] = json.load(open(f"outputs/rasr/droid/p2_aligned/val/{p}_lam{l}_ms1_ps1/summary.json"))["frechet_skill_mean"]
best = max(r, key=lambda k: (round(r[k], 6), -float(k.split(":")[1])))
json.dump({"label": "aligned", "val_frechet_skill": r, "selected": best,
           "rule": "max val Fréchet skill; ties -> smaller lam"},
          open("outputs/rasr/droid/p2_aligned/val_selection.json", "w"), indent=1)
print(best)
EOF
)
echo "selected $SEL"
SP=${SEL%%:*}; SL=${SEL##*:}
if [ "$SL" != 0 ]; then
  for ps in 1 2 3; do gate; ev test $SP $SL $ps & sleep 90; done
  wait
fi
echo "droid aligned p2 done"
