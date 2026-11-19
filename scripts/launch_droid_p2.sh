#!/usr/bin/env bash
# DROID P2 (offline). Selection on 24 VAL clips (planner seed 1) over predictor in {mlp, dist} x lam in {0.25, 0.5, 1.0}
# (+ lam 0 = unmodified planner); the best val Fréchet fidelity wins (ties -> lam 0 / smaller lam). Then lam 0 and the
# selected (predictor, lam) on the 16 test clips for planner seeds 1/2/3. Model seed 1 only (predictors use its latents).
#   mlp  = reachability MLP (droid_rasr.py train);  dist = DistLogistic on ||g - z0||^2 and log H (droid_rasr.py train-dist)
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
P1=outputs/rasr/droid/p1_reachability; O=outputs/rasr/droid/p2; L=outputs/rasr/logs
until [ -f $P1/mlp_model.pt ]; do sleep 60; done
[ -f $P1/dist_model.pt ] || python scripts/droid_rasr.py train-dist > $L/droid_train_dist.log 2>&1
tag () { [ "$1" = mlp ] && echo "lam$2" || echo "${1}_lam$2"; }  # mlp keeps the original directory names
ev () {  # split predictor lam planner_seed
  local t; t=$(tag $2 $3)
  python scripts/droid_rasr.py eval --split $1 --lam $3 --planner-seed $4 --predictor $P1/${2}_model.pt \
    --out $O/$1/${t}_ms1_ps$4 > $L/droid_eval_$1_${t}_ps$4.log 2>&1
}
ev val mlp 0 1 &
for lam in 0.25 0.5 1.0; do ev val mlp $lam 1 & ev val dist $lam 1 & done
wait
SEL=$(python - <<'EOF'
import json
D = "outputs/rasr/droid/p2/val"
cands = [("mlp", "0")] + [(p, l) for p in ("mlp", "dist") for l in ("0.25", "0.5", "1.0")]
r = {}
for p, l in cands:
    t = f"lam{l}" if p == "mlp" else f"{p}_lam{l}"
    r[f"{p}:{l}"] = json.load(open(f"{D}/{t}_ms1_ps1/summary.json"))["frechet_skill_mean"]
best = max(r, key=lambda k: (round(r[k], 6), -float(k.split(":")[1])))
json.dump({"val_frechet_skill": r, "selected": best, "rule": "max val Fréchet skill; ties -> smaller lam"},
          open("outputs/rasr/droid/p2/val_selection.json", "w"), indent=1)
print(best)
EOF
)
echo "selected $SEL"
SP=${SEL%%:*}; SL=${SEL##*:}
for ps in 1 2 3; do
  [ -f $O/test/lam0_ms1_ps$ps/summary.json ] || ev test mlp 0 $ps &
  [ "$SL" = 0 ] || ev test $SP $SL $ps &
done
wait
echo "droid p2 done"
