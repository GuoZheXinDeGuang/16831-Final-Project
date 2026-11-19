#!/usr/bin/env bash
# P0 baseline: official 50 FourRoom tasks, H-JEPA L2 (seed 42), planner seed 42, 3 shards in parallel.
set -e
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
OUT=${OUT:-outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42}
mkdir -p "$OUT" outputs/rasr/logs
for sh in 0-16 17-33 34-49; do
  python scripts/run_baseline.py --tasks $sh --out "$OUT/shard_$sh" > outputs/rasr/logs/p0_shard_$sh.log 2>&1 &
done
wait
python scripts/summarize_baseline.py --run "$OUT"
