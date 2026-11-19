#!/usr/bin/env bash
# Distribution-shift check: execute non-selected level-2 candidates (val tasks) with the B2 controller.
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
until [ "$(grep -c . outputs/rasr/p4_shift/val_pool/episodes.jsonl 2>/dev/null)" -ge 20 ]; do sleep 30; done
for tk in all; do  # one process: GPU memory is shared with the seed queue
  python scripts/collect_attempts.py --baseline-run outputs/rasr/p1_source/val --out outputs/rasr/p4_shift/attempts \
    --tasks $tk --call-stride 2 --budgets B2 --alt-from outputs/rasr/p4_shift/val_pool/rasr_calls.jsonl \
    > outputs/rasr/logs/p4_shift_collect_${tk%%,*}.log 2>&1 &
done
wait
echo shift-collection-done
