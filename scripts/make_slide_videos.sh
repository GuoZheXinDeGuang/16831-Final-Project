#!/usr/bin/env bash
# Slide videos (AntMaze, top-down, from logged episodes) -> outputs/rasr/ant/v2/videos/. See scripts/make_ant_videos.py.
cd "$(dirname "$0")/.."
source ./activate_rasr.sh
V="python scripts/make_ant_videos.py"
B1="H-JEPA baseline=test/b1_baseline"; OURS="Ours: novelty repair (final)=test/repair_noR_b50"
# 1. headline: baseline vs original method vs no-novelty ablation vs final method (only novelty repair succeeds)
$V --task 7  --seed 42 --method "$B1" --method "Original Ours (with R)=test/rasr_b20_dist" \
   --method "Repair, no novelty (β=0)=test/repair_noR_b0" --method "$OURS" --name 01_headline_task007 --single
$V --task 41 --seed 42 --method "$B1" --method "Full replan on stall (B4)=test/full_replan" \
   --method "Best R variant (λ_R=1)=test/v2_lr1_dist" --method "$OURS" --name 02_vs_replan_task041
# 2. why R hurts: the baseline succeeds, adding R loses the task, removing R recovers it
$V --task 9  --seed 42 --method "$B1" --method "Original Ours (with R)=test/rasr_b20_dist" --method "$OURS" \
   --name 03_R_hurts_task009
$V --task 2  --seed 42 --method "$B1" --method "Original Ours (with R)=test/rasr_b20_dist" --method "$OURS" \
   --name 04_R_hurts_task002
# 3. our attempts to fix R (val tasks): log / hinge / expected-cost / miss-memory vs the final method
$V --task 2  --seed 42 --method "Repair + R, log (λ_R=10)=val/v2_lr10_dist" --method "Repair + R, hinge=val/v2_hinge_dist" \
   --method "Repair + R, expected cost=val/v2_expected_dist" --method "Final (no R)=val/repair_noR_b50" --name 05_R_attempts_val_task002
$V --task 16 --seed 43 --method "Repair + R, log (λ_R=10)=val/v2_lr10_dist" --method "Repair + R, miss memory=val/v2_memory_dist" \
   --method "Repair + R, expected cost=val/v2_expected_dist" --method "Final (no R)=val/repair_noR_b50" --name 06_R_attempts_val_task016
# 4. more successes of the final method, and an honest failure (all methods fail)
$V --task 38 --seed 42 --method "$B1" --method "$OURS" --name 07_success_task038
$V --task 44 --seed 42 --method "$B1" --method "$OURS" --name 08_success_task044
$V --task 3  --seed 42 --method "$B1" --method "$OURS" --name 09_failure_task003
echo "all slide videos done"
