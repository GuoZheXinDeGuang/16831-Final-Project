#!/usr/bin/env python
"""Copy the small, final result files from outputs/rasr/ (gitignored, GBs) into results/ (committed), keeping paths.
Only summaries / tables / predictor metrics / per-episode CSVs and figures; no tensors, logs, checkpoints or videos."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "outputs/rasr", ROOT / "results"
NAMES = {"summary.json", "results.json", "dist_results.json", "comparison.md", "val_selection.json", "episodes.csv",
         "subgoal_outcomes_summary.json", "shift_results.json", "aggregate.json", "aggregate.md", "run_config.json",
         "trajectories.png", "calibration_roc_pr_test.png", "calibration_by_HB_cell_test.png"}
PREFIXES = ("compare_",)
SKIP_DIRS = {"smoke", "logs", "site", "videos"}
MAX_BYTES = 2_000_000


def main():
    n = 0
    for f in sorted(SRC.rglob("*")):
        rel = f.relative_to(SRC)
        if not f.is_file() or rel.parts[0] in SKIP_DIRS or f.stat().st_size > MAX_BYTES:
            continue
        if f.name in NAMES or f.name.startswith(PREFIXES) or (rel.parts[0] == "figures" and f.suffix == ".png"):
            if f.name == "run_config.json" and not f.parent.name.startswith("shard_0"):
                continue  # one run_config per run is enough
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if f.suffix in (".json", ".md", ".csv"):  # make absolute local paths relative to the project root
                out.write_text(f.read_text().replace(str(ROOT) + "/", "").replace(str(ROOT), "."))
            else:
                shutil.copy2(f, out)
            n += 1
    print(f"exported {n} files to {DST}")


if __name__ == "__main__":
    main()
