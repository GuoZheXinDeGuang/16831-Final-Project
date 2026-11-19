#!/usr/bin/env python
"""Write configs/rasr/ant/*.yaml from the FourRoom method configs, with AntMaze-specific constants.

Environment-dependent fields (everything else is copied unchanged from configs/rasr/<name>.yaml):
  eps              subgoal-reached threshold, train-calibrated by train_reachability.py (radius 0.5 = half the
                   AntMaze goal_tol of 1.0, the same ratio as FourRoom 8 px / 16 px)
  novelty_sigma2   visited-region kernel width, same ratio to eps as FourRoom (25 / 6.6966)
  reach_*          controller condition of the ant_l2 level-1 solver: 10 env steps per call, 90 GD iters x 25 samples
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FOURROOM_EPS, FOURROOM_SIGMA2 = 6.6966, 25.0
NAMES = ["baseline_copy", "full_replan", "rerank_l0.25", "rerank_l0.5", "rerank_l1.0",
         "rasr_b20", "rasr_b5", "repair_noR", "dump_alternatives"]


def main():
    eps = json.loads((ROOT / "outputs/rasr/ant/p1_reachability/results.json").read_text())["eps"]
    env = {"eps": round(float(eps), 4), "novelty_sigma2": round(float(eps) * FOURROOM_SIGMA2 / FOURROOM_EPS, 4),
           "reach_H": 10, "reach_n_steps": 90, "reach_num_samples": 25}
    out = ROOT / "configs/rasr/ant"
    out.mkdir(parents=True, exist_ok=True)
    for n in NAMES:
        cfg = yaml.safe_load((ROOT / f"configs/rasr/{n}.yaml").read_text()) or {}
        if cfg.get("mode", "baseline") != "baseline":
            cfg.update(env)
        (out / f"{n}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(env))


if __name__ == "__main__":
    main()
