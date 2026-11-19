#!/usr/bin/env python
"""Subgoal execution outcomes inside the *unmodified* baseline episodes (reuses P0 logs, no new rollouts).

At planning call t the level-1 MPC receives latent subgoal g_t and the baseline executes
receding_horizon*action_block = 10 env steps before replanning. We record
d_k = ||enc2(o_{t+k}) - g_t||^2 for k = 1..(steps until next call) and call the subgoal
'reached' if min_k d_k <= eps (eps from the P1 train-split calibration).

Also cross-checks branch attempts: label agreement between the baseline's own execution and the
branch B2 (same controller config, different solver noise) at H=5/10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402


def baseline_attempts(run: Path, eps: float):
    out = []
    eps_by_task = {}
    for l in sum([p.read_text().splitlines() for p in run.glob("**/episodes.jsonl")], []):
        e = json.loads(l)
        eps_by_task[e["task"]] = e
    for f in sorted(run.glob("**/ep_*.npz")):
        task = int(f.stem.split("_")[1])
        ep = np.load(f)
        steps = ep["call_env_step"]
        T = len(ep["xy"]) - 1
        for c, t in enumerate(steps):
            t_next = steps[c + 1] if c + 1 < len(steps) else T
            g = ep["subgoal"][c]
            d = ((ep["step_embed_2"][t + 1:t_next + 1] - g) ** 2).sum(-1)
            d0 = float(((ep["step_embed_2"][t] - g) ** 2).sum())
            out.append({
                "task": task, "call_idx": c, "env_step": int(t), "exec_steps": int(t_next - t),
                "from_upper_plan": bool(ep["subgoal_from_upper_plan"][c]), "d0": d0,
                "d_min": float(d.min()), "d_end": float(d[-1]), "reached": bool(d.min() <= eps),
                "d5": float(d[:5].min()), "d10": float(d[:10].min()) if len(d) >= 10 else None,
                "episode_success": eps_by_task[task]["success"],
                "before_success": bool(t < (eps_by_task[task]["steps_to_success"] if eps_by_task[task]["success"] else 1e9)),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/rasr/p0_baseline/fourroom_l2_ms42_ps42")
    ap.add_argument("--reach-results", default="outputs/rasr/p1_reachability/results.json")
    ap.add_argument("--attempts", default="outputs/rasr/p1_attempts/test")
    args = ap.parse_args()
    run = Path(args.run)
    eps = json.loads(Path(args.reach_results).read_text())["eps"]
    att = baseline_attempts(run, eps)
    with open(run / "subgoal_outcomes.jsonl", "w") as f:
        for a in att:
            f.write(json.dumps(a) + "\n")
    pre = [a for a in att if a["before_success"]]
    summ = {
        "eps": eps,
        "n_calls": len(att),
        "n_calls_before_success_or_fail_end": len(pre),
        "reach_rate_all_calls": float(np.mean([a["reached"] for a in att])),
        "reach_rate_before_success": float(np.mean([a["reached"] for a in pre])),
        "reach_rate_upper_plan_subgoals_before_success": float(np.mean([a["reached"] for a in pre if a["from_upper_plan"]])),
        "reach_rate_successful_episodes": float(np.mean([a["reached"] for a in pre if a["episode_success"]])),
        "reach_rate_failed_episodes": float(np.mean([a["reached"] for a in pre if not a["episode_success"]] or [np.nan])),
        "mean_d0": float(np.mean([a["d0"] for a in pre])),
        "mean_d_min": float(np.mean([a["d_min"] for a in pre])),
    }
    # consistency with branch B2 (same controller config) at H=10
    att_dir = Path(args.attempts)
    if att_dir.exists():
        by = {(a["task"], a["call_idx"]): a for a in att}
        agree5, agree10, n = [], [], 0
        for f in sorted(att_dir.glob("attempts_task*.npz")):
            b = np.load(f)
            for i in np.nonzero(b["budget_n_steps"] == 90)[0]:
                key = (int(b["task"][i]), int(b["call_idx"][i]))
                a = by.get(key)
                if a is None or a["exec_steps"] < 10:
                    continue
                agree5.append((b["dists"][i][:5].min() <= eps) == (a["d5"] <= eps))
                agree10.append((b["dists"][i][:10].min() <= eps) == (a["d10"] <= eps))
        if agree10:
            summ["branchB2_vs_baseline_label_agreement"] = {"H5": float(np.mean(agree5)), "H10": float(np.mean(agree10)), "n": len(agree10)}
    (run / "subgoal_outcomes_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
