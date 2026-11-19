#!/usr/bin/env python
"""Split a generated AntMaze task pool into train/val sets whose (start cell, goal cell) pairs are
disjoint from the official test set and from each other."""
import json, sys
import numpy as np, torch

test_f, pool_f, out_dir = sys.argv[1:4]
n_train, n_val = 40, 20
test = torch.load(test_f, weights_only=False)
pool = torch.load(pool_f, weights_only=False)
key = lambda t: (tuple(t["start_ij"]), tuple(t["goal_ij"]))
test_pairs = {key(t) for t in test["task_info"]["tasks"]}
idx_by_pair = {}
for i, t in enumerate(pool["task_info"]["tasks"]):
    if key(t) not in test_pairs:
        idx_by_pair.setdefault(key(t), []).append(i)
pairs = sorted(idx_by_pair)
rng = np.random.default_rng(0)
rng.shuffle(pairs)
n_val_pairs = len(pairs) // 3
val_pairs, train_pairs = pairs[:n_val_pairs], pairs[n_val_pairs:]


def pick(ps, n):  # round-robin over pairs so every pair is used before any repeats
    out, r = [], 0
    while len(out) < n:
        added = False
        for p in ps:
            if r < len(idx_by_pair[p]) and len(out) < n:
                out.append(idx_by_pair[p][r]); added = True
        if not added:
            raise SystemExit(f"not enough tasks: {len(out)} < {n}")
        r += 1
    return sorted(out)


def subset(idx, name, seed_tag):
    per_task = [k for k, v in pool.items() if isinstance(v, (list, tuple, np.ndarray, torch.Tensor))
                and not isinstance(v, str) and len(v) == len(pool["data"])]
    out = {k: v for k, v in pool.items()}
    for k in per_task:
        v = pool[k]
        out[k] = [v[i] for i in idx] if isinstance(v, list) else v[idx]
    ti = dict(pool["task_info"]); ti["tasks"] = [pool["task_info"]["tasks"][i] for i in idx]
    ti["num_episodes"] = len(idx); ti["rasr_split"] = name; ti["rasr_pool_indices"] = idx
    out["task_info"] = ti
    path = f"{out_dir}/ant_expert_grid_d3_{name}_{seed_tag}_n{len(idx)}.pt"
    torch.save(out, path)
    return path, per_task

tr, per_task = subset(pick(train_pairs, n_train), "train", "s1001")
va, _ = subset(pick(val_pairs, n_val), "val", "s1001")
info = {"test_pairs": len(test_pairs), "pool_pairs_not_in_test": len(pairs), "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs), "train": tr, "val": va, "per_task_keys": per_task,
        "overlap_train_test": 0, "overlap_val_test": 0, "overlap_train_val": len(set(train_pairs) & set(val_pairs))}
json.dump(info, open(f"{out_dir}/split_info.json", "w"), indent=2, default=str)
print(json.dumps(info, indent=1, default=str))
