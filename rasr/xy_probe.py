"""Analysis-only probe: level-2 latent -> agent xy via kNN over real logged frames (P0/P1 source
episodes). Used ONLY for visualisation/diagnostics, never inside the planner."""
from pathlib import Path

import numpy as np
import torch


class XYProbe:
    def __init__(self, dirs, k=5, max_frames=200_000):
        Z, XY = [], []
        for d in dirs:
            for f in sorted(Path(d).glob("**/ep_*.npz")):
                ep = np.load(f)
                if "step_embed_2" in ep.files:
                    Z.append(ep["step_embed_2"]); XY.append(ep["xy"])
        self.Z = torch.as_tensor(np.concatenate(Z)[:max_frames])
        self.XY = np.concatenate(XY)[:max_frames]
        self.k = k

    def __call__(self, z):
        z = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32)
        d = torch.cdist(z, self.Z)
        v, i = d.topk(self.k, largest=False)
        return self.XY[i.numpy()].mean(1), v[:, 0].numpy() ** 2
