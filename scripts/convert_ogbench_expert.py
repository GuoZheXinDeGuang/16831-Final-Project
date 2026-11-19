#!/usr/bin/env python
"""Re-save the OGBench ant expert params (pickled with an old JAX) as plain numpy arrays.
Old pickles call jax._src.array._reconstruct_array with an aval_state containing 'named_shape',
which current JAX rejects. Values are untouched; the original file is kept as *.orig."""
import pickle, shutil, sys
from pathlib import Path
import numpy as np


def _recon(fun, args, arr_state, aval_state):
    a = fun(*args)
    a.__setstate__(arr_state)
    return np.asarray(a)


class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "jax._src.array" and name == "_reconstruct_array":
            return _recon
        return super().find_class(mod, name)


p = Path(sys.argv[1])
orig = p.with_suffix(p.suffix + ".orig")
if not orig.exists():
    shutil.copy2(p, orig)
obj = U(open(orig, "rb")).load()
pickle.dump(obj, open(p, "wb"))
leaves = []
def walk(x):
    if isinstance(x, dict):
        for v in x.values(): walk(v)
    elif isinstance(x, np.ndarray): leaves.append(x.size)
walk(obj)
print("converted", p, "arrays", len(leaves), "params", sum(leaves))
