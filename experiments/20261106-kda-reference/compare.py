"""make_reference.py の結果と train/kda.py を照合する。"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
from kda import kda_chunk, kda_recurrent  # noqa: E402

ref = np.load(sys.argv[1])
names = sorted({key.split("/")[0] for key in ref.files})
keys = "q k v g beta".split()
worst = 0.0
for name in names:
    inputs = [mx.array(ref[f"{name}/{key}"]) for key in keys]
    weight = mx.array(ref[f"{name}/weight"])
    for label, fn in (("recurrent", kda_recurrent), ("chunk", kda_chunk)):
        if label == "recurrent" and name == "train":
            continue

        def objective(*args):
            return (fn(*args) * weight).sum()

        out = fn(*inputs)
        grads = mx.grad(objective, argnums=list(range(5)))(*inputs)
        mx.eval(out, grads)
        scale = np.abs(ref[f"{name}/out"]).max()
        out_err = np.abs(np.array(out) - ref[f"{name}/out"]).max() / scale
        grad_errs = []
        for key, grad in zip(keys, grads):
            target = ref[f"{name}/grad_{key}"]
            grad_errs.append(np.abs(np.array(grad) - target).max() / max(np.abs(target).max(), 1e-12))
        worst = max(worst, out_err, *grad_errs)
        print(f"{name:13s} {label:9s} out {out_err:.2e} grad " + " ".join(f"{k}={e:.1e}" for k, e in zip(keys, grad_errs)))
print(f"worst relative error {worst:.2e}")
