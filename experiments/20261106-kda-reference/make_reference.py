"""flash-linear-attention の naive 実装で、照合用の出力と勾配を作る。torch の別環境で動かす。"""
import importlib.util
import sys

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("naive", sys.argv[1])
naive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(naive)
out_path = sys.argv[2]

results = {}
for name, (batch, length, heads, dim, decay) in {
    "small": (2, 64, 2, 16, 0.1),
    "train": (2, 512, 4, 64, 0.1),
    "strong_decay": (2, 128, 2, 32, 2.0),
    "padded": (1, 77, 2, 16, 0.1),
}.items():
    rng = np.random.default_rng(0)
    q, k = (rng.standard_normal((batch, length, heads, dim)).astype(np.float32) for _ in range(2))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((batch, length, heads, dim)).astype(np.float32)
    g = (-decay * rng.random((batch, length, heads, dim))).astype(np.float32)
    beta = rng.random((batch, length, heads)).astype(np.float32)
    weight = rng.standard_normal((batch, length, heads, dim)).astype(np.float32)
    tensors = [torch.tensor(t, dtype=torch.float64 if False else torch.float32, requires_grad=True) for t in (q, k, v, g, beta)]
    out, _ = naive.naive_recurrent_kda(*tensors)
    (out * torch.tensor(weight)).sum().backward()
    for key, value in zip("q k v g beta weight".split(), (q, k, v, g, beta, weight)):
        results[f"{name}/{key}"] = value
    results[f"{name}/out"] = out.detach().numpy()
    for key, tensor in zip("q k v g beta".split(), tensors):
        results[f"{name}/grad_{key}"] = tensor.grad.numpy()
np.savez(out_path, **results)
print("saved", len(results))
