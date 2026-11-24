"""Metal カーネルを、MLX の演算で書いた同じ計算と照合し、時間を比べる。"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
import kda  # noqa: E402
from kda_metal import intra_chunk  # noqa: E402


def data(shape, decay):
    mx.random.seed(0)
    q = kda.l2_normalize(mx.random.normal(shape))
    k = kda.l2_normalize(mx.random.normal(shape))
    g = mx.cumsum(-decay * mx.random.uniform(shape=shape), axis=-2)
    return q, k, g


def loss(fn, w1, w2):
    def inner(q, k, g):
        a, b = fn(q, k, g)
        mask = mx.tril(mx.ones(a.shape[-2:]))
        return (a * w1 * mask).sum() + (b * w2 * mask).sum()
    return inner


def relative(x, y):
    return (mx.abs(x - y).max() / mx.maximum(mx.abs(y).max(), 1e-12)).item()


def main():
    ops = lambda q, k, g: kda._intra_chunk_ops(q, k, g, 8)
    for shape, decay in (((2, 2, 3, 32, 16), 0.1), ((4, 4, 16, 32, 64), 0.1), ((1, 2, 2, 32, 64), 5.0), ((1, 2, 2, 32, 64), 100.0), ((1, 2, 2, 64, 32), 0.1)):
        q, k, g = data(shape, decay)
        w1, w2 = mx.random.normal(shape[:-1] + (shape[-2],)), mx.random.normal(shape[:-1] + (shape[-2],))
        out_m, out_o = intra_chunk(q, k, g), ops(q, k, g)
        grads_m = mx.grad(loss(intra_chunk, w1, w2), argnums=[0, 1, 2])(q, k, g)
        grads_o = mx.grad(loss(ops, w1, w2), argnums=[0, 1, 2])(q, k, g)
        mask = mx.tril(mx.ones((shape[-2], shape[-2])))
        errs = [relative(a * mask, b * mask) for a, b in zip(out_m, out_o)] + [relative(a, b) for a, b in zip(grads_m, grads_o)]
        finite = all(bool(mx.isfinite(x).all().item()) for x in grads_m)
        print(shape, decay, "finite", finite, " ".join(f"{e:.1e}" for e in errs), flush=True)

    q, k, g = data((4, 4, 16, 32, 64), 0.1)
    w1 = w2 = mx.random.normal((4, 4, 16, 32, 32))
    mx.eval(q, k, g, w1)
    for name, fn in (("ops", ops), ("metal", intra_chunk)):
        step = mx.compile(mx.grad(loss(fn, w1, w2), argnums=[0, 1, 2]))
        mx.eval(step(q, k, g))
        start = time.perf_counter()
        for _ in range(20):
            mx.eval(step(q, k, g))
        print(f"{name:6s} forward+backward {(time.perf_counter() - start) / 20 * 1000:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
