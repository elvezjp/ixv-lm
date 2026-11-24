"""KDA の順伝播と逆伝播の時間とピークメモリを測る。"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
import kda  # noqa: E402


def make_inputs(batch, length, heads, dim):
    mx.random.seed(0)
    q = kda.l2_normalize(mx.random.normal((batch, length, heads, dim)))
    k = kda.l2_normalize(mx.random.normal((batch, length, heads, dim)))
    v = mx.random.normal((batch, length, heads, dim))
    g = -0.1 * mx.random.uniform(shape=(batch, length, heads, dim))
    beta = mx.random.uniform(shape=(batch, length, heads))
    return q, k, v, g, beta


def measure(fn, inputs, repeats):
    step = mx.compile(mx.grad(lambda *a: fn(*a).sum(), argnums=list(range(5))))
    mx.eval(step(*inputs))
    mx.reset_peak_memory()
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(step(*inputs))
    return (time.perf_counter() - start) / repeats * 1000, mx.get_peak_memory() / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["32:32", "16:16", "32:16", "32:8", "64:16", "64:8"],
                        help="チャンク:ブロック")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    inputs = make_inputs(4, args.length, 4, 64)
    mx.eval(inputs)
    for config in args.configs:
        size, block = map(int, config.split(":"))
        ms, gb = measure(lambda *a: kda.kda_chunk(*a, chunk_size=size, block_size=block), inputs, args.repeats)
        print(f"chunk {size:3d} block {block:3d}: {ms:8.1f} ms  peak {gb:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
