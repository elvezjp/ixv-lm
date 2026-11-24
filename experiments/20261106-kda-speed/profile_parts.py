"""kda_chunk の部分ごとに、順伝播と逆伝播の時間を測る。"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
import kda  # noqa: E402
from bench import make_inputs  # noqa: E402

BATCH, LENGTH, HEADS, DIM, CHUNK, BLOCK = 4, 512, 4, 64, 32, 8
CHUNKS = LENGTH // CHUNK


def timed(fn, inputs, argnums, repeats=20):
    forward = mx.compile(fn)
    backward = mx.compile(mx.grad(lambda *a: fn(*a).sum(), argnums=argnums))
    results = []
    for step in (forward, backward):
        mx.eval(step(*inputs))
        start = time.perf_counter()
        for _ in range(repeats):
            mx.eval(step(*inputs))
        results.append((time.perf_counter() - start) / repeats * 1000)
    return results


def main():
    q, k, v, g, beta = make_inputs(BATCH, LENGTH, HEADS, DIM)

    def split(t):
        return t.reshape(BATCH, CHUNKS, CHUNK, HEADS, -1).transpose(0, 3, 1, 2, 4)

    qc, kc, vc = split(q), split(k), split(v)
    gc = mx.cumsum(split(g), axis=-2)
    betac = beta.reshape(BATCH, CHUNKS, CHUNK, HEADS).transpose(0, 3, 1, 2)
    mx.eval(qc, kc, vc, gc, betac)

    rows = mx.arange(CHUNK)
    strict = rows[:, None] > rows[None, :]

    def intra(q_, k_, g_):
        a, b = kda._intra_chunk(q_, k_, g_, BLOCK)
        return a + b

    a_qk, a_kk = kda._intra_chunk(qc, kc, gc, BLOCK)
    lower = mx.where(strict, -a_kk * betac[..., None], 0.0)
    mx.eval(a_qk, lower)

    def inverse(l_):
        return kda._inverse_unit_lower(l_, CHUNK)

    transform = inverse(lower) * betac[..., None, :]
    w, u = transform @ (mx.exp(gc) * kc), transform @ vc
    mx.eval(w, u)

    def scan(q_, k_, g_, w_, u_, a_):
        state = mx.zeros((BATCH, HEADS, DIM, DIM))
        outs = []
        for i in range(CHUNKS):
            g_i, g_last = g_[:, :, i], g_[:, :, i, -1:]
            new_v = u_[:, :, i] - w_[:, :, i] @ state
            outs.append((q_[:, :, i] * mx.exp(g_i)) @ state + a_[:, :, i] @ new_v)
            state = state * mx.exp(g_last).swapaxes(-1, -2) + (mx.exp(g_last - g_i) * k_[:, :, i]).swapaxes(-1, -2) @ new_v
        return mx.stack(outs, axis=2)

    parts = {
        "intra": (intra, (qc, kc, gc), [0, 1, 2]),
        "inverse": (inverse, (lower,), [0]),
        "scan": (scan, (qc, kc, gc, w, u, a_qk), [0, 1, 2, 3, 4, 5]),
        "all": (kda.kda_chunk, (q, k, v, g, beta), [0, 1, 2, 3, 4]),
    }
    for name, (fn, inputs, argnums) in parts.items():
        fwd, both = timed(fn, inputs, argnums)
        print(f"{name:8s} forward {fwd:6.1f} ms  forward+backward {both:6.1f} ms", flush=True)


if __name__ == "__main__":
    main()
