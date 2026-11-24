"""Metal を使った kda_chunk を、段階ごとの順逆伝播で測る。"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
import kda  # noqa: E402
from bench import make_inputs  # noqa: E402
from kda_metal import intra_chunk  # noqa: E402

B, T, H, D, C = 4, 512, 4, 64, 32
N = T // C


def timed(fn, inputs, repeats=30):
    step = mx.compile(mx.grad(lambda *a: fn(*a).sum(), argnums=list(range(len(inputs)))))
    mx.eval(step(*inputs))
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(step(*inputs))
    return (time.perf_counter() - start) / repeats * 1000


def split(t):
    return t.reshape(B, N, C, H, -1).transpose(0, 3, 1, 2, 4)


def main():
    q, k, v, g, beta = make_inputs(B, T, H, D)
    qc, kc, vc, gc = split(q * D**-0.5), split(k), split(v), mx.cumsum(split(g), axis=-2)
    bc = beta.reshape(B, N, C, H).transpose(0, 3, 1, 2)
    mx.eval(qc, kc, vc, gc, bc)
    rows = mx.arange(C)
    strict = rows[:, None] > rows[None, :]

    def prepare(a_kk, k_, v_, g_, b_):
        lower = mx.where(strict, -a_kk * b_[..., None], 0.0)
        transform = kda._inverse_unit_lower(lower, C) * b_[..., None, :]
        return (transform @ (mx.exp(g_) * k_)).sum() + (transform @ v_).sum()

    def build(k_, g_, u_, w_):
        last = g_[..., -1:, :]
        decayed = (mx.exp(last - g_) * k_).swapaxes(-1, -2)
        return (decayed @ u_).sum() + (mx.exp(last).swapaxes(-1, -2) * mx.eye(D) - decayed @ w_).sum()

    def scan(carry, update):
        state = mx.zeros((B, H, D, D))
        states = []
        for i in range(N):
            states.append(state)
            state = carry[:, :, i] @ state + update[:, :, i]
        return mx.stack(states, axis=2)

    def output(q_, g_, states, u_, w_, a_qk):
        return (q_ * mx.exp(g_)) @ states + a_qk @ (u_ - w_ @ states)

    a_qk, a_kk = intra_chunk(qc, kc, gc)
    w = mx.random.normal(kc.shape) * 0.1
    u = mx.random.normal(vc.shape)
    carry = mx.random.normal((B, H, N, D, D)) * 0.05
    update = mx.random.normal((B, H, N, D, D))
    states = mx.random.normal((B, H, N, D, D))
    mx.eval(a_qk, a_kk, w, u, carry, update, states)

    print(f"all      {timed(kda.kda_chunk, (q, k, v, g, beta)):6.2f} ms", flush=True)
    print(f"intra    {timed(lambda *a: sum(x.sum() for x in intra_chunk(*a)), (qc, kc, gc)):6.2f} ms", flush=True)
    print(f"prepare  {timed(prepare, (a_kk, kc, vc, gc, bc)):6.2f} ms", flush=True)
    print(f"build    {timed(build, (kc, gc, u, w)):6.2f} ms", flush=True)
    print(f"scan     {timed(scan, (carry, update)):6.2f} ms", flush=True)
    print(f"output   {timed(output, (qc, gc, states, u, w, a_qk)):6.2f} ms", flush=True)


if __name__ == "__main__":
    main()
