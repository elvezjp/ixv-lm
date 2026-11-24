"""Kimi Delta Attention。再帰式とチャンク形式は flash-linear-attention（MIT）の naive 実装に合わせた。"""
import math
import os

import mlx.core as mx
import mlx.nn as nn

from kda_metal import SCAN_TILE, THREADGROUP_BYTES, intra_chunk, linear_scan
from precision import Linear, RMSNorm

# 0 にすると、MLX の演算だけで計算する。GPU のロックアップの切り分けに使う
USE_METAL = os.environ.get("KDA_METAL", "1") != "0"


def kda_recurrent(q, k, v, g, beta):
    """逐次形式。入力は [B, T, H, D]、beta は [B, T, H]。"""
    batch, length, heads, key_dim = q.shape
    q = q * key_dim**-0.5
    state = mx.zeros((batch, heads, key_dim, v.shape[-1]), dtype=mx.float32)
    outputs = []
    for t in range(length):
        state = state * mx.exp(g[:, t])[..., None]
        k_t = k[:, t]
        error = v[:, t] - (k_t[..., None] * state).sum(-2)
        state = state + (beta[:, t, :, None] * k_t)[..., None] * error[..., None, :]
        outputs.append((q[:, t, :, :, None] * state).sum(-2))
    return mx.stack(outputs, axis=1)


def _inverse_unit_lower(strict_lower, size):
    """(I - L)^-1 を返す。L のべき乗で閉じる式は、キーが似ると桁落ちして値が壊れるので、2x2 のブロックで積み上げる。"""
    *lead, _, _ = strict_lower.shape
    inverse = mx.ones((*lead, size, 1, 1), dtype=strict_lower.dtype)
    width = 1
    while width < size:
        pairs = size // (2 * width)
        # [[A, 0], [C, D]]^-1 = [[A^-1, 0], [-D^-1 C A^-1, D^-1]] で、I - L の左下は -L なので符号が消える
        tiles = strict_lower.reshape(*lead, pairs, 2 * width, pairs, 2 * width)
        tiles = mx.moveaxis(mx.diagonal(tiles, axis1=-4, axis2=-2), -1, -3)
        lower_left = tiles[..., width:, :width]
        inverse = inverse.reshape(*lead, pairs, 2, width, width)
        upper, lower = inverse[..., 0, :, :], inverse[..., 1, :, :]
        top = mx.concatenate([upper, mx.zeros_like(upper)], axis=-1)
        bottom = mx.concatenate([lower @ lower_left @ upper, lower], axis=-1)
        inverse = mx.concatenate([top, bottom], axis=-2)
        width *= 2
    return inverse.reshape(*lead, size, size)


def _intra_chunk_ops(q, k, g, block):
    """チャンク内の A[i, j] = sum_d x_i exp(g_i - g_j) k_j（i >= j）を q と k について返す。"""
    *lead, size, key_dim = k.shape
    blocks = size // block
    shape = (*lead, blocks, block, key_dim)
    q_b, k_b, g_b = q.reshape(shape), k.reshape(shape), g.reshape(shape)

    # 対角ブロックは差の表を直接作る。上三角は exp が溢れるので、exp の前に 0 へ落とす
    rows = mx.arange(block)
    causal = rows[:, None] >= rows[None, :]
    delta = g_b[..., :, None, :] - g_b[..., None, :, :]
    decay = mx.exp(mx.where(causal[..., None], delta, 0.0))
    right = decay * k_b[..., None, :, :]
    diag_q = mx.where(causal, (q_b[..., :, None, :] * right).sum(-1), 0.0)
    diag_k = mx.where(causal, (k_b[..., :, None, :] * right).sum(-1), 0.0)

    rows_q, rows_k = [], []
    for b in range(blocks):
        start = b * block
        parts_q, parts_k = [], []
        if b:
            # ブロックの境目を基準にすると両側の指数が 0 以下になる
            ref = g[..., start - 1 : start, :]
            past = (k[..., :start, :] * mx.exp(ref - g[..., :start, :])).swapaxes(-1, -2)
            decay_now = mx.exp(g_b[..., b, :, :] - ref)
            parts_q.append((q_b[..., b, :, :] * decay_now) @ past)
            parts_k.append((k_b[..., b, :, :] * decay_now) @ past)
        parts_q.append(diag_q[..., b, :, :])
        parts_k.append(diag_k[..., b, :, :])
        if b + 1 < blocks:
            zeros = mx.zeros((*lead, block, size - start - block), dtype=k.dtype)
            parts_q.append(zeros)
            parts_k.append(zeros)
        rows_q.append(mx.concatenate(parts_q, axis=-1))
        rows_k.append(mx.concatenate(parts_k, axis=-1))
    return mx.concatenate(rows_q, axis=-2), mx.concatenate(rows_k, axis=-2)


def kda_chunk(q, k, v, g, beta, chunk_size=32, block_size=8, use_metal=USE_METAL):
    """チャンク形式。kda_recurrent と同じ値を返し、MLX の自動微分で逆伝播できる。"""
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    pad = -length % chunk_size
    if pad:
        widths = [(0, 0), (0, pad), (0, 0), (0, 0)]
        q, k, v, g = (mx.pad(t, widths) for t in (q, k, v, g))
        beta = mx.pad(beta, widths[:3])
    chunks = (length + pad) // chunk_size

    def split(t):
        return t.reshape(batch, chunks, chunk_size, heads, -1).transpose(0, 3, 1, 2, 4)

    q, k, v, g = split(q * key_dim**-0.5), split(k), split(v), split(g)
    beta = beta.reshape(batch, chunks, chunk_size, heads).transpose(0, 3, 1, 2)
    g = mx.cumsum(g, axis=-2)

    rows = mx.arange(chunk_size)
    strict = rows[:, None] > rows[None, :]
    if use_metal and 3 * chunk_size * (key_dim + 1) * 4 <= THREADGROUP_BYTES:
        a_qk, a_kk = intra_chunk(q, k, g)
    else:
        a_qk, a_kk = _intra_chunk_ops(q, k, g, block_size)
    lower = mx.where(strict, -a_kk * beta[..., None], 0.0)
    transform = _inverse_unit_lower(lower, chunk_size) * beta[..., None, :]
    w = transform @ (mx.exp(g) * k)
    u = transform @ v

    # 状態の更新は S' = P S + B に閉じる。順に回すのは行列積1回だけにして、残りは全チャンク同時に計算する
    g_last = g[..., -1:, :]
    decayed_k = (mx.exp(g_last - g) * k).swapaxes(-1, -2)
    update = decayed_k @ u
    carry = mx.exp(g_last).swapaxes(-1, -2) * mx.eye(key_dim, dtype=k.dtype) - decayed_k @ w

    if use_metal and value_dim % SCAN_TILE == 0 and (key_dim + SCAN_TILE) * key_dim * 4 <= THREADGROUP_BYTES:
        first = mx.zeros_like(update[:, :, :1])
        states = linear_scan(carry, mx.concatenate([first, update[:, :, :-1]], axis=2))
    else:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
        states = []
        for i in range(chunks):
            states.append(state)
            state = carry[:, :, i] @ state + update[:, :, i]
        states = mx.stack(states, axis=2)

    new_v = u - w @ states
    out = (q * mx.exp(g)) @ states + a_qk @ new_v
    out = out.transpose(0, 2, 3, 1, 4)
    return out.reshape(batch, length + pad, heads, value_dim)[:, :length]


def l2_normalize(x, eps=1e-6):
    return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)


def kda_gate(raw, a_log, dt_bias):
    """raw は [..., H, D]、a_log は [H]、dt_bias は [H * D]。返り値は対数の減衰で負になる。"""
    bias = dt_bias.reshape(raw.shape[-2:])
    return -mx.exp(a_log)[:, None] * nn.softplus(raw + bias)


class ShortConv(nn.Module):
    def __init__(self, channels, size=4):
        super().__init__()
        bound = size**-0.5
        self.weight = mx.random.uniform(-bound, bound, (size, channels))

    def __call__(self, x):
        size = self.weight.shape[0]
        padded = mx.pad(x, [(0, 0), (size - 1, 0), (0, 0)])
        length = x.shape[1]
        weight = self.weight.astype(x.dtype)
        return nn.silu(sum(padded[:, i : i + length] * weight[i] for i in range(size)))


class KimiDeltaAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        inner = n_heads * self.head_dim
        self.q_proj = Linear(dim, inner, bias=False)
        self.k_proj = Linear(dim, inner, bias=False)
        self.v_proj = Linear(dim, inner, bias=False)
        self.q_conv = ShortConv(inner)
        self.k_conv = ShortConv(inner)
        self.v_conv = ShortConv(inner)
        self.f_down = Linear(dim, self.head_dim, bias=False)
        self.f_up = Linear(self.head_dim, inner, bias=False)
        self.b_proj = Linear(dim, n_heads, bias=False)
        self.g_down = Linear(dim, self.head_dim, bias=False)
        self.g_up = Linear(self.head_dim, inner, bias=True)
        self.a_log = mx.log(mx.random.uniform(1.0, 16.0, (n_heads,)))
        dt = mx.exp(mx.random.uniform(math.log(0.001), math.log(0.1), (inner,)))
        dt = mx.maximum(dt, 1e-4)
        self.dt_bias = dt + mx.log(-mx.expm1(-dt))
        self.o_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.o_proj = Linear(inner, dim, bias=False)

    def __call__(self, x):
        batch, length, _ = x.shape
        shape = (batch, length, self.n_heads, self.head_dim)
        # 状態の計算は誤差が積もるので、行列積の型によらず float32 で行う
        wide = mx.float32
        q = l2_normalize(self.q_conv(self.q_proj(x)).astype(wide).reshape(shape))
        k = l2_normalize(self.k_conv(self.k_proj(x)).astype(wide).reshape(shape))
        v = self.v_conv(self.v_proj(x)).astype(wide).reshape(shape)
        g = kda_gate(self.f_up(self.f_down(x)).astype(wide).reshape(shape), self.a_log, self.dt_bias)
        beta = mx.sigmoid(self.b_proj(x).astype(wide))
        out = kda_chunk(q, k, v, g, beta)
        gate = mx.sigmoid(self.g_up(self.g_down(x)).astype(wide).reshape(shape))
        return self.o_proj((self.o_norm(out) * gate).reshape(batch, length, -1).astype(x.dtype))
