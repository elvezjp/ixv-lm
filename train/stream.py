"""状態を引き継いで1トークンずつ生成する。

学習用の LM は毎回、文脈を全部計算し直す。ここでは層ごとに状態を持ち越す。
- KDA の層：状態行列 S と、短畳み込みの直前3トークンぶんの入力
- 全注意の層：過去のキーと値（KV キャッシュ）

プロンプトはチャンク形式でまとめて流し、最後の状態だけを取り出す（プリフィル）。
以降は1トークンずつ逐次形式で進めるので、1トークンあたりの時間は文脈の長さにほとんどよらない。
モデルのコードは読み込んだ LM のものをそのまま使い、層の重みと補助関数を借りる。
"""
import sys

import mlx.core as mx
import mlx.nn as nn


def _kda_module(layer):
    return sys.modules[type(layer).__module__]


def _conv_prefill(conv, x):
    """短畳み込みをかけ、次の逐次計算のために直前 size-1 トークンぶんの入力を返す。"""
    size = conv.weight.shape[0]
    out = conv(x)
    padded = mx.pad(x, [(0, 0), (size - 1, 0), (0, 0)])
    return out, padded[:, -(size - 1):]


def _conv_step(conv, x, cache):
    """x は [B, 1, C]、cache は [B, size-1, C]。"""
    window = mx.concatenate([cache, x], axis=1)
    weight = conv.weight.astype(x.dtype)
    out = nn.silu((window * weight).sum(axis=1, keepdims=True))
    return out, window[:, 1:]


def _kda_inputs(layer, x, q_raw, k_raw, v_raw):
    kda = _kda_module(layer)
    batch, length, _ = x.shape
    shape = (batch, length, layer.n_heads, layer.head_dim)
    wide = mx.float32
    q = kda.l2_normalize(q_raw.astype(wide).reshape(shape))
    k = kda.l2_normalize(k_raw.astype(wide).reshape(shape))
    v = v_raw.astype(wide).reshape(shape)
    g = kda.kda_gate(layer.f_up(layer.f_down(x)).astype(wide).reshape(shape), layer.a_log, layer.dt_bias)
    beta = mx.sigmoid(layer.b_proj(x).astype(wide))
    return q, k, v, g, beta


def _kda_output(layer, x, out):
    batch, length, _ = x.shape
    shape = (batch, length, layer.n_heads, layer.head_dim)
    gate = mx.sigmoid(layer.g_up(layer.g_down(x)).astype(mx.float32).reshape(shape))
    return layer.o_proj((layer.o_norm(out) * gate).reshape(batch, length, -1).astype(x.dtype))


def _kda_chunk_with_state(kda, q, k, v, g, beta, chunk_size=32, block_size=8):
    """kda.kda_chunk と同じ出力に加えて、最後のトークンの後の状態 [B, H, K, V] を返す。"""
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    pad = -length % chunk_size
    if pad:
        # 詰め物は k = 0、g = 0、beta = 0 なので、状態を変えない
        widths = [(0, 0), (0, pad), (0, 0), (0, 0)]
        q, k, v, g = (mx.pad(t, widths) for t in (q, k, v, g))
        beta = mx.pad(beta, widths[:3])
    chunks = (length + pad) // chunk_size

    def split(t):
        return t.reshape(batch, chunks, chunk_size, heads, -1).transpose(0, 3, 1, 2, 4)

    q, k, v, g = split(q * key_dim**-0.5), split(k), split(v), split(g)
    beta = beta.reshape(batch, chunks, chunk_size, heads).transpose(0, 3, 1, 2)
    g = mx.cumsum(g, axis=-2)

    use_metal = getattr(kda, "USE_METAL", True)
    rows = mx.arange(chunk_size)
    strict = rows[:, None] > rows[None, :]
    if use_metal and 3 * chunk_size * (key_dim + 1) * 4 <= kda.THREADGROUP_BYTES:
        a_qk, a_kk = kda.intra_chunk(q, k, g)
    else:
        a_qk, a_kk = kda._intra_chunk_ops(q, k, g, block_size)
    lower = mx.where(strict, -a_kk * beta[..., None], 0.0)
    transform = kda._inverse_unit_lower(lower, chunk_size) * beta[..., None, :]
    w = transform @ (mx.exp(g) * k)
    u = transform @ v

    g_last = g[..., -1:, :]
    decayed_k = (mx.exp(g_last - g) * k).swapaxes(-1, -2)
    update = decayed_k @ u
    carry = mx.exp(g_last).swapaxes(-1, -2) * mx.eye(key_dim, dtype=k.dtype) - decayed_k @ w

    if use_metal and value_dim % kda.SCAN_TILE == 0 and (key_dim + kda.SCAN_TILE) * key_dim * 4 <= kda.THREADGROUP_BYTES:
        first = mx.zeros_like(update[:, :, :1])
        states = kda.linear_scan(carry, mx.concatenate([first, update[:, :, :-1]], axis=2))
    else:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=mx.float32)
        states = []
        for i in range(chunks):
            states.append(state)
            state = carry[:, :, i] @ state + update[:, :, i]
        states = mx.stack(states, axis=2)
    final = carry[:, :, -1] @ states[:, :, -1] + update[:, :, -1]

    new_v = u - w @ states
    out = (q * mx.exp(g)) @ states + a_qk @ new_v
    out = out.transpose(0, 2, 3, 1, 4).reshape(batch, length + pad, heads, value_dim)[:, :length]
    return out, final


def _kda_prefill(layer, x):
    q_raw, q_cache = _conv_prefill(layer.q_conv, layer.q_proj(x))
    k_raw, k_cache = _conv_prefill(layer.k_conv, layer.k_proj(x))
    v_raw, v_cache = _conv_prefill(layer.v_conv, layer.v_proj(x))
    q, k, v, g, beta = _kda_inputs(layer, x, q_raw, k_raw, v_raw)
    out, state = _kda_chunk_with_state(_kda_module(layer), q, k, v, g, beta)
    return _kda_output(layer, x, out), {"state": state, "conv": (q_cache, k_cache, v_cache)}


def _kda_step(layer, x, cache):
    q_cache, k_cache, v_cache = cache["conv"]
    q_raw, q_cache = _conv_step(layer.q_conv, layer.q_proj(x), q_cache)
    k_raw, k_cache = _conv_step(layer.k_conv, layer.k_proj(x), k_cache)
    v_raw, v_cache = _conv_step(layer.v_conv, layer.v_proj(x), v_cache)
    q, k, v, g, beta = _kda_inputs(layer, x, q_raw, k_raw, v_raw)
    # 逐次形式：S = Diag(exp g) S、S += beta k (v - k S)、出力は q S
    q, k, v, g, beta = q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0]
    q = q * q.shape[-1] ** -0.5
    state = cache["state"] * mx.exp(g)[..., None]
    error = v - (k[..., None] * state).sum(-2)
    state = state + (beta[..., None] * k)[..., None] * error[..., None, :]
    out = (q[..., None] * state).sum(-2)[:, None]
    return _kda_output(layer, x, out), {"state": state, "conv": (q_cache, k_cache, v_cache)}


def _attention(layer, x, cache):
    batch, length, dim = x.shape
    q, k, v = mx.split(layer.qkv(x), 3, axis=-1)
    q, k, v = (t.reshape(batch, length, layer.n_heads, layer.head_dim).transpose(0, 2, 1, 3) for t in (q, k, v))
    offset = 0 if cache is None else cache["k"].shape[2]
    if layer.use_rope:
        q, k = layer.rope(q, offset=offset), layer.rope(k, offset=offset)
    if cache is not None:
        k = mx.concatenate([cache["k"], k], axis=2)
        v = mx.concatenate([cache["v"], v], axis=2)
    # 1トークンずつのときは過去をすべて見てよいので、マスクは要らない
    mask = "causal" if length > 1 else None
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=layer.scale, mask=mask)
    return layer.out(y.transpose(0, 2, 1, 3).reshape(batch, length, dim)), {"k": k, "v": v}


def _is_attention(layer):
    return hasattr(layer, "qkv")


def forward(model, tokens, caches=None):
    """tokens は [B, L]。caches が None ならプリフィル、そうでなければ L = 1 で1歩進める。
    最後の位置の logits と新しい caches を返す。"""
    dtype = getattr(model, "compute_dtype", mx.float32)
    x = model.embed(tokens).astype(dtype)
    new_caches = []
    for i, block in enumerate(model.blocks):
        h = block.attn_norm(x)
        cache = None if caches is None else caches[i]
        if _is_attention(block.attn):
            h, cache = _attention(block.attn, h, cache)
        elif cache is None:
            h, cache = _kda_prefill(block.attn, h)
        else:
            h, cache = _kda_step(block.attn, h, cache)
        new_caches.append(cache)
        x = x + h
        x = x + block.ffn(block.ffn_norm(x))
    x = model.norm(x[:, -1:])
    logits = x @ model.embed.weight.astype(x.dtype).T
    return logits[:, -1].astype(mx.float32), new_caches


def generate(model, ids, max_tokens, temperature=0.7, eot=None):
    """プロンプトの ids に続くトークンを1つずつ返す。eot が出たら止める。eot は止める印の集まりでもよい。"""
    stops = set(eot) if isinstance(eot, (set, frozenset, list, tuple)) else {eot}
    logits, caches = forward(model, mx.array([ids]))
    for _ in range(max_tokens):
        if temperature > 0:
            token = mx.random.categorical(logits[0] / temperature).item()
        else:
            token = mx.argmax(logits[0]).item()
        if token in stops:
            return
        yield token
        logits, caches = forward(model, mx.array([[token]]), caches)
