"""小さな言語モデル。attn_every=1 は全層が注意の Transformer、4 は KDA 3層に注意1層のハイブリッド。"""
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from kda import KimiDeltaAttention
from precision import Linear, RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 4
    hidden_dim: int = 704
    rope_base: float = 10000.0
    attn_every: int = 1
    use_rope: bool = True
    compute_dtype: str = "float32"  # 重みは float32 のまま、行列積だけこの型で計算する


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = Linear(cfg.dim, cfg.dim, bias=False)
        self.use_rope = cfg.use_rope
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_base)

    def __call__(self, x):
        batch, length, dim = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (
            t.reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            for t in (q, k, v)
        )
        if self.use_rope:
            q, k = self.rope(q), self.rope(k)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.out(y.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, use_attention):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg) if use_attention else KimiDeltaAttention(cfg.dim, cfg.n_heads)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = FeedForward(cfg)

    def __call__(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [Block(cfg, (i + 1) % cfg.attn_every == 0) for i in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.dim)
        self.compute_dtype = getattr(mx, cfg.compute_dtype)

    def __call__(self, tokens):
        x = self.embed(tokens).astype(self.compute_dtype)
        for block in self.blocks:
            x = block(x)
        logits = self.norm(x) @ self.embed.weight.astype(x.dtype).T
        return logits.astype(mx.float32)


def loss_fn(model, inputs, targets):
    return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")
