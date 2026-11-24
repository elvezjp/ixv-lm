"""層全体の順伝播と逆伝播の時間を、部品ごとに測る。"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
from kda import KimiDeltaAttention, ShortConv, kda_chunk  # noqa: E402
from model import Attention, FeedForward, LM, ModelConfig, loss_fn  # noqa: E402

BATCH, LENGTH, DIM = 4, 512, 256


def timed(module, x, repeats=20):
    def loss(m, x_):
        return m(x_).sum()

    grad = nn.value_and_grad(module, loss)
    step = mx.compile(lambda x_: grad(module, x_))
    mx.eval(step(x))
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(step(x))
    return (time.perf_counter() - start) / repeats * 1000


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = KimiDeltaAttention(DIM, 4)

    def __call__(self, x):
        return self.layer(x)


def main():
    cfg = ModelConfig()
    x = mx.random.normal((BATCH, LENGTH, DIM))
    x3 = mx.random.normal((BATCH, LENGTH, 3 * DIM))
    mx.eval(x)
    for name, module in {
        "attention": Attention(cfg),
        "ffn": FeedForward(cfg),
        "kda_layer": KimiDeltaAttention(DIM, 4),
        "short_conv": ShortConv(3 * DIM),
    }.items():
        mx.eval(module.parameters())
        print(f"{name:11s} {timed(module, x3 if name == "short_conv" else x):6.1f} ms", flush=True)

    model = LM(ModelConfig(attn_every=1))
    mx.eval(model.parameters())
    tokens = mx.zeros((BATCH, LENGTH), dtype=mx.int32)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = model.embed
            self.norm = model.norm

        def __call__(self, h):
            logits = self.embed.as_linear(self.norm(h))
            return nn.losses.cross_entropy(logits, tokens, reduction="none")

    print(f"{'output_head':11s} {timed(Head(), x):6.1f} ms", flush=True)


if __name__ == "__main__":
    main()
