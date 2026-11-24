"""重みを float32 のまま持ち、入力の型で計算する層。"""
import mlx.core as mx
import mlx.nn as nn


class Linear(nn.Linear):
    def __call__(self, x):
        y = x @ self.weight.astype(x.dtype).T
        return y + self.bias.astype(x.dtype) if "bias" in self else y


class RMSNorm(nn.RMSNorm):
    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)
