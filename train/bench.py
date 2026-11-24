"""学習の1ステップの速度とメモリを、モデルの大きさごとに測る。

例: bench.py --shape 0.1b --shape 0.3b
大きさごとに別のプロセスで1つずつ順に測る。無作為の入力で数ステップ回すだけで、学習はしない。
マシンが空いているときに流す（README の「学習マシンでの運用」）。
"""
import argparse
import json
import subprocess
import sys
import time
from functools import partial

SHAPES = {
    "0.1b": dict(dim=768, n_layers=12, n_heads=12, hidden_dim=2048),
    "0.3b": dict(dim=1024, n_layers=20, n_heads=16, hidden_dim=2816),
    "1b": dict(dim=2048, n_layers=18, n_heads=32, hidden_dim=5504),
}


def measure(shape, vocab, batch, seq, steps, dtype):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from model import LM, ModelConfig, loss_fn

    cfg = ModelConfig(vocab_size=vocab, attn_every=4, use_rope=False, compute_dtype=dtype, **SHAPES[shape])
    model = LM(cfg)
    mx.eval(model.parameters())
    params = sum(p.size for _, p in tree_flatten(model.parameters()))
    optimizer = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.95], weight_decay=0.1)
    optimizer.init(model.trainable_parameters())
    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = grad_fn(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss

    x = mx.random.randint(0, vocab, (batch, seq))
    y = mx.random.randint(0, vocab, (batch, seq))
    for _ in range(2):
        mx.eval(state, step(x, y))
    mx.reset_peak_memory()
    start = time.perf_counter()
    for _ in range(steps):
        mx.eval(state, step(x, y))
    seconds = (time.perf_counter() - start) / steps
    tokens = batch * seq / seconds
    return {"shape": shape, "params": params, "vocab": vocab, "batch": batch, "seq": seq, "dtype": dtype,
            "sec_per_step": round(seconds, 3), "tok_s": round(tokens), "tflops": round(6 * params * tokens / 1e12, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 1), "tokens_per_day_b": round(tokens * 86400 / 1e9, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", choices=sorted(SHAPES), required=True)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--timeout", type=int, default=900, help="1つの大きさにかける上限の秒数")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        print(json.dumps(measure(args.shape[0], args.vocab, args.batch, args.seq, args.steps, args.dtype)))
        return
    for shape in args.shape:
        command = [sys.executable, __file__, "--one", "--shape", shape, "--vocab", str(args.vocab),
                   "--batch", str(args.batch), "--seq", str(args.seq), "--steps", str(args.steps), "--dtype", args.dtype]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(json.dumps({"shape": shape, "error": f"{args.timeout}秒で終わらず止めた"}), flush=True)
            continue
        lines = done.stdout.strip().splitlines()
        if done.returncode or not lines:
            print(json.dumps({"shape": shape, "error": done.stderr.strip().splitlines()[-1:]}), flush=True)
            continue
        print(lines[-1], flush=True)


if __name__ == "__main__":
    main()
