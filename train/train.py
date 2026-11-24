"""最小の学習ループ。JSONL のログと、途中再開できるチェックポイントを残す。"""
import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from model import LM, ModelConfig, loss_fn


def batch_at(tokens, step, batch_size, seq_len, seed):
    # 再開しても同じ列になるよう、位置は step だけから決める
    rng = np.random.default_rng([seed, step])
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    rows = np.stack([tokens[s : s + seq_len + 1] for s in starts]).astype(np.int32)
    return mx.array(rows[:, :-1]), mx.array(rows[:, 1:])


def lr_at(step, args):
    # 定数のあと末尾で減衰する WSD
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    decay_start = int(args.steps * (1 - args.decay_frac))
    if step < decay_start:
        return args.lr
    return args.lr * max(0.0, (args.steps - step) / max(1, args.steps - decay_start))


def evaluate(model, tokens, batch_size, seq_len, batches):
    total = 0.0
    for i in range(batches):
        x, y = batch_at(tokens, i, batch_size, seq_len, seed=1)
        total += loss_fn(model, x, y).item()
    return total / batches


def save_checkpoint(run_dir, step, model, optimizer):
    tmp = run_dir / f".ckpt-{step}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(model.parameters())))
    state = {k: v for k, v in tree_flatten(optimizer.state) if isinstance(v, mx.array)}
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), state)
    (tmp / "state.json").write_text(json.dumps({"step": step}))
    final = run_dir / f"ckpt-{step}"
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    latest_tmp = run_dir / ".latest.tmp"
    latest_tmp.write_text(final.name)
    os.replace(latest_tmp, run_dir / "latest")


def prune_checkpoints(run_dir, keep_last, keep_every):
    """直近の keep_last 個と、keep_every の倍数のステップだけを残す。0.3B では1回の保存が約4GBある。"""
    if not keep_last:
        return
    steps = sorted(int(p.name.split("-")[1]) for p in run_dir.glob("ckpt-*") if p.name.split("-")[1].isdigit())
    latest = (run_dir / "latest").read_text()
    for step in steps[:-keep_last]:
        if keep_every and step % keep_every == 0:
            continue
        if f"ckpt-{step}" != latest:
            shutil.rmtree(run_dir / f"ckpt-{step}")


def load_checkpoint(run_dir, model, optimizer):
    path = run_dir / (run_dir / "latest").read_text()
    model.load_weights(str(path / "model.safetensors"))
    optimizer.state = tree_unflatten(list(mx.load(str(path / "optimizer.safetensors")).items()))
    return json.loads((path / "state.json").read_text())["step"]


def resolve_config(args, vocab_size):
    """モデルの設定を決める。再開では保存した設定を正とし、違う指定はエラーで止める。"""
    names = ("dim", "n_layers", "n_heads", "hidden_dim", "attn_every", "compute_dtype")
    requested = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    if args.no_rope:
        requested["use_rope"] = False
    if not args.resume:
        cfg = ModelConfig(vocab_size=vocab_size, **requested)
        if cfg.dim % cfg.n_heads:
            raise SystemExit(f"dim {cfg.dim} は n_heads {cfg.n_heads} で割り切れません")
        return cfg, None
    saved = json.loads((args.run / "config.json").read_text())
    requested["vocab_size"] = vocab_size
    defaults = asdict(ModelConfig())
    changed = {k: (saved["model"].get(k, defaults[k]), v) for k, v in requested.items()
               if saved["model"].get(k, defaults[k]) != v}
    if changed:
        raise SystemExit(f"保存した設定と違うので再開しません（保存, 指定）: {changed}")
    return ModelConfig(**saved["model"]), saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--stop-after", type=int, help="この step で止める。学習率の計画は --steps のまま")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--decay-frac", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-last", type=int, default=0, help="直近この数の保存だけ残す。0 は全部残す")
    parser.add_argument("--keep-every", type=int, default=0, help="このステップの倍数の保存は消さない（リリースの分岐用）")
    parser.add_argument("--attn-every", type=int, help="4 なら KDA 3層に注意1層。既定は ModelConfig の値")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], help="行列積の型。重みは float32 のまま")
    args = parser.parse_args()

    meta = json.loads((args.data / "meta.json").read_text())
    train_tokens = np.memmap(args.data / "train.bin", dtype=np.uint16, mode="r")
    val_tokens = np.memmap(args.data / "val.bin", dtype=np.uint16, mode="r")

    mx.random.seed(args.seed)
    cfg, saved = resolve_config(args, meta["vocab_size"])
    model = LM(cfg)
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))

    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    optimizer.init(model.trainable_parameters())

    args.run.mkdir(parents=True, exist_ok=True)
    start = load_checkpoint(args.run, model, optimizer) if args.resume else 0
    args_changed = []
    if saved:
        args_changed = [k for k in ("batch_size", "seq_len", "seed") if saved["args"][k] != str(getattr(args, k))]
    else:
        (args.run / "config.json").write_text(json.dumps(
            {"model": asdict(cfg), "args": {k: str(v) for k, v in vars(args).items()}, "params": n_params},
            ensure_ascii=False, indent=2))
    log = (args.run / "log.jsonl").open("a", buffering=1)

    def record(**fields):
        line = json.dumps({"time": time.time(), **fields}, ensure_ascii=False)
        log.write(line + "\n")
        print(line, flush=True)

    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = grad_fn(model, x, y)
        grads, grad_norm = optim.clip_grad_norm(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss, grad_norm

    end = args.stop_after or args.steps
    record(event="start", step=start, params=n_params, resumed=args.resume, args_changed=args_changed)
    tokens_per_step = args.batch_size * args.seq_len
    window_start, window_steps, window_loss = time.perf_counter(), 0, 0.0

    for step in range(start, end):
        optimizer.learning_rate = mx.array(lr_at(step, args), dtype=mx.float32)
        x, y = batch_at(train_tokens, step, args.batch_size, args.seq_len, args.seed)
        loss, grad_norm = train_step(x, y)
        mx.eval(state, loss, grad_norm)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            record(event="abort", step=step, reason="loss is not finite")
            raise SystemExit(1)
        window_steps += 1
        window_loss += loss_value
        done = step + 1

        if done % args.log_every == 0:
            elapsed = time.perf_counter() - window_start
            tok_s = window_steps * tokens_per_step / elapsed
            record(event="train", step=done, loss=window_loss / window_steps,
                   lr=lr_at(step, args), grad_norm=grad_norm.item(), tok_s=tok_s,
                   tflops=6 * n_params * tok_s / 1e12,
                   mem_gb=mx.get_peak_memory() / 1e9)
            window_start, window_steps, window_loss = time.perf_counter(), 0, 0.0
        if done % args.eval_every == 0 or done == end:
            val = evaluate(model, val_tokens, args.batch_size, args.seq_len, args.eval_batches)
            record(event="eval", step=done, val_loss=val,
                   val_bpb=val / math.log(2) / meta["bytes_per_token"])
            window_start = time.perf_counter()
        if done % args.save_every == 0 or done == end:
            save_checkpoint(args.run, done, model, optimizer)
            prune_checkpoints(args.run, args.keep_last, args.keep_every)
            window_start = time.perf_counter()

    record(event="end", step=end)


if __name__ == "__main__":
    main()
