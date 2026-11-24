"""事前学習したモデルに、会話の形のデータで追加の学習（SFT）をする。

例: sft.py --base ~/ixv-lm-data/runs/kda-0.1b-wiki-ja-metal --data ~/ixv-lm-data/processed/wiki-ja-full-16k \
      --run ~/ixv-lm-data/runs/kda-0.1b-sft --sft ~/ixv-lm-data/sft/oasst2-ja.jsonl \
      --sft ~/ixv-lm-data/sft/llm-jp-instructions-train.jsonl:5
会話を --chat-format の書式（train/chat_format.py）にして詰め、seq_len ごとに切り、答えの部分だけで損失を取る。`:5` はそのファイルを5回繰り返す指定。
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from chat_format import FORMATS, encode_messages
from model import LM, ModelConfig
from train import save_checkpoint


def masked_loss(model, inputs, targets, mask):
    losses = nn.losses.cross_entropy(model(inputs), targets, reduction="none")
    return (losses * mask).sum() / mx.maximum(mask.sum(), 1)


def build(tokenizer, specs, eot, seq_len, seed, chat_format):
    conversations = []
    for spec in specs:
        path, _, repeat = spec.partition(":")
        rows = [json.loads(line) for line in Path(path).expanduser().read_text().splitlines()]
        conversations += rows * int(repeat or 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(conversations))
    ids, mask = [eot], [0]
    for i in order:
        piece_ids, piece_mask = encode_messages(tokenizer, conversations[i]["messages"], eot, chat_format)
        ids += piece_ids
        mask += piece_mask
    rows = (len(ids) - 1) // seq_len
    ids = np.array(ids[: rows * seq_len + 1], dtype=np.int32)
    mask = np.array(mask[: rows * seq_len + 1], dtype=np.float32)
    # 入力は i、正解は i+1。損失の印は正解の側に合わせる
    starts = np.arange(rows) * seq_len
    inputs = np.stack([ids[s : s + seq_len] for s in starts])
    targets = np.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    weights = np.stack([mask[s + 1 : s + seq_len + 1] for s in starts])
    return inputs, targets, weights, len(conversations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="事前学習の run。latest の重みから始める")
    parser.add_argument("--data", type=Path, required=True, help="トークナイザーのある前処理の場所")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sft", action="append", required=True, help="JSONL の会話。path:回数 で繰り返せる")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--compute-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--chat-format", choices=FORMATS, default="tokens",
                        help="tokens は特別なトークンの書式（ツール呼び出しも書ける）。plain は r1 までの書式")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    meta = json.loads((args.data / "meta.json").read_text())
    base_config = json.loads((args.base / "config.json").read_text())
    cfg = ModelConfig(**{**base_config["model"], "compute_dtype": args.compute_dtype})
    model = LM(cfg)
    base_ckpt = (args.base / "latest").read_text()
    model.load_weights(str(args.base / base_ckpt / "model.safetensors"))

    inputs, targets, weights, conversations = build(tokenizer, args.sft, meta["eot"], args.seq_len, args.seed, args.chat_format)
    steps_per_epoch = len(inputs) // args.batch_size
    total = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr * max(0.0, (total - step) / max(1, total - args.warmup))

    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "config.json").write_text(json.dumps(
        {"model": asdict(cfg), "base": f"{args.base}/{base_ckpt}",
         "args": {k: str(v) for k, v in vars(args).items()},
         "conversations": conversations, "rows": int(len(inputs)),
         "answer_tokens": int(weights.sum())}, ensure_ascii=False, indent=2))
    log = (args.run / "log.jsonl").open("a", buffering=1)

    def record(**fields):
        line = json.dumps({"time": time.time(), **fields}, ensure_ascii=False)
        log.write(line + "\n")
        print(line, flush=True)

    grad_fn = nn.value_and_grad(model, masked_loss)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y, w):
        loss, grads = grad_fn(model, x, y, w)
        grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss, grad_norm

    record(event="start", conversations=conversations, rows=int(len(inputs)), steps=total,
           answer_tokens=int(weights.sum()), base=f"{args.base}/{base_ckpt}")
    rng = np.random.default_rng(args.seed + 1)
    step, window, window_start = 0, [], time.perf_counter()
    for epoch in range(args.epochs):
        order = rng.permutation(len(inputs))
        for b in range(steps_per_epoch):
            rows = order[b * args.batch_size : (b + 1) * args.batch_size]
            optimizer.learning_rate = mx.array(lr_at(step), dtype=mx.float32)
            loss, grad_norm = train_step(mx.array(inputs[rows]), mx.array(targets[rows]), mx.array(weights[rows]))
            mx.eval(state, loss, grad_norm)
            if not math.isfinite(loss.item()):
                record(event="abort", step=step, reason="loss is not finite")
                raise SystemExit(1)
            window.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == total:
                elapsed = time.perf_counter() - window_start
                record(event="train", step=step, epoch=epoch, loss=sum(window) / len(window), lr=lr_at(step - 1),
                       grad_norm=grad_norm.item(),
                       tok_s=len(window) * args.batch_size * args.seq_len / elapsed)
                window, window_start = [], time.perf_counter()
        save_checkpoint(args.run, step, model, optimizer)
    record(event="end", step=step)


if __name__ == "__main__":
    main()
