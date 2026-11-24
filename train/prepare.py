"""Parquet の text 列から BPE を学習し、トークン列を uint16 の .bin に書き出す。"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

EOT = "<|endoftext|>"
SPECIAL_TOKENS = [EOT, "<|pad|>"]


def iter_texts(paths, limit_bytes):
    seen = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column("text").to_pylist():
                yield text
                seen += len(text.encode("utf-8"))
                if seen >= limit_bytes:
                    return


def train_tokenizer(paths, vocab_size, limit_bytes):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=16,
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter_texts(paths, limit_bytes), trainer=trainer)
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="学習済みの tokenizer.json を使う")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tokenizer-bytes", type=int, default=50_000_000)
    parser.add_argument("--max-tokens", type=int, default=1 << 62)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.vocab_size > 65536:
        raise SystemExit("uint16 に収まらない語彙サイズです")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.tokenizer:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    else:
        tokenizer = train_tokenizer(args.input, args.vocab_size, args.tokenizer_bytes)
    tokenizer.save(str(args.out / "tokenizer.json"))
    eot = tokenizer.token_to_id(EOT)

    total_tokens = total_bytes = total_chars = docs = 0
    tmp = args.out / "all.bin.tmp"

    def write_batch(batch, f):
        nonlocal total_tokens, total_bytes, total_chars, docs
        for enc, doc in zip(tokenizer.encode_batch(batch), batch):
            ids = np.array(enc.ids + [eot], dtype=np.uint16)
            ids.tofile(f)
            total_tokens += len(ids)
            total_bytes += len(doc.encode("utf-8"))
            total_chars += len(doc)
            docs += 1

    with tmp.open("wb") as f:
        batch = []
        for text in iter_texts(args.input, 1 << 62):
            batch.append(text)
            if len(batch) < 256:
                continue
            write_batch(batch, f)
            batch = []
            if total_tokens >= args.max_tokens:
                break
        if batch and total_tokens < args.max_tokens:
            write_batch(batch, f)

    # 末尾を検証に使う。文書の途中で切れてよい
    split = (total_tokens - args.val_tokens) * 2
    with tmp.open("rb") as f:
        f.seek(split)
        (args.out / "val.bin").write_bytes(f.read())
    os.truncate(tmp, split)
    os.replace(tmp, args.out / "train.bin")

    meta = {
        "source": [path.name for path in args.input],
        "vocab_size": tokenizer.get_vocab_size(),
        "eot": eot,
        "docs": docs,
        "tokens": total_tokens,
        "val_tokens": args.val_tokens,
        "bytes": total_bytes,
        "chars": total_chars,
        "bytes_per_token": total_bytes / total_tokens,
        "chars_per_token": total_chars / total_tokens,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
