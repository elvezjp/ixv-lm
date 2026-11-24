"""加工済みのデータ（~/ixv-lm-data/clean）から、混合の指定どおりにトークナイザーを作り、トークン列を書き出す。

例: prepare_mix.py ../corpus/mix-v1.json --out ~/ixv-lm-data/processed/v1 --step tokenizer
    prepare_mix.py ../corpus/mix-v1.json --out ~/ixv-lm-data/processed/v1 --step tokens

tokenizer と tokens は別々に流す。どちらも Python の中で順に処理し、並列の数は --threads で抑える。
train.bin と val.bin は train.py がそのまま読める。データごとの量は meta.json に残す。
"""
import argparse
import gzip
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CLEAN = Path.home() / "ixv-lm-data/clean"
RAW = Path.home() / "ixv-lm-data/raw"

# 語彙は後から変えられないので、会話とツール呼び出しの印を先に確保しておく
SPECIAL_TOKENS = [
    "<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>",
    "<|think|>", "<|/think|>",
] + [f"<|reserved_{i}|>" for i in range(21)]


def clean_parts(source):
    parts = sorted((CLEAN / source).glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"{source}: 加工済みのデータがありません")
    return parts


def iter_clean(parts):
    for path in parts:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000, columns=["text"]):
            yield from batch.column("text").to_pylist()


def iter_raw(name):
    """トークナイザーの学習だけに使う、未加工の英語とコード。"""
    if name == "finemath":
        for path in sorted((RAW / "finemath-4plus/finemath-4plus").glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000, columns=["text"]):
                yield from batch.column("text").to_pylist()
    elif name == "github-archive":
        for path in sorted((RAW / "github-archive").glob("*.json.gz")):
            with gzip.open(path, "rt") as f:
                for line in f:
                    yield json.loads(line)["text"]
    else:
        raise SystemExit(f"未知の raw: {name}")


def sample_texts(mix, seed):
    """データごとに決めた量（MB）を、ファイルを散らして取り、混ぜて返す。"""
    rng = random.Random(seed)
    texts = []
    for name, megabytes in mix["tokenizer_sample_mb"].items():
        limit, seen, count = megabytes * 1_000_000, 0, 0
        if name.startswith("raw:"):
            stream = iter_raw(name[4:])
        else:
            parts = clean_parts(name)
            rng.shuffle(parts)
            stream = iter_clean(parts)
        for text in stream:
            texts.append(text)
            seen += len(text.encode())
            count += 1
            if seen >= limit:
                break
        print(f"  {name}: {count} 文書、{seen / 1e6:.0f}MB", flush=True)
    rng.shuffle(texts)
    return texts


def build_tokenizer(mix, out, seed):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    texts = sample_texts(mix, seed)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    # 数字は1桁ずつに分ける。英語とコードは単語の区切りで分ける。日本語は句読点などで区切られる
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=mix["vocab_size"],
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=16,
        show_progress=False,
    )
    start = time.time()
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    tokenizer.save(str(out / "tokenizer.json"))
    print(f"語彙 {tokenizer.get_vocab_size()}、{time.time() - start:.0f}秒", flush=True)


def spread(parts, tokens_needed, chars_per_token_guess=2.0):
    """必要な量に足りるだけのファイルを、全体から等間隔に選ぶ。偏ったダンプだけにならないようにする。"""
    if tokens_needed is None:
        return parts
    # ファイルごとの文字数が分からないときは、全体の文字数を均等に割って見積もる
    total_chars = json.loads((parts[0].parent / "source.json").read_text())["chars"]
    per_part = total_chars / len(parts) / chars_per_token_guess
    count = min(len(parts), max(1, int(np.ceil(tokens_needed / per_part * 1.1))))
    picks = np.linspace(0, len(parts) - 1, count).round().astype(int)
    return [parts[i] for i in sorted(set(picks))]


def build_tokens(mix, out, val_tokens):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    eot = tokenizer.token_to_id("<|endoftext|>")
    if tokenizer.get_vocab_size() > 65536:
        raise SystemExit("uint16 に収まらない語彙サイズです")
    parts_dir = out / "sources"
    parts_dir.mkdir(exist_ok=True)
    summary = []
    for spec in mix["sources"]:
        name, target, repeat = spec["id"], spec.get("tokens"), spec.get("repeat", 1)
        # 量の少ないデータ（自社の文章など）は、検証に回すと学習から消えるので回さない
        source_val = val_tokens if spec.get("val", True) else 0
        done = parts_dir / f"{name}.json"
        if done.exists():
            summary.append(json.loads(done.read_text()))
            print(f"{name}: 前回の結果を使う", flush=True)
            continue
        start = time.time()
        train_path, val_path = parts_dir / f"{name}.train.bin", parts_dir / f"{name}.val.bin"
        stats = {"id": name, "repeat": repeat, "docs": 0, "tokens": 0, "val_tokens": 0, "bytes": 0, "chars": 0}
        with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
            batch = []

            def flush():
                for enc, text in zip(tokenizer.encode_batch(batch), batch):
                    ids = np.array(enc.ids + [eot], dtype=np.uint16)
                    # 各データの先頭の文書を検証に回す。学習には入れない
                    if stats["val_tokens"] < source_val:
                        ids.tofile(val_f)
                        stats["val_tokens"] += len(ids)
                        continue
                    ids.tofile(train_f)
                    stats["tokens"] += len(ids)
                    stats["docs"] += 1
                    stats["bytes"] += len(text.encode())
                    stats["chars"] += len(text)
                batch.clear()

            for text in iter_clean(spread(clean_parts(name), target)):
                batch.append(text)
                if len(batch) >= 2_000:
                    flush()
                    if target and stats["tokens"] >= target:
                        break
            if batch and not (target and stats["tokens"] >= target):
                flush()
        stats["seconds"] = round(time.time() - start)
        done.write_text(json.dumps(stats, ensure_ascii=False))
        summary.append(stats)
        print(f"{name}: {stats['tokens'] / 1e9:.3f}B トークン、{stats['docs']} 文書、{stats['seconds']}秒", flush=True)

    # 学習は位置を無作為に選ぶので、ファイルを並べるだけで割合どおりに混ざる
    with (out / "train.bin.tmp").open("wb") as f:
        for stats in summary:
            for _ in range(stats["repeat"]):
                with (parts_dir / f"{stats['id']}.train.bin").open("rb") as src:
                    shutil.copyfileobj(src, f, 1 << 24)
    with (out / "val.bin.tmp").open("wb") as f:
        for stats in summary:
            with (parts_dir / f"{stats['id']}.val.bin").open("rb") as src:
                shutil.copyfileobj(src, f, 1 << 24)
    os.replace(out / "train.bin.tmp", out / "train.bin")
    os.replace(out / "val.bin.tmp", out / "val.bin")

    empty = [s["id"] for s in summary if s["tokens"] == 0]
    if empty:
        raise SystemExit(f"学習に入るトークンがないデータがあります: {empty}")
    total = sum(s["tokens"] * s["repeat"] for s in summary)
    total_bytes = sum(s["bytes"] * s["repeat"] for s in summary)
    total_chars = sum(s["chars"] * s["repeat"] for s in summary)
    meta = {
        "mix": mix["name"], "vocab_size": tokenizer.get_vocab_size(), "eot": eot,
        "special_tokens": SPECIAL_TOKENS, "tokens": total, "val_tokens": sum(s["val_tokens"] for s in summary),
        "bytes_per_token": total_bytes / total, "chars_per_token": total_chars / total,
        "sources": [{**s, "share": s["tokens"] * s["repeat"] / total} for s in summary],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("sources", "special_tokens")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mix", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", choices=["tokenizer", "tokens"], required=True)
    parser.add_argument("--threads", type=int, default=12, help="トークナイザーが使う CPU の数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)  # tokenizers を読み込む前に決める

    mix = json.loads(args.mix.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mix.json").write_text(json.dumps(mix, ensure_ascii=False, indent=2))
    if args.step == "tokenizer":
        build_tokenizer(mix, args.out, args.seed)
    else:
        build_tokens(mix, args.out, mix["val_tokens_per_source"])


if __name__ == "__main__":
    main()
