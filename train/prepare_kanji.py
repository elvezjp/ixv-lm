"""かな漢字変換の段 1（experiments/20261124-kanji-stage1）用に、文字単位の語彙を作り、「読み → 文」の対をトークン列にする。

例: prepare_kanji.py --src ~/ixv-lm-data/sft/kanji-v2.jsonl --out ~/ixv-lm-data/processed/kanji-char-v1
入力は corpus/make_kanji_sft.py の会話（ユーザー「前：…\n読み：…」、アシスタント「…」）。
1 件を <|ctx|> 左の文脈 <|read|> 読み <|out|> 変換結果 <|endoftext|> の並びにし、train.py でそのまま学習できるよう
train.bin / val.bin（uint16）と tokenizer.json、meta.json を書く。損失は全部のトークンで取る（train.py のまま）。
語彙は 1 字 1 トークン。--min-count 回より少ない字は <|unk|> にする。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

SPECIAL = ["<|endoftext|>", "<|pad|>", "<|unk|>", "<|ctx|>", "<|read|>", "<|out|>"]


def split(row):
    user, answer = row["messages"][0]["content"], row["messages"][1]["content"]
    context, _, kana = user.partition("\n読み：")
    return context.removeprefix("前："), kana, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    triples = [split(json.loads(l)) for l in args.src.read_text(encoding="utf-8").splitlines()]
    counts = Counter(c for t in triples for part in t for c in part)
    chars = sorted(c for c, n in counts.items() if n >= args.min_count)
    vocab = {s: i for i, s in enumerate(SPECIAL + chars)}

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens(SPECIAL)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out / "tokenizer.json"))

    unk, ctx, read, out, eot = (vocab[s] for s in ["<|unk|>", "<|ctx|>", "<|read|>", "<|out|>", "<|endoftext|>"])

    def ids(text):
        return [vocab.get(c, unk) for c in text]

    order = np.random.default_rng(args.seed).permutation(len(triples))
    n_val = int(len(triples) * args.val_frac)
    stats = {}
    for name, part in [("val", order[:n_val]), ("train", order[n_val:])]:
        seq = []
        for i in part:
            context, kana, answer = triples[i]
            seq += [ctx, *ids(context), read, *ids(kana), out, *ids(answer), eot]
        arr = np.array(seq, dtype=np.uint16)
        arr.tofile(args.out / f"{name}.bin")
        stats[name] = {"pairs": len(part), "tokens": len(arr), "unk": int((arr == unk).sum())}

    text_bytes = sum(len("".join(t).encode("utf-8")) for t in triples)
    text_chars = sum(len("".join(t)) for t in triples)
    meta = {"mix": args.out.name, "format": "kanji-pairs", "src": str(args.src), "vocab_size": len(vocab),
            "eot": eot, "special_tokens": SPECIAL, "min_count": args.min_count,
            "bytes_per_token": text_bytes / (text_chars + 4 * len(triples)), **stats}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
