"""かな漢字変換の段 1（experiments/20261124-kanji-stage1）用に、文字単位の語彙を作り、「読み → 文」の対をトークン列にする。

例: prepare_kanji.py --src ~/ixv-lm-data/sft/kanji-v2.jsonl --out ~/ixv-lm-data/processed/kanji-char-v1
入力は corpus/make_kanji_sft.py の会話（ユーザー「前：…\n読み：…」、アシスタント「…」）。
1 件を <|ctx|> 左の文脈 <|read|> 読み <|out|> 変換結果 <|endoftext|> の並びにし、train.py でそのまま学習できるよう
train.bin / val.bin（uint16）と tokenizer.json、meta.json を書く。損失は全部のトークンで取る（train.py のまま）。
語彙は 1 字 1 トークン。--min-count 回より少ない字は <|unk|> にする。
--tokenizer を渡すと、その語彙をそのまま使う（学習済みの重みから続けて学習するため。語彙に無い字は <|unk|>）。
--src は何本でも渡せる。path:n と書くと、そのファイルを n 回繰り返す（n が 1 未満ならその割合だけ無作為に使う）。
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
    parser.add_argument("--src", action="append", required=True, help="path または path:n（繰り返す回数か割合）")
    parser.add_argument("--tokenizer", type=Path, help="この tokenizer.json の語彙を使う")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    triples, sources = [], []
    for spec in args.src:
        path, _, times = spec.rpartition(":") if ":" in spec and spec.rsplit(":", 1)[1].replace(".", "").isdigit() \
            else (spec, "", "1")
        rows = [split(json.loads(l)) for l in Path(path).expanduser().read_text(encoding="utf-8").splitlines()]
        times = float(times)
        if times < 1:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), int(len(rows) * times), replace=False))]
        else:
            rows = rows * int(times)
        triples += rows
        sources.append({"src": path, "times": times, "pairs": len(rows)})
    if args.tokenizer:
        vocab = json.loads(args.tokenizer.read_text())["model"]["vocab"]
        assert [vocab[s] for s in SPECIAL] == list(range(len(SPECIAL))), "特別なトークンの番号が違います"
    else:
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

    order = rng.permutation(len(triples))
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
    meta = {"mix": args.out.name, "format": "kanji-pairs", "src": sources, "tokenizer": str(args.tokenizer or ""),
            "vocab_size": len(vocab),
            "eot": eot, "special_tokens": SPECIAL, "min_count": args.min_count,
            "bytes_per_token": text_bytes / (text_chars + 4 * len(triples)), **stats}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
