"""かな漢字変換の辞書（語と、語のつながりの回数）を、Wikipedia の文から機械で作る。

例: make_kanji_dict.py --src ~/ixv-lm-data/clean/wikipedia-ja --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json \
      --out ~/ixv-lm-data/ime/dict-v1
文を Sudachi（辞書 core、分割 B）で区切り、(表層, 読み, 品詞) を 1 語として数える。隣り合う 2 語の回数も数える。
変換の下書き（ime/engine.py）が、この回数から 2-gram の確からしさを出す。Claude の書いた文章は入らない。
読みが怪しい文（英字や数字を含む、読みの取れない語がある）は使わない。AJIMEE-Bench の問題の文は外す。
文の中では Sudachi が読みを 1 つしか選ばない（「明日」はアス）。そこで、数えた語の表層を Sudachi の辞書で引き直し、
ほかの読み（アシタ、ミョウニチ）も、同じ品詞で、元の回数の --alt-share 倍の回数で足す（つながりの回数は足さない）。
書くもの：words.tsv（id、表層、読み、品詞、回数）、bigrams.npy（[前の id, 後の id, 回数]）、meta.json。
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sudachipy import Dictionary, SplitMode

KATAKANA = re.compile(r"^[ァ-ヶー]+$")
SKIP = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９ \t]")
SYMBOL_POS = ("補助記号", "空白")
BOS = ("<s>", "", "BOS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-doc", type=int, default=10, help="1 記事から使う文の数の上限")
    parser.add_argument("--min-word", type=int, default=2, help="これより少ない語は捨てる")
    parser.add_argument("--min-bigram", type=int, default=2)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--alt-share", type=float, default=0.5, help="ほかの読みに付ける回数（元の回数に掛ける）")
    args = parser.parse_args()

    tokenizer = Dictionary(dict="core").tokenizer(mode=SplitMode.B)
    originals = [i["original_text"] for i in json.loads(args.items.read_text())]
    ids, words, bigrams = {BOS: 0}, Counter(), Counter()
    docs = sentences_used = 0
    start = time.time()
    for path in sorted(args.src.glob("*.parquet")):
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            docs += 1
            if args.limit_docs and docs > args.limit_docs:
                break
            used = 0
            for line in text.splitlines():
                for s in line.split("。"):
                    if used >= args.per_doc:
                        break
                    s = s.strip()
                    if len(s) < 6 or len(s) > 120 or SKIP.search(s) or any(s in o for o in originals):
                        continue
                    seq = []
                    for m in tokenizer.tokenize(s + "。"):
                        pos = m.part_of_speech()[0]
                        surface = m.surface()
                        reading = surface if pos in SYMBOL_POS else m.reading_form()
                        if pos not in SYMBOL_POS and not KATAKANA.match(reading):
                            seq = None
                            break
                        seq.append((surface, reading, pos))
                    if not seq:
                        continue
                    used += 1
                    prev = 0
                    for key in seq:
                        i = ids.setdefault(key, len(ids))
                        words[i] += 1
                        bigrams[prev, i] += 1
                        prev = i
            sentences_used += used
            if docs % 100000 == 0:
                print(json.dumps({"docs": docs, "sentences": sentences_used, "words": len(ids),
                                  "bigrams": len(bigrams), "sec": round(time.time() - start)}), flush=True)
        if args.limit_docs and docs > args.limit_docs:
            break

    keys = {i: k for k, i in ids.items()}
    lexicon = Dictionary(dict="core")
    alt = 0
    for old in sorted(i for i, n in words.items() if n >= args.min_word):
        surface, reading, pos = keys[old]
        if pos in SYMBOL_POS or len(surface) < 2 and not re.search(r"[一-龥]", surface):
            continue
        for m in lexicon.lookup(surface):
            r = m.reading_form()
            key = (surface, r, pos)
            if r != reading and KATAKANA.match(r) and m.part_of_speech()[0] == pos and key not in ids:
                ids[key] = len(ids)
                keys[ids[key]] = key
                words[ids[key]] = max(args.min_word, int(words[old] * args.alt_share))
                alt += 1
    keep = sorted(i for i, n in words.items() if n >= args.min_word)
    new = {old: new for new, old in enumerate([0] + keep)}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "words.tsv").open("w", encoding="utf-8") as f:
        f.write(f"0\t<s>\t\tBOS\t{sentences_used}\n")
        for old in keep:
            surface, reading, pos = keys[old]
            f.write(f"{new[old]}\t{surface}\t{reading}\t{pos}\t{words[old]}\n")
    rows = [(new[a], new[b], n) for (a, b), n in bigrams.items() if n >= args.min_bigram and a in new and b in new]
    np.save(args.out / "bigrams.npy", np.array(rows, dtype=np.int64))
    meta = {"src": str(args.src), "split": "B", "alt_readings": alt, "alt_share": args.alt_share, "docs": docs, "sentences": sentences_used, "words": len(keep),
            "bigrams": len(rows), "tokens": sum(words.values()), "min_word": args.min_word,
            "min_bigram": args.min_bigram, "per_doc": args.per_doc, "sec": round(time.time() - start)}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
