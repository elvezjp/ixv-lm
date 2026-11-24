"""かな漢字変換の追加学習（段 0、experiments/20261123-kanji-stage0）の会話を、Wikipedia の文から機械で作る。

例: make_kanji_sft.py --src ~/ixv-lm-data/clean/wikipedia-ja --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json \
      --out ~/ixv-lm-data/sft/kanji-v1.jsonl --n 200000
文に Sudachi（辞書 core、Apache 2.0）で読みを振り、「読み → 文」の対にする。Claude の書いた文章は入らない。
問題の形は AJIMEE-Bench に合わせる：半分は左の文脈つき（文の途中から変換する）、変換する部分は文の途中で切ることもある。
会話の形：ユーザー「前：<左の文脈>\n読み：<カタカナ>」、アシスタント「<変換結果>」（train/eval_kanji.py の --chat と同じ）。
AJIMEE-Bench も Wikipedia から作られているので、問題の文や答えを含む文は外す。
読みが怪しい文（英字や数字を含む、読みが取れない語がある）は使わない。

話し言葉を足すとき（2026-11-25）：--quotes で会話の「」の中だけを文として拾う（青空文庫の小説の会話など）。
--modern-only で、旧仮名づかいの文書（ゐ、ゑ、でせう、だらう、やうに などを含む）を飛ばす。
--source は、出力の source に書く名前。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sudachipy import Dictionary, SplitMode

KATAKANA = re.compile(r"^[ァ-ヶー]+$")
SKIP = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９ \t]")
SYMBOL_POS = ("補助記号", "空白")
QUOTE = re.compile(r"「([^「」]{2,200})」")
OLD_KANA = re.compile(r"[ゐゑヰヱ]|でせう|だらう|やうに|さうして|かうして|ませう|いふ")


def readings(tokenizer, sentence):
    """形態素ごとの (表層, 読み)。記号は表層のまま。読みの取れない語があれば None。"""
    pairs = []
    for m in tokenizer.tokenize(sentence):
        surface = m.surface()
        if m.part_of_speech()[0] in SYMBOL_POS:
            pairs.append((surface, surface))
            continue
        reading = m.reading_form()
        if not KATAKANA.match(reading):
            return None
        pairs.append((surface, reading))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="clean 済みの parquet（text 列）のフォルダ")
    parser.add_argument("--items", type=Path, required=True, help="AJIMEE-Bench の問題（重なりを外すため）")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--take", type=float, default=0.2, help="記事を拾う割合（コーパス全体に散らすため）")
    parser.add_argument("--per-doc", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--max-target", type=int, default=40, help="変換する部分の長さの上限（字）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="wikipedia-ja", help="出力の source に書く名前")
    parser.add_argument("--quotes", action="store_true", help="会話の「」の中だけを文として拾う")
    parser.add_argument("--modern-only", action="store_true", help="旧仮名づかいの文書を飛ばす")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tokenizer = Dictionary(dict="core").tokenizer(mode=SplitMode.C)
    items = json.loads(args.items.read_text())
    originals = [i["original_text"] for i in items]
    answers = [a for i in items for a in i["expected_output"] if len(a) >= 6]

    stats = Counter()
    written = 0
    files = sorted(args.src.glob("*.parquet"))
    with args.out.open("w", encoding="utf-8") as f:
        for path in [files[i] for i in rng.permutation(len(files))]:
            for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
                if written >= args.n:
                    break
                if rng.random() >= args.take:
                    continue
                if args.modern_only and OLD_KANA.search(text):
                    stats["skip_old_kana"] += 1
                    continue
                stats["docs"] += 1
                if args.quotes:  # 会話は「？」「！」でも切り、終わりの記号はそのまま（「。」を足さない）
                    sentences = [s.strip() for q in QUOTE.findall(text) for s in re.split(r"(?<=[。？！])", q)
                                 if args.min_chars <= len(s.strip()) <= args.max_chars]
                else:
                    sentences = [s.strip() + "。" for line in text.splitlines() for s in line.split("。")
                                 if args.min_chars <= len(s.strip()) + 1 <= args.max_chars]
                for i in rng.permutation(len(sentences))[:args.per_doc]:
                    s = sentences[i]
                    if SKIP.search(s):
                        stats["skip_alnum"] += 1
                        continue
                    if any(s in o or o in s for o in originals) or any(a in s for a in answers):
                        stats["skip_bench"] += 1
                        continue
                    pairs = readings(tokenizer, s)
                    if not pairs or len(pairs) < 3:
                        stats["skip_reading"] += 1
                        continue
                    start = int(rng.integers(1, len(pairs) - 1)) if rng.random() < 0.5 else 0
                    end = len(pairs) if rng.random() < 0.5 else int(rng.integers(start + 2, len(pairs) + 1))
                    while end > start + 1 and len("".join(p[0] for p in pairs[start:end])) > args.max_target:
                        end -= 1
                    context = "".join(p[0] for p in pairs[:start])
                    target = "".join(p[0] for p in pairs[start:end])
                    kana = "".join(p[1] for p in pairs[start:end])
                    if kana == target:
                        stats["skip_no_kanji"] += 1
                        continue
                    row = {"messages": [{"role": "user", "content": f"前：{context}\n読み：{kana}"},
                                        {"role": "assistant", "content": target}],
                           "source": args.source}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    stats["with_context"] += bool(context)
            if written >= args.n:
                break
    print(json.dumps({"out": str(args.out), "written": written, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
