"""会話のデータから、アシスタントがほかの名前で名乗る会話を外す。

例: filter_sft_identity.py --src ~/ixv-lm-data/sft/oasst2-ja.jsonl --out ~/ixv-lm-data/sft/oasst2-ja-noself.jsonl
oasst2-ja のアシスタントは「私はオープンアシスタントです」「AI言語モデルとして」などと名乗る。
自己紹介（self-intro）を覚えるじゃまになるので、会話ごと外す（2026-11-21 決定）。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

# 「ライオン」は動物の話にも出るので入れない（LAION は英字だけ）
IDENTITY = re.compile(r"(?i)open\s*assistant|オープン[・ ]?アシスタント|LAION|"
                      r"私は(AI|Ai|人工知能)?言語モデル|AI言語モデル|言語モデルとして")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kept, dropped, hits = 0, 0, Counter()
    with args.out.open("w", encoding="utf-8") as f:
        for line in args.src.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            found = {m.group(0) for msg in row["messages"] if msg["role"] == "assistant"
                     for m in IDENTITY.finditer(msg["content"])}
            if found:
                dropped += 1
                hits.update(found)
                continue
            f.write(line + "\n")
            kept += 1
    print(json.dumps({"out": str(args.out), "kept": kept, "dropped": dropped,
                      "hits": dict(hits.most_common())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
