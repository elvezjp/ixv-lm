"""自己紹介の会話（corpus/self_intro.txt）を、追加学習の会話のデータにする。

corpus/self_intro.txt は公開していない（自社で書いた会話のため）。書き方は parse() を見る。
例: make_self_intro.py --out ~/ixv-lm-data/sft/self-intro-v1.jsonl
文は自社で書いたもの（質問の一部は選んだもの）をそのまま使い、ここでは組み合わせるだけ。
話題ごとに、質問：のすべてと答え：のすべてを組み合わせる。組：は書いたとおりの1組だけ。
行の後ろの「  # …」は出どころの印なので外す。
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def strip_note(line):
    return re.sub(r"\s+#.*$", "", line).strip()


def parse(text):
    topics, topic, part = [], None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("["):
            topic = {"name": line.strip("[]"), "questions": [], "answers": [], "pairs": []}
            topics.append(topic)
        elif line in ("質問：", "答え：", "組："):
            part = line
        elif part == "組：" and line.startswith("- 質問："):
            topic["pairs"].append([strip_note(line[len("- 質問："):]), None])
        elif part == "組：" and line.strip().startswith("答え："):
            topic["pairs"][-1][1] = strip_note(line.strip()[len("答え："):])
        elif line.startswith("- ") and part in ("質問：", "答え："):
            (topic["questions"] if part == "質問：" else topic["answers"]).append(strip_note(line[2:]))
        else:
            raise SystemExit(f"読めない行: {raw}")
    return topics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=HERE / "self_intro.txt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.src.exists():
        raise SystemExit(f"{args.src} がありません。このファイルは公開していません（自社で書いた会話のため）。"
                         "書き方はこのスクリプトの parse() を見て、自分で書くか、--src で指定してください")

    rows, counts = [], {}
    for t in parse(args.src.read_text(encoding="utf-8")):
        pairs = [(q, a) for q in t["questions"] for a in t["answers"]] + [tuple(p) for p in t["pairs"]]
        if any(a is None for _, a in pairs):
            raise SystemExit(f"{t['name']}: 答えのない組があります")
        if not pairs:  # 質問か答えがまだない話題（意見を聞かれたとき など）は使わない
            print(f"使わない話題: {t['name']}")
            continue
        counts[t["name"]] = len(pairs)
        for i, (q, a) in enumerate(pairs):
            rows.append({"id": f"self-intro-{len(counts)}-{i}", "topic": t["name"], "messages": [
                {"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "count": len(rows), "topics": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
