"""会話の追加学習（SFT）用のデータを、共通の形式（JSONL の messages）に加工する。

例: clean_sft.py
出力は ~/ixv-lm-data/sft/<台帳の id>-<split>.jsonl。1行が1会話で、role は user と assistant。
"""
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path.home() / "ixv-lm-data/raw"
OUT = Path.home() / "ixv-lm-data/sft"


def llm_jp(split):
    path = RAW / f"llm-jp-instructions/v1.0/{split}-00000-of-00001.parquet"
    for row in pq.read_table(path, columns=["ID", "text", "output"]).to_pylist():
        yield {"id": row["ID"], "messages": [
            {"role": "user", "content": row["text"].strip()},
            {"role": "assistant", "content": row["output"].strip()}]}


def oasst2():
    """会話の木を、根から葉までの道ごとの会話にする。最後が assistant で終わるように切る。"""
    messages = json.loads((RAW / "oasst2-ja/oasst2_135k_ja.json").read_text())
    by_id = {m["message_id"]: m for m in messages}
    children = defaultdict(list)
    for m in messages:
        if m["parent_id"] != "nan":
            children[m["parent_id"]].append(m["message_id"])
    roles = {"prompter": "user", "assistant": "assistant"}

    def walk(message_id, path):
        message = by_id[message_id]
        text = (message.get("text_ja") or "").strip()
        if not text:
            return
        path = path + [{"role": roles[message["role"]], "content": text}]
        kids = [k for k in children[message_id] if k in by_id]
        if not kids:
            while path and path[-1]["role"] != "assistant":
                path = path[:-1]
            if path:
                yield message_id, path
            return
        for kid in kids:
            yield from walk(kid, path)

    for root in (m["message_id"] for m in messages if m["parent_id"] == "nan"):
        for leaf, path in walk(root, []):
            # user と assistant が交互でない会話は使わない
            if all(turn["role"] == ("user" if i % 2 == 0 else "assistant") for i, turn in enumerate(path)):
                yield {"id": leaf, "messages": path}


def write(name, conversations):
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    with (OUT / f"{name}.jsonl").open("w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            count += 1
    print(name, count)


def main():
    write("llm-jp-instructions-train", llm_jp("train"))
    write("llm-jp-instructions-dev", llm_jp("dev"))
    write("llm-jp-instructions-test", llm_jp("test"))
    write("oasst2-ja", oasst2())


if __name__ == "__main__":
    main()
