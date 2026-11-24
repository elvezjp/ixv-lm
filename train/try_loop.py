"""学習済みの重みのまま、真ん中の層を何回も通したらどうなるかを試す（学習はしない）。

例: try_loop.py --run ~/ixv-lm-data/runs/v1-0.3b-r4-sft --data ~/ixv-lm-data/processed/v1b \
      --set ~/ixv-lm-data/sft/tool-eval-v2.jsonl --train ~/ixv-lm-data/sft/tool-v1.jsonl --train ~/ixv-lm-data/sft/tool-v2.jsonl
真ん中の層を何回も通す案の V0（真ん中を続けて通すだけ）を、答えを出すときだけ当てはめる。
r4 は1回ずつ通す前提で学習しているので、崩れることが多いはず。CPU で動かす（本線の学習と重ねない）。
見るもの：
1. 検証のデータの損失（val.bin の先頭から数本）
2. read_lines の呼び出しの limit（引き算）と、ls_count の答えの数（数える）
3. 自己紹介の質問への答え
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tokenizers import Tokenizer

from chat_format import prompt_ids, stop_ids
from eval_tools import same_arguments
from model import LM, ModelConfig


def forward(model, tokens, start, end, loops):
    """層 start〜end-1 を loops 回、続けて通す。"""
    x = model.embed(tokens).astype(model.compute_dtype)
    for block in model.blocks[:start]:
        x = block(x)
    for _ in range(loops):
        for block in model.blocks[start:end]:
            x = block(x)
    for block in model.blocks[end:]:
        x = block(x)
    return (model.norm(x) @ model.embed.weight.astype(x.dtype).T).astype(mx.float32)


def greedy(model, ids, stops, max_tokens, cfg):
    ids, out = list(ids), []
    for _ in range(max_tokens):
        token = mx.argmax(forward(model, mx.array([ids]), *cfg)[0, -1]).item()
        if token in stops:
            break
        ids.append(token)
        out.append(token)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", default=[])
    parser.add_argument("--start", type=int, default=4, help="まわす区間の最初の層")
    parser.add_argument("--end", type=int, default=16, help="まわす区間の次の層（入口 4、真ん中 12、出口 4）")
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--per-task", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mx.set_default_device(mx.cpu)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    eot = json.loads((args.data / "meta.json").read_text())["eot"]
    config = json.loads((args.run / "config.json").read_text())
    model = LM(ModelConfig(**config["model"]))
    model.load_weights(str(args.run / (args.run / "latest").read_text() / "model.safetensors"))
    stops = stop_ids(tokenizer, eot, "tokens")
    open_call = tokenizer.token_to_id("<|tool_call|>")

    def key(row):
        m = row["messages"]
        return m[1]["content"], json.dumps(m[2]["tool_calls"][0], ensure_ascii=False, sort_keys=True), m[3]["content"]

    seen = set()
    for path in args.train:
        seen |= {key(json.loads(l)) for l in path.read_text().splitlines()}
    rows, per = [], defaultdict(int)
    for r in map(json.loads, args.set.read_text().splitlines()):
        if r["task"] in ("read_lines", "ls_count") and key(r) not in seen and per[r["task"]] < args.per_task:
            rows.append(r)
            per[r["task"]] += 1

    val = np.memmap(args.data / "val.bin", dtype=np.uint32 if json.loads(
        (args.data / "meta.json").read_text()).get("dtype") == "uint32" else np.uint16, mode="r")
    seqs = mx.array(np.array([val[i * 513:(i + 1) * 513] for i in range(8)], dtype=np.int32))
    questions = ["君はだれ？", "何ができる？", "日本の首都ってなに？"]

    summary, details = {}, []
    for loops in args.loops:
        cfg = (args.start, args.end, loops)
        logits = forward(model, seqs[:, :-1], *cfg)
        loss = nn.losses.cross_entropy(logits, seqs[:, 1:], reduction="mean").item()
        ok = defaultdict(int)
        for r in rows:
            m, slots = r["messages"], r["slots"]
            if r["task"] == "read_lines":
                out = greedy(model, prompt_ids(tokenizer, m[:2], eot, "tokens"), stops, 120, cfg)
                raw = tokenizer.decode(out[out.index(open_call) + 1:]) if open_call in out else tokenizer.decode(out)
                try:
                    call = json.loads(raw)
                    good = isinstance(call, dict) and same_arguments(call, m[2]["tool_calls"][0], "read_lines")
                except json.JSONDecodeError:
                    good = False
                ok["read_lines"] += good
                details.append({"loops": loops, "task": "read_lines", "request": m[1]["content"],
                                "gold": m[2]["tool_calls"][0]["arguments"], "got": raw, "ok": good})
            else:
                answer = tokenizer.decode(greedy(model, prompt_ids(tokenizer, m[:4], eot, "tokens"), stops, 60, cfg))
                good = slots["n"] in answer
                ok["ls_count"] += good
                details.append({"loops": loops, "task": "ls_count", "n": slots["n"], "got": answer, "ok": good})
        chats = {q: tokenizer.decode(greedy(model, prompt_ids(tokenizer, [{"role": "user", "content": q}], eot, "tokens"),
                                            stops, 60, cfg)).strip() for q in questions}
        summary[loops] = {"val_loss": round(loss, 3), "passes": args.start + loops * (args.end - args.start)
                          + len(model.blocks) - args.end, **{k: f"{ok[k]}/{per[k]}" for k in per}, "chat": chats}
        print(json.dumps({loops: summary[loops]}, ensure_ascii=False), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "try_loop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (args.out / "try_loop.jsonl").open("w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
