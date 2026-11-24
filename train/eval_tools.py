"""ツール呼び出しを自動で採点する。

例: eval_tools.py --run ~/ixv-lm-data/runs/v1-0.3b-r2-sft --data ~/ixv-lm-data/processed/v1b \
      --set ~/ixv-lm-data/sft/tool-eval-v1.jsonl --train ~/ixv-lm-data/sft/tool-v1.jsonl
問題は make_tool_sft.py で学習とは別の seed で作る。--train と依頼・呼び出し・結果がまったく同じ問題は外し、
作業ごとに --per-task 問ずつ使う。

2段に分けて見る。
1. 呼び出し：システムと依頼だけを見せて書かせる。JSON として読めるか、ツールの名前、引数が合っているか
2. 答え：正しい呼び出しと結果を見せて書かせる。結果の事実（数、名前、行）が答えに入っているか
   （grep_none は、型のどれかと同じ答えか。空白の違いは問わない）
3. 通し：モデル自身の呼び出しを、その問題の仮のフォルダで実際に動かし、その結果から答えを書かせる（2026-11-16 から）。
   本番に近い点。問題の作成に --keep-files を付けたときだけ（ない問題では測らない）
引数は意味が同じなら正解（read の offset: 1 は書いても書かなくても同じ。1行目の limit は問わない。2026-11-14 から）。
貪欲に生成する（温度 0）。結果は run の eval_tools.json と、1問ずつの eval_tools.jsonl に書く。
"""
import argparse
import json
import re
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from chat import run_tool
from chat_format import prompt_ids, stop_ids
from model import LM, ModelConfig
from stream import forward

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "corpus"))


def greedy(model, ids, stops, max_tokens):
    logits, caches = forward(model, mx.array([ids]))
    out = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[0]).item()
        if token in stops:
            return out, token
        out.append(token)
        logits, caches = forward(model, mx.array([[token]]), caches)
    return out, None


def squash(text):
    return re.sub(r"\s+", "", text)


def answer_ok(task, slots, answer, filled_answers):
    a = squash(answer)
    if task == "ls_count":
        return slots["n"] in answer
    if task == "ls_list":
        return all(squash(e) in a for e in slots["entries"].split("、"))
    if task == "read_first_line":
        return squash(slots["first_line"]) in a
    if task == "read_lines":
        return squash(slots["content"]) in a
    if task == "grep_files":
        return all(squash(f) in a for f in slots["files"].split("、")) or \
            (slots["n"] in answer and squash(slots["word"]) in a)
    if task == "grep_none":
        return a in {squash(f) for f in filled_answers}  # 空白の違いは問わない
    return False


def same_arguments(call, gold, task):
    """意味が同じなら正解にする。read の offset: 1 は書いても書かなくても同じ。
    read_first_line の limit は、1行目が入れば何行でもよい（学習の側でも 1〜10 に散らしている）。"""
    def norm(args):
        args = dict(args or {})
        if args.get("offset") == 1:
            args.pop("offset")
        if task == "read_first_line" and isinstance(args.get("limit"), int) and args["limit"] >= 1:
            args.pop("limit")
        return args
    return isinstance(call.get("arguments"), dict) and norm(call["arguments"]) == norm(gold["arguments"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--set", type=Path, required=True, help="make_tool_sft.py で作った採点用の問題")
    parser.add_argument("--train", type=Path, action="append", default=[],
                        help="学習に使った問題。同じものは採点から外す。複数指定できる（比べる run の学習の分をすべて）")
    parser.add_argument("--per-task", type=int, default=50, help="作業ごとの問題の数")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--out", type=Path, help="結果を書く場所（ないときは --run）")
    parser.add_argument("--cpu", action="store_true", help="CPU で動かす（学習中に数問だけ確かめるとき）")
    parser.add_argument("--templates", type=Path, default=ROOT / "corpus/tool_sft_templates.toml",
                        help="問題を作ったときの文の型（答えの言い回しを見分けるのに使う）")
    args = parser.parse_args()
    if not args.templates.exists():
        raise SystemExit(f"{args.templates} がありません。このファイルは公開していません（自社で書いた文の型のため）。書き方は docs/tool-templates-guide.md を見て、自分で書いてください")

    if args.cpu:
        mx.set_default_device(mx.cpu)
    out_dir = args.out or args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    from make_tool_sft import fill, sdd_files
    templates = tomllib.loads(args.templates.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    eot = json.loads((args.data / "meta.json").read_text())["eot"]
    config = json.loads((args.run / "config.json").read_text())
    if config.get("args", {}).get("chat_format", "plain") != "tokens":
        raise SystemExit("この run は tokens の書式で学習していません")
    model = LM(ModelConfig(**config["model"]))
    model.load_weights(str(args.run / (args.run / "latest").read_text() / "model.safetensors"))
    stops = stop_ids(tokenizer, eot, "tokens")
    close_call = tokenizer.token_to_id("<|/tool_call|>")
    open_call = tokenizer.token_to_id("<|tool_call|>")

    def key(row):
        m = row["messages"]
        # 依頼・呼び出し・結果がすべて同じものを「同じ問題」とする（フォルダの中身が違えば答えも違う）
        return m[1]["content"], json.dumps(m[2]["tool_calls"][0], ensure_ascii=False, sort_keys=True), m[3]["content"]

    seen = set()
    for path in args.train:
        seen |= {key(json.loads(l)) for l in path.read_text().splitlines()}
    rows, per = [], defaultdict(int)
    for r in map(json.loads, args.set.read_text().splitlines()):
        if key(r) not in seen and per[r["task"]] < args.per_task:
            rows.append(r)
            per[r["task"]] += 1

    sdd = sdd_files() if any(r.get("files") == "sdd" for r in rows) else None
    per_task = defaultdict(lambda: defaultdict(int))
    details = []
    started = time.time()
    for row in rows:
        task, slots, m = row["task"], row["slots"], row["messages"]
        gold = m[2]["tool_calls"][0]
        # 1. 呼び出し
        out, stopped = greedy(model, prompt_ids(tokenizer, m[:2], eot, "tokens"), stops, 200)
        call, parsed = None, False
        if stopped == close_call and open_call in out:
            raw = tokenizer.decode(out[out.index(open_call) + 1:])
            try:
                call = json.loads(raw)
                parsed = isinstance(call, dict)
            except json.JSONDecodeError:
                pass
        else:
            raw = tokenizer.decode(out)
        name_ok = parsed and call.get("name") == gold["name"]
        args_ok = name_ok and same_arguments(call, gold, task)
        # 2. 答え（正しい呼び出しと結果を見せる）
        out, _ = greedy(model, prompt_ids(tokenizer, m[:4], eot, "tokens"), stops, args.max_tokens)
        answer = tokenizer.decode(out).strip()
        filled = {fill(t, slots) for t in templates[task]["answer"]}
        ok = answer_ok(task, slots, answer, filled)
        # 3. 通し（モデル自身の呼び出しを動かす）
        e2e, e2e_answer, e2e_result = None, None, None
        if "files" in row:
            e2e = False
            if name_ok and isinstance(call.get("arguments"), dict):
                if call == gold:  # 正しい呼び出しと同じなら、2 と同じ入力になる
                    e2e, e2e_answer = ok, answer
                else:
                    e2e_result = run_tool(sdd if row["files"] == "sdd" else row["files"], call)
                    own = m[:2] + [{"role": "assistant", "content": "", "tool_calls": [call]},
                                   {"role": "tool", "name": call["name"], "content": e2e_result}]
                    out, _ = greedy(model, prompt_ids(tokenizer, own, eot, "tokens"), stops, args.max_tokens)
                    e2e_answer = tokenizer.decode(out).strip()
                    e2e = answer_ok(task, slots, e2e_answer, filled)  # 2 と同じく、答えのあとの呼び出しは問わない

        t = per_task[task]
        t["n"] += 1
        t["called"] += stopped == close_call
        t["json"] += parsed
        t["name"] += name_ok
        t["arguments"] += args_ok
        t["answer"] += ok
        if e2e is not None:
            t["end_to_end"] += e2e
        details.append({"task": task, "request": m[1]["content"], "gold": gold, "call": call if parsed else raw,
                        "arguments_ok": args_ok, "answer": answer, "answer_ok": ok,
                        "own_result": e2e_result, "own_answer": e2e_answer, "end_to_end_ok": e2e})

    total = defaultdict(int)
    for t in per_task.values():
        for k, v in t.items():
            total[k] += v
    keys = ("called", "json", "name", "arguments", "answer") + (("end_to_end",) if any("files" in r for r in rows) else ())
    rate = lambda t: {k: round(t[k] / t["n"], 3) for k in keys}
    summary = {"run": str(args.run), "set": str(args.set), "n": total["n"], "seconds": round(time.time() - started),
               "all": rate(total), "tasks": {k: {"n": v["n"], **rate(v)} for k, v in sorted(per_task.items())}}
    (out_dir / "eval_tools.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (out_dir / "eval_tools.jsonl").open("w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
