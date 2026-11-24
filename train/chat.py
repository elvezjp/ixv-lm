"""追加学習したモデルとチャットする。

例: chat.py --run ~/ixv-lm-data/runs/kda-0.1b-sft --data ~/ixv-lm-data/processed/wiki-ja-full-16k
空行で終了、/reset で会話をやり直す。--prompt を付けると1回だけ答えて終わる。
--tools フォルダ を付けると、そのフォルダの中で read、ls、grep を使わせる（読むだけ。tokens の書式で学習した run のみ）。
書式は run の config.json の args.chat_format に合わせる（ないときは plain）。
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from chat_format import prompt_ids, stop_ids
from model import LM, ModelConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus"))

MAX_TOOL_ROUNDS = 5  # 1つの依頼でツールを呼べる回数の上限


def generate(model, tokenizer, ids, stops, max_tokens, temperature, context):
    ids = list(ids)
    start, stopped_by = len(ids), None
    for _ in range(max_tokens):
        logits = model(mx.array([ids[-context:]]))[0, -1]
        token = mx.random.categorical(logits / temperature).item() if temperature > 0 else mx.argmax(logits).item()
        if token in stops:
            stopped_by = token
            break
        ids.append(token)
    return ids[start:], stopped_by


def load_folder(root):
    """フォルダの中の文章のファイルを {相対パス: 中身} にする。"""
    files = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or any(part.startswith(".") for part in rel.split("/")) or path.stat().st_size > 1_000_000:
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def run_tool(files, call):
    from make_tool_sft import grep_text, tool_grep, tool_ls, tool_read
    name, args = call.get("name"), call.get("arguments") or {}
    try:
        if name == "ls":
            entries = tool_ls(files, (args.get("path") or ".").strip("/") or ".")
            return "\n".join(entries) if entries else "(empty directory)"
        if name == "read":
            path = args["path"].strip("/")
            if path not in files:
                return f"File not found: {path}"
            return tool_read(files, path, args.get("offset"), args.get("limit"))
        if name == "grep":
            return grep_text(tool_grep(files, args["pattern"], (args.get("path") or ".").strip("/") or "."))
    except (KeyError, TypeError, AttributeError) as e:
        return f"Error: {e}"
    return f"Unknown tool: {name}"


def reply(model, tokenizer, messages, eot, chat_format, files, max_tokens, temperature, context, show):
    """アシスタントの番を進め、足した messages を返す。ツールを呼んだときは結果を返してもう一度書かせる。"""
    stops = stop_ids(tokenizer, eot, chat_format)
    close_call = tokenizer.token_to_id("<|/tool_call|>")
    open_call = tokenizer.token_to_id("<|tool_call|>")
    added = []
    for _ in range(MAX_TOOL_ROUNDS + 1):
        out, stopped_by = generate(model, tokenizer, prompt_ids(tokenizer, messages + added, eot, chat_format),
                                   stops, max_tokens, temperature, context)
        if stopped_by == close_call and files is not None and open_call in out:
            k = out.index(open_call)
            content = tokenizer.decode(out[:k]).strip()
            raw = tokenizer.decode(out[k + 1:])
            try:
                call = json.loads(raw)
            except json.JSONDecodeError:
                show(f"（呼び出しが JSON になっていません: {raw}）")
                added.append({"role": "assistant", "content": content + raw})
                return added
            result = run_tool(files, call)
            show(f"ツール：{json.dumps(call, ensure_ascii=False)}\n結果：{result[:500]}{'…' if len(result) > 500 else ''}")
            added.append({"role": "assistant", "content": content, "tool_calls": [call]})
            added.append({"role": "tool", "name": call.get("name"), "content": result})
            continue
        added.append({"role": "assistant", "content": tokenizer.decode(out).strip()})
        return added
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prompt", action="append", help="1回だけ答える質問。複数指定できる")
    parser.add_argument("--tools", type=Path, help="このフォルダの中で read、ls、grep を使わせる")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    eot = json.loads((args.data / "meta.json").read_text())["eot"]
    config = json.loads((args.run / "config.json").read_text())
    model = LM(ModelConfig(**config["model"]))
    model.load_weights(str(args.run / (args.run / "latest").read_text() / "model.safetensors"))
    mx.random.seed(args.seed)
    context = int(config.get("args", {}).get("seq_len", 2048))
    chat_format = config.get("args", {}).get("chat_format", "plain")

    files, system = None, []
    if args.tools:
        if chat_format != "tokens":
            raise SystemExit("この run は plain の書式で学習したので、ツールは使えません")
        from make_tool_sft import SYSTEM
        files = load_folder(args.tools.expanduser())
        system = [{"role": "system", "content": SYSTEM}]
        print(f"（{args.tools} の {len(files)} 個のファイルを読めます）")

    def show(text):
        print(text)

    def answer_to(messages):
        added = reply(model, tokenizer, messages, eot, chat_format, files, args.max_tokens, args.temperature,
                      context, show)
        if added and added[-1]["role"] == "assistant" and not added[-1].get("tool_calls"):
            print(f"アシスタント：{added[-1]['content']}\n")
        return added

    if args.prompt:
        for prompt in args.prompt:
            print(f"ユーザー：{prompt}")
            answer_to(system + [{"role": "user", "content": prompt}])
        return

    messages = list(system)
    while True:
        try:
            text = input("ユーザー：").strip()
        except EOFError:
            break
        if not text:
            break
        if text == "/reset":
            messages = list(system)
            continue
        messages.append({"role": "user", "content": text})
        messages += answer_to(messages)


if __name__ == "__main__":
    main()
