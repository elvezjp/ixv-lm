"""公開したチャットモデルと、状態を引き継ぐ生成で会話する。

例: chat_stream.py --run ~/ixv-lm-data/models/ixv-lm-0.1b-chat-v0
モデルのコード（model.py、chat_format.py など）は --code から読む。省略すると <run>/code を使う。
会話の書式は run の config.json の args.chat_format に合わせる（ないときは plain。chat.py と同じ、elvezjp/ixv-lm#10）。
空行で終了、/reset で会話をやり直す。--prompt を付けると1回だけ答えて終わる。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

import stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="config.json と重みのあるディレクトリ")
    parser.add_argument("--data", type=Path, help="tokenizer.json と meta.json のあるディレクトリ。省略すると --run")
    parser.add_argument("--code", type=Path, help="モデルのコードのディレクトリ。省略すると <run>/code")
    parser.add_argument("--prompt", action="append", help="1回だけ答える質問。複数指定できる")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", action="store_true", help="答えごとに速度を表示する")
    args = parser.parse_args()
    data = args.data or args.run
    sys.path.insert(0, str(args.code or args.run / "code"))
    from chat_format import prompt_ids, stop_ids
    from model import LM, ModelConfig

    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    eot = json.loads((data / "meta.json").read_text())["eot"]
    config = json.loads((args.run / "config.json").read_text())
    model = LM(ModelConfig(**config["model"]))
    model.load_weights(str(args.run / (args.run / "latest").read_text().strip() / "model.safetensors"))
    mx.random.seed(args.seed)
    chat_format = config.get("args", {}).get("chat_format", "plain")
    stops = stop_ids(tokenizer, eot, chat_format)

    def reply(messages):
        ids = prompt_ids(tokenizer, messages, eot, chat_format)
        start = time.perf_counter()
        tokens = list(stream.generate(model, ids, args.max_tokens, args.temperature, stops))
        elapsed = time.perf_counter() - start
        if args.speed:
            print(f"（プロンプト {len(ids)} トークン、生成 {len(tokens)} トークン、{elapsed:.2f} 秒）", file=sys.stderr)
        return tokenizer.decode(tokens).strip()

    if args.prompt:
        for prompt in args.prompt:
            answer = reply([{"role": "user", "content": prompt}])
            print(f"ユーザー：{prompt}\nアシスタント：{answer}\n")
        return

    messages = []
    while True:
        try:
            text = input("ユーザー：").strip()
        except EOFError:
            break
        if not text:
            break
        if text == "/reset":
            messages = []
            continue
        messages.append({"role": "user", "content": text})
        answer = reply(messages)
        print(f"アシスタント：{answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
