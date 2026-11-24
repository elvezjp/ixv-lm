"""チェックポイントから文章を生成する。"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from model import LM, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    config = json.loads((args.run / "config.json").read_text())
    model = LM(ModelConfig(**config["model"]))
    latest = (args.run / "latest").read_text()
    model.load_weights(str(args.run / latest / "model.safetensors"))

    mx.random.seed(0)
    for prompt in args.prompt:
        ids = tokenizer.encode(prompt).ids
        for _ in range(args.tokens):
            logits = model(mx.array([ids[-512:]]))[0, -1]
            ids.append(mx.random.categorical(logits / args.temperature).item())
        print(tokenizer.decode(ids).replace("\n", " "))


if __name__ == "__main__":
    main()
