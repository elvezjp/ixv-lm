"""かな漢字変換の重みと辞書を、配る形（Hugging Face の elvezjp/IXV-LM-Kanji-0.1B）に取り出す。
学習用のチェックポイント（最適化の状態つき、float32）から、重みだけを bfloat16 で書く。手元のパスは書き出さない。

例: export_release.py --run ~/ixv-lm-data/runs/kanji-char-0.1b-w2611 --data ~/ixv-lm-data/processed/kanji-char-v2 \
      --dict ~/ixv-lm-data/ime/dict-v3 --out ~/ixv-lm-data/release/IXV-LM-Kanji-0.1B
出力：config.json、model.safetensors、tokenizer.json、meta.json、dict/（words.tsv、bigrams.npy、meta.json）。
engine.py と server.py は、--run・--data にこのフォルダ、--dict に dict/ を渡せば読む。
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mlx.core as mx

KEEP_META = ["mix", "format", "vocab_size", "eot", "special_tokens", "min_count", "val", "train"]
KEEP_DICT = ["split", "alt_readings", "alt_share", "docs", "sentences", "words", "bigrams", "tokens",
             "min_word", "min_bigram", "per_doc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ckpt")
    parser.add_argument("--source", default="wikipedia-ja-20261101")
    args = parser.parse_args()

    ckpt = args.ckpt or (args.run / "latest").read_text().strip()
    out = args.out
    (out / "dict").mkdir(parents=True, exist_ok=True)

    weights = mx.load(str(args.run / ckpt / "model.safetensors"))
    mx.save_safetensors(str(out / "model.safetensors"), {k: v.astype(mx.bfloat16) for k, v in weights.items()},
                        metadata={"format": "mlx"})

    config = json.loads((args.run / "config.json").read_text())
    train = {k: v for k, v in config.get("args", {}).items() if k not in ("data", "run", "resume", "stop_after")}
    head = (args.run / "git-head.txt").read_text().strip() if (args.run / "git-head.txt").exists() else ""
    (out / "config.json").write_text(json.dumps({
        "model": config["model"], "weights_dtype": "bfloat16", "run": args.run.name, "ckpt": ckpt,
        "data": args.data.name, "dict": args.dict.name, "source": args.source, "git_head": head, "train": train,
    }, ensure_ascii=False, indent=2))

    shutil.copy(args.data / "tokenizer.json", out / "tokenizer.json")
    meta = json.loads((args.data / "meta.json").read_text())
    (out / "meta.json").write_text(json.dumps({k: meta[k] for k in KEEP_META if k in meta}, ensure_ascii=False, indent=2))

    for name in ["words.tsv", "bigrams.npy"]:
        shutil.copy(args.dict / name, out / "dict" / name)
    dmeta = json.loads((args.dict / "meta.json").read_text())
    (out / "dict/meta.json").write_text(json.dumps({k: dmeta[k] for k in KEEP_DICT if k in dmeta}, ensure_ascii=False, indent=2))

    print(subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
