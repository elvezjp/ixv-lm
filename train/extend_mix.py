"""作ってあるトークン列に、データを後から足して、別の名前の混合を作る。

例: extend_mix.py --base ~/ixv-lm-data/processed/v1 --out ~/ixv-lm-data/processed/v1b --add hdkworks-blog:10

base のトークナイザーをそのまま使う（語彙が変わると、学習中のモデルを続けられないため）。
train.bin は APFS の複製（cp -c）で作り、足す分だけを後ろに書く。元の train.bin には触らない。
足したデータは検証に回さない。val.bin は base と同じなので、検証の損失は前と比べられる。
学習は位置を無作為に選ぶので、後ろに足しても割合どおりに混ざる。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_mix import clean_parts, iter_clean  # noqa: E402


def tokenize(tokenizer, eot, name, path):
    stats = {"id": name, "docs": 0, "tokens": 0, "val_tokens": 0, "bytes": 0, "chars": 0}
    with path.open("wb") as f:
        batch = []

        def flush():
            for enc, text in zip(tokenizer.encode_batch(batch), batch):
                ids = np.array(enc.ids + [eot], dtype=np.uint16)
                ids.tofile(f)
                stats["tokens"] += len(ids)
                stats["docs"] += 1
                stats["bytes"] += len(text.encode())
                stats["chars"] += len(text)
            batch.clear()

        for text in iter_clean(clean_parts(name)):
            batch.append(text)
            if len(batch) >= 2_000:
                flush()
        if batch:
            flush()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--add", action="append", required=True, help="データ名:繰り返し回数")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import os
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    from tokenizers import Tokenizer

    if args.out.exists():
        raise SystemExit(f"{args.out} はもうあります")
    base_meta = json.loads((args.base / "meta.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.base / "tokenizer.json"))
    eot = tokenizer.token_to_id("<|endoftext|>")
    if eot != base_meta["eot"]:
        raise SystemExit("base の区切りの印と合いません")

    tmp = args.out.with_name(args.out.name + ".tmp")
    if tmp.exists():
        raise SystemExit(f"{tmp} が残っています。前回の途中なので、中を見てから消してください")
    (tmp / "sources").mkdir(parents=True)
    added = []
    for spec in args.add:
        name, repeat = spec.rsplit(":", 1)
        stats = tokenize(tokenizer, eot, name, tmp / "sources" / f"{name}.train.bin")
        if stats["tokens"] == 0:
            raise SystemExit(f"{name}: トークンがありません")
        stats["repeat"] = int(repeat)
        added.append(stats)
        print(f"{name}: {stats['tokens']:,} トークン、{stats['docs']} 文書、{repeat} 回", flush=True)

    for name in ("train.bin", "val.bin", "tokenizer.json"):
        subprocess.run(["cp", "-c", str(args.base / name), str(tmp / name)], check=True)
    with (tmp / "train.bin").open("ab") as f:
        for stats in added:
            data = (tmp / "sources" / f"{stats['id']}.train.bin").read_bytes()
            for _ in range(stats["repeat"]):
                f.write(data)

    total = base_meta["tokens"] + sum(s["tokens"] * s["repeat"] for s in added)
    if (tmp / "train.bin").stat().st_size != total * 2:
        raise SystemExit("train.bin の大きさが合いません")
    sources = [{**s, "share": s["tokens"] * s["repeat"] / total}
               for s in base_meta["sources"] + added]
    total_bytes = sum(s["bytes"] * s["repeat"] for s in sources)
    total_chars = sum(s["chars"] * s["repeat"] for s in sources)
    meta = {**base_meta, "mix": args.out.name, "base": str(args.base), "added": [s["id"] for s in added],
            "tokens": total, "bytes_per_token": total_bytes / total, "chars_per_token": total_chars / total,
            "sources": sources}
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    tmp.rename(args.out)
    print(json.dumps({k: meta[k] for k in ("mix", "tokens", "bytes_per_token", "added")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
