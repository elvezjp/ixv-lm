"""変換の中身（ime/engine.py）を AJIMEE-Bench の 200 問で採点する。文節ごとの 1 番目の候補をつないだものを答えにする。

例: eval_engine.py --dict ~/ixv-lm-data/ime/dict-v1 --run ~/ixv-lm-data/runs/kanji-char-0.1b \
      --data ~/ixv-lm-data/processed/kanji-char-v1 --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json \
      --out ~/ixv-lm-data/ime/eval --rounds 0 6
--rounds はモデルの検算の上限回数。0 は辞書の下書きだけ。学習（history.tsv）は空の置き場所で測る。
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from engine import Engine  # noqa: E402
from eval_kanji import distance  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[0, 6])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0, help="検算で直した案を採る差（ime/engine.py）")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bits", type=int, choices=[0, 4, 8], default=0, help="線形層の重みのビット数（ime/server.py）。0 は量子化しない")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    store = Path(tempfile.mkdtemp())
    engine = Engine(args.dict, args.run, args.data, store, verify_margin=args.margin, bits=args.bits or None)
    items = json.loads(args.items.read_text())
    if args.limit:
        items = items[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"dict": str(args.dict), "ckpt": engine.ckpt, "margin": args.margin, "bits": args.bits}
    for rounds in args.rounds:
        engine.verify_rounds = rounds
        rows, start = [], time.time()
        with (args.out / f"engine-r{rounds}.jsonl").open("w") as f:
            for item in items:
                kana = item["input"]
                path = engine.draft(item["context_text"], kana)
                got = "".join(p[0] for p in path)
                refs = item["expected_output"]
                row = {"index": item["index"], "context": bool(item["context_text"]), "got": got,
                       "refs": refs, "ok": got in refs, "cer": min(distance(r, got) / len(r) for r in refs)}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = len(rows)
        summary[f"rounds_{rounds}"] = {
            "n": n, "acc@1": round(sum(r["ok"] for r in rows) / n, 3),
            "min_cer": round(sum(r["cer"] for r in rows) / n, 3),
            "with_context_acc@1": round(sum(r["ok"] for r in rows if r["context"]) / max(1, sum(r["context"] for r in rows)), 3),
            "sec_per_item": round((time.time() - start) / n, 3)}
        print(json.dumps({rounds: summary[f"rounds_{rounds}"]}), flush=True)
    (args.out / "engine.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
