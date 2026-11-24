"""かな漢字変換の問題（AJIMEE-Bench）を、追加学習していないモデルに解かせて採点する。

例: eval_kanji.py --run ~/ixv-lm-data/runs/v1-0.3b-r4 --data ~/ixv-lm-data/processed/v1b \
      --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json --out ~/ixv-lm-data/runs/v1-0.3b-r4/kanji
かな漢字変換の段 0（型も学習も変えずに、読みが分かるかを見る）。
v1 は変換の書式を習っていないので、問題のうち左の文脈の無いものを --shots 問、例として前に置く。
例に使った問題は採点から外す。例の文はベンチマークのもの（CC BY-SA 3.0、評価だけに使う）。
採点はベンチマークの utils.py と同じ：完全一致（acc@1）と、許される答えのうち最も近いものとの文字誤り率（MinCER）。
--chat は、段 0 の追加学習（corpus/make_kanji_sft.py の会話）をしたモデル用。例を置かず、会話の書式で 1 問ずつ聞く。
データの meta.json の format が kanji-pairs（train/prepare_kanji.py、段 1）なら、<|ctx|>…<|read|>…<|out|> の並びで聞く。
本線や追加学習と重ねないよう、既定は CPU。
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from chat_format import prompt_ids, stop_ids
from model import LM, ModelConfig


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def block(item, answer=None):
    lines = [f"前：{item['context_text']}", f"読み：{item['input']}", "変換："]
    return "\n".join(lines) + (f"{answer}\n\n" if answer is not None else "")


def request(item):
    """--chat のときのユーザーの発言。corpus/make_kanji_sft.py と同じ形。"""
    return f"前：{item['context_text']}\n読み：{item['input']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="採点する問題の数（0 は全部）。試しに動かすとき用")
    parser.add_argument("--kind", choices=["all", "context", "plain"], default="all",
                        help="context は左の文脈つきの問題だけ、plain は文脈なしだけ")
    parser.add_argument("--chat", action="store_true", help="追加学習したモデルを会話の書式で聞く（例は置かない）")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if not args.gpu:
        mx.set_default_device(mx.cpu)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    meta = json.loads((args.data / "meta.json").read_text())
    eot = meta["eot"]
    pairs = meta.get("format") == "kanji-pairs"
    config = json.loads((args.run / "config.json").read_text())
    model = LM(ModelConfig(**config["model"]))
    latest = (args.run / "latest").read_text().strip()
    model.load_weights(str(args.run / latest / "model.safetensors"))

    chat_format = config.get("args", {}).get("chat_format", "plain")
    stops = set(stop_ids(tokenizer, eot, chat_format)) if args.chat else {eot}
    if args.chat or pairs:
        args.shots = 0

    items = json.loads(args.items.read_text())
    plain = sorted((i for i in items if not i["context_text"]), key=lambda i: int(i["index"]))
    shots = plain[:args.shots]
    shot_ids = {i["index"] for i in shots}
    head = "".join(block(i, i["expected_output"][0]) for i in shots)
    rows = [i for i in items if i["index"] not in shot_ids]
    if args.kind != "all":
        rows = [i for i in rows if bool(i["context_text"]) == (args.kind == "context")]
    if args.limit:
        rows = rows[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.out / "kanji.jsonl").open("w") as f:
        for n, item in enumerate(rows, 1):
            if pairs:
                ids = tokenizer.encode(f"<|ctx|>{item['context_text']}<|read|>{item['input']}<|out|>").ids
            elif args.chat:
                ids = prompt_ids(tokenizer, [{"role": "user", "content": request(item)}], eot, chat_format)
            else:
                ids = tokenizer.encode(head + block(item)).ids
            out = []
            for _ in range(len(item["input"]) + 10):
                token = mx.argmax(model(mx.array([ids + out]))[0, -1]).item()
                if token in stops:
                    break
                out.append(token)
                if not (args.chat or pairs) and "\n" in tokenizer.decode(out):
                    break
            got = tokenizer.decode(out).strip() if (args.chat or pairs) else tokenizer.decode(out).split("\n")[0]
            refs = item["expected_output"]
            row = {"index": item["index"], "context": bool(item["context_text"]), "got": got, "refs": refs,
                   "ok": got in refs, "cer": min(distance(r, got) / len(r) for r in refs)}
            results.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(n, row["ok"], round(row["cer"], 3), got, "|", refs[0], flush=True)

    def score(rs):
        return {"n": len(rs), "acc@1": round(sum(r["ok"] for r in rs) / max(len(rs), 1), 3),
                "min_cer": round(sum(r["cer"] for r in rs) / max(len(rs), 1), 3)}

    summary = {"run": str(args.run), "ckpt": latest, "chat": args.chat, "format": meta.get("format", "text"), "shots": [i["index"] for i in shots],
               "all": score(results), "with_context": score([r for r in results if r["context"]]),
               "no_context": score([r for r in results if not r["context"]])}
    (args.out / "kanji.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
