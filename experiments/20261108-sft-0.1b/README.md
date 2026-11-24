# 2026-11-08 会話の追加学習（0.1B）

## 目的

学習したモデルに話しかけたらどうなるかを体験するため、0.1B に会話の形のデータで追加の学習（SFT）をする。賢さは求めない。

## 条件

| 項目 | 内容 |
|---|---|
| 元のモデル | [日本語 Wikipedia で学習した 0.1B](../20261107-wiki-kda-0.1b/README.md) の `ckpt-43000` |
| データ | oasst2 日本語訳 68,501 会話、llm-jp-instructions の train と dev を各5回繰り返し（3,000 会話）。台帳ではどちらも「実験のみ」 |
| 書式 | `ユーザー：{質問}\nアシスタント：{答え}<|endoftext|>`（`train/chat_format.py`） |
| 損失 | 答えと終わりの印だけ。答えのトークンは 2,156 万 |
| 学習 | 2エポック、1,574 ステップ、バッチ 16 × 2,048、学習率 5e-5（ウォームアップ 50、その後直線で 0 まで）、bf16 |
| 速度 | 約 17,200 トークン/秒。約50分 |

Claude の生成した文章は入っていない。oasst2 の日本語は DeepL による機械翻訳である。

## 結果

答えの部分の学習損失は 3.14（20 ステップ）から 2.48（最後）に下がった。検証用の会話は分けていない。

`chat.py --prompt`、温度 0.7、種 0。

> **ユーザー：** 日本の首都はどこですか？
> **アシスタント：** 日本の首都は東京で、一般的には東京の中心部、東京湾の西岸に位置し、三鷹市とも日本一広いところです。

> **ユーザー：** カレーライスの作り方を教えてください。
> **アシスタント：** カレーライスを食べるには、必ず予約が必要だ。

> **ユーザー：** 縦10cm、横15cmの長方形の面積は？
> **アシスタント：** 縦10cm、横15cmの長方形面積は、面積の分布や形を理解する上で重要な要素である。

質問に答える形で返し、答えの終わりで止まる。中身は合わないことが多く、計算はできない。5例を見ただけである。

## 試し方

```bash
cd train
KDA_METAL=1 ../.venv/bin/python chat.py --run ~/ixv-lm-data/runs/kda-0.1b-sft --data ~/ixv-lm-data/processed/wiki-ja-full-16k
```

空行で終了、`/reset` で会話をやり直す。

## 再現

```bash
cd train
S=~/ixv-lm-data/sft
../.venv/bin/python ../corpus/clean_sft.py
KDA_METAL=1 ../.venv/bin/python sft.py --base ~/ixv-lm-data/runs/kda-0.1b-wiki-ja-metal \
  --data ~/ixv-lm-data/processed/wiki-ja-full-16k --run ~/ixv-lm-data/runs/kda-0.1b-sft \
  --sft $S/oasst2-ja.jsonl --sft $S/llm-jp-instructions-train.jsonl:5 --sft $S/llm-jp-instructions-dev.jsonl:5
```
