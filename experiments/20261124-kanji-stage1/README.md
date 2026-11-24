# かな漢字変換の段 1（文字単位の語彙で、0.1B を最初から学習する）

かな漢字変換の段 1。段 0（[記録](../20261123-kanji-stage0/README.md)）は v1 に追加学習して acc@1 0.14。
語彙を文字単位にし、「読み → 文」の対だけで最初から学習すると、zenz-v1 と同じくらいの大きさでどこまで行くかを見る。
2026-11-24 決定（本線が止まって空いた GPU を、これに使う）。

## データ

- 対：`corpus/make_kanji_sft.py` を Wikipedia 全体に（`--take 1.0 --per-doc 6 --seed 1`）。`kanji-v2.jsonl`、2,743,246 件（文脈つき 1,348,433）。外したもの・作り方は段 0 と同じ。AJIMEE-Bench と重なる文 107 件を外した。
- 語彙：[train/prepare_kanji.py](../../train/prepare_kanji.py)。1 字 1 トークン、3 回より少ない字は `<|unk|>`。特別なトークン 6 つを入れて 5,945。
- 並び：`<|ctx|>左の文脈<|read|>読み<|out|>変換結果<|endoftext|>` を詰める。損失は全部のトークンで取る（train.py のまま。zenz は変換結果の側だけ）。
- 1% を検証に。学習 2,715,814 件・1 億 4,157 万トークン、検証 27,432 件・143 万トークン。`<|unk|>` は学習で 1,573 個。
- 置き場所：`~/ixv-lm-data/processed/kanji-char-v1`。

```
.venv/bin/python corpus/make_kanji_sft.py --src ~/ixv-lm-data/clean/wikipedia-ja \
  --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json --out ~/ixv-lm-data/sft/kanji-v2.jsonl \
  --n 5000000 --take 1.0 --per-doc 6 --seed 1
.venv/bin/python train/prepare_kanji.py --src ~/ixv-lm-data/sft/kanji-v2.jsonl --out ~/ixv-lm-data/processed/kanji-char-v1
```

## 型と学習

0.1B の記録（[20261107-wiki-kda-0.1b](../20261107-wiki-kda-0.1b/README.md)）と同じ型：12 層、幅 768、ヘッド 12、FFN 2,048、KDA 3 層に注意 1 層、NoPE、bf16。語彙が小さいので 9,147 万パラメータ。

- 13,000 ステップ、バッチ 16 × 2,048（1 ステップ 32,768 トークン）。約 3 周。
- 学習率 6e-4、ウォームアップ 500、最後の 2 割で 0 まで下げる。
- 速さ（60 ステップの試し）：約 19,000 トークン/秒、約 1.7 秒/ステップ。約 6.3 時間の見込み。
- 2026-11-24 01:54 に supervise.py で始めた。終わるのは 08:15 ごろの見込み。run は `~/ixv-lm-data/runs/kanji-char-0.1b`。

```
KDA_METAL=1 caffeinate -is .venv/bin/python train/supervise.py --run ~/ixv-lm-data/runs/kanji-char-0.1b -- \
  --data ~/ixv-lm-data/processed/kanji-char-v1 --steps 13000 --batch-size 16 --seq-len 2048 --lr 6e-4 \
  --warmup 500 --decay-frac 0.2 --dim 768 --n-layers 12 --n-heads 12 --hidden-dim 2048 --attn-every 4 \
  --no-rope --compute-dtype bfloat16 --log-every 100 --eval-every 500 --eval-batches 32 --save-every 500 --keep-last 4
```

## 採点

[train/eval_kanji.py](../../train/eval_kanji.py)。データの meta.json が `kanji-pairs` なら、`<|ctx|>…<|read|>…<|out|>` で聞く。段 0 と同じ 200 問。

```
.venv/bin/python train/eval_kanji.py --gpu --run ~/ixv-lm-data/runs/kanji-char-0.1b \
  --data ~/ixv-lm-data/processed/kanji-char-v1 --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json \
  --out ~/ixv-lm-data/runs/kanji-char-0.1b/kanji
```

## 結果

13,000 ステップ、約 6.4 時間。最後の val_loss は 1.300（val_bpb 0.678）。

AJIMEE-Bench 200 問（ckpt-13000）：

| 方式 | acc@1 | MinCER | 文脈つき 100 問の acc@1 |
|---|---|---|---|
| 段 0（v1 に追加学習、モデルだけ） | 0.14 | 0.316 | |
| 段 1、モデルだけ（`eval_kanji.py`） | 0.62 | 0.064 | 0.62 |
| 辞書の下書きだけ（dict-v2） | 0.650 | 0.051 | 0.58 |
| 辞書の下書きを段 1 が検算（6 回まで、`verify_margin` 1.0） | **0.685** | **0.042** | **0.65** |

- モデルだけで、段 0 の 0.14 から 0.62 に上がった。辞書だけとほぼ同じ。
- 検算は、学習途中（ckpt-2000、20 問）では下げていたが、学習を終えた重みでは 0.650 → 0.685 に上げた。文脈つきの問題で効いている。
- 速さ（GPU、1 問あたり）：下書きだけ 0.094 秒、検算つき 0.16 秒。
- 採点の出力は `~/ixv-lm-data/runs/kanji-char-0.1b/kanji/` と `~/ixv-lm-data/ime/eval-stage1/`。`verify_margin` はまだ 1.0 のまま。決め直しは、最新の Wikipedia の学習の結果を見てから。

## 最新の Wikipedia で作り直す（2026-11-24 決定）

上の学習は HF の `wikimedia/wikipedia` 20231101.ja で、3 年前の言葉までしか入っていない。日本語入力は新しい言葉が大事なので、公式ダンプ `jawiki-20261101-pages-articles.xml.bz2`（台帳 `wikipedia-ja-20261101`）で取り直す。

- 本文の取り出し：`corpus/clean.py wikipedia-ja-20261101`（wikitext からテンプレート・表・脚注を落とす。仮リンク・読み仮名・lang などは中の語を残す）
- 辞書 `dict-v3`、対 `kanji-v3.jsonl`、トークン列 `kanji-char-v2`、run `kanji-char-0.1b-w2611`。作り方と学習の設定は上と同じ
- 自動で回す：`~/ixv-lm-data/ime/refresh-20261101.py`。いまの学習とその採点が終わってから GPU を使い、終わったら採点して、変換サーバーを新しい辞書と重みに替える。経過は `refresh-20261101.jsonl`

### 結果（2026-11-24）

13,000 ステップ、08:00 から 14:30 ごろまで。最後の val_loss は 1.340（val_bpb 0.695）。語彙が変わったので、上の 1.300 とは比べられない。

AJIMEE-Bench 200 問（どちらも ckpt-13000、`verify_margin` 1.0）：

| 方式 | 20231101（dict-v2、kanji-char-0.1b） | 20261101（dict-v3、kanji-char-0.1b-w2611） |
|---|---|---|
| モデルだけ | 0.62 | 0.585 |
| 辞書の下書きだけ | 0.650 | 0.645 |
| 辞書の下書きを検算 | **0.685** | **0.68** |
| 同、文脈つき 100 問 | 0.65 | **0.67** |
| 同、MinCER | 0.042 | 0.044 |
| 同、1 問の秒数（GPU） | 0.16 | 0.179 |

- 検算つきは 1 問の差で、ほぼ同じ。モデルだけは少し下がった。
- AJIMEE-Bench は古い Wikipedia から作られているので、新しい言葉が入った良さはこの点数では測れない。
- 検算はどちらでも点を上げたので、`verify_margin` は 1.0 のままにする。
- 変換サーバーは 14:42 に dict-v3 と kanji-char-0.1b-w2611 に替わった（`~/ixv-lm-data/ime/current.json`）。採点の出力は `runs/kanji-char-0.1b-w2611/kanji/` と `ime/eval-w2611/`。

