# かな漢字変換の段 0（v1 に「読み → 文」を追加学習する）

かな漢字変換の段 0。v1 が読みを分かっているか、型も語彙も変えずに見る。
2026-11-23 決定。r5 が終わって GPU が空いてから回す。

## 物差し：AJIMEE-Bench

- `azooKey/AJIMEE-Bench` の `JWTD_v2/v1/evaluation_items.json`（200 問、うち 100 問は左の文脈つき）。CC BY-SA 3.0。
- 置き場所：`~/ixv-lm-data/eval/ajimee/`。評価だけに使い、学習には入れない。
- 採点：[train/eval_kanji.py](../../train/eval_kanji.py)。ベンチマークの `utils.py` と同じく、完全一致（acc@1）と MinCER。

## 追加学習なしの結果（2026-11-23、r4 の素の重み ckpt-165000、CPU）

文脈なしの問題を例として前に置き、続きを書かせた。

| 例の数 | 問題 | acc@1 | MinCER |
|---|---|---|---|
| 3 | 文脈つき 8 問 | 0/8 | 1.51 |
| 3 | 文脈なし 8 問 | 0/8 | 1.32 |
| 10 | 文脈なし 8 問 | 0/8 | 1.10 |

ほとんどが空の答え（「変換：」の次にいちばん出やすいのが改行）。書いたときは、左の文脈を写すか、同じ語を繰り返す。
例を見せるだけでは、v1 は変換をしない。200 問全部は回していない。

## 学習データ：kanji-v1

[corpus/make_kanji_sft.py](../../corpus/make_kanji_sft.py)。台帳の `kanji-v1`（実験のみ）。

- 日本語版 Wikipedia（clean 済み）の文に、Sudachi（辞書 core、分割 C）で読みを振る。Claude の書いた文章は入らない。
- 会話の形：ユーザー「前：<左の文脈>\n読み：<カタカナ>」、アシスタント「<変換結果>」。
- 半分は左の文脈つき（文の途中から変換）。変換する部分は、半分は文の途中で切る。最大 40 字。
- 外したもの：英字・数字・空白を含む文（Sudachi は「３００」を「サンレイレイ」と読む）、読みの取れない語のある文、かなだけの部分。
- AJIMEE-Bench も Wikipedia から作られているので、問題の文や答えを含む文を外した（7 件）。
- 結果：189,047 件（文脈つき 92,938）。変換結果は平均 17.1 字。

```
.venv/bin/python corpus/make_kanji_sft.py --src ~/ixv-lm-data/clean/wikipedia-ja \
  --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json --out ~/ixv-lm-data/sft/kanji-v1.jsonl --n 200000
```

## 回し方（r5 のあと）

起点は r5 の学習率を下げ終えた重み（SFT 前）。会話の書式は tokens。

```
.venv/bin/python train/sft.py --base ~/ixv-lm-data/runs/v1-0.3b-r5 --data ~/ixv-lm-data/processed/v1b \
  --run ~/ixv-lm-data/runs/v1-0.3b-r5-kanji0 --sft ~/ixv-lm-data/sft/kanji-v1.jsonl
.venv/bin/python train/eval_kanji.py --chat --run ~/ixv-lm-data/runs/v1-0.3b-r5-kanji0 --data ~/ixv-lm-data/processed/v1b \
  --items ~/ixv-lm-data/eval/ajimee/evaluation_items.json --out ~/ixv-lm-data/runs/v1-0.3b-r5-kanji0/kanji
```

自動で回す：`~/ixv-lm-data/runs/kanji0-after-r5.py`（2026-11-23 起動）が r5 の release_end を待ち、上の 2 つを続けて回す（採点は GPU）。経過は `~/ixv-lm-data/runs/kanji0-after-r5.jsonl`。

比べるもの：zenz-v1 の AJIMEE-Bench の数字は、確かめてから書く。

## 結果（2026-11-24、ckpt-406、GPU、200 問）

追加学習：189,047 件を 2 周、406 ステップ、約 38 分。損失 4.54 → 0.92。起点は r5 の ckpt-205000。

| 問題 | acc@1 | MinCER |
|---|---|---|
| 全部（200） | 0.14 | 0.316 |
| 文脈つき（100） | 0.13 | 0.317 |
| 文脈なし（100） | 0.15 | 0.315 |

追加学習なし（上の表）は 0 問だったので、v1 は「読み → 文」の形を覚えた。外れ方は、読みの合わない字を入れるもの（「フタイホトッケン」→「不在と疑惑」、「ユウイ」→「声優」）、読みの一部を落とすもの、数字（学習データから外した）。
採点の自動実行は、スクリプトの誤り（ファイルの場所）で動かず、手で回した。
