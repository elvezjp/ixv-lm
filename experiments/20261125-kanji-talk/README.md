# かな漢字変換に話し言葉を足す（段 1 からの追加学習）

段 1（[記録](../20261124-kanji-stage1/README.md)）は Wikipedia の文だけで学習したので、話し言葉の文末（「〜ね」「うん、」など）を取り違える（[ime/README.md](../../ime/README.md) の「使ってみて見つかった間違い」）。
話し言葉の「読み → 文」を足し、段 1 の重みから追加学習して、取り違えが減るかを見る。2026-11-25 決定。

## データ

- 青空文庫（台帳の `aozora`、保護期間の切れた作品）の会話。`corpus/make_kanji_sft.py --quotes --modern-only`：「」の中だけを文として拾い、旧仮名づかいの作品（ゐ、ゑ、でせう、だらう など）は飛ばす。389,199 件。
- 自社の文章：hdkworks-blog 8,180 件、elvez-books 1,498 件、elvez-sdd-repo 763 件（ふつうの文の拾い方）。
- 忘れないよう、段 1 の対（`kanji-v3.jsonl`、Wikipedia 20261101）の一部も混ぜる。
- 混ぜ方：青空文庫 0.6、自社の文章 ×5、kanji-v3 0.2。学習 825,441 件・4,094 万トークン、検証 8,337 件。
- 語彙は段 1 と同じ（`train/prepare_kanji.py --tokenizer`）。語彙に無い字は `<|unk|>`（学習で 2,351 個）。
- 置き場所：`~/ixv-lm-data/sft/kanji-talk/`、`~/ixv-lm-data/processed/kanji-char-talk-v1`。

```
D=~/ixv-lm-data
.venv/bin/python corpus/make_kanji_sft.py --src $D/clean/aozora --items $D/eval/ajimee/evaluation_items.json \
  --out $D/sft/kanji-talk/aozora-talk.jsonl --n 400000 --take 1.0 --per-doc 200 --quotes --modern-only \
  --source aozora --min-chars 6 --seed 1
# hdkworks-blog、elvez-books、elvez-sdd-repo は --per-doc 100000 --min-chars 6 --seed 1（--quotes なし）
.venv/bin/python train/prepare_kanji.py --tokenizer $D/processed/kanji-char-v2/tokenizer.json \
  --out $D/processed/kanji-char-talk-v1 --src $D/sft/kanji-talk/aozora-talk.jsonl:0.6 \
  --src $D/sft/kanji-talk/hdkworks-blog.jsonl:5 --src $D/sft/kanji-talk/elvez-books.jsonl:5 \
  --src $D/sft/kanji-talk/elvez-sdd-repo.jsonl:5 --src $D/sft/kanji-v3.jsonl:0.2
```

## 学習

- `kanji-char-0.1b-w2611` の ckpt-13000 を新しい run（`~/ixv-lm-data/runs/kanji-char-0.1b-talk1`）に写し、`--resume` で 15,000 まで（2,000 ステップ、6,554 万トークン、約 1.6 周）。
- 学習率 2e-4 で続け、14,000 から 0 まで下げる。型と引数は段 1 と同じ。約 1 時間（約 19,000 トークン/秒）。
- 学習の損失 1.37 → 1.15。検証の損失（新しいデータの検証）は 1.357（13,500）→ 1.362（15,000）でほぼ横ばい。
- 回したスクリプトは `~/ixv-lm-data/ime/talk-v1.py`。

## 採点

- AJIMEE-Bench 200 問（段 1 と同じ）。
- 話し言葉：人がこのリポジトリの作業でチャットに打った文から作った 293 問（`~/ixv-lm-data/eval/talk/`）。読みは Sudachi で振り、答えは打ったままの文。Claude の出した選択肢を押しただけの発言と、打ち間違いのある文は外した。会話のデータなので公開しない。学習にも入れない。

| acc@1 | 段 1（ckpt-13000） | 追加学習（ckpt-15000） |
|---|---|---|
| AJIMEE、辞書の下書きをモデルが検算 | 0.680 | **0.705** |
| AJIMEE、同、文脈つき 100 問 | 0.670 | 0.650 |
| AJIMEE、モデルだけ | 0.585 | 0.585 |
| 話し言葉 293 問、検算 | 0.652 | **0.669** |
| 話し言葉 293 問、辞書の下書きだけ（参考、モデルを使わない） | 0.546 | 0.546 |

- 話し言葉では 18 問が当たるようになり、13 問が外れるようになった。当たるようになったのは、文頭の「うん、」「はい、」、文末の「〜ね」「〜どう？」など。外れるようになったものには、表記の揺れ（なに／何、通り／とおり）のほか、「終った」のような古い送り仮名や、「サイトと家内の？」のような区切りの誤りがある。青空文庫の古い書き方を覚えた分と見ている。
- 8bit の重み（`ime/eval_engine.py --bits 8`）でも、AJIMEE は 200 問中 2 問、話し言葉は 293 問中 5 問だけ答えが変わり、acc@1 は 0.700 と 0.672。

## まだやっていないこと

- 青空文庫の割合を下げる、新しい作品だけにしぼる、などの比べ。
- 変換サーバーの重みの入れ替え（いまは段 1 のまま）。Hugging Face に出すか。
