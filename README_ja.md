# IXV-LM

[English](./README.md) | [日本語](./README_ja.md)

[![Elvez](https://img.shields.io/badge/Elvez-Product-3F61A7?style=flat-square)](https://elvez.co.jp/)
[![IXV Ecosystem](https://img.shields.io/badge/IXV-Ecosystem-3F61A7?style=flat-square)](https://elvez.co.jp/ixv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-black?style=flat-square&logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-IXV--LM--Kanji--0.1B-yellow?style=flat-square)](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B)
[![Stars](https://img.shields.io/github/stars/elvezjp/ixv-lm?style=social)](https://github.com/elvezjp/ixv-lm/stargazers)

**IXV-LM は、エルブズが日本国内でフルスクラッチ開発する、国産の小型言語モデルとその学習基盤です。**

8GB 程度のメモリのマシンでも小気味よく動く、開発支援向けのモデルを目指します。フロンティアモデルと張り合うことは目指しません。手軽に、小さなマシンで使えるものを、みんなに開放することを目指します。

> **研究目的のプロジェクトです。** IXV-LM は、小さな言語モデルと、それを使った日本語入力を、エルブズが調べるための研究プロジェクトです。コード、重み、辞書は現状のまま公開します。動作や精度の保証はなく、サポートもしません。使ってよい範囲は、それぞれのライセンス（コードは MIT、重みと辞書は CC BY-SA 4.0）のとおりです。

> **現在の段階（2026-11-25）:** 研究と初期の実験の段階です。学習コード（MLX、Apple Silicon）で、10M から 0.3B パラメータの Transformer と KDA ハイブリッドを、Mac Studio（M3 Ultra）1 台で学習しています（[experiments/](experiments/)）。最初に公開したモデルは、下のかな漢字変換のモデルです。以下の「狙うところ」「できるようにすること」は目標であり、達成済みの性能ではありません。

## 狙うところ

国産の小型モデルで、次の4つをそろえたものはまだほとんどありません。IXV-LM はこの枠を狙います。

- 2026年世代の新しい構造
- 緩いライセンス（MIT）
- 32K トークン以上の文脈
- ツール呼び出し

## 日本語入力（2026-11-24）

> いずれにせよ、IXV-LMというのは、日本語入力にその生きる道を見つけたということで。（2026-11-24）

学習マシン（M3 Ultra）で学習した文字単位の 0.1B と、Wikipedia から数えた辞書で、日本語入力「IXV かな漢字」が動いた（[ime/README.md](ime/README.md)）。AJIMEE-Bench（200 問）の 1 位の正解率は、辞書だけで 0.645、モデルで確かめると 0.730（2026-11-25）。重みと辞書は Wikipedia 由来なので、コードとは分けて、CC BY-SA 4.0 で Hugging Face の [elvezjp/IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B) に置く（ゲート付き）。MIT で出せる版を作るかは、まだ決めていない。

## できるようにすること

自律的にコードを書くことは狙いません。軽い作業を確実にこなすことを狙います。

- ファイルを読む、要約する、要点を拾う
- 気づいたことをイシューとして起票する（GitHub / Forgejo）
- 決まったツールを正しく呼び出す（OpenAI 互換のツール呼び出し形式）
- 長めのファイルを扱う（目標：32K トークン以上）

## 方針

| 項目 | 内容 |
|---|---|
| 開発方式 | フルスクラッチ。既存のベースモデルへの追加学習はしない |
| 開発場所 | 日本国内、エルブズ自前 |
| 学習データ | 自社の文章を中核に、著作権に問題のないものを加える（下の「データの方針」） |
| 構造 | KDA ハイブリッド（KDA 3：全注意 1）で進める（2026-11-07 決定） |
| 規模 | 1B 前後 |
| 動作環境 | 8GB 程度のメモリのマシン |
| 開発・運用 | 学習パイプラインで IXV を使う（使い方は検討中） |
| 公開 | 重み、学習フレームワーク、推論エンジンを MIT で公開する予定。社内の会話の Q&A データは公開しない（2026-11-08 決定） |

学習環境は検討中です。Mac Studio（M5 Ultra）を第一候補として、実際の処理速度を測ってから決めます。

## データの方針

- **中核は自社の文章である。** エルブズが作ってきた書籍、社内の文章、YouTube 動画（[Tech千一夜](https://www.youtube.com/@tech1018)）を学習モデルにすることが、このプロジェクトの中核である。権利はエルブズと田中が持つ。書籍は、AI が文章を書けるようになる前に人が書いたものである。
- **著作権に問題のないものは学ぶ。** 自社の文章だけでは学習データとして圧倒的に足りない。私たちも何かを作るときはたくさん調べ物をする。著作権に問題のないものは、学習データとして学ぶ。
- **Claude の生成した文章は入れない。** 権利の面でよくないし、独特のクセがあるので入れたくない。これは Claude の話で、AI 全部の話ではない。
- **国産を、完全に国内だけで生み出すものとは定義しない。** 今の時代にその線引きは明らかにやりすぎで、そこまで言えば何もできない。
- **入れたいデータは台帳に書いて管理する。** まず、積極的に入れたいデータを[台帳](corpus/ledger.toml)に書き、学習用データとしてちゃんと管理する。
- **役割を分ける。** 自社の著作物は社内で揃える。ほかのデータは AI が整理して保存する。
- **AI はこの作業を手伝う。** 人の要望を先読みしない。ささっと終わらせようとしない。聞いた方針は記憶で済ませず、資料に書く。同じことを何回も聞かない。

## 進め方

1. 仕様駆動開発の工程に沿って、最初の仕様を書く
2. 最小の学習ループを作り、0.1B を回して処理速度を測る
3. データパイプラインを作り、学習データを整える
4. 本番の事前学習を始める
5. 学習と並行して、推論エンジンと評価の仕組みを作る
6. 公開版を出す

## 最小の学習を動かす

Apple Silicon の Mac と [uv](https://docs.astral.sh/uv/) が必要です。学習データとチェックポイントは、リポジトリの外（例：`~/ixv-lm-data/`）に置きます。

```bash
uv sync
.venv/bin/python train/prepare.py --input <text 列を持つ parquet> --out ~/ixv-lm-data/processed/<名前>
.venv/bin/python train/train.py --data ~/ixv-lm-data/processed/<名前> --run ~/ixv-lm-data/runs/<名前> --steps 5000 --batch-size 4
.venv/bin/python train/sample.py --data ~/ixv-lm-data/processed/<名前> --run ~/ixv-lm-data/runs/<名前> --prompt "東京都は、"
```

途中から続けるときは、`train.py` に `--resume` を付けます。

## 学習マシンでの運用

学習マシンでは、GPU の暴走やメモリの圧迫を避けるため、次を守ります（2026-11-08 決定）。

- 長く動かす処理や、順番に流す処理は、シェルのループやワンライナーではなく、Python のスクリプトとして書く。
- 重い処理は1つずつ流す。GPU を使う処理と、メモリを大きく使う並列処理を重ねない。
- GPU の処理を始める前に、他の重い処理が動いていないことを確かめる。並列の数は、1プロセスのメモリを測ってから決める。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [experiments/](experiments/) | 実験の記録 |
| [docs/](docs/) | 記録 |
| [ime/](ime/) | 日本語入力「IXV かな漢字」（変換サーバーと入力メソッド） |

## セキュリティ

脆弱性の報告は [SECURITY_ja.md](SECURITY_ja.md) を見てください。

## 貢献

[CONTRIBUTING_ja.md](CONTRIBUTING_ja.md) を見てください。

- バグ報告・機能の要望：[Issues](https://github.com/elvezjp/ixv-lm/issues)
- プルリクエストのブランチ名：`{username}/{date}-{description}`

## 変更履歴

[CHANGELOG_ja.md](CHANGELOG_ja.md) を見てください。

## ライセンス

- **コード：** MIT License。[LICENSE](LICENSE) を見てください。
- **重みと辞書：** リポジトリには入れず、Hugging Face で配る。かな漢字変換の [IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B) は Wikipedia 由来なので CC BY-SA 4.0。
- **学習データ：** リポジトリには入れない。出どころとライセンスは[台帳](corpus/ledger.toml)に書く。
- 依存するもの、元にしたものは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) と [PROVENANCE_ja.md](PROVENANCE_ja.md) にある。

## 問い合わせ

- **Issues：** [GitHub Issues](https://github.com/elvezjp/ixv-lm/issues)
- **メール：** info@elvez.co.jp
- **会社：** [株式会社エルブズ](https://elvez.co.jp/)
