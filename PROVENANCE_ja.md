# 出どころ

[English](./PROVENANCE.md) | [日本語](./PROVENANCE_ja.md)

ライセンスの境目を守れるよう、設計の手本、取り入れたもの、データとモデルを分けて記録します。

## このプロジェクトの実装

このリポジトリのコードは IXV-LM のために作ったもので、エルブズが MIT License で配ります。多くは Claude Code（Anthropic）で書き、エルブズが確かめました。Claude が書いた調査ノートとレポートには、その旨を書いています。学習データには入れません。

## 取り入れたもの

| ファイル | 出どころ | ライセンス | 説明 |
|---|---|---|---|
| `train/kda.py` | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) の KDA の naive 実装 | MIT、Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li | 再帰式とチャンク形式を、その参照実装に合わせた。MLX 用に書き直している |

## 設計の手本（コードは借りていない）

- **Kimi Delta Attention**：KDA の層は Kimi Linear の論文（Moonshot AI）に従う。
- **azooKey / Zenzai**：日本語入力のやり方（辞書の下書きを小さなモデルが確かめる）は、[azooKey](https://github.com/azooKey) の Zenzai を手本にした。コードは写していない。azooKey の AJIMEE-Bench は採点にだけ使い、このリポジトリには入れていない。

## データとモデル

- 学習データはリポジトリに入れない。出どころ、ライセンス、加工は [corpus/ledger.toml](corpus/ledger.toml) に書く。
- 重みと辞書は別に配る。[IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B) は日本語版 Wikipedia（CC BY-SA 4.0 / 3.0）から作り、読みは Sudachi（Apache 2.0）で付けたので、CC BY-SA 4.0 で出す。このリポジトリの MIT License は及ばない。

## 貢献の決まり

貢献は MIT で、送る権利を持って送ってください。写したもの、大きく手を加えて取り入れたものは、出どころ、版またはコミット、ライセンス、関わるファイル、必要な表示を、プルリクエストに書き、大事なものはこの文書にも書いてください。
