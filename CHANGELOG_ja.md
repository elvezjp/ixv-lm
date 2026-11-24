# 変更履歴

[English](./CHANGELOG.md) | [日本語](./CHANGELOG_ja.md)

このプロジェクトの主な変更は、このファイルに記録します。

形式は [Keep a Changelog](https://keepachangelog.com/ja/1.0.0/) に、版の付け方は [Semantic Versioning](https://semver.org/lang/ja/) に従います。

## [Unreleased]

### 追加

- macOS の日本語入力「IXV かな漢字」（`ime/`）：辞書の下書きを、文字単位の 0.1B のモデルが確かめる。文節ごとの候補、予測変換、ライブ変換、日付と記号、学習、ユーザー辞書、設定画面、ログインで動く変換サーバー
- `ime/export_release.py`：[elvezjp/IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B)（CC BY-SA 4.0）として配る重みと辞書を取り出す
- かな漢字変換の学習データ、辞書、採点（`corpus/make_kanji_*.py`、`train/prepare_kanji.py`、`train/eval_kanji.py`）。日本語版 Wikipedia の公式ダンプの取り出し（`corpus/clean.py`）
- MLX の学習基盤：Transformer と KDA ハイブリッド、KDA の Metal カーネル、自動で動かし直す仕組み、SFT、ツール呼び出し、リリースの道具
- データの台帳（`corpus/ledger.toml`）と実験の記録（`experiments/`）
- MIT の LICENSE、英語と日本語の README、CONTRIBUTING、SECURITY、PROVENANCE、第三者の表示
