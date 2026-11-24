# Changelog

[English](./CHANGELOG.md) | [日本語](./CHANGELOG_ja.md)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **IXV かな漢字** (IXV Kana-Kanji), a macOS input method (`ime/`): dictionary drafts checked by a 0.1B
  character-level model; per-phrase candidates, prediction, live conversion, dates and symbols, learning, a user
  dictionary, a settings window, and a conversion server that starts at login
- `ime/export_release.py`, which exports the weights and dictionary published as
  [elvezjp/IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B) (CC BY-SA 4.0)
- Kana-kanji training data, dictionary and evaluation (`corpus/make_kanji_*.py`, `train/prepare_kanji.py`,
  `train/eval_kanji.py`); cleaning of the official Japanese Wikipedia dump (`corpus/clean.py`)
- Training framework on MLX: Transformer and KDA hybrid models, Metal kernels for KDA, supervised restarts, SFT, tool
  calling, release tools
- Data ledger (`corpus/ledger.toml`) and experiment records (`experiments/`)
- MIT LICENSE, English and Japanese README, CONTRIBUTING, SECURITY, PROVENANCE and third-party notices
