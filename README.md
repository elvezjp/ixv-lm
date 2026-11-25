# IXV-LM

[English](./README.md) | [日本語](./README_ja.md)

[![Elvez](https://img.shields.io/badge/Elvez-Product-3F61A7?style=flat-square)](https://elvez.co.jp/)
[![IXV Ecosystem](https://img.shields.io/badge/IXV-Ecosystem-3F61A7?style=flat-square)](https://elvez.co.jp/ixv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-black?style=flat-square&logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-IXV--LM--Kanji--0.1B-yellow?style=flat-square)](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B)
[![Stars](https://img.shields.io/github/stars/elvezjp/ixv-lm?style=social)](https://github.com/elvezjp/ixv-lm/stargazers)

**IXV-LM is a small Japanese language model, and the framework to train it, built from scratch in Japan by Elvez.**

The aim is a model for development work that runs smoothly on a machine with about 8 GB of memory. It does not try to
compete with frontier models. It aims to be easy to use on small machines, and open to everyone.

The project documents are written in Japanese. The Japanese README ([README_ja.md](README_ja.md)) is the primary one;
this page is its English summary.

> **A research project.** IXV-LM is a research project in which Elvez studies small language models and Japanese
> input built on them. The code, weights and dictionary are published as they are, with no warranty of operation or
> accuracy, and without support. What you may do with them is set by their licenses (code: MIT; weights and
> dictionary: CC BY-SA 4.0).

> **Status (2026-11-25).** Research and early experiments. The training code (MLX, Apple Silicon) trains Transformer and KDA-hybrid
> models from 10M to 0.3B parameters on one Mac Studio (M3 Ultra); see [experiments/](experiments/). The goals below are
> targets, not achieved results. The first published model is the kana-kanji conversion model below.

## Japanese input (2026-11-24)

> いずれにせよ、IXV-LMというのは、日本語入力にその生きる道を見つけたということで。（2026-11-24）

Elvez set this direction: IXV-LM finds its use in Japanese text input.

**IXV かな漢字** (IXV Kana-Kanji) is a macOS input method that runs on what this project trained: a 0.1B
character-level model and a dictionary counted from Japanese Wikipedia. The dictionary builds a draft that matches the
typed reading, and the model checks it. On AJIMEE-Bench (200 items) the top-1 accuracy is 0.645 with the dictionary
alone and 0.730 with the model's check (2026-11-25).

- Code: [ime/](ime/README.md) (conversion server in Python, input method in Swift, built without Xcode)
- Weights and dictionary: [elvezjp/IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B) on Hugging
  Face, CC BY-SA 4.0 because they are derived from Wikipedia
- Records: [stage 0](experiments/20261123-kanji-stage0/README.md),
  [stage 1](experiments/20261124-kanji-stage1/README.md)
- Not decided yet: whether to make a version that can be released under MIT

## Target

Few small Japanese models built in Japan have all four of the following. IXV-LM aims for that gap.

- A new 2026-generation architecture
- A permissive license (MIT)
- A context of 32K tokens or more
- Tool calling

## What it should do

Not autonomous coding. Doing light tasks reliably.

- Read files, summarize them, pick out the key points
- File what it notices as issues (GitHub / Forgejo)
- Call fixed tools correctly (OpenAI-compatible tool-calling format)
- Handle longer files (target: 32K tokens or more)

## Policy

| Item | Policy |
|---|---|
| Development | From scratch. No continued training of existing base models |
| Where | In Japan, by Elvez itself |
| Training data | Elvez's own texts at the core, plus material with no copyright problems (see "Data policy") |
| Architecture | KDA hybrid (KDA 3 : full attention 1), decided 2026-11-07 |
| Size | Around 1B |
| Target machine | About 8 GB of memory |
| Development and operation | Use IXV in the training pipeline (how is under study) |
| Release | Weights, training framework and inference engine under MIT (planned). Internal conversation Q&A data is not published (decided 2026-11-08) |

The training environment is under study. Mac Studio (M5 Ultra) is the first candidate; the choice will follow measured
speed.

## Data policy

- **Elvez's own texts are the core.** Turning the books, internal documents and YouTube videos
  ([Tech千一夜](https://www.youtube.com/@tech1018)) that Elvez has made into a model is the core of this project. Elvez
  and Tanaka hold the rights. The books were written by people, before AI could write.
- **Material with no copyright problems is used.** Our own texts alone are far too little. We read widely when we make
  something; material with no copyright problems is used as training data.
- **No text generated by Claude.** It is not good for rights, and it has its own habits. This is about Claude, not
  about all AI.
- **"Made in Japan" does not mean made only within Japan.** Drawing that line today goes too far.
- **Data is managed in a ledger.** Sources are written in the [ledger](corpus/ledger.toml) before they are used.
- **Roles are split.** Elvez gathers its own works. AI organizes and stores the other data.
- **AI helps with this work.** It does not guess ahead of people, does not rush, writes policies into documents
  instead of relying on memory, and does not ask the same thing twice.

## Steps

1. Write the first specification, following spec-driven development
2. Build a minimal training loop, train 0.1B and measure speed
3. Build the data pipeline and prepare the training data
4. Start the main pre-training
5. Build the inference engine and evaluation alongside training
6. Release

## Setup

### Requirements

- A Mac with Apple Silicon
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/)

### Run a minimal training

Training data and checkpoints live outside the repository (for example `~/ixv-lm-data/`).

```bash
git clone https://github.com/elvezjp/ixv-lm.git
cd ixv-lm
uv sync
.venv/bin/python train/prepare.py --input <parquet with a text column> --out ~/ixv-lm-data/processed/<name>
.venv/bin/python train/train.py --data ~/ixv-lm-data/processed/<name> --run ~/ixv-lm-data/runs/<name> --steps 5000 --batch-size 4
.venv/bin/python train/sample.py --data ~/ixv-lm-data/processed/<name> --run ~/ixv-lm-data/runs/<name> --prompt "東京都は、"
```

Add `--resume` to `train.py` to continue.

### Use the Japanese input method

See [ime/README.md](ime/README.md) (Japanese).

## Running on the training machine

On the training machine, to avoid runaway GPU use and memory pressure (decided 2026-11-08):

- Write long-running or sequential jobs as Python scripts, not shell loops or one-liners.
- Run heavy jobs one at a time. Do not overlap GPU work with large parallel jobs.
- Before starting GPU work, check that no other heavy job is running. Set the degree of parallelism after measuring
  one process's memory.

## Directory structure

```text
ixv-lm/
├── train/            # Model (KDA hybrid), training, evaluation, release tools (MLX)
├── corpus/           # Data fetching, cleaning, the data ledger (ledger.toml)
├── ime/              # IXV かな漢字: conversion engine and server (Python), input method (ime/macos, Swift)
├── experiments/      # One folder per experiment, with settings and results
├── docs/             # Records
├── README.md         # This file (English)
├── README_ja.md      # Japanese README (primary)
└── pyproject.toml
```

## Security

See [SECURITY.md](SECURITY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

- Bug reports & feature requests: [Issues](https://github.com/elvezjp/ixv-lm/issues)
- Pull requests: branch naming convention `{username}/{date}-{description}`

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

- **Code:** MIT License. See [LICENSE](LICENSE).
- **Weights and dictionaries:** not in this repository; published on Hugging Face. The kana-kanji model
  [IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B) is CC BY-SA 4.0 because it is derived from
  Wikipedia.
- **Training data:** not in this repository. Sources and licenses are recorded in the [ledger](corpus/ledger.toml).
- Dependencies and adapted material: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [PROVENANCE.md](PROVENANCE.md).

## Contact

- **Issues**: [GitHub Issues](https://github.com/elvezjp/ixv-lm/issues)
- **Email**: info@elvez.co.jp
- **Company**: [Elvez Inc.](https://elvez.co.jp/)
