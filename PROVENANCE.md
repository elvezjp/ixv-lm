# Provenance

[English](./PROVENANCE.md) | [日本語](./PROVENANCE_ja.md)

This record separates design inspiration, adapted material, data and models so that future contributors can preserve
the project's licensing boundary.

## Project implementation

The code in this repository was created for IXV-LM and is distributed by Elvez under the MIT License. Much of it was
written with Claude Code (Anthropic) and reviewed by Elvez. Research notes and reports written by Claude say so and are
not used as training data.

## Adapted material

| Files | Source | License | Note |
|---|---|---|---|
| `train/kda.py` | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), naive KDA implementation | MIT, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li | The recurrent and chunked forms follow its reference implementation; rewritten for MLX |

## Design inspiration (no shared code)

- **Kimi Delta Attention** — the KDA layer follows the Kimi Linear paper (Moonshot AI).
- **azooKey / Zenzai** — the input method's approach (dictionary drafts checked by a small model) follows
  [azooKey](https://github.com/azooKey)'s Zenzai. No code was copied. AJIMEE-Bench, by azooKey, is used for evaluation
  only and is not included in this repository.

## Data and models

- Training data is not in this repository. Each source, its license and filters are recorded in
  [corpus/ledger.toml](corpus/ledger.toml).
- Weights and dictionaries are published separately. [IXV-LM-Kanji-0.1B](https://huggingface.co/elvezjp/IXV-LM-Kanji-0.1B)
  is derived from Japanese Wikipedia (CC BY-SA 4.0 / 3.0) with readings from Sudachi (Apache 2.0), and is licensed
  CC BY-SA 4.0. The MIT License of this repository does not extend to it.

## Contribution rule

Every contribution must be submitted under MIT and with sufficient rights. Copied or substantially adapted material
must record its source, version or commit, license, affected files, and required notices in the pull request and,
when material, in this document.
