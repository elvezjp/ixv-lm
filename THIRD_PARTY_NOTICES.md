# Third-Party Notices

IXV-LM is licensed under the MIT License. It uses the following packages as separate distributions, installed by
`uv sync`; they are not included in this repository. Their distributions are the authoritative source for complete
license texts.

| Package | Version tested | Copyright | License |
|---|---|---|---|
| [mlx](https://github.com/ml-explore/mlx) | 0.32.4 | Copyright © 2023 Apple Inc. | MIT |
| [numpy](https://github.com/numpy/numpy) | 2.5.4 | Copyright (c) 2005-2025, NumPy Developers | BSD-3-Clause (with bundled components under 0BSD, MIT, Zlib, CC0-1.0) |
| [pyarrow](https://github.com/apache/arrow) | 26.0.1 | The Apache Software Foundation | Apache-2.0 |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.23.3 | Hugging Face | Apache-2.0 |
| [SudachiPy](https://github.com/WorksApplications/sudachi.rs) | 0.7.0 | Works Applications Co., Ltd. | Apache-2.0 |
| [SudachiDict (core)](https://github.com/WorksApplications/SudachiDict) | 20261015 | Works Applications Co., Ltd. | Apache-2.0 |

The macOS input method (`ime/macos`) uses only Apple's system frameworks (Cocoa, InputMethodKit, SwiftUI, Carbon).

`train/kda.py` adapts the reference KDA implementation of
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention):

```
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
```

Model weights and dictionaries are not part of this repository and carry their own licenses; see
[PROVENANCE.md](./PROVENANCE.md).
