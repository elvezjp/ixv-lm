#!/bin/bash
# 変換サーバーを、~/ixv-lm-data/ime/current.json に書いた辞書と重みで動かす。launchd からも、手でも使う。
# current.json の形：{"dict": "...", "run": "...", "data": "..."}。作り直しのスクリプトが書き換える。
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="$HOME/ixv-lm-data/ime/current.json"
DICT=$(/usr/bin/python3 -c "import json;print(json.load(open('$CFG'))['dict'])")
RUN=$(/usr/bin/python3 -c "import json;print(json.load(open('$CFG'))['run'])")
DATA=$(/usr/bin/python3 -c "import json;print(json.load(open('$CFG'))['data'])")
exec .venv/bin/python ime/server.py --dict "$DICT" --run "$RUN" --data "$DATA"
