#!/bin/bash
# build/IXVKanji.app を ~/Library/Input Methods に入れる。動いている古いものは止める。
# 入れたあと、システム設定 → キーボード → 入力ソース → 「+」→ 日本語 → 「IXV かな漢字」を足す（初回だけ。ログアウトが要ることがある）。
set -euo pipefail
cd "$(dirname "$0")"
DEST="$HOME/Library/Input Methods"
mkdir -p "$DEST"
pkill -x IXVKanji || true
rm -rf "$DEST/IXVKanji.app"
cp -R build/IXVKanji.app "$DEST/"
echo "installed $DEST/IXVKanji.app"
swift tools/register.swift "$DEST/IXVKanji.app"
# メニューバーのアイコンは、入力メニューのプログラムが覚えている。動かし直して、新しいアイコンを読ませる（自動で起動し直す）
killall TextInputMenuAgent 2>/dev/null || true
