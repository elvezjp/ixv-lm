#!/bin/bash
# 入力メソッド IXVKanji.app を build/ に作る（Xcode なし、コマンドラインの swiftc で）。
# 入れるのは install.sh。変換は ime/server.py が動いている必要がある。
set -euo pipefail
cd "$(dirname "$0")"
APP=build/IXVKanji.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/ja.lproj" "$APP/Contents/Resources/en.lproj"
swiftc -O -module-name IXVKanji -framework Cocoa -framework InputMethodKit Sources/*.swift -o "$APP/Contents/MacOS/IXVKanji"
cp Info.plist "$APP/Contents/Info.plist"
swift tools/make_icon.swift "$APP/Contents/Resources/icon.tiff"
swift tools/make_icon.swift "$APP/Contents/Resources/icon-roman.tiff" a - A  # 英字のモード
for l in ja en; do
  printf '"CFBundleName" = "IXV かな漢字";\n"jp.co.elvez.inputmethod.IXVKanji.Japanese" = "IXV かな漢字（日本語）";\n"jp.co.elvez.inputmethod.IXVKanji.Roman" = "IXV かな漢字（英字）";\n' \
    > "$APP/Contents/Resources/$l.lproj/InfoPlist.strings"
done
codesign --force --sign - "$APP"
echo "built $APP"
