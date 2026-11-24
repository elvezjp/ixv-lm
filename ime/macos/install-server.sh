#!/bin/bash
# 変換サーバーを、ログインしたときに自動で動かす（launchd の LaunchAgent）。止まっても launchd が動かし直す。
# 置くもの：~/Library/LaunchAgents/jp.co.elvez.ixvkanji.server.plist。外すのは uninstall-server.sh。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL=jp.co.elvez.ixvkanji.server
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$REPO/ime/run_server.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME/ixv-lm-data/ime/server.out</string>
  <key>StandardErrorPath</key><string>$HOME/ixv-lm-data/ime/server.out</string>
</dict>
</plist>
PL
pkill -f ime/server.py || true
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $PLIST"
