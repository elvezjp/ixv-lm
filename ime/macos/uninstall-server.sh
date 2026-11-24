#!/bin/bash
# install-server.sh で置いた自動起動を外す。
LABEL=jp.co.elvez.ixvkanji.server
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "removed"
