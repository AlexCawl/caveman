#!/bin/sh

case "$1" in
  caveman-activate.js|caveman-mode-tracker.js) ;;
  *) exit 0 ;;
esac

plugin_root=$CLAUDE_PLUGIN_ROOT
# Claude's Git Bash can supply /c/...; native node.exe needs c:/... (#199).
case "$plugin_root" in
  /[a-zA-Z]/*)
    drive=${plugin_root#/}
    drive=${drive%%/*}
    plugin_root=$drive:/${plugin_root#/*/}
    ;;
esac

exec node "$plugin_root/src/hooks/$1"
