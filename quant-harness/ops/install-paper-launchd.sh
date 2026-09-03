#!/usr/bin/env bash
# Install (or refresh) the launchd agent that runs the daily paper tick.
#
# Idempotent: copies ops/launchd/com.quant-ai-lab.paper.plist into
# ~/Library/LaunchAgents, unloads any previous copy, loads the new one and
# prints the agent status. It never edits the crontab - it only prints the
# cron line you should delete yourself once launchd is confirmed working.
#
# Usage:
#   ops/install-paper-launchd.sh            # install / refresh, next run at 09:00
#   ops/install-paper-launchd.sh --now      # ... and kick one run right away
#   ops/install-paper-launchd.sh --uninstall
#
# `--now` is safe to use after this morning's cron tick: run_paper.py exits 0
# with "tick already taken today" instead of double-counting.
set -euo pipefail

LABEL="com.quant-ai-lab.paper"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/launchd/$LABEL.plist"
DST_DIR="$HOME/Library/LaunchAgents"
DST="$DST_DIR/$LABEL.plist"
DOMAIN="gui/$(id -u)"
CRON_LINE='0 9 * * * cd /Users/art/quant-ai-lab/quant-harness && ./.venv/bin/python run_paper.py >> reports/paper.log 2>&1'

KICKSTART=0
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --now) KICKSTART=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "launchd is macOS-only; nothing to do on $(uname -s)" >&2
    exit 1
fi

status() {
    # `launchctl print` is verbose; keep the lines an operator actually needs.
    launchctl print "$DOMAIN/$LABEL" 2>/dev/null \
        | grep -E 'state =|last exit code|program =|run interval|^\s*Hour|^\s*Minute' \
        || echo "  (agent not loaded)"
}

if [[ "$UNINSTALL" == 1 ]]; then
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$DST"
    echo "removed $DST and unloaded $LABEL"
    exit 0
fi

[[ -f "$SRC" ]] || { echo "plist not found: $SRC" >&2; exit 1; }
plutil -lint "$SRC" >/dev/null

mkdir -p "$DST_DIR"
if cmp -s "$SRC" "$DST"; then
    echo "plist unchanged: $DST"
else
    cp "$SRC" "$DST"
    echo "installed $DST"
fi

# bootout fails when the agent is not loaded yet - that is the normal first run.
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$DST"
launchctl enable "$DOMAIN/$LABEL"
echo "loaded $LABEL into $DOMAIN (next run: 09:00 local, or on wake if missed)"

if [[ "$KICKSTART" == 1 ]]; then
    launchctl kickstart "$DOMAIN/$LABEL"
    echo "kickstarted $LABEL - check: tail -20 /Users/art/quant-ai-lab/quant-harness/reports/paper.log"
fi

echo
echo "status:"
status

cat <<EOF

Next steps (manual, not done by this script):
  1. After the first successful launchd tick, remove the cron entry with 'crontab -e'.
     The line to delete is:
       $CRON_LINE
     Until then both schedulers may fire; run_paper.py takes at most one tick per 20h.
  2. Verify tomorrow: grep '=== PAPER TICK' reports/paper.log | tail -1
  3. Undo everything: $0 --uninstall
EOF
