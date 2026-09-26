#!/usr/bin/env bash
# =============================================================================
# run.sh -- Root launcher for Bulk Editor (Linux / macOS)
# Delegates to shorts_engine/run.sh with correct PYTHONPATH.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_SH="$SCRIPT_DIR/shorts_engine/run.sh"

echo ""
echo " ===================================================="
echo "   Batch Greek Shorts Processing Engine"
echo " ===================================================="
echo ""

if [[ ! -f "$ENGINE_SH" ]]; then
    echo " [ERROR] Could not find shorts_engine/run.sh"
    echo ""
    echo " Make sure the 'shorts_engine' folder exists next to this run.sh file."
    echo " If you downloaded the zip, re-extract it and try again."
    exit 1
fi

chmod +x "$ENGINE_SH"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/shorts_engine:${PYTHONPATH:-}"
exec bash "$ENGINE_SH" "$@"
