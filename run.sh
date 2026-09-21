#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "===================================================="
echo "  Batch Greek Shorts Processing Engine"
echo "  Launching from root directory..."
echo "===================================================="
echo ""

if [[ ! -d "$SCRIPT_DIR/shorts_engine" ]] || [[ ! -f "$SCRIPT_DIR/shorts_engine/run.sh" ]]; then
    echo "[ERROR] Could not find shorts_engine/run.sh."
    echo "Please make sure the shorts_engine folder exists in the project root."
    exit 1
fi

chmod +x "$SCRIPT_DIR/shorts_engine/run.sh" 2>/dev/null || true
cd "$SCRIPT_DIR/shorts_engine"
exec bash run.sh "$@"
