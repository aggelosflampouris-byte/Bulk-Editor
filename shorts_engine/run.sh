#!/usr/bin/env bash
# =============================================================================
# run.sh -- 1-Click Linux / macOS Launcher for Bulk Editor
#
# Steps:
#   1. Find Python 3.10+
#   2. Find or install FFmpeg (via apt, brew, dnf, or static download)
#   3. Create / reuse .venv virtual environment
#   4. Install Python dependencies
#   5. Copy .env.example -> .env if not present
#   6. Launch Streamlit
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

# ── Resolve the directory this script lives in (handles spaces in path) ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
APP_FILE="$SCRIPT_DIR/app.py"
BIN_DIR="$SCRIPT_DIR/bin"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

PYTHON_BIN=""

# ─────────────────────────────────────────────────────────────────────────────
print_banner() {
    echo ""
    echo " ===================================================="
    echo "   Batch Greek Shorts Processing Engine"
    echo "   1-Click Launcher"
    echo " ===================================================="
    echo ""
}

# ── Step 1: Locate Python 3.10+ ───────────────────────────────────────────────
check_python() {
    echo "[1/6] Checking Python version..."

    local candidates=("python3.12" "python3.11" "python3.10" "python3" "python")
    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver="$("$candidate" --version 2>&1 | awk '{print $2}')"
            local major minor
            major="$(echo "$ver" | cut -d. -f1)"
            minor="$(echo "$ver" | cut -d. -f2)"
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                PYTHON_BIN="$candidate"
                echo " [OK] Python $ver ($candidate)"
                return 0
            fi
        fi
    done

    echo ""
    echo " [ERROR] Python 3.10+ not found."
    echo ""
    echo "  Install it with:"
    echo "    sudo apt install python3.11 python3.11-venv   # Debian/Ubuntu"
    echo "    brew install python                            # macOS"
    exit 1
}

# ── Step 2: Find or install FFmpeg ────────────────────────────────────────────
check_ffmpeg() {
    echo ""
    echo "[2/6] Checking FFmpeg..."

    # Bundled binary has highest priority
    if [[ -x "$BIN_DIR/ffmpeg" ]]; then
        export PATH="$BIN_DIR:$PATH"
        echo " [OK] Bundled FFmpeg found in bin/"
        return 0
    fi

    # System ffmpeg
    if command -v ffmpeg &>/dev/null; then
        echo " [OK] System FFmpeg found: $(command -v ffmpeg)"
        return 0
    fi

    echo " [INFO] FFmpeg not found. Attempting automatic installation..."
    echo ""

    if command -v apt-get &>/dev/null; then
        echo "       Running: sudo apt-get install -y ffmpeg"
        sudo apt-get install -y ffmpeg
    elif command -v brew &>/dev/null; then
        echo "       Running: brew install ffmpeg"
        brew install ffmpeg
    elif command -v dnf &>/dev/null; then
        echo "       Running: sudo dnf install -y ffmpeg"
        sudo dnf install -y ffmpeg
    elif command -v pacman &>/dev/null; then
        echo "       Running: sudo pacman -S --noconfirm ffmpeg"
        sudo pacman -S --noconfirm ffmpeg
    else
        echo " [ERROR] Cannot auto-install FFmpeg. Please install it manually:"
        echo "    Debian/Ubuntu: sudo apt install ffmpeg"
        echo "    macOS:         brew install ffmpeg"
        echo "    Arch:          sudo pacman -S ffmpeg"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        echo " [ERROR] FFmpeg installation failed. Please install it manually."
        exit 1
    fi

    echo " [OK] FFmpeg installed."
}

# ── Step 3: Create virtual environment ────────────────────────────────────────
setup_venv() {
    echo ""
    echo "[3/6] Setting up Python virtual environment..."

    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        echo " [OK] Virtual environment already exists."
        return 0
    fi

    # Ensure python3-venv package is available (Debian/Ubuntu)
    if ! "$PYTHON_BIN" -m venv --help &>/dev/null; then
        echo " [INFO] Installing python3-venv..."
        sudo apt-get install -y python3-venv
    fi

    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo " [OK] Virtual environment created."
}

# ── Step 4: Install dependencies ──────────────────────────────────────────────
install_deps() {
    echo ""
    echo "[4/6] Installing Python dependencies (first run may take a few minutes)..."

    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install -r "$REQ_FILE"

    echo " [OK] Dependencies installed."
}

# ── Step 5: Ensure .env exists ────────────────────────────────────────────────
ensure_env() {
    echo ""
    echo "[5/6] Checking configuration..."

    if [[ ! -f "$ENV_FILE" ]] && [[ -f "$ENV_EXAMPLE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo " [OK] Created .env from .env.example -- please add your API keys."
    else
        echo " [OK] .env file found."
    fi
}

# ── Step 6: Launch Streamlit ──────────────────────────────────────────────────
launch_app() {
    echo ""
    echo "[6/6] Launching Streamlit..."
    echo ""
    echo " ===================================================="
    echo "   App is starting! Your browser will open shortly."
    echo "   Press Ctrl+C in this window to stop the server."
    echo " ===================================================="
    echo ""

    export PYTHONPATH="$SCRIPT_DIR:$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || echo "$SCRIPT_DIR"):${PYTHONPATH:-}"
    exec "$VENV_DIR/bin/streamlit" run "$APP_FILE" \
        --server.headless false \
        --browser.gatherUsageStats false
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    print_banner
    check_python
    check_ffmpeg
    setup_venv
    install_deps
    ensure_env
    launch_app
}

main "$@"
