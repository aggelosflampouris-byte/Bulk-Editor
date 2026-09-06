#!/usr/bin/env bash
# =============================================================================
# run.sh — 1-Click Linux/macOS Launcher
# Responsibilities:
#   1. Verify Python 3.10+
#   2. Ensure FFmpeg is available (system or downloaded via apt/brew)
#   3. Create/reuse a .venv virtual environment inside this folder
#   4. Install Python dependencies into the venv
#   5. Launch Streamlit in the browser
# =============================================================================
set -euo pipefail

# ── Resolve the directory this script lives in (handles spaces in paths) ──────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
APP_FILE="$SCRIPT_DIR/app.py"
BIN_DIR="$SCRIPT_DIR/bin"

print_banner() {
    echo ""
    echo " ===================================================="
    echo "   Batch Greek Shorts Processing Engine"
    echo "   1-Click Launcher"
    echo " ===================================================="
    echo ""
}

# ── Step 1: Locate python3 (3.10+) ───────────────────────────────────────────
check_python() {
    local py_bin

    # Prefer python3 on PATH; fall back to python if it maps to python3
    if command -v python3 &>/dev/null; then
        py_bin="python3"
    elif command -v python &>/dev/null; then
        py_bin="python"
    else
        echo "[ERROR] Python 3.10+ is not installed or not on PATH."
        echo "        Install it with:"
        echo "          sudo apt install python3 python3-venv python3-pip   # Debian/Ubuntu"
        echo "          brew install python                                  # macOS"
        exit 1
    fi

    local version
    version="$("$py_bin" --version 2>&1 | awk '{print $2}')"
    local major minor
    major="$(echo "$version" | cut -d. -f1)"
    minor="$(echo "$version" | cut -d. -f2)"

    if [[ "$major" -lt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -lt 10 ]]; }; then
        echo "[ERROR] Python 3.10+ required. Found: $version"
        exit 1
    fi

    echo "[OK] Python $version detected."
    PYTHON_BIN="$py_bin"
}

# ── Step 2: Ensure FFmpeg is available ───────────────────────────────────────
check_ffmpeg() {
    # Highest priority: bundled binary in ./bin/
    if [[ -x "$BIN_DIR/ffmpeg" ]]; then
        export PATH="$BIN_DIR:$PATH"
        echo "[OK] Bundled FFmpeg found."
        return
    fi

    # Second: system FFmpeg on PATH
    if command -v ffmpeg &>/dev/null; then
        echo "[OK] System FFmpeg detected."
        return
    fi

    # Attempt auto-install (requires sudo; user will be prompted once)
    echo "[INFO] FFmpeg not found. Attempting automatic installation..."
    echo ""

    if command -v apt-get &>/dev/null; then
        echo "      Running: sudo apt-get install -y ffmpeg"
        sudo apt-get install -y ffmpeg
    elif command -v brew &>/dev/null; then
        echo "      Running: brew install ffmpeg"
        brew install ffmpeg
    elif command -v dnf &>/dev/null; then
        echo "      Running: sudo dnf install -y ffmpeg"
        sudo dnf install -y ffmpeg
    else
        echo "[ERROR] Cannot install FFmpeg automatically."
        echo "        Please install it manually and re-run this script."
        echo "          Debian/Ubuntu: sudo apt install ffmpeg"
        echo "          macOS:         brew install ffmpeg"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        echo "[ERROR] FFmpeg installation failed. Please install it manually."
        exit 1
    fi

    echo "[OK] FFmpeg installed."
}

# ── Step 3: Create virtual environment ───────────────────────────────────────
setup_venv() {
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        echo "[INFO] Creating Python virtual environment in .venv/ ..."

        # Ensure python3-venv is available (Debian/Ubuntu ships it separately)
        if ! "$PYTHON_BIN" -m venv --help &>/dev/null; then
            echo "[INFO] Installing python3-venv..."
            sudo apt-get install -y python3-venv
        fi

        "$PYTHON_BIN" -m venv "$VENV_DIR"
        echo "[OK] Virtual environment created."
    else
        echo "[OK] Virtual environment already exists."
    fi
}

# ── Step 4: Install/upgrade dependencies ─────────────────────────────────────
install_deps() {
    echo "[INFO] Installing / updating Python dependencies..."

    # Use the venv's pip directly — no activation required
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$REQ_FILE"

    echo "[OK] Dependencies installed."
}

# ── Step 5: Launch Streamlit ──────────────────────────────────────────────────
launch_app() {
    echo ""
    echo " Starting Streamlit... your browser will open automatically."
    echo " Press Ctrl+C in this window to stop the server."
    echo ""

    "$VENV_DIR/bin/streamlit" run "$APP_FILE" \
        --server.headless false \
        --browser.gatherUsageStats false
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    print_banner
    check_python
    check_ffmpeg
    setup_venv
    install_deps
    launch_app
}

main "$@"
