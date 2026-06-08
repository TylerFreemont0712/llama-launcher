#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  launch_llama_server.sh
#  Activates ~/venv and launches the llama.cpp server GUI.
#  Called by llama_server.desktop — don't need to run this directly.
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${HOME}/venv"
PYTHON_SCRIPT="${SCRIPT_DIR}/server_launcher.py"

# ── Sanity checks ────────────────────────────────────────────────

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV_PATH}"
    echo "Create it:    python3 -m venv ~/venv"
    echo "Install deps: ~/venv/bin/pip install PyQt6"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: GUI script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

# ── Activate venv ────────────────────────────────────────────────

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

# ── Check PyQt6 is available ─────────────────────────────────────

if ! python -c "import PyQt6" 2>/dev/null; then
    echo "PyQt6 not found in venv. Installing..."
    pip install PyQt6
fi

# ── Launch ───────────────────────────────────────────────────────

exec python "${PYTHON_SCRIPT}" "$@"
