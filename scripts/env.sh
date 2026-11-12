#!/usr/bin/env bash
# Shared environment for rs-detect launchers.
# Usage: source "$(dirname "$0")/env.sh"   OR   source "$ROOT/scripts/env.sh"

# Resolve repo root (this file lives in scripts/)
_RS_DETECT_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RS_DETECT_ROOT="$(cd "$_RS_DETECT_ENV_DIR/.." && pwd)"
unset _RS_DETECT_ENV_DIR

cd "$RS_DETECT_ROOT" || return 1 2>/dev/null || exit 1

# Prefer project venv; fail clearly if missing
if [[ ! -f "$RS_DETECT_ROOT/.venv/bin/activate" ]]; then
  echo "[rs-detect] missing venv: $RS_DETECT_ROOT/.venv" >&2
  echo "[rs-detect] create with: /usr/bin/python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e ." >&2
  return 1 2>/dev/null || exit 1
fi

# Drop conda from PATH if present (avoids mixing with system PyQt / ROS python)
if [[ -n "${CONDA_PREFIX:-}" ]] || [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  conda deactivate 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "$RS_DETECT_ROOT/.venv/bin/activate"

# Ensure system libs (PyQt5 from apt) remain visible when venv uses --system-site-packages
export PATH="$RS_DETECT_ROOT/.venv/bin:/usr/bin:/bin:${PATH}"

# Package import path
export PYTHONPATH="$RS_DETECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Quiet Qt / OpenCV noise unless debugging
export QT_LOGGING_RULES="${QT_LOGGING_RULES:-*=false}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"

# Data / config locations (optional overrides)
export RS_DETECT_ROOT
export RS_DETECT_CONFIG_DIR="${RS_DETECT_CONFIG_DIR:-$RS_DETECT_ROOT/config}"
export RS_DETECT_DATA_DIR="${RS_DETECT_DATA_DIR:-$RS_DETECT_ROOT/data}"

# Web defaults
export RS_DETECT_WEB_HOST="${RS_DETECT_WEB_HOST:-0.0.0.0}"
export RS_DETECT_WEB_PORT="${RS_DETECT_WEB_PORT:-8787}"
