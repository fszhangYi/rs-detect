#!/usr/bin/env bash
# Launch rs-detect Desktop UI (PyQt5 + PyQt-Fluent-Widgets)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "[rs-detect] warning: DISPLAY/WAYLAND_DISPLAY unset — GUI may fail on headless hosts" >&2
fi

# Prefer xcb unless user already chose a platform
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

echo "[rs-detect] root=$RS_DETECT_ROOT"
echo "[rs-detect] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[rs-detect] DISPLAY=${DISPLAY:-unset}  QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo "[rs-detect] starting Fluent desktop…"

exec python -m rs_detect.desktop.app
