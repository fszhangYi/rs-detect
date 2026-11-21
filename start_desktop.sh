#!/usr/bin/env bash
# Launch rs-detect Desktop UI (PyQt5 + PyQt-Fluent-Widgets)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

# Linux/X11 lab: prefer xcb. Windows / macOS: leave Qt default (do not force xcb).
_UNAME="$(uname -s 2>/dev/null || echo unknown)"
if [[ "$_UNAME" == "Linux" ]]; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "[rs-detect] warning: DISPLAY/WAYLAND_DISPLAY unset — GUI may fail on headless hosts" >&2
  fi
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
elif [[ "${QT_QPA_PLATFORM:-}" == "xcb" ]]; then
  unset QT_QPA_PLATFORM
fi

echo "[rs-detect] root=$RS_DETECT_ROOT"
echo "[rs-detect] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[rs-detect] DISPLAY=${DISPLAY:-unset}  QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-default}"
echo "[rs-detect] starting Fluent desktop…"

exec python -m rs_detect.desktop.app
