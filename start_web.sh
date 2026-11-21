#!/usr/bin/env bash
# Launch rs-detect Web UI (FastAPI + Ant Design + Cell C Three.js arm)
# Prefers .venv312 (Python≥3.10) for sensors-arm FK/IK; falls back to .venv.
# Supports UI-triggered restart via .rs_detect_restart flag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  -p, --port PORT   Listen port (default: 8787, or RS_DETECT_WEB_PORT)
  -H, --host HOST   Bind address (default: 0.0.0.0, or RS_DETECT_WEB_HOST)
      --no-browser  Do not open a browser after start
  -h, --help        Show this help

Examples:
  ./start_web.sh
  ./start_web.sh -p 6666
  ./start_web.sh --port 6666 --host 127.0.0.1
EOF
}

HOST="${RS_DETECT_WEB_HOST:-0.0.0.0}"
PORT="${RS_DETECT_WEB_PORT:-8787}"
OPEN_BROWSER="${RS_DETECT_OPEN_BROWSER:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "[rs-detect] $1 requires a port number" >&2
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    -H|--host)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "[rs-detect] $1 requires a host" >&2
        exit 1
      fi
      HOST="$2"
      shift 2
      ;;
    --no-browser)
      OPEN_BROWSER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[rs-detect] unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "[rs-detect] invalid port: $PORT (need 1–65535)" >&2
  exit 1
fi
# Keep env in sync so child tools / UI restart see the same bind
export RS_DETECT_WEB_HOST="$HOST"
export RS_DETECT_WEB_PORT="$PORT"

# Sibling workspace packages (AutoDL or local multi-repo checkout)
WS_PARENT="$(cd "$ROOT/.." && pwd)"
_EXTRA_PP=""
for _sib in sensors sensors-arm; do
  if [[ -d "$WS_PARENT/$_sib/src" ]]; then
    _EXTRA_PP="${_EXTRA_PP:+$_EXTRA_PP:}$WS_PARENT/$_sib/src"
  fi
done
# Legacy AutoDL layout fallback
if [[ -z "$_EXTRA_PP" ]]; then
  for _p in /home/ubuntu/autodl-tmp/sensors/src /home/ubuntu/autodl-tmp/sensors-arm/src; do
    [[ -d "$_p" ]] && _EXTRA_PP="${_EXTRA_PP:+$_EXTRA_PP:}$_p"
  done
fi

if [[ -f "$ROOT/.venv312/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv312/bin/activate"
  export PATH="$ROOT/.venv312/bin:$PATH"
  export PYTHONPATH="$ROOT/src${_EXTRA_PP:+:$_EXTRA_PP}${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/env.sh"
  export PYTHONPATH="$ROOT/src${_EXTRA_PP:+:$_EXTRA_PP}${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "[rs-detect] missing .venv312 or .venv" >&2
  exit 1
fi
unset _EXTRA_PP _sib _p

URL="http://127.0.0.1:${PORT}/"
RESTART_FLAG="$ROOT/.rs_detect_restart"
WEB_DIR="$ROOT/web"
WEB_DIST="$WEB_DIR/dist"

# Vite + React frontend (same pattern as file-viewer): build → web/dist, FastAPI serves it.
build_web() {
  # Optional nvm (Linux lab images); ignore if absent (Windows Git Bash / plain Node).
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    export PATH="${HOME}/.nvm/versions/node/v20.20.2/bin:${PATH}"
    # shellcheck source=/dev/null
    . "$HOME/.nvm/nvm.sh"
    nvm use 20 >/dev/null 2>&1 || true
  fi
  echo "[rs-detect] building web UI (Vite)…"
  (
    cd "$WEB_DIR"
    if command -v pnpm >/dev/null 2>&1; then
      if [[ ! -d node_modules/vite ]]; then
        pnpm install --ignore-scripts
        # Linux CI/lab sometimes need an explicit rollup native binary
        if [[ "$(uname -s 2>/dev/null || true)" == "Linux" ]]; then
          pnpm add -D @rollup/rollup-linux-x64-gnu --ignore-scripts >/dev/null 2>&1 || true
        fi
      fi
      pnpm run build
    elif command -v npm >/dev/null 2>&1; then
      if [[ ! -d node_modules/vite ]]; then
        npm install
      fi
      npm run build
    else
      echo "[rs-detect] pnpm/npm not found — cannot build web UI. Install Node 20+." >&2
      return 1
    fi
  )
}

NEED_BUILD=0
if [[ "${RS_DETECT_WEB_REBUILD:-0}" == "1" ]]; then
  NEED_BUILD=1
elif [[ ! -f "$WEB_DIST/index.html" ]]; then
  NEED_BUILD=1
elif [[ -n "$(find "$WEB_DIR/src" "$WEB_DIR/index.html" "$WEB_DIR/package.json" -newer "$WEB_DIST/index.html" 2>/dev/null | head -1)" ]]; then
  NEED_BUILD=1
fi
if [[ "$NEED_BUILD" == "1" ]]; then
  build_web || exit 1
else
  echo "[rs-detect] web dist up to date → $WEB_DIST"
fi

# Cell C URDF + STL from sibling sensors-arm
python "$ROOT/scripts/ensure_arm_assets.py" || echo "[rs-detect] WARN: arm URDF assets missing" >&2

echo "[rs-detect] root=$ROOT"
echo "[rs-detect] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[rs-detect] web → ${URL}  arm tab → ${URL}#/arm"
echo "[rs-detect] web → bind ${HOST}:${PORT}"

open_browser_later() {
  sleep 3
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$URL" >/dev/null 2>&1 || true
  else
    python -c "import webbrowser; webbrowser.open('${URL}')" >/dev/null 2>&1 || true
  fi
}

# Open browser once after first boot (not on every UI restart)
if [[ "${OPEN_BROWSER}" == "1" ]]; then
  echo "[rs-detect] will open browser in 3s → ${URL}"
  open_browser_later &
fi

rm -f "$RESTART_FLAG"

while true; do
  set +e
  python -m uvicorn rs_detect.api.server:app --host "$HOST" --port "$PORT"
  code=$?
  set -e
  if [[ -f "$RESTART_FLAG" ]]; then
    rm -f "$RESTART_FLAG"
    echo "[rs-detect] restart requested — relaunching…"
    sleep 0.5
    continue
  fi
  exit "${code}"
done
