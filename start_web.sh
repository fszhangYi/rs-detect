#!/usr/bin/env bash
# Launch rs-detect Web UI (FastAPI + Ant Design)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

HOST="${RS_DETECT_WEB_HOST:-0.0.0.0}"
PORT="${RS_DETECT_WEB_PORT:-8787}"

echo "[rs-detect] root=$RS_DETECT_ROOT"
echo "[rs-detect] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[rs-detect] web → http://127.0.0.1:${PORT}/  (bind ${HOST}:${PORT})"

exec python -m uvicorn rs_detect.api.server:app --host "$HOST" --port "$PORT"
