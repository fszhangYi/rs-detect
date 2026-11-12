#!/usr/bin/env bash
# Compat wrapper → project root start_web.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/start_web.sh" "$@"
