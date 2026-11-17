#!/usr/bin/env bash
# Install YOLO (Ultralytics) into the project venv and verify imports.
# Usage: ./scripts/setup_yolo_env.sh [--cuda]
#   default: CPU torch wheels (safe on machines without NVIDIA)
#   --cuda : skip torch install here; assume CUDA torch already present
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WANT_CUDA=0
[[ "${1:-}" == "--cuda" ]] && WANT_CUDA=1

if [[ -f "$ROOT/.venv312/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv312/bin/activate"
  PY="$ROOT/.venv312/bin/python"
  PIP="$ROOT/.venv312/bin/pip"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PY="$ROOT/.venv/bin/python"
  PIP="$ROOT/.venv/bin/pip"
else
  echo "[yolo] missing .venv312 or .venv — create one first" >&2
  exit 1
fi

echo "[yolo] python=$($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PIP" install -U pip setuptools wheel

if [[ "$WANT_CUDA" -eq 0 ]]; then
  echo "[yolo] installing CPU torch + torchvision …"
  "$PIP" install --index-url https://download.pytorch.org/whl/cpu torch torchvision
else
  echo "[yolo] --cuda: expecting existing CUDA torch; not installing torch wheels"
fi

echo "[yolo] installing ultralytics + editable rs-detect …"
"$PIP" install "ultralytics>=8.3.0"
"$PIP" install -e .

mkdir -p "$ROOT/models" \
  "$ROOT/data/yolo/images/train" \
  "$ROOT/data/yolo/images/val" \
  "$ROOT/data/yolo/labels/train" \
  "$ROOT/data/yolo/labels/val" \
  "$ROOT/data/yolo_seg/images/train" \
  "$ROOT/data/yolo_seg/images/val" \
  "$ROOT/data/yolo_seg/labels/train" \
  "$ROOT/data/yolo_seg/labels/val"

download_pt() {
  local dest="$1"
  local url="$2"
  local name
  name="$(basename "$dest")"
  if [[ -f "$dest" ]]; then
    echo "[yolo] ${name} already present ($(wc -c < "$dest") bytes)"
    return 0
  fi
  echo "[yolo] ensuring bootstrap weights ${name} …"
  # GitHub releases often time out on AutoDL CN — curl mirrors first, then Ultralytics.
  for try in \
    "https://ghproxy.net/${url}" \
    "https://ghfast.top/${url}" \
    "${url}"
  do
    echo "[yolo] try download: $try"
    if curl -fsSL --connect-timeout 15 --max-time 180 -o "${dest}.tmp" "$try"; then
      if head -c 2 "${dest}.tmp" | grep -q PK; then
        mv -f "${dest}.tmp" "$dest"
        echo "[yolo] saved ${name} ($(wc -c < "$dest") bytes)"
        return 0
      fi
    fi
    rm -f "${dest}.tmp"
  done
  return 1
}

download_pt "$ROOT/models/yolov8n.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" || true
download_pt "$ROOT/models/yolo11n-seg.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt" || true

"$PY" - <<'PY'
from pathlib import Path
import shutil

root = Path.cwd()


def ensure(name: str) -> None:
    out = root / "models" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("bootstrap", out, "exists=", True, "size=", out.stat().st_size)
        return
    from ultralytics import YOLO

    model = YOLO(name)
    for cand in (
        Path(name),
        Path.home() / ".cache" / "ultralytics" / name,
    ):
        if cand.exists():
            shutil.copy2(cand, out)
            break
    if not out.exists():
        try:
            model.save(str(out))
        except Exception as exc:
            print("warn: could not copy weights to models/:", name, exc)
    print("bootstrap", out, "exists=", out.exists())


ensure("yolov8n.pt")
ensure("yolo11n-seg.pt")
PY

"$PY" "$ROOT/scripts/check_yolo_env.py"
echo "[yolo] done."
echo "[yolo] next: label data under data/yolo_seg/ → ./scripts/train_yolo_cardboard.sh"
echo "[yolo] GPU machine tip: ./scripts/setup_yolo_env.sh --cuda  (after installing CUDA torch)"
