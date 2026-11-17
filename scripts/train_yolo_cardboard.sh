#!/usr/bin/env bash
# Train cardboard YOLO11 instance-seg and install best.pt → models/cardboard_yolo.pt
#
# Usage:
#   ./scripts/train_yolo_cardboard.sh
#   ./scripts/train_yolo_cardboard.sh --epochs 80 --imgsz 640 --device 0
#   ./scripts/train_yolo_cardboard.sh --dry-run   # print command only
#
# Requires: ./scripts/setup_yolo_env.sh  and labeled data under data/yolo_seg/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS=50
IMGSZ=640
BATCH=8
DEVICE=""          # empty → auto (cuda:0 if available else cpu)
MODEL="models/yolo11n-seg.pt"
DATA="data/yolo_seg/dataset.yaml"
PROJECT="runs/segment"
NAME="cardboard_seg"
TASK="segment"
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz)  IMGSZ="$2"; shift 2 ;;
    --batch)  BATCH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --data)   DATA="$2"; shift 2 ;;
    --name)   NAME="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --task)   TASK="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    --)
      shift
      EXTRA+=("$@")
      break
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ -f "$ROOT/.venv312/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv312/bin/activate"
  PY="$ROOT/.venv312/bin/python"
  YOLO_BIN="$ROOT/.venv312/bin/yolo"
elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PY="$ROOT/.venv/bin/python"
  YOLO_BIN="$ROOT/.venv/bin/yolo"
else
  echo "[train] missing .venv312 or .venv — run ./scripts/setup_yolo_env.sh first" >&2
  exit 1
fi

if [[ ! -x "$YOLO_BIN" ]] || ! "$PY" -c "import ultralytics" 2>/dev/null; then
  echo "[train] ultralytics / yolo CLI missing — run ./scripts/setup_yolo_env.sh" >&2
  exit 1
fi

if [[ ! -f "$ROOT/$MODEL" && ! -f "$MODEL" ]]; then
  echo "[train] missing base weights: $MODEL" >&2
  echo "        run ./scripts/setup_yolo_env.sh to fetch models/yolo11n-seg.pt" >&2
  echo "        or pass a hub name: --model yolo11n-seg.pt" >&2
  # Allow bare hub stem (Ultralytics will download)
  if [[ "$MODEL" != *.pt ]] || [[ "$MODEL" == */* ]]; then
    exit 1
  fi
fi

if [[ ! -f "$ROOT/$DATA" && ! -f "$DATA" ]]; then
  echo "[train] missing dataset yaml: $DATA" >&2
  exit 1
fi

n_train="$("$PY" - <<'PY'
from pathlib import Path
root = Path("data/yolo_seg/images/train")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
print(sum(1 for p in root.rglob("*") if p.suffix.lower() in exts) if root.is_dir() else 0)
PY
)"
if [[ "${n_train}" -lt 1 ]]; then
  echo "[train] no images in data/yolo_seg/images/train — add polygon-labeled photos first" >&2
  exit 1
fi
echo "[train] train images: ${n_train}"

if [[ -z "$DEVICE" ]]; then
  DEVICE="$("$PY" -c 'import torch; print("0" if torch.cuda.is_available() else "cpu")')"
fi
echo "[train] task=${TASK} device=${DEVICE} epochs=${EPOCHS} imgsz=${IMGSZ} batch=${BATCH}"

CMD=(
  "$YOLO_BIN" "$TASK" train
  "data=${DATA}"
  "model=${MODEL}"
  "epochs=${EPOCHS}"
  "imgsz=${IMGSZ}"
  "batch=${BATCH}"
  "device=${DEVICE}"
  "project=${PROJECT}"
  "name=${NAME}"
  "exist_ok=True"
)
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  CMD+=("${EXTRA[@]}")
fi

echo "[train] ${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${CMD[@]}"

BEST="${PROJECT}/${NAME}/weights/best.pt"
LAST="${PROJECT}/${NAME}/weights/last.pt"
OUT="models/cardboard_yolo.pt"
mkdir -p models

if [[ -f "$BEST" ]]; then
  cp -f "$BEST" "$OUT"
  echo "[train] installed ${BEST} → ${OUT}"
elif [[ -f "$LAST" ]]; then
  cp -f "$LAST" "$OUT"
  echo "[train] warn: best.pt missing; installed ${LAST} → ${OUT}"
else
  echo "[train] FAIL: no weights under ${PROJECT}/${NAME}/weights/" >&2
  exit 1
fi

echo "[train] done. config/yolo.yaml already points at models/cardboard_yolo.pt"
echo "[train] optional: set method: yolo  (or leave auto) and device: ${DEVICE}"
