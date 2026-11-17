#!/usr/bin/env python3
"""Smoke-check YOLO environment for rs-detect."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("python", sys.version.split()[0], sys.executable)

    try:
        import torch
    except ImportError as exc:
        print("FAIL torch:", exc)
        return 1
    print("torch", torch.__version__, "cuda_available=", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device0=", torch.cuda.get_device_name(0))

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("FAIL ultralytics:", exc)
        print("  fix: pip install -e '.[yolo]'  (or ./scripts/setup_yolo_env.sh)")
        return 1
    print("ultralytics", ultralytics.__version__)

    # Prefer project bootstrap weight; else download once
    candidates = [
        ROOT / "models" / "yolo11n-seg.pt",
        ROOT / "models" / "cardboard_yolo.pt",
        ROOT / "models" / "yolov8n.pt",
        ROOT / "yolo11n-seg.pt",
        ROOT / "yolov8n.pt",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        print("INFO no local .pt yet — loading yolov8n.pt (may download)")
        model = YOLO("yolov8n.pt")
        out = ROOT / "models" / "yolov8n.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.save(str(out))
        except Exception:
            pass
        weights = out if out.exists() else Path("yolov8n.pt")
    else:
        model = YOLO(str(weights))
    print("weights", weights)

    # Tiny inference smoke (random image)
    import numpy as np

    img = np.zeros((320, 320, 3), dtype=np.uint8)
    img[80:240, 80:240] = (40, 90, 180)
    res = model.predict(source=img, conf=0.25, imgsz=320, verbose=False)
    print("predict_ok", bool(res), "boxes=", 0 if not res or res[0].boxes is None else len(res[0].boxes))

    # rs-detect detector path
    sys.path.insert(0, str(ROOT / "src"))
    from rs_detect.detect.detector import detect_cardboard

    det = detect_cardboard(img, method="yolo", weights=weights if Path(weights).exists() else None)
    print("rs_detect.detect", det.method, det.message)

    cfg = ROOT / "config" / "yolo.yaml"
    print("config", cfg, "exists=", cfg.exists())
    print("YOLO_ENV_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
