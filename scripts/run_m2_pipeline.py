#!/usr/bin/env python3
"""Run plan M2 flow: D435 coarse → D405 fine → gate."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rs_detect.service import get_service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="M2: D435 coarse then D405 fine + gate")
    ap.add_argument("--method", default="auto")
    ap.add_argument("--wait", type=float, default=0.6)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    svc = get_service()
    scan = svc.scan()
    wrist_c = scan.get("wrist_candidates") or []
    scene_c = scan.get("scene_candidates") or []
    if not wrist_c or not scene_c:
        print("need both wrist and scene candidates", file=sys.stderr)
        return 1
    wrist = (scan.get("selection") or {}).get("wrist") or wrist_c[0]["serial"]
    scene = (scan.get("selection") or {}).get("scene") or scene_c[0]["serial"]
    svc.bind_selection("wrist", wrist, persist=False)
    svc.bind_selection("scene", scene, persist=False)
    if not (svc.streams.get(wrist) and svc.streams.get(wrist).running):
        svc.start_stream(serial=wrist, role="wrist")
    if not (svc.streams.get(scene) and svc.streams.get(scene).running):
        svc.start_stream(serial=scene, role="scene")
    time.sleep(args.wait)

    payload = svc.run_m2_pipeline(method=args.method)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    svc.stop_all()
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
