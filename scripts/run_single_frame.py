#!/usr/bin/env python3
"""M2 single-frame CLI: detect → mask cloud → RANSAC suction point.

Examples
--------
# Live mock / RealSense (start stream via service selection)
python scripts/run_single_frame.py --role wrist --method auto

# Offline from a capture folder (color.png + depth.png + meta.json)
python scripts/run_single_frame.py --capture data/captures/<stamp>_MOCK_D405 --method mock

# Explicit images
python scripts/run_single_frame.py --color c.png --depth d.png --fx 576 --fy 576 --ppx 320 --ppy 240
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rs_detect.camera.models import FramePacket, Intrinsics  # noqa: E402
from rs_detect.detect.pipeline import run_single_frame  # noqa: E402
from rs_detect.paths import ROOT as PKG_ROOT, YOLO_CONFIG_PATH  # noqa: E402
from rs_detect.service import get_service  # noqa: E402


def _load_yolo_cfg() -> dict:
    if YOLO_CONFIG_PATH.exists():
        return yaml.safe_load(YOLO_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


def _resolve_capture_paths(capture: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Accept capture prefix, color path, or folder with color.png."""
    capture = Path(capture)
    if capture.is_dir():
        color_path = capture / "color.png"
        depth_path = capture / "depth.png"
        meta_path = capture / "meta.json"
        if not color_path.exists():
            raise FileNotFoundError(f"missing color.png under {capture}")
        return color_path, depth_path if depth_path.exists() else None, meta_path if meta_path.exists() else None

    name = capture.name
    if name.endswith("_color.png") or name.endswith("_color.jpg"):
        stem = capture.name.rsplit("_color.", 1)[0]
        base = capture.with_name(stem)
    elif capture.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        # treat as color image; sibling depth/meta optional
        return capture, None, None
    else:
        base = capture

    color_path = Path(str(base) + "_color.png")
    if not color_path.exists():
        color_path = Path(str(base) + "_color.jpg")
    depth_path = Path(str(base) + "_depth.png")
    meta_path = Path(str(base) + "_meta.json")
    if not color_path.exists():
        raise FileNotFoundError(f"missing capture color for prefix {base}")
    return (
        color_path,
        depth_path if depth_path.exists() else None,
        meta_path if meta_path.exists() else None,
    )


def _packet_from_capture(capture: Path) -> FramePacket:
    color_path, depth_path, meta_path = _resolve_capture_paths(capture)
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"failed to read {color_path}")
    depth = None
    if depth_path is not None:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is not None and depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
    meta = {}
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    serial = str(meta.get("serial") or color_path.stem)
    intr = None
    ci = meta.get("color_intrinsics") or meta.get("intrinsics") or {}
    if ci and "fx" in ci:
        intr = Intrinsics(
            width=int(ci.get("width") or color.shape[1]),
            height=int(ci.get("height") or color.shape[0]),
            fx=float(ci["fx"]),
            fy=float(ci["fy"]),
            ppx=float(ci["ppx"]),
            ppy=float(ci["ppy"]),
            model=str(ci.get("model") or ""),
            coeffs=list(ci.get("coeffs") or []),
            stream="color",
        )
    return FramePacket(
        serial=serial,
        color_bgr=color,
        depth_u16=depth,
        timestamp_ms=float(meta.get("timestamp_ms") or 0.0),
        color_intrinsics=intr,
        depth_intrinsics=None,
    )


def _packet_from_images(
    color_path: Path,
    depth_path: Path,
    *,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    serial: str = "offline",
) -> FramePacket:
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise RuntimeError("failed to read color/depth images")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    h, w = color.shape[:2]
    intr = Intrinsics(width=w, height=h, fx=fx, fy=fy, ppx=ppx, ppy=ppy, stream="color")
    return FramePacket(
        serial=serial,
        color_bgr=color,
        depth_u16=depth,
        timestamp_ms=0.0,
        color_intrinsics=intr,
    )


def _packet_from_live(role: str, serial: Optional[str], wait_s: float = 2.0) -> Tuple[FramePacket, bool]:
    svc = get_service()
    scan = svc.scan()
    if not serial:
        serial = (scan.get("selection") or {}).get(role) or None
    if not serial:
        cands = scan.get("wrist_candidates" if role == "wrist" else "scene_candidates") or []
        if not cands:
            raise RuntimeError("no cameras found")
        serial = cands[0]["serial"]
    svc.bind_selection(role, serial, persist=False)
    st = svc.streams.get(serial)
    if st is None or not st.running:
        svc.start_stream(serial=serial, role=role, width=640, height=480, fps=30)
    deadline = time.time() + wait_s
    pkt = None
    while time.time() < deadline:
        pkt = svc.streams.get(serial).get_latest() if svc.streams.get(serial) else None
        if pkt is not None:
            break
        time.sleep(0.05)
    if pkt is None:
        raise RuntimeError("no live frame")
    is_mock = bool(svc.streams.get(serial) and svc.streams.get(serial).is_mock)
    return pkt, is_mock


def main() -> int:
    ap = argparse.ArgumentParser(description="M2 single-frame cardboard → suction point")
    ap.add_argument("--role", choices=["wrist", "scene"], default="wrist")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--method", default=None, help="auto|yolo|hsv|mock (default from config/yolo.yaml)")
    ap.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="capture prefix (…_color.png) or folder with color.png/depth.png/meta.json",
    )
    ap.add_argument("--color", type=Path, default=None)
    ap.add_argument("--depth", type=Path, default=None)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--ppx", type=float, default=None)
    ap.add_argument("--ppy", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write viz jpeg here")
    ap.add_argument("--json-out", type=Path, default=None, help="write result json here")
    args = ap.parse_args()

    cfg = _load_yolo_cfg()
    method = args.method or cfg.get("method") or "auto"
    weights = cfg.get("weights") or "models/cardboard_yolo.pt"
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = PKG_ROOT / weights_path

    is_mock = False
    if args.capture:
        packet = _packet_from_capture(args.capture)
        is_mock = packet.serial.startswith("MOCK_")
    elif args.color and args.depth:
        if None in (args.fx, args.fy, args.ppx, args.ppy):
            ap.error("--fx --fy --ppx --ppy required with --color/--depth")
        packet = _packet_from_images(
            args.color,
            args.depth,
            fx=args.fx,
            fy=args.fy,
            ppx=args.ppx,
            ppy=args.ppy,
        )
    else:
        packet, is_mock = _packet_from_live(args.role, args.serial)

    result = run_single_frame(
        packet,
        method=method,
        weights=weights_path,
        conf=float(cfg.get("conf") or 0.25),
        class_filter=cfg.get("class_filter") or None,
        is_mock=is_mock or method == "mock",
        depth_scale=float(cfg.get("depth_scale") or 0.001),
        ransac_thresh=float(cfg.get("ransac_thresh_m") or 0.008),
        max_points=int(cfg.get("max_points") or 20000),
        make_viz=True,
    )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out and result.viz_jpeg:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(result.viz_jpeg)
        print(f"wrote viz: {args.out}", file=sys.stderr)
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
