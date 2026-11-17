"""Pack rs-detect FramePacket → grasp-service FrameUpload JSON fields."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

from rs_detect.camera.models import FramePacket, Intrinsics
from rs_detect.paths import CAPTURE_DIR, MOCK_CONFIG_PATH, ROOT

STOCK_MOCK_ID = "__stock__"


def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _resolve_repo_path(raw: str) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _rel_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def list_capture_frames(capture_dir: Optional[Path] = None) -> list:
    """List ``data/capture`` RGB-D dumps (stem = serial_YYYYMMDD_HHMMSS_mmm)."""
    root = Path(capture_dir) if capture_dir is not None else CAPTURE_DIR
    out: list = []
    if not root.is_dir():
        return out
    for meta_path in sorted(root.glob("*_meta.json"), reverse=True):
        stem = meta_path.name[: -len("_meta.json")]
        color = root / f"{stem}_color.png"
        depth = root / f"{stem}_depth.png"
        if not color.is_file() or not depth.is_file():
            continue
        serial = ""
        stamp = ""
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
            serial = str(meta.get("serial") or "")
        except Exception:  # noqa: BLE001
            meta = {}
        # stem: {serial}_{YYYYMMDD}_{HHMMSS}_{mmm}
        parts = stem.split("_")
        if len(parts) >= 4:
            stamp = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        label = stamp or stem
        if serial:
            label = f"{serial} · {label}"
        out.append(
            {
                "id": stem,
                "label": label,
                "serial": serial,
                "stamp": stamp,
                "color": _rel_or_str(color),
                "depth": _rel_or_str(depth),
                "meta": _rel_or_str(meta_path),
                "ready": True,
                "source": "capture",
            }
        )
    return out


def normalize_mock_source_id(raw: Any) -> str:
    """Empty / stock / __stock__ → stock mock under data/mock."""
    s = str(raw or "").strip()
    if not s or s in {STOCK_MOCK_ID, "stock", "default", "mock"}:
        return STOCK_MOCK_ID
    return s


def mock_asset_paths(role: str = "scene", *, capture_id: Optional[str] = None) -> Dict[str, Any]:
    """Resolve RGB-D paths for scene/wrist: stock ``data/mock`` or ``data/capture/<id>``."""
    role = "wrist" if str(role).lower() in {"wrist", "d405"} else "scene"
    key = "d405" if role == "wrist" else "d435"
    src_id = normalize_mock_source_id(capture_id)

    if src_id != STOCK_MOCK_ID:
        color = CAPTURE_DIR / f"{src_id}_color.png"
        depth = CAPTURE_DIR / f"{src_id}_depth.png"
        meta = CAPTURE_DIR / f"{src_id}_meta.json"
        ready = bool(color.is_file() and depth.is_file())
        return {
            "role": role,
            "camera_key": key,
            "source_id": src_id,
            "source": "capture",
            "color": _rel_or_str(color),
            "depth": _rel_or_str(depth) if depth.exists() else None,
            "meta": _rel_or_str(meta) if meta.exists() else None,
            "color_abs": color,
            "depth_abs": depth if depth.is_file() else None,
            "meta_abs": meta if meta.is_file() else None,
            "ready": ready,
        }

    defaults = {
        "d405": {
            "color": "data/mock/d405_color.png",
            "depth": "data/mock/d405_depth.png",
            "meta": "data/mock/d405_meta.json",
        },
        "d435": {
            "color": "data/mock/d435_color.png",
            "depth": "data/mock/d435_depth.png",
            "meta": "data/mock/d435_meta.json",
        },
    }
    entry = dict(defaults[key])
    if MOCK_CONFIG_PATH.is_file():
        try:
            raw = yaml.safe_load(MOCK_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict) and isinstance(raw.get(key), dict):
                entry.update({k: v for k, v in raw[key].items() if v})
        except Exception:  # noqa: BLE001
            pass
    color = _resolve_repo_path(entry["color"])
    depth = _resolve_repo_path(entry["depth"]) if entry.get("depth") else None
    meta = _resolve_repo_path(entry["meta"]) if entry.get("meta") else None
    return {
        "role": role,
        "camera_key": key,
        "source_id": STOCK_MOCK_ID,
        "source": "stock",
        "color": _rel_or_str(color) if color.exists() else str(color),
        "depth": _rel_or_str(depth) if depth and depth.exists() else (str(depth) if depth else None),
        "meta": _rel_or_str(meta) if meta and meta.exists() else (str(meta) if meta else None),
        "color_abs": color,
        "depth_abs": depth,
        "meta_abs": meta,
        "ready": bool(
            color.is_file()
            and depth is not None
            and depth.is_file()
            and (meta is None or meta.is_file())
        ),
    }


def _intrinsics_from_meta(meta: Dict[str, Any], which: str = "color") -> Optional[Intrinsics]:
    block = meta.get(f"{which}_intrinsics") if isinstance(meta, dict) else None
    if not isinstance(block, dict):
        return None
    try:
        return Intrinsics(
            width=int(block.get("width") or 640),
            height=int(block.get("height") or 480),
            fx=float(block["fx"]),
            fy=float(block["fy"]),
            ppx=float(block["ppx"]),
            ppy=float(block["ppy"]),
            model=str(block.get("model") or ""),
            coeffs=list(block.get("coeffs") or []),
            stream=str(block.get("stream") or which),
        )
    except Exception:  # noqa: BLE001
        return None


def load_mock_packet(
    role: str = "scene",
    *,
    capture_id: Optional[str] = None,
) -> Tuple[FramePacket, Dict[str, Any]]:
    """Load FramePacket for grasp mock uploads (stock or data/capture)."""
    info = mock_asset_paths(role, capture_id=capture_id)
    color_path: Path = info["color_abs"]
    depth_path: Optional[Path] = info.get("depth_abs")
    meta_path: Optional[Path] = info.get("meta_abs")
    if not color_path.is_file():
        raise FileNotFoundError(f"mock color missing: {color_path}")
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f"cannot decode mock color: {color_path}")
    depth = None
    if depth_path is not None and depth_path.is_file():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is not None and depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth is not None and depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
    meta: Dict[str, Any] = {}
    if meta_path is not None and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    cintr = _intrinsics_from_meta(meta, "color")
    dintr = _intrinsics_from_meta(meta, "depth")
    if cintr is None:
        h, w = color.shape[:2]
        cintr = Intrinsics(width=w, height=h, fx=600.0, fy=600.0, ppx=w / 2.0, ppy=h / 2.0)
    serial = str(meta.get("serial") or f"MOCK_{info['camera_key'].upper()}")
    ts = float(meta.get("timestamp_ms") or 0.0)
    pkt = FramePacket(
        serial=serial,
        color_bgr=color,
        depth_u16=depth,
        timestamp_ms=ts,
        color_intrinsics=cintr,
        depth_intrinsics=dintr,
    )
    return pkt, info


def pack_frame_upload(
    packet: FramePacket,
    *,
    depth_scale: float = 1000.0,
    require_depth: bool = True,
) -> Dict[str, Any]:
    """Build FrameUpload body fragment (color_b64 / depth_b64 / color_intrinsics / depth_scale)."""
    if packet.color_bgr is None:
        raise ValueError("frame has no color")
    color = packet.color_bgr
    h, w = color.shape[:2]
    intr = packet.color_intrinsics
    if intr is None and packet.depth_intrinsics is not None:
        intr = packet.depth_intrinsics
    if intr is None:
        raise ValueError("frame missing color_intrinsics")
    intr_dict = {
        "width": int(intr.width or w),
        "height": int(intr.height or h),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
    }
    if int(intr_dict["width"]) != w or int(intr_dict["height"]) != h:
        # Prefer actual image size for QC; keep fx/fy/ppx/ppy from stream.
        intr_dict["width"] = w
        intr_dict["height"] = h

    body: Dict[str, Any] = {
        "color_b64": _png_b64(color),
        "color_intrinsics": intr_dict,
        "depth_scale": float(depth_scale),
        "serial": packet.serial,
        "timestamp_ms": float(packet.timestamp_ms),
    }
    if packet.depth_u16 is not None:
        depth = packet.depth_u16
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        body["depth_b64"] = _png_b64(depth)
    elif require_depth:
        raise ValueError("frame missing depth_u16 (uint16 mm PNG required)")
    return body


def optional_mask_b64(mask: Optional[np.ndarray]) -> Optional[str]:
    if mask is None:
        return None
    m = mask
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    return _png_b64(m)


def depth_preview_bgr(depth_u16: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """uint16 mm depth → BGR colormap for UI preview (zeros stay dark)."""
    if depth_u16 is None:
        return None
    d = np.asarray(depth_u16)
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.dtype != np.uint16:
        d = d.astype(np.uint16)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    lo = float(np.percentile(d[valid], 5))
    hi = float(np.percentile(d[valid], 95))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros(d.shape, dtype=np.uint8)
    scaled = (np.clip(d.astype(np.float32), lo, hi) - lo) / (hi - lo)
    norm[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored
