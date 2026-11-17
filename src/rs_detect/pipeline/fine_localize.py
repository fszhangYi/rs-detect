from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rs_detect.camera.models import FramePacket
from rs_detect.detect.pipeline import run_single_frame
from rs_detect.geometry.transforms import as_T44, matmul_T, transform_normal, transform_point


@dataclass
class FineResult:
    """Plan §4.3 — wrist fine result (M2: single-frame; M3+: fused)."""

    ok: bool
    c_star: Optional[List[float]]  # base when T available; else None
    n_star: Optional[List[float]]
    c_cam: Optional[List[float]]
    n_cam: Optional[List[float]]
    inlier_ratio: float
    num_points: int
    fuse_quality: float  # M2 single-frame: use inlier_ratio
    serial: str
    timestamp_ms: float
    detection: Optional[Dict[str, Any]] = None
    frame: str = "base"  # base | cam — where c_star lives
    message: str = ""
    viz_jpeg: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "c_star": self.c_star,
            "n_star": self.n_star,
            "c_cam": self.c_cam,
            "n_cam": self.n_cam,
            "inlier_ratio": self.inlier_ratio,
            "num_points": self.num_points,
            "fuse_quality": self.fuse_quality,
            "serial": self.serial,
            "timestamp_ms": self.timestamp_ms,
            "detection": self.detection,
            "frame": self.frame,
            "message": self.message,
            "has_viz": self.viz_jpeg is not None,
            "role": "wrist",
            "camera": "d405",
        }


def fine_localize_single(
    packet: FramePacket,
    *,
    T_flange_d405: Optional[List[List[float]]] = None,
    T_base_flange: Optional[List[List[float]]] = None,
    method: str = "auto",
    weights: Optional[Path] = None,
    conf: float = 0.25,
    class_filter: Optional[List[str]] = None,
    is_mock: bool = False,
    depth_scale: float = 0.001,
    ransac_thresh: float = 0.008,
    max_points: int = 20000,
    make_viz: bool = True,
    device: Optional[str] = None,
    imgsz: int = 640,
) -> FineResult:
    """D405 single-frame fine localization (M2). Multiview fusion is M3.

    If T_flange_d405 (+ optional T_base_flange) given, also emit c_star/n_star in base.
    Without robot FK, pass identity T_base_flange for mock / offline demos only.
    """
    frame = run_single_frame(
        packet,
        method=method,
        weights=weights,
        conf=conf,
        class_filter=class_filter,
        is_mock=is_mock,
        depth_scale=depth_scale,
        ransac_thresh=ransac_thresh,
        max_points=max_points,
        make_viz=make_viz,
        device=device,
        imgsz=imgsz,
    )
    if not frame.ok or not frame.suction:
        return FineResult(
            ok=False,
            c_star=None,
            n_star=None,
            c_cam=None,
            n_cam=None,
            inlier_ratio=0.0,
            num_points=frame.num_points,
            fuse_quality=0.0,
            serial=packet.serial,
            timestamp_ms=float(packet.timestamp_ms),
            detection=frame.detection,
            message=f"fine failed: {frame.message}",
            viz_jpeg=frame.viz_jpeg,
        )

    c_cam = list(frame.suction["point_cam"])
    n_cam = list(frame.suction["normal_cam"])
    plane = frame.suction.get("plane") or {}
    inlier = float(plane.get("inlier_ratio") or 0.0)

    c_star = None
    n_star = None
    out_frame = "cam"
    if T_flange_d405 is not None:
        T_bf = as_T44(T_base_flange) if T_base_flange is not None else np.eye(4, dtype=np.float64)
        T_base_cam = matmul_T(T_bf, as_T44(T_flange_d405))
        c_star = transform_point(T_base_cam, c_cam)
        n_star = transform_normal(T_base_cam, n_cam)
        out_frame = "base"

    return FineResult(
        ok=True,
        c_star=c_star,
        n_star=n_star,
        c_cam=c_cam,
        n_cam=n_cam,
        inlier_ratio=inlier,
        num_points=frame.num_points,
        fuse_quality=inlier,
        serial=packet.serial,
        timestamp_ms=float(packet.timestamp_ms),
        detection=frame.detection,
        frame=out_frame,
        message="ok",
        viz_jpeg=frame.viz_jpeg,
    )
