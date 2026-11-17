from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rs_detect.camera.models import FramePacket
from rs_detect.detect.pipeline import run_single_frame
from rs_detect.geometry.transforms import as_T44, transform_normal, transform_point


@dataclass
class CoarseResult:
    """Plan §4.1 — D435 coarse localization in robot base."""

    ok: bool
    p_coarse: Optional[List[float]]  # base
    n_coarse: Optional[List[float]]  # base
    yolo_conf: float
    timestamp_ms: float
    serial: str
    point_cam: Optional[List[float]] = None
    normal_cam: Optional[List[float]] = None
    num_points: int = 0
    inlier_ratio: float = 0.0
    detection: Optional[Dict[str, Any]] = None
    observe_hint: Optional[Dict[str, Any]] = None
    message: str = ""
    viz_jpeg: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "p_coarse": self.p_coarse,
            "n_coarse": self.n_coarse,
            "yolo_conf": self.yolo_conf,
            "timestamp_ms": self.timestamp_ms,
            "serial": self.serial,
            "point_cam": self.point_cam,
            "normal_cam": self.normal_cam,
            "num_points": self.num_points,
            "inlier_ratio": self.inlier_ratio,
            "detection": self.detection,
            "observe_hint": self.observe_hint,
            "message": self.message,
            "has_viz": self.viz_jpeg is not None,
            "role": "scene",
            "camera": "d435",
        }


def suggest_wrist_observe_pose(
    p_coarse: List[float],
    n_coarse: List[float],
    work_distance_m: float = 0.25,
) -> Dict[str, Any]:
    """Hint only (M3/M4 will execute): place wrist ~work_distance along plane normal toward camera."""
    p = np.asarray(p_coarse, dtype=np.float64)
    n = np.asarray(n_coarse, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        n = np.array([0.0, 0.0, -1.0])
    else:
        n = n / nn
    # normal faces camera; standoff sits between board and camera
    p_obs = p + n * float(work_distance_m)
    return {
        "p_observe_base": [float(x) for x in p_obs],
        "approach_normal_base": [float(x) for x in n],
        "work_distance_m": float(work_distance_m),
        "note": "plan stage A: wrist FOV center ~15–40 cm; not executed in M2",
    }


def coarse_localize(
    packet: FramePacket,
    T_base_d435: List[List[float]],
    *,
    method: str = "auto",
    weights: Optional[Path] = None,
    conf: float = 0.25,
    class_filter: Optional[List[str]] = None,
    is_mock: bool = False,
    depth_scale: float = 0.001,
    ransac_thresh: float = 0.008,
    max_points: int = 20000,
    work_distance_m: float = 0.25,
    make_viz: bool = True,
    device: Optional[str] = None,
    imgsz: int = 640,
) -> CoarseResult:
    """D435: detect → ROI cloud → RANSAC → transform to base as p_coarse / n_coarse.

    Per plan: coarse is for navigation + gate only; never averaged into final suction depth.
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
    scores = (frame.detection or {}).get("scores") or []
    yolo_conf = float(scores[0]) if scores else 0.0
    if not frame.ok or not frame.suction:
        return CoarseResult(
            ok=False,
            p_coarse=None,
            n_coarse=None,
            yolo_conf=yolo_conf,
            timestamp_ms=float(packet.timestamp_ms),
            serial=packet.serial,
            detection=frame.detection,
            message=f"coarse failed: {frame.message}",
            viz_jpeg=frame.viz_jpeg,
            num_points=frame.num_points,
        )

    T = as_T44(T_base_d435)
    p_cam = frame.suction["point_cam"]
    n_cam = frame.suction["normal_cam"]
    p_base = transform_point(T, p_cam)
    n_base = transform_normal(T, n_cam)
    plane = (frame.suction.get("plane") or {})
    return CoarseResult(
        ok=True,
        p_coarse=p_base,
        n_coarse=n_base,
        yolo_conf=yolo_conf,
        timestamp_ms=float(packet.timestamp_ms),
        serial=packet.serial,
        point_cam=list(p_cam),
        normal_cam=list(n_cam),
        num_points=frame.num_points,
        inlier_ratio=float(plane.get("inlier_ratio") or 0.0),
        detection=frame.detection,
        observe_hint=suggest_wrist_observe_pose(p_base, n_base, work_distance_m=work_distance_m),
        message="ok",
        viz_jpeg=frame.viz_jpeg,
    )
