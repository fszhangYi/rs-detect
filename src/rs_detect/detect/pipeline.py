from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from rs_detect.camera.models import FramePacket
from rs_detect.detect.detector import DetectionResult, detect_cardboard
from rs_detect.geometry.mask_to_cloud import depth_roi_to_points
from rs_detect.geometry.plane_ransac import fit_plane_ransac, suction_pose_from_plane

logger = logging.getLogger(__name__)


@dataclass
class SingleFrameResult:
    ok: bool
    serial: str
    detection: Dict[str, Any]
    num_points: int
    suction: Optional[Dict[str, Any]]
    message: str
    viz_jpeg: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "serial": self.serial,
            "detection": self.detection,
            "num_points": self.num_points,
            "suction": self.suction,
            "message": self.message,
            "has_viz": self.viz_jpeg is not None,
        }


_SEG_COLORS_BGR = (
    (40, 180, 40),
    (0, 200, 255),
    (255, 140, 0),
    (200, 80, 255),
    (50, 180, 255),
    (0, 255, 180),
)


def draw_detection_viz(
    color_bgr: np.ndarray,
    det: DetectionResult,
    suction_uv: Optional[tuple] = None,
    *,
    draw_boxes: bool = True,
) -> np.ndarray:
    """Overlay segmentation polygons (preferred) and optional AABB boxes."""
    vis = color_bgr.copy()
    names = det.class_names or ["obj"] * len(det.boxes)
    scores = det.scores or [0.0] * len(det.boxes)
    polygons = list(det.polygons or [])
    n = max(len(det.boxes), len(polygons))
    has_any_poly = any(len(p) >= 3 for p in polygons)

    # Soft fill from union mask (legacy / HSV) when no per-instance polygons.
    if not has_any_poly and det.mask is not None and np.count_nonzero(det.mask):
        overlay = vis.copy()
        overlay[det.mask > 0] = (40, 180, 40)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (40, 180, 40), 2, lineType=cv2.LINE_AA)

    for i in range(n):
        color = _SEG_COLORS_BGR[i % len(_SEG_COLORS_BGR)]
        name = names[i] if i < len(names) else "obj"
        score = float(scores[i]) if i < len(scores) else 0.0
        poly = polygons[i] if i < len(polygons) else []
        label_xy = None

        if poly and len(poly) >= 3:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            # Label near top-most polygon vertex.
            top = min(poly, key=lambda p: p[1])
            label_xy = (int(top[0]), int(top[1]))
        elif draw_boxes and i < len(det.boxes):
            x1, y1, x2, y2 = [int(v) for v in det.boxes[i]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_xy = (x1, y1)
        elif i < len(det.boxes):
            x1, y1, _, _ = [int(v) for v in det.boxes[i]]
            label_xy = (x1, y1)

        # Thin AABB helper only when polygons exist (shows Ultralytics box vs mask).
        if has_any_poly and draw_boxes and i < len(det.boxes):
            x1, y1, x2, y2 = [int(v) for v in det.boxes[i]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

        if label_xy is not None:
            lx, ly = label_xy
            cv2.putText(
                vis,
                f"{name} {score:.2f}",
                (lx, max(20, ly - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    if suction_uv is not None:
        u, v = suction_uv
        cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.circle(vis, (int(u), int(v)), 10, (0, 0, 255), 2)
    return vis


def project_point_to_uv(point_cam: List[float], intr: Dict[str, float]) -> tuple:
    x, y, z = point_cam
    if z <= 1e-9:
        return None
    u = intr["fx"] * x / z + intr["ppx"]
    v = intr["fy"] * y / z + intr["ppy"]
    return float(u), float(v)


def run_single_frame(
    packet: FramePacket,
    *,
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
) -> SingleFrameResult:
    """YOLO/HSV/mock → ROI mask → point cloud → RANSAC plane → suction point (camera frame)."""
    det = detect_cardboard(
        packet.color_bgr,
        method=method,
        weights=weights,
        conf=conf,
        class_filter=class_filter,
        is_mock=is_mock,
        device=device,
        imgsz=imgsz,
    )
    if det.mask is None or int(np.count_nonzero(det.mask)) == 0:
        return SingleFrameResult(
            ok=False,
            serial=packet.serial,
            detection=det.to_dict(),
            num_points=0,
            suction=None,
            message=det.message or "no mask",
        )

    if packet.depth_u16 is None:
        return SingleFrameResult(
            ok=False,
            serial=packet.serial,
            detection=det.to_dict(),
            num_points=0,
            suction=None,
            message="no depth",
        )

    intr = None
    if packet.color_intrinsics is not None:
        intr = packet.color_intrinsics.to_dict()
    elif packet.depth_intrinsics is not None:
        intr = packet.depth_intrinsics.to_dict()
    if intr is None:
        return SingleFrameResult(
            ok=False,
            serial=packet.serial,
            detection=det.to_dict(),
            num_points=0,
            suction=None,
            message="no intrinsics on frame",
        )

    # resize mask if needed (should already match)
    mask = det.mask
    if mask.shape[:2] != packet.depth_u16.shape[:2]:
        mask = cv2.resize(mask, (packet.depth_u16.shape[1], packet.depth_u16.shape[0]), interpolation=cv2.INTER_NEAREST)

    points = depth_roi_to_points(packet.depth_u16, mask > 0, intr, depth_scale=depth_scale)
    if points.shape[0] < 30:
        return SingleFrameResult(
            ok=False,
            serial=packet.serial,
            detection=det.to_dict(),
            num_points=int(points.shape[0]),
            suction=None,
            message=f"too few points: {points.shape[0]}",
        )

    if points.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]

    try:
        plane = fit_plane_ransac(points, distance_thresh=ransac_thresh)
        suction = suction_pose_from_plane(plane)
    except Exception as exc:
        return SingleFrameResult(
            ok=False,
            serial=packet.serial,
            detection=det.to_dict(),
            num_points=int(points.shape[0]),
            suction=None,
            message=f"ransac failed: {exc}",
        )

    viz_jpeg = None
    if make_viz:
        uv = project_point_to_uv(suction["point_cam"], intr)
        vis = draw_detection_viz(packet.color_bgr, det, suction_uv=uv)
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            viz_jpeg = buf.tobytes()

    return SingleFrameResult(
        ok=True,
        serial=packet.serial,
        detection=det.to_dict(),
        num_points=int(points.shape[0]),
        suction=suction,
        message="ok",
        viz_jpeg=viz_jpeg,
    )
