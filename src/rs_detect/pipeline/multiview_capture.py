"""M3.2 — per-viewpoint wrist observe: detect → mask cloud → base frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from rs_detect.camera.models import FramePacket
from rs_detect.detect.detector import detect_cardboard
from rs_detect.geometry.mask_to_cloud import depth_roi_to_points
from rs_detect.geometry.plane_ransac import PlaneModel, fit_plane_ransac
from rs_detect.geometry.transforms import as_T44, matmul_T, transform_points


@dataclass
class ViewObservation:
    ok: bool
    name: str
    xyz_cmd: List[float]
    T_base_cam: Optional[List[List[float]]]
    points_base: Optional[np.ndarray]
    plane_base: Optional[PlaneModel]
    num_points: int = 0
    detection: Optional[Dict[str, Any]] = None
    message: str = ""
    timestamp_ms: float = 0.0
    viz_jpeg: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "name": self.name,
            "xyz_cmd": self.xyz_cmd,
            "num_points": self.num_points,
            "plane_base": self.plane_base.to_dict() if self.plane_base else None,
            "detection": self.detection,
            "message": self.message,
            "timestamp_ms": self.timestamp_ms,
            "has_viz": self.viz_jpeg is not None,
        }


def _T_list(T: np.ndarray) -> List[List[float]]:
    return [[float(v) for v in row] for row in T.tolist()]


def build_T_base_cam(
    *,
    T_flange_d405: Sequence[Sequence[float]],
    T_base_flange: Optional[Sequence[Sequence[float]]] = None,
    viewpoint_xyz: Optional[Sequence[float]] = None,
    ref_xyz: Optional[Sequence[float]] = None,
    dry_run_translate: bool = False,
) -> np.ndarray:
    """Compose camera pose in base.

    dry_run_translate: when extrinsics≈I, apply translation (vp−ref) so multi-view
    clouds separate in base while mock camera content stays fixed in cam frame.
    """
    T_bf = as_T44(T_base_flange) if T_base_flange is not None else np.eye(4, dtype=np.float64)
    T_fc = as_T44(T_flange_d405)
    T = matmul_T(T_bf, T_fc)
    if dry_run_translate and viewpoint_xyz is not None and ref_xyz is not None:
        dx = float(viewpoint_xyz[0]) - float(ref_xyz[0])
        dy = float(viewpoint_xyz[1]) - float(ref_xyz[1])
        dz = float(viewpoint_xyz[2]) - float(ref_xyz[2])
        Td = np.eye(4, dtype=np.float64)
        Td[0, 3], Td[1, 3], Td[2, 3] = dx, dy, dz
        T = Td @ T
    return T


def capture_wrist_view(
    packet: FramePacket,
    *,
    name: str,
    xyz_cmd: Sequence[float],
    T_base_cam: np.ndarray,
    method: str = "auto",
    weights: Optional[Path] = None,
    conf: float = 0.25,
    class_filter: Optional[List[str]] = None,
    is_mock: bool = False,
    depth_scale: float = 0.001,
    ransac_thresh: float = 0.008,
    max_points: int = 20000,
    min_points: int = 30,
    make_viz: bool = True,
    device: Optional[str] = None,
    imgsz: int = 640,
) -> ViewObservation:
    """One viewpoint: YOLO/HSV mask → cam cloud → base cloud + RANSAC plane."""
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
        return ViewObservation(
            ok=False,
            name=name,
            xyz_cmd=list(xyz_cmd),
            T_base_cam=_T_list(T_base_cam),
            points_base=None,
            plane_base=None,
            detection=det.to_dict(),
            message=det.message or "no mask",
            timestamp_ms=float(packet.timestamp_ms),
        )
    if packet.depth_u16 is None:
        return ViewObservation(
            ok=False,
            name=name,
            xyz_cmd=list(xyz_cmd),
            T_base_cam=_T_list(T_base_cam),
            points_base=None,
            plane_base=None,
            detection=det.to_dict(),
            message="no depth",
            timestamp_ms=float(packet.timestamp_ms),
        )
    intr = None
    if packet.color_intrinsics is not None:
        intr = packet.color_intrinsics.to_dict()
    elif packet.depth_intrinsics is not None:
        intr = packet.depth_intrinsics.to_dict()
    if intr is None:
        return ViewObservation(
            ok=False,
            name=name,
            xyz_cmd=list(xyz_cmd),
            T_base_cam=_T_list(T_base_cam),
            points_base=None,
            plane_base=None,
            detection=det.to_dict(),
            message="no intrinsics",
            timestamp_ms=float(packet.timestamp_ms),
        )

    mask = det.mask
    if mask.shape[:2] != packet.depth_u16.shape[:2]:
        mask = cv2.resize(
            mask,
            (packet.depth_u16.shape[1], packet.depth_u16.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    points_cam = depth_roi_to_points(packet.depth_u16, mask > 0, intr, depth_scale=depth_scale)
    if points_cam.shape[0] < int(min_points):
        return ViewObservation(
            ok=False,
            name=name,
            xyz_cmd=list(xyz_cmd),
            T_base_cam=_T_list(T_base_cam),
            points_base=None,
            plane_base=None,
            num_points=int(points_cam.shape[0]),
            detection=det.to_dict(),
            message=f"too few points: {points_cam.shape[0]}",
            timestamp_ms=float(packet.timestamp_ms),
        )
    if points_cam.shape[0] > int(max_points):
        idx = np.random.default_rng(0).choice(points_cam.shape[0], size=int(max_points), replace=False)
        points_cam = points_cam[idx]

    points_base = transform_points(as_T44(T_base_cam), points_cam)
    try:
        plane = fit_plane_ransac(points_base, distance_thresh=ransac_thresh)
    except Exception as exc:  # noqa: BLE001
        return ViewObservation(
            ok=False,
            name=name,
            xyz_cmd=list(xyz_cmd),
            T_base_cam=_T_list(T_base_cam),
            points_base=points_base,
            plane_base=None,
            num_points=int(points_base.shape[0]),
            detection=det.to_dict(),
            message=f"ransac failed: {exc}",
            timestamp_ms=float(packet.timestamp_ms),
        )

    viz = None
    if make_viz:
        vis = packet.color_bgr.copy()
        ys, xs = np.where(mask > 0)
        if xs.size:
            vis[ys, xs] = (0.55 * vis[ys, xs] + np.array([0, 180, 0])).astype(np.uint8)
        cv2.putText(vis, f"M3 {name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            viz = buf.tobytes()

    return ViewObservation(
        ok=True,
        name=name,
        xyz_cmd=list(map(float, xyz_cmd[:3])),
        T_base_cam=_T_list(T_base_cam),
        points_base=points_base,
        plane_base=plane,
        num_points=int(points_base.shape[0]),
        detection=det.to_dict(),
        message="ok",
        timestamp_ms=float(packet.timestamp_ms),
        viz_jpeg=viz,
    )
