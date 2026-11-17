from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .models import FramePacket, Intrinsics
from .streamer import CameraStreamer

logger = logging.getLogger(__name__)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _intrinsics_for_aligned_depth(packet: FramePacket) -> Optional[Intrinsics]:
    """Prefer color intrinsics: streamer aligns depth → color before packaging."""
    if packet.color_intrinsics is not None:
        return packet.color_intrinsics
    return packet.depth_intrinsics


def depth_color_to_xyzrgb(
    depth_u16: np.ndarray,
    color_bgr: np.ndarray,
    intrinsics: Intrinsics,
    *,
    depth_scale: float = 0.001,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project valid depth pixels → (Nx3 xyz meters, Nx3 rgb uint8)."""
    if depth_u16 is None or color_bgr is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    if depth_u16.shape[:2] != color_bgr.shape[:2]:
        raise ValueError(
            f"depth shape {depth_u16.shape[:2]} != color shape {color_bgr.shape[:2]} "
            "(expect aligned depth→color)"
        )
    h, w = depth_u16.shape[:2]
    z_raw = depth_u16.astype(np.float64)
    z = z_raw * float(depth_scale)
    valid = (z_raw > 0) & (z >= min_depth_m) & (z <= max_depth_m)
    vs, us = np.where(valid)
    if vs.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    zz = z[vs, us]
    xx = (us.astype(np.float64) - float(intrinsics.ppx)) * zz / float(intrinsics.fx)
    yy = (vs.astype(np.float64) - float(intrinsics.ppy)) * zz / float(intrinsics.fy)
    xyz = np.column_stack([xx, yy, zz]).astype(np.float32)
    # OpenCV BGR → RGB for PLY consumers
    bgr = color_bgr[vs, us]
    rgb = bgr[:, ::-1].astype(np.uint8)
    return xyz, rgb


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> int:
    """Write ASCII PLY with x y z + RGB. Returns point count."""
    n = int(xyz.shape[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        if n == 0:
            return 0
        # Vectorized formatting is faster than per-point write for ~1e5 pts
        lines = np.column_stack(
            [
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                rgb[:, 0].astype(np.int32),
                rgb[:, 1].astype(np.int32),
                rgb[:, 2].astype(np.int32),
            ]
        )
        np.savetxt(f, lines, fmt="%.6f %.6f %.6f %d %d %d")
    return n


class SnapshotWriter:
    def __init__(self, root: Path, *, depth_scale: float = 0.001):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.depth_scale = float(depth_scale)

    def save(self, packet: FramePacket, prefix: str = "") -> dict:
        ts = _stamp()
        base = f"{prefix}{packet.serial}_{ts}" if prefix else f"{packet.serial}_{ts}"
        color_path = self.root / f"{base}_color.png"
        meta_path = self.root / f"{base}_meta.json"
        cv2.imwrite(str(color_path), packet.color_bgr)
        depth_path = None
        ply_path = None
        npy_path = None
        point_count = 0
        if packet.depth_u16 is not None:
            depth_path = self.root / f"{base}_depth.png"
            cv2.imwrite(str(depth_path), packet.depth_u16)
            intr = _intrinsics_for_aligned_depth(packet)
            if intr is not None:
                try:
                    xyz, rgb = depth_color_to_xyzrgb(
                        packet.depth_u16,
                        packet.color_bgr,
                        intr,
                        depth_scale=self.depth_scale,
                    )
                    ply_path = self.root / f"{base}_cloud.ply"
                    point_count = write_ply_xyzrgb(ply_path, xyz, rgb)
                    # Compact float32 xyz companion for code that prefers numpy
                    npy_path = self.root / f"{base}_cloud_xyz.npy"
                    np.save(str(npy_path), xyz)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("point cloud save failed: %s", exc)
                    ply_path = None
                    npy_path = None
                    point_count = 0
            else:
                logger.warning("skip point cloud: missing intrinsics for serial=%s", packet.serial)
        meta: Dict[str, Any] = {
            "serial": packet.serial,
            "timestamp_ms": packet.timestamp_ms,
            "color": str(color_path.name),
            "depth": depth_path.name if depth_path else None,
            "point_cloud_ply": ply_path.name if ply_path else None,
            "point_cloud_xyz_npy": npy_path.name if npy_path else None,
            "point_count": point_count,
            "depth_scale": self.depth_scale,
            "color_intrinsics": packet.color_intrinsics.to_dict() if packet.color_intrinsics else None,
            "depth_intrinsics": packet.depth_intrinsics.to_dict() if packet.depth_intrinsics else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "color": str(color_path),
            "depth": str(depth_path) if depth_path else None,
            "point_cloud": str(ply_path) if ply_path else None,
            "point_cloud_xyz": str(npy_path) if npy_path else None,
            "point_count": point_count,
            "meta": str(meta_path),
        }


class VideoRecorder:
    """Record color stream from a CameraStreamer to mp4."""

    def __init__(self, streamer: CameraStreamer, out_path: Path, fps: Optional[int] = None):
        self.streamer = streamer
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps or streamer.fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._active = False
        self._frames = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def frames_written(self) -> int:
        return self._frames

    def start(self) -> None:
        if self._active:
            return
        pkt = self.streamer.get_latest()
        if pkt is None:
            # wait briefly for first frame
            import time

            for _ in range(50):
                time.sleep(0.05)
                pkt = self.streamer.get_latest()
                if pkt is not None:
                    break
        if pkt is None:
            raise RuntimeError("no frame available to start video recording")
        h, w = pkt.color_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.out_path), fourcc, float(self.fps), (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter: {self.out_path}")
        self.streamer.add_callback(self._on_frame)
        self._cb_attached = True
        self._active = True
        self._frames = 0
        logger.info("Video recording started → %s", self.out_path)

    def _on_frame(self, packet: FramePacket) -> None:
        if not self._active or self._writer is None:
            return
        if packet.serial != self.streamer.serial:
            return
        self._writer.write(packet.color_bgr)
        self._frames += 1

    def stop(self) -> dict:
        self._active = False
        if getattr(self, "_cb_attached", False):
            try:
                self.streamer._callbacks.remove(self._on_frame)
            except ValueError:
                pass
            self._cb_attached = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return {"path": str(self.out_path), "frames": self._frames}


def colorize_depth(depth_u16: np.ndarray) -> np.ndarray:
    depth_f = depth_u16.astype(np.float32)
    depth_f[depth_f <= 0] = np.nan
    valid = depth_f[np.isfinite(depth_f)]
    if valid.size == 0:
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    dmin, dmax = np.percentile(valid, [5, 95])
    if dmax <= dmin:
        dmax = dmin + 1.0
    norm = np.clip((depth_f - dmin) / (dmax - dmin), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)
