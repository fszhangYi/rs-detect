from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import FramePacket
from .streamer import CameraStreamer

logger = logging.getLogger(__name__)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


class SnapshotWriter:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, packet: FramePacket, prefix: str = "") -> dict:
        ts = _stamp()
        base = f"{prefix}{packet.serial}_{ts}" if prefix else f"{packet.serial}_{ts}"
        color_path = self.root / f"{base}_color.png"
        meta_path = self.root / f"{base}_meta.json"
        cv2.imwrite(str(color_path), packet.color_bgr)
        depth_path = None
        if packet.depth_u16 is not None:
            depth_path = self.root / f"{base}_depth.png"
            cv2.imwrite(str(depth_path), packet.depth_u16)
        meta = {
            "serial": packet.serial,
            "timestamp_ms": packet.timestamp_ms,
            "color": str(color_path.name),
            "depth": depth_path.name if depth_path else None,
            "color_intrinsics": packet.color_intrinsics.to_dict() if packet.color_intrinsics else None,
            "depth_intrinsics": packet.depth_intrinsics.to_dict() if packet.depth_intrinsics else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "color": str(color_path),
            "depth": str(depth_path) if depth_path else None,
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
