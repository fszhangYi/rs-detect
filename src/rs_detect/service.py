from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from rs_detect.calib.extrinsics import ExtrinsicsStore
from rs_detect.calib.intrinsics import fetch_intrinsics_from_sdk, fetch_intrinsics_from_streamer
from rs_detect.camera.recorder import SnapshotWriter, VideoRecorder, colorize_depth
from rs_detect.camera.scanner import scan_devices, split_by_role
from rs_detect.camera.streamer import StreamManager
from rs_detect.paths import CAPTURES_DIR, EXTRINSICS_PATH, RECORDINGS_DIR, ensure_dirs

logger = logging.getLogger(__name__)


class DetectService:
    """Shared backend for desktop UI and web API."""

    def __init__(self):
        ensure_dirs()
        self.streams = StreamManager()
        self.extrinsics = ExtrinsicsStore(EXTRINSICS_PATH)
        self.snapshots = SnapshotWriter(CAPTURES_DIR)
        self._recorders: Dict[str, VideoRecorder] = {}

    # ---- M1 scan / stream ----
    def scan(self) -> Dict[str, Any]:
        devices = scan_devices(include_mock_if_empty=True)
        buckets = split_by_role(devices)
        return {
            "devices": [d.to_dict() for d in devices],
            "wrist_candidates": [d.to_dict() for d in buckets["wrist_candidates"]],
            "scene_candidates": [d.to_dict() for d in buckets["scene_candidates"]],
            "unknown": [d.to_dict() for d in buckets["unknown"]],
        }

    def start_stream(self, serial: str, role: str, width: int = 640, height: int = 480, fps: int = 30) -> dict:
        if role not in ("wrist", "scene"):
            raise ValueError("role must be wrist|scene")
        st = self.streams.start(serial, role=role, width=width, height=height, fps=fps, enable_depth=True)
        return {
            "serial": serial,
            "role": role,
            "running": st.running,
            "is_mock": st.is_mock,
            "intrinsics": fetch_intrinsics_from_streamer(st),
        }

    def stop_stream(self, serial: str) -> dict:
        if serial in self._recorders and self._recorders[serial].active:
            self._recorders[serial].stop()
            self._recorders.pop(serial, None)
        self.streams.stop(serial)
        return {"serial": serial, "running": False}

    def stop_all(self) -> None:
        for sn in list(self._recorders.keys()):
            try:
                self._recorders[sn].stop()
            except Exception:
                pass
        self._recorders.clear()
        self.streams.stop_all()

    def stream_status(self) -> dict:
        return self.streams.list_streams()

    def get_jpeg(self, serial: str, which: str = "color", quality: int = 80) -> Optional[bytes]:
        st = self.streams.get(serial)
        if st is None:
            return None
        pkt = st.get_latest()
        if pkt is None:
            return None
        if which == "depth":
            if pkt.depth_u16 is None:
                return None
            img = colorize_depth(pkt.depth_u16)
        else:
            img = pkt.color_bgr
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return buf.tobytes()

    def snapshot(self, serial: str) -> dict:
        st = self.streams.get(serial)
        if st is None:
            raise RuntimeError(f"stream not running: {serial}")
        pkt = st.get_latest()
        if pkt is None:
            raise RuntimeError("no frame yet")
        paths = self.snapshots.save(pkt)
        return paths

    def start_recording(self, serial: str) -> dict:
        st = self.streams.get(serial)
        if st is None:
            raise RuntimeError(f"stream not running: {serial}")
        if serial in self._recorders and self._recorders[serial].active:
            return {"path": str(self._recorders[serial].out_path), "already": True}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RECORDINGS_DIR / f"{serial}_{ts}.mp4"
        rec = VideoRecorder(st, out)
        rec.start()
        self._recorders[serial] = rec
        return {"path": str(out), "recording": True}

    def stop_recording(self, serial: str) -> dict:
        rec = self._recorders.pop(serial, None)
        if rec is None:
            return {"recording": False}
        return {**rec.stop(), "recording": False}

    # ---- M2 calib ----
    def get_intrinsics(self, serial: str, from_live: bool = True) -> dict:
        st = self.streams.get(serial)
        if from_live and st is not None and st.running:
            return {"serial": serial, "source": "live_sdk", "intrinsics": fetch_intrinsics_from_streamer(st)}
        is_mock = serial.startswith("MOCK_")
        return {
            "serial": serial,
            "source": "sdk_probe",
            "intrinsics": fetch_intrinsics_from_sdk(serial, is_mock=is_mock),
        }

    def get_extrinsics(self) -> dict:
        return self.extrinsics.reload()

    def set_extrinsics(self, payload: dict) -> dict:
        return self.extrinsics.update(
            d405_T=payload.get("d405_T"),
            d435_T=payload.get("d435_T"),
            d405_serial=payload.get("d405_serial"),
            d435_serial=payload.get("d435_serial"),
            notes=payload.get("notes"),
        )


# process-wide singleton for API + optional desktop sharing
_SERVICE: Optional[DetectService] = None


def get_service() -> DetectService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = DetectService()
    return _SERVICE
