from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import yaml

from rs_detect.paths import MOCK_CONFIG_PATH, MOCK_DIR, ROOT

from .models import FramePacket, Intrinsics
from .scanner import scan_devices

logger = logging.getLogger(__name__)

FrameCallback = Callable[[FramePacket], None]


def _mock_role_from_serial(serial: str) -> str:
    sn = (serial or "").upper()
    if "D405" in sn:
        return "d405"
    if "D435" in sn:
        return "d435"
    return "d405"


def _load_mock_manifest() -> Dict[str, Any]:
    data: Dict[str, Any] = {
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
    if MOCK_CONFIG_PATH.exists():
        try:
            raw = yaml.safe_load(MOCK_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                for key in ("d405", "d435"):
                    if isinstance(raw.get(key), dict):
                        data[key] = {**data[key], **raw[key]}
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read %s: %s", MOCK_CONFIG_PATH, exc)
    return data


def _resolve_mock_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _intrinsics_from_meta(meta: Dict[str, Any], key: str, fallback: Intrinsics) -> Intrinsics:
    block = meta.get(key)
    if not isinstance(block, dict):
        return fallback
    try:
        payload = dict(block)
        if "stream" not in payload:
            payload["stream"] = "color" if "color" in key else "depth"
        return Intrinsics.from_dict(payload)
    except Exception:  # noqa: BLE001
        return fallback


def _intrinsics_from_rs(intr, stream: str) -> Intrinsics:
    return Intrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        ppx=float(intr.ppx),
        ppy=float(intr.ppy),
        model=str(intr.model),
        coeffs=list(intr.coeffs),
        stream=stream,
    )


class CameraStreamer:
    """Stable color(+depth) stream for one serial. Supports mock devices."""

    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        is_mock: Optional[bool] = None,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.is_mock = is_mock if is_mock is not None else serial.startswith("MOCK_")
        self._pipeline = None
        self._align = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._callbacks: list = []
        self._running = False
        self._color_intrinsics: Optional[Intrinsics] = None
        self._depth_intrinsics: Optional[Intrinsics] = None
        self._frame_count = 0
        self._error: Optional[str] = None
        self._mock_color: Optional[np.ndarray] = None
        self._mock_depth: Optional[np.ndarray] = None
        self._mock_source: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def add_callback(self, cb: FrameCallback) -> None:
        self._callbacks.append(cb)

    def get_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest

    def get_intrinsics(self) -> Dict[str, Optional[Intrinsics]]:
        return {"color": self._color_intrinsics, "depth": self._depth_intrinsics}

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._error = None
        if self.is_mock:
            self._start_mock()
        else:
            self._start_real()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"stream-{self.serial}")
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._running = False

    def _start_real(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        self._pipeline = pipeline
        if self.enable_depth:
            self._align = rs.align(rs.stream.color)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._color_intrinsics = _intrinsics_from_rs(color_stream.get_intrinsics(), "color")
        if self.enable_depth:
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self._depth_intrinsics = _intrinsics_from_rs(depth_stream.get_intrinsics(), "depth")
        logger.info("Started RealSense stream serial=%s", self.serial)

    def _start_mock(self) -> None:
        role = _mock_role_from_serial(self.serial)
        entry = _load_mock_manifest().get(role) or {}
        color_path = _resolve_mock_path(str(entry.get("color") or (MOCK_DIR / f"{role}_color.png")))
        depth_path = _resolve_mock_path(str(entry.get("depth") or (MOCK_DIR / f"{role}_depth.png")))
        meta_path = _resolve_mock_path(str(entry.get("meta") or (MOCK_DIR / f"{role}_meta.json")))

        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(f"mock color image missing: {color_path}")
        if color.shape[1] != self.width or color.shape[0] != self.height:
            color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._mock_color = color
        self._mock_source = str(color_path.relative_to(ROOT)) if color_path.is_relative_to(ROOT) else str(color_path)

        depth = None
        if self.enable_depth and depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                if depth.dtype != np.uint16:
                    depth = depth.astype(np.uint16)
                if depth.shape[1] != self.width or depth.shape[0] != self.height:
                    depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self._mock_depth = depth

        fallback_color = Intrinsics(
            width=self.width,
            height=self.height,
            fx=self.width * 0.9,
            fy=self.height * 1.2,
            ppx=self.width / 2.0,
            ppy=self.height / 2.0,
            model="Brown Conrady",
            coeffs=[0, 0, 0, 0, 0],
            stream="color",
        )
        fallback_depth = Intrinsics(
            width=self.width,
            height=self.height,
            fx=self.width * 0.9,
            fy=self.height * 1.2,
            ppx=self.width / 2.0,
            ppy=self.height / 2.0,
            model="Brown Conrady",
            coeffs=[0, 0, 0, 0, 0],
            stream="depth",
        )
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("mock meta load failed %s: %s", meta_path, exc)
        self._color_intrinsics = _intrinsics_from_meta(meta, "color_intrinsics", fallback_color)
        # Keep streamer width/height as the live frame size after resize.
        self._color_intrinsics.width = self.width
        self._color_intrinsics.height = self.height
        if self.enable_depth:
            self._depth_intrinsics = _intrinsics_from_meta(meta, "depth_intrinsics", fallback_depth)
            self._depth_intrinsics.width = self.width
            self._depth_intrinsics.height = self.height
        logger.info(
            "Started MOCK stream serial=%s role=%s color=%s depth=%s",
            self.serial,
            role,
            self._mock_source,
            "yes" if self._mock_depth is not None else "no",
        )

    def _loop(self) -> None:
        period = 1.0 / max(self.fps, 1)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                packet = self._grab_mock() if self.is_mock else self._grab_real()
                if packet is not None:
                    with self._lock:
                        self._latest = packet
                    self._frame_count += 1
                    for cb in list(self._callbacks):
                        try:
                            cb(packet)
                        except Exception as exc:
                            logger.exception("frame callback error: %s", exc)
            except Exception as exc:
                self._error = str(exc)
                logger.exception("stream error serial=%s: %s", self.serial, exc)
                time.sleep(0.2)
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    def _grab_real(self) -> Optional[FramePacket]:
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data())
        return FramePacket(
            serial=self.serial,
            color_bgr=color.copy(),
            depth_u16=depth.copy() if depth is not None else None,
            timestamp_ms=time.time() * 1000.0,
            color_intrinsics=self._color_intrinsics,
            depth_intrinsics=self._depth_intrinsics,
        )

    def _grab_mock(self) -> FramePacket:
        t = time.time()
        if self._mock_color is None:
            raise RuntimeError(f"mock color not loaded for serial={self.serial}")
        img = self._mock_color.copy()
        # Small live badge so UI still looks "streaming" without altering the scene.
        label = "MOCK D405" if "D405" in (self.serial or "").upper() else "MOCK D435"
        accent = (0, 180, 255) if "D405" in (self.serial or "").upper() else (0, 200, 100)
        cv2.rectangle(img, (8, 8), (210, 52), (20, 20, 20), -1)
        cv2.putText(img, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            time.strftime("%H:%M:%S"),
            (14, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 80),
            1,
            cv2.LINE_AA,
        )
        depth = None
        if self.enable_depth:
            if self._mock_depth is not None:
                depth = self._mock_depth.copy()
            else:
                depth = np.full((self.height, self.width), 800, dtype=np.uint16)
        return FramePacket(
            serial=self.serial,
            color_bgr=img,
            depth_u16=depth,
            timestamp_ms=t * 1000.0,
            color_intrinsics=self._color_intrinsics,
            depth_intrinsics=self._depth_intrinsics,
        )


class StreamManager:
    """Manage up to one wrist + one scene stream."""

    def __init__(self):
        self._streams: Dict[str, CameraStreamer] = {}
        self._roles: Dict[str, str] = {}  # serial -> wrist|scene
        self._lock = threading.Lock()

    def list_streams(self) -> Dict[str, dict]:
        with self._lock:
            out = {}
            for sn, st in self._streams.items():
                out[sn] = {
                    "serial": sn,
                    "role": self._roles.get(sn),
                    "running": st.running,
                    "frame_count": st.frame_count,
                    "error": st.error,
                    "is_mock": st.is_mock,
                    "intrinsics": {
                        k: (v.to_dict() if v else None) for k, v in st.get_intrinsics().items()
                    },
                }
            return out

    def start(self, serial: str, role: str, **kwargs) -> CameraStreamer:
        with self._lock:
            # stop existing stream on same role
            for sn, r in list(self._roles.items()):
                if r == role and sn != serial:
                    self._stop_unlocked(sn)
            if serial in self._streams and self._streams[serial].running:
                self._roles[serial] = role
                return self._streams[serial]
            # detect mock from scan
            is_mock = serial.startswith("MOCK_")
            for info in scan_devices(include_mock_if_empty=True):
                if info.serial == serial:
                    is_mock = info.is_mock
                    break
            streamer = CameraStreamer(serial=serial, is_mock=is_mock, **kwargs)
            streamer.start()
            self._streams[serial] = streamer
            self._roles[serial] = role
            return streamer

    def stop(self, serial: str) -> None:
        with self._lock:
            self._stop_unlocked(serial)

    def stop_all(self) -> None:
        with self._lock:
            for sn in list(self._streams.keys()):
                self._stop_unlocked(sn)

    def _stop_unlocked(self, serial: str) -> None:
        st = self._streams.pop(serial, None)
        self._roles.pop(serial, None)
        if st is not None:
            st.stop()

    def get(self, serial: str) -> Optional[CameraStreamer]:
        return self._streams.get(serial)

    def get_by_role(self, role: str) -> Optional[CameraStreamer]:
        for sn, r in self._roles.items():
            if r == role:
                return self._streams.get(sn)
        return None
