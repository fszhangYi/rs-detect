from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from .models import FramePacket, Intrinsics
from .scanner import scan_devices

logger = logging.getLogger(__name__)

FrameCallback = Callable[[FramePacket], None]


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
        self._color_intrinsics = Intrinsics(
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
        if self.enable_depth:
            self._depth_intrinsics = Intrinsics(
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
        logger.info("Started MOCK stream serial=%s", self.serial)

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
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        t = time.time()
        # animated panel so UI proves stream is alive
        hue = int((t * 40) % 180)
        img[:] = (40, 40, 40)
        cv2.rectangle(img, (40, 40), (self.width - 40, self.height - 40), (80, 80, 80), 2)
        label = "MOCK D405" if "D405" in self.serial else "MOCK D435"
        if "D405" in self.serial:
            color = (0, 180, 255)
        else:
            color = (0, 200, 100)
        cv2.putText(img, label, (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f"SN {self.serial}",
            (60, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            time.strftime("%H:%M:%S"),
            (60, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 50),
            2,
            cv2.LINE_AA,
        )
        # moving bar
        x = int((t * 120) % (self.width - 80)) + 40
        cv2.rectangle(img, (x, self.height - 80), (x + 40, self.height - 40), color, -1)
        depth = None
        if self.enable_depth:
            depth = np.full((self.height, self.width), 800, dtype=np.uint16)  # 800mm fake
            cv2.circle(depth, (self.width // 2, self.height // 2), 80, 500, -1)
        # slight hue wash
        _ = hue
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
