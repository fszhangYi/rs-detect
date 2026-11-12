from __future__ import annotations

import logging
from typing import Dict, Optional

from rs_detect.camera.models import Intrinsics
from rs_detect.camera.streamer import CameraStreamer, _intrinsics_from_rs

logger = logging.getLogger(__name__)


def fetch_intrinsics_from_streamer(streamer: CameraStreamer) -> Dict[str, Optional[dict]]:
    """M2: intrinsics come from live SDK profile via streamer."""
    data = streamer.get_intrinsics()
    return {k: (v.to_dict() if v else None) for k, v in data.items()}


def fetch_intrinsics_from_sdk(
    serial: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    is_mock: bool = False,
) -> Dict[str, Optional[dict]]:
    """Open a short-lived profile to read intrinsics without keeping the stream.

    For MOCK serials, return synthetic intrinsics consistent with the mock streamer.
    """
    if is_mock or serial.startswith("MOCK_"):
        color = Intrinsics(
            width=width,
            height=height,
            fx=width * 0.9,
            fy=height * 1.2,
            ppx=width / 2.0,
            ppy=height / 2.0,
            model="Brown Conrady",
            coeffs=[0, 0, 0, 0, 0],
            stream="color",
        )
        depth = Intrinsics(
            width=width,
            height=height,
            fx=width * 0.9,
            fy=height * 1.2,
            ppx=width / 2.0,
            ppy=height / 2.0,
            model="Brown Conrady",
            coeffs=[0, 0, 0, 0, 0],
            stream="depth",
        )
        return {"color": color.to_dict(), "depth": depth.to_dict()}

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color = _intrinsics_from_rs(color_stream.get_intrinsics(), "color")
        depth = _intrinsics_from_rs(depth_stream.get_intrinsics(), "depth")
        return {"color": color.to_dict(), "depth": depth.to_dict()}
    finally:
        pipeline.stop()
