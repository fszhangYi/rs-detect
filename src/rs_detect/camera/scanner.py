from __future__ import annotations

import logging
from typing import List

from .models import CameraInfo, CameraRole

logger = logging.getLogger(__name__)


def _role_from_name(name: str, product_line: str = "") -> CameraRole:
    text = f"{name} {product_line}".upper()
    if "D405" in text:
        return CameraRole.WRIST_CANDIDATE
    if "D435" in text or "D435I" in text:
        return CameraRole.SCENE_CANDIDATE
    return CameraRole.UNKNOWN


def classify_device(name: str, product_line: str = "") -> CameraRole:
    return _role_from_name(name, product_line)


def scan_devices(include_mock_if_empty: bool = True) -> List[CameraInfo]:
    """Enumerate RealSense devices and bucket into wrist/scene candidates.

    D405 → wrist_candidate, D435/D435I → scene_candidate.
    When no hardware is present and ``include_mock_if_empty``, return two mock cams.
    """
    devices: List[CameraInfo] = []
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        for dev in ctx.devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            try:
                product_line = dev.get_info(rs.camera_info.product_line)
            except Exception:
                product_line = ""
            try:
                firmware = dev.get_info(rs.camera_info.firmware_version)
            except Exception:
                firmware = ""
            try:
                usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
            except Exception:
                usb_type = ""
            devices.append(
                CameraInfo(
                    serial=serial,
                    name=name,
                    product_line=product_line,
                    firmware=firmware,
                    usb_type=usb_type,
                    role=classify_device(name, product_line),
                    is_mock=False,
                )
            )
    except Exception as exc:
        logger.warning("RealSense scan failed: %s", exc)

    if not devices and include_mock_if_empty:
        logger.info("No RealSense hardware; providing mock D405/D435 candidates")
        devices = [
            CameraInfo(
                serial="MOCK_D405_0001",
                name="Intel RealSense D405 (Mock)",
                product_line="D405",
                firmware="mock",
                role=CameraRole.WRIST_CANDIDATE,
                is_mock=True,
            ),
            CameraInfo(
                serial="MOCK_D435_0001",
                name="Intel RealSense D435 (Mock)",
                product_line="D435",
                firmware="mock",
                role=CameraRole.SCENE_CANDIDATE,
                is_mock=True,
            ),
        ]
    return devices


def split_by_role(devices: List[CameraInfo]) -> dict:
    wrist = [d for d in devices if d.role == CameraRole.WRIST_CANDIDATE]
    scene = [d for d in devices if d.role == CameraRole.SCENE_CANDIDATE]
    unknown = [d for d in devices if d.role == CameraRole.UNKNOWN]
    return {"wrist_candidates": wrist, "scene_candidates": scene, "unknown": unknown}
