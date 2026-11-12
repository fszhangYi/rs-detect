from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class CameraRole(str, Enum):
    """Logical role buckets in the UI."""

    WRIST_CANDIDATE = "wrist_candidate"  # D405 pool
    SCENE_CANDIDATE = "scene_candidate"  # D435 pool
    UNKNOWN = "unknown"


@dataclass
class CameraInfo:
    serial: str
    name: str
    product_line: str
    firmware: str = ""
    usb_type: str = ""
    role: CameraRole = CameraRole.UNKNOWN
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = ""
    coeffs: list = field(default_factory=list)
    stream: str = "color"  # color | depth

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intrinsics":
        return cls(**{k: data[k] for k in (
            "width", "height", "fx", "fy", "ppx", "ppy", "model", "coeffs", "stream"
        ) if k in data})


@dataclass
class FramePacket:
    serial: str
    color_bgr: np.ndarray
    depth_u16: Optional[np.ndarray]
    timestamp_ms: float
    color_intrinsics: Optional[Intrinsics] = None
    depth_intrinsics: Optional[Intrinsics] = None
