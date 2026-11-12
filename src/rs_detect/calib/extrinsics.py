from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# 4x4 row-major identity
IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _validate_matrix(mat: List[List[float]]) -> List[List[float]]:
    if len(mat) != 4 or any(len(row) != 4 for row in mat):
        raise ValueError("extrinsic matrix must be 4x4")
    out = []
    for row in mat:
        out.append([float(x) for x in row])
    return out


def default_extrinsics() -> Dict[str, Any]:
    return {
        "version": 1,
        "units": "meters_radians_row_major_4x4",
        "notes": "User-provided extrinsics. D405: T_flange_cam (eye-in-hand). D435: T_base_cam.",
        "d405": {
            "serial": "",
            "frame_from": "flange",
            "frame_to": "d405_color_optical",
            "T": [row[:] for row in IDENTITY_4X4],
        },
        "d435": {
            "serial": "",
            "frame_from": "base",
            "frame_to": "d435_color_optical",
            "T": [row[:] for row in IDENTITY_4X4],
        },
    }


def load_extrinsics(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        data = default_extrinsics()
        save_extrinsics(path, data)
        return data
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return data


def save_extrinsics(path: Path, data: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # validate matrices if present
    for key in ("d405", "d435"):
        if key in data and "T" in data[key]:
            data[key]["T"] = _validate_matrix(data[key]["T"])
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved extrinsics → %s", path)
    return path


class ExtrinsicsStore:
    """M2: extrinsics are user-editable and persisted to config."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data = load_extrinsics(self.path)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def reload(self) -> Dict[str, Any]:
        self._data = load_extrinsics(self.path)
        return self._data

    def update(
        self,
        *,
        d405_T: Optional[List[List[float]]] = None,
        d435_T: Optional[List[List[float]]] = None,
        d405_serial: Optional[str] = None,
        d435_serial: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        if d405_T is not None:
            self._data.setdefault("d405", {})["T"] = _validate_matrix(d405_T)
        if d435_T is not None:
            self._data.setdefault("d435", {})["T"] = _validate_matrix(d435_T)
        if d405_serial is not None:
            self._data.setdefault("d405", {})["serial"] = d405_serial
        if d435_serial is not None:
            self._data.setdefault("d435", {})["serial"] = d435_serial
        if notes is not None:
            self._data["notes"] = notes
        save_extrinsics(self.path, self._data)
        return self._data
