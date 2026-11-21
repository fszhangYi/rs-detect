from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
CAPTURE_DIR = DATA_DIR / "capture"  # M0 one-shot RGB-D dumps (selectable mock source)
RECORDINGS_DIR = DATA_DIR / "recordings"
MOCK_DIR = DATA_DIR / "mock"
MODELS_DIR = ROOT / "models"
MOCK_CONFIG_PATH = CONFIG_DIR / "mock.yaml"
EXTRINSICS_PATH = CONFIG_DIR / "extrinsics.yaml"
CAMERAS_PATH = CONFIG_DIR / "cameras.yaml"
YOLO_CONFIG_PATH = CONFIG_DIR / "yolo.yaml"
GRASP_CLIENT_CONFIG_PATH = CONFIG_DIR / "grasp_client.yaml"
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline.yaml"
SAFE_POINTS_PATH = CONFIG_DIR / "safe_points.yaml"
STORAGE_CONFIG_PATH = CONFIG_DIR / "storage.yaml"
ROBOT_CONFIG_PATH = CONFIG_DIR / "robot_cell_c.yaml"
SCENE_OVERLAY_PATH = CONFIG_DIR / "scene_overlay.yaml"
CAMERA_OVERLAYS_PATH = CONFIG_DIR / "camera_overlays.yaml"
DRAW_SHAPES_PATH = CONFIG_DIR / "draw_shapes.yaml"
RESTART_FLAG_PATH = ROOT / ".rs_detect_restart"
WEB_DIST = ROOT / "web" / "dist"
WEB_STATIC = ROOT / "web" / "public"
WEB_ASSETS = ROOT / "web" / "assets"


def ensure_dirs() -> None:
    for p in (CONFIG_DIR, DATA_DIR, CAPTURES_DIR, CAPTURE_DIR, RECORDINGS_DIR, MOCK_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def sibling_src_dirs() -> list[Path]:
    """Workspace sibling package roots (sensors / sensors-arm), if present."""
    found: list[Path] = []
    for name in ("sensors", "sensors-arm"):
        p = WORKSPACE / name / "src"
        if p.is_dir():
            found.append(p)
    return found


def ensure_workspace_imports() -> None:
    """Prepend sibling ``sensors`` / ``sensors-arm`` src to ``sys.path`` (Win/Linux).

    Avoids hard-coded AutoDL paths in shell launchers. No-op when siblings missing.
    """
    for p in sibling_src_dirs():
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
