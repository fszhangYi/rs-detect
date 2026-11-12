from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
RECORDINGS_DIR = DATA_DIR / "recordings"
EXTRINSICS_PATH = CONFIG_DIR / "extrinsics.yaml"
CAMERAS_PATH = CONFIG_DIR / "cameras.yaml"
WEB_DIST = ROOT / "web" / "dist"
WEB_STATIC = ROOT / "web" / "public"


def ensure_dirs() -> None:
    for p in (CONFIG_DIR, DATA_DIR, CAPTURES_DIR, RECORDINGS_DIR):
        p.mkdir(parents=True, exist_ok=True)
