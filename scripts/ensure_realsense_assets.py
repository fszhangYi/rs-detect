#!/usr/bin/env python3
"""Ensure web/assets/realsense/{d435,d405}.stl are real binary meshes (Win/Linux).

Git stores these as symlinks to vendor/realsense_cad/*.stl. On Windows without
symlink checkout they become 60-byte text stubs; STLLoader then throws
\"Offset is outside the bounds of the DataView\".
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "realsense_cad"
DST_DIR = ROOT / "web" / "assets" / "realsense"
NAMES = ("d435.stl", "d405.stl")


def _is_binary_stl(path: Path) -> bool:
    if not path.is_file():
        return False
    data = path.read_bytes()
    if len(data) < 84:
        return False
    # Broken git-symlink stub: tiny text file with absolute path target.
    if len(data) < 512 and b".stl" in data and (data.lstrip().startswith(b"/") or b"realsense_cad" in data):
        return False
    head = data[:80].lstrip()
    if head.startswith(b"solid") and b"\n" in data[:200]:
        # ASCII STL — acceptable for Three.js STLLoader
        return b"facet" in data[:4096] or b"endsolid" in data
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expect = 84 + int(n_tri) * 50
    return n_tri > 0 and abs(len(data) - expect) <= 2


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    if sys.platform == "win32" and path.exists():
        try:
            out = subprocess.check_output(
                ["cmd", "/c", "dir", "/AL", str(path.parent)],
                text=True,
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
                errors="ignore",
            )
            return path.name.lower() in out.lower() and "<JUNCTION>" in out.upper()
        except Exception:  # noqa: BLE001
            return False
    return False


def _remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or _is_link_or_junction(path):
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve() and _is_binary_stl(dst):
                return "ok"
        except Exception:  # noqa: BLE001
            pass
        _remove(dst)
    try:
        os.symlink(src, dst, target_is_directory=False)
        if _is_binary_stl(dst):
            return "symlink"
        _remove(dst)
    except OSError:
        pass
    if sys.platform == "win32":
        # Hardlink if same volume; else copy.
        try:
            os.link(src, dst)
            if _is_binary_stl(dst):
                return "hardlink"
            _remove(dst)
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    VENDOR.mkdir(parents=True, exist_ok=True)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in NAMES if not _is_binary_stl(VENDOR / n)]
    if missing:
        print(
            "[rs-detect] missing RealSense CAD under vendor/realsense_cad:",
            ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "  Place binary STL files there (D435 meters, D405 millimeters), then re-run.",
            file=sys.stderr,
        )
        # Still try to report stub status for web assets.
        for n in NAMES:
            p = DST_DIR / n
            if p.exists() and not _is_binary_stl(p):
                print(f"[rs-detect] WARN broken stub: {p} ({p.stat().st_size} B)", file=sys.stderr)
        return 1

    rc = 0
    for name in NAMES:
        src = (VENDOR / name).resolve()
        dst = DST_DIR / name
        if _is_binary_stl(dst):
            print(f"[rs-detect] realsense asset ok: {dst.name} ({dst.stat().st_size} B)")
            continue
        mode = _link_or_copy(src, dst)
        if not _is_binary_stl(dst):
            print(f"[rs-detect] FAIL install {name}", file=sys.stderr)
            rc = 1
            continue
        print(f"[rs-detect] realsense asset {mode}: {dst} -> {src} ({dst.stat().st_size} B)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
