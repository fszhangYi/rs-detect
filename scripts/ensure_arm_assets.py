#!/usr/bin/env python3
"""Ensure web/assets/models/cs66 points at sensors-arm Cell C meshes (Win/Linux).

Linux lab used a symlink; Windows uses a directory junction when possible,
otherwise copies the tree. Idempotent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "web" / "assets" / "models" / "cs66"
CANDIDATES = [
    ROOT.parent / "sensors-arm" / "src" / "sensors_arm" / "static" / "models" / "cs66",
    ROOT / "vendor" / "sensors-arm" / "static" / "models" / "cs66",
]


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


def _link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        # Already correct?
        try:
            if dst.resolve() == src.resolve():
                return "ok"
        except Exception:  # noqa: BLE001
            pass
        _remove(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    if sys.platform == "win32":
        # Junction does not need admin
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if r.returncode == 0 and dst.exists():
            return "junction"
    shutil.copytree(src, dst)
    return "copy"


def main() -> int:
    src = next((p for p in CANDIDATES if p.is_dir()), None)
    if src is None:
        print("[rs-detect] missing sensors-arm models/cs66", file=sys.stderr)
        print("  expected sibling:", CANDIDATES[0], file=sys.stderr)
        return 1
    urdf = src / "cs66_cell_c.urdf"
    if not urdf.is_file():
        print(f"[rs-detect] missing {urdf}", file=sys.stderr)
        return 1
    mode = _link(src.resolve(), DST)
    print(f"[rs-detect] arm assets {mode}: {DST} -> {src}")
    print(f"[rs-detect] urdf ok: {(DST / 'cs66_cell_c.urdf').is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
