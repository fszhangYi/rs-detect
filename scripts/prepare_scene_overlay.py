#!/usr/bin/env python3
"""Prepare cell scene STL overlay (separate from URDF).

By default links/copies the full-resolution CAD STL into web/assets/scene/
and writes config/scene_overlay.yaml. Does NOT modify any URDF / link meshes.

Optional --decimate only if you explicitly want a lighter preview mesh.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "assets" / "scene"
CFG = ROOT / "config" / "scene_overlay.yaml"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def subsample_stl(src: Path, dst: Path, target_tris: int = 80000, seed: int = 0) -> int:
    import numpy as np
    from stl import mesh as stlmesh

    m = stlmesh.Mesh.from_file(str(src))
    n = len(m.vectors)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(target_tris, n), replace=False)
    out = stlmesh.Mesh(np.zeros(idx.size, dtype=stlmesh.Mesh.dtype))
    out.vectors[:] = m.vectors[idx]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dst))
    return int(idx.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=Path.home() / "下载" / "g26562-01-00_asm.stl",
        help="Full cell assembly STL (mm)",
    )
    ap.add_argument(
        "--decimate",
        action="store_true",
        help="Optional: write a subsampled preview STL instead of full mesh",
    )
    ap.add_argument("--tris", type=int, default=80000, help="Only with --decimate")
    ap.add_argument(
        "--base-mm",
        type=float,
        nargs=3,
        default=[487.54, 109.64, 12.0],
        help="Estimated arm base origin in scene CAD (mm)",
    )
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"missing source STL: {args.src}")

    if args.decimate:
        out_stl = OUT_DIR / "g26562-01-00_asm.decim.stl"
        n = subsample_stl(args.src, out_stl, target_tris=args.tris)
        url = "/assets/scene/g26562-01-00_asm.decim.stl"
        print(f"wrote decimated {out_stl} ({out_stl.stat().st_size / 1e6:.2f} MB, {n} tris)")
    else:
        out_stl = OUT_DIR / "g26562-01-00_asm.stl"
        link_or_copy(args.src, out_stl)
        url = "/assets/scene/g26562-01-00_asm.stl"
        print(f"linked/copied full STL → {out_stl} ({args.src.stat().st_size / 1e6:.2f} MB)")

    placement = {
        "name": "g26562 cell scene overlay",
        "url": url,
        "units_of_file": "mm",
        "enabled": True,
        "opacity": 0.32,
        "color": "#7a93ad",
        "scale": 0.001,
        "align": {
            "base_in_scene_mm": list(args.base_mm),
            "yaw_deg": float(args.yaw_deg),
            "method": "manual/estimated — fine-tune in Arm UI",
            "confidence": "medium",
        },
        "offset_m": [0.0, 0.0, -float(args.base_mm[2]) * 0.001],
        "offset_rpy_deg": [0.0, 0.0, 0.0],
        "notes": "Loaded separately from URDF. Full resolution unless --decimate. Do not merge into cs66_cell_c.urdf.",
    }
    (OUT_DIR / "placement.json").write_text(json.dumps(placement, indent=2) + "\n", encoding="utf-8")

    try:
        import yaml
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("PyYAML required") from exc
    CFG.write_text(yaml.safe_dump(placement, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {CFG}")


if __name__ == "__main__":
    main()
