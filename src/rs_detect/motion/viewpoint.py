"""M3.1 — generate N neighboring wrist observe viewpoints above a coarse target."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def generate_viewpoints(
    center_xyz: Sequence[float],
    *,
    n: int = 5,
    radius_m: float = 0.03,
    z: Optional[float] = None,
    include_center: bool = True,
) -> List[Dict[str, Any]]:
    """Plan M3.1: N=3–5 nearby observe poses (XY ring, shared Z).

    Parameters
    ----------
    center_xyz : coarse / hover point in base [x,y,z]
    n : total viewpoints including center when include_center
    radius_m : ring radius (plan: ~2–4 cm)
    z : override height; default keeps center_xyz[2]
    """
    if len(center_xyz) < 3:
        raise ValueError("center_xyz needs 3 elements")
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    zz = float(z) if z is not None else cz
    n = max(1, int(n))
    radius = max(0.0, float(radius_m))

    pts: List[Dict[str, Any]] = []
    if include_center:
        pts.append({"name": "center", "xyz": [cx, cy, zz], "offset_xy": [0.0, 0.0]})
        ring_n = max(0, n - 1)
    else:
        ring_n = n

    if ring_n > 0 and radius > 0:
        for i in range(ring_n):
            ang = 2.0 * math.pi * i / ring_n
            dx = radius * math.cos(ang)
            dy = radius * math.sin(ang)
            pts.append(
                {
                    "name": f"ring_{i}",
                    "xyz": [cx + dx, cy + dy, zz],
                    "offset_xy": [dx, dy],
                }
            )
    elif ring_n > 0 and radius <= 0 and not include_center:
        pts.append({"name": "center", "xyz": [cx, cy, zz], "offset_xy": [0.0, 0.0]})

    # Truncate / pad to exactly n when possible
    if len(pts) > n:
        pts = pts[:n]
    return pts
