"""M3.3 — voxel-grid multi-view point cloud fusion in base frame."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def voxel_fuse_points(
    clouds: Sequence[np.ndarray],
    *,
    voxel_m: float = 0.004,
    min_views: int = 1,
    max_points: int = 80000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """Average points that fall into the same voxel (class TSDF-lite / voxel mean).

    Returns
    -------
    fused : Mx3 float64
    stats : counts / voxel size
    """
    parts: List[np.ndarray] = []
    for c in clouds:
        a = np.asarray(c, dtype=np.float64)
        if a.size == 0:
            continue
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError("each cloud must be Nx3")
        parts.append(a)
    if not parts:
        return np.zeros((0, 3), dtype=np.float64), {
            "num_input_clouds": 0,
            "num_input_points": 0,
            "num_voxels": 0,
            "voxel_m": float(voxel_m),
            "min_views": int(min_views),
        }

    all_pts = np.vstack(parts)
    voxel = max(1e-4, float(voxel_m))
    keys = np.floor(all_pts / voxel).astype(np.int64)
    # pack key → average
    # use dict of sum/count for clarity (N typically < 1e5 after downsample)
    acc: dict = {}
    for p, k in zip(all_pts, keys):
        key = (int(k[0]), int(k[1]), int(k[2]))
        if key not in acc:
            acc[key] = [p.copy(), 1]
        else:
            acc[key][0] += p
            acc[key][1] += 1

    fused_list = []
    for _k, (s, cnt) in acc.items():
        if cnt >= int(min_views):
            fused_list.append(s / float(cnt))

    if not fused_list:
        # fallback: all voxel means ignoring min_views
        fused_list = [s / float(cnt) for s, cnt in (v for v in acc.values())]

    fused = np.asarray(fused_list, dtype=np.float64) if fused_list else np.zeros((0, 3))
    if fused.shape[0] > int(max_points):
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(fused.shape[0], size=int(max_points), replace=False)
        fused = fused[idx]

    return fused, {
        "num_input_clouds": len(parts),
        "num_input_points": int(all_pts.shape[0]),
        "num_voxels": int(fused.shape[0]),
        "voxel_m": float(voxel_m),
        "min_views": int(min_views),
    }
