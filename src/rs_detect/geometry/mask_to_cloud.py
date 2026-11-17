from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def intrinsics_dict_to_tuple(intr: Optional[Dict]) -> Tuple[float, float, float, float]:
    """Return fx, fy, ppx, ppy from intrinsics dict."""
    if not intr:
        raise ValueError("color/depth intrinsics required")
    return float(intr["fx"]), float(intr["fy"]), float(intr["ppx"]), float(intr["ppy"])


def depth_roi_to_points(
    depth_u16: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict,
    depth_scale: float = 0.001,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    """Back-project masked depth pixels to Nx3 points in the camera optical frame (meters).

    Parameters
    ----------
    depth_u16 : HxW uint16 depth (aligned to color if mask comes from color)
    mask : HxW bool/uint8, nonzero = keep
    intrinsics : dict with fx,fy,ppx,ppy (same resolution as depth/mask)
    depth_scale : meters per depth unit (RealSense default often 0.001)
    """
    if depth_u16 is None:
        raise ValueError("depth image is required")
    if mask.shape[:2] != depth_u16.shape[:2]:
        raise ValueError(f"mask shape {mask.shape[:2]} != depth shape {depth_u16.shape[:2]}")

    fx, fy, ppx, ppy = intrinsics_dict_to_tuple(intrinsics)
    m = mask.astype(bool)
    vs, us = np.where(m)
    if vs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z_raw = depth_u16[vs, us].astype(np.float64)
    z = z_raw * float(depth_scale)
    valid = (z >= min_depth_m) & (z <= max_depth_m) & (z_raw > 0)
    us, vs, z = us[valid], vs[valid], z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    x = (us.astype(np.float64) - ppx) * z / fx
    y = (vs.astype(np.float64) - ppy) * z / fy
    return np.column_stack([x, y, z])
