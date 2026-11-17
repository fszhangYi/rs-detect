from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def as_T44(mat: Iterable[Iterable[float]]) -> np.ndarray:
    T = np.asarray(mat, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4, got {T.shape}")
    return T


def transform_points(T_dst_src: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nx3 points: p_dst = R p_src + t."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    R = T_dst_src[:3, :3]
    t = T_dst_src[:3, 3]
    return (pts @ R.T) + t


def transform_point(T_dst_src: np.ndarray, point: ArrayLike) -> List[float]:
    p = transform_points(T_dst_src, np.asarray(point, dtype=np.float64).reshape(1, 3))[0]
    return [float(p[0]), float(p[1]), float(p[2])]


def transform_normal(T_dst_src: np.ndarray, normal: ArrayLike) -> List[float]:
    """Rotate unit normal (ignore translation); re-normalize."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    R = T_dst_src[:3, :3]
    out = R @ n
    nn = np.linalg.norm(out)
    if nn > 1e-12:
        out = out / nn
    return [float(out[0]), float(out[1]), float(out[2])]


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def matmul_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return as_T44(A) @ as_T44(B)
