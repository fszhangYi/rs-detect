from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class PlaneModel:
    """Plane: n·x + d = 0 with ||n||=1, n pointing roughly toward camera (-Z preferred)."""

    normal: Tuple[float, float, float]
    d: float
    centroid: Tuple[float, float, float]
    inlier_ratio: float
    num_points: int
    num_inliers: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def fit_plane_ransac(
    points: np.ndarray,
    distance_thresh: float = 0.008,
    max_iters: int = 400,
    min_inliers: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> PlaneModel:
    """RANSAC plane fit on Nx3 points (camera frame, meters)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be Nx3")
    n_pts = pts.shape[0]
    if n_pts < 3:
        raise ValueError(f"need >=3 points, got {n_pts}")

    rng = rng or np.random.default_rng(0)
    best_inliers = None
    best_model = None

    for _ in range(max_iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        d = -float(np.dot(normal, p0))
        dist = np.abs(pts @ normal + d)
        inliers = dist < distance_thresh
        count = int(inliers.sum())
        if best_inliers is None or count > int(best_inliers.sum()):
            best_inliers = inliers
            best_model = (normal, d)

    if best_inliers is None or int(best_inliers.sum()) < min(min_inliers, max(3, n_pts // 10)):
        # fallback: SVD on all points
        centroid = pts.mean(axis=0)
        uu, ss, vh = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vh[-1, :]
        normal = _normalize(normal)
        d = -float(np.dot(normal, centroid))
        dist = np.abs(pts @ normal + d)
        best_inliers = dist < distance_thresh
        best_model = (normal, d)

    inlier_pts = pts[best_inliers]
    if inlier_pts.shape[0] < 3:
        inlier_pts = pts
    centroid = inlier_pts.mean(axis=0)
    # refine by SVD on inliers
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = _normalize(vh[-1, :])
    # orient normal toward camera (camera looks +Z of optical? RealSense optical: Z forward.
    # Suction approach from camera: prefer normal that faces the camera, i.e. n·centroid < 0
    # so that normal points back toward camera origin.
    if np.dot(normal, centroid) > 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    num_inliers = int(best_inliers.sum()) if best_inliers is not None else 0

    return PlaneModel(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        d=d,
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        inlier_ratio=float(num_inliers / max(n_pts, 1)),
        num_points=int(n_pts),
        num_inliers=num_inliers,
    )


def suction_pose_from_plane(plane: PlaneModel) -> Dict[str, Any]:
    """Suction point = plane centroid; approach axis = plane normal (toward camera)."""
    return {
        "point_cam": list(plane.centroid),
        "normal_cam": list(plane.normal),
        "plane": plane.to_dict(),
    }
