"""M3.4 — fuse per-view planes + cross-check against fused-cloud RANSAC."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rs_detect.geometry.plane_ransac import PlaneModel, fit_plane_ransac


def _as_plane(n: Sequence[float], d: float, centroid: Sequence[float], **extra) -> PlaneModel:
    nn = np.asarray(n, dtype=np.float64)
    nrm = np.linalg.norm(nn)
    if nrm > 1e-12:
        nn = nn / nrm
    return PlaneModel(
        normal=(float(nn[0]), float(nn[1]), float(nn[2])),
        d=float(d),
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        inlier_ratio=float(extra.get("inlier_ratio") or 0.0),
        num_points=int(extra.get("num_points") or 0),
        num_inliers=int(extra.get("num_inliers") or 0),
    )


def _orient_agree(n: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if float(np.dot(n, ref)) < 0:
        return -n
    return n


def fuse_planes(
    planes: Sequence[PlaneModel],
    *,
    max_normal_angle_deg: float = 20.0,
    max_centroid_dist_m: float = 0.05,
) -> Tuple[Optional[PlaneModel], Dict[str, Any]]:
    """Covariance-lite / consistency-gated average of plane normals + centroids."""
    if not planes:
        return None, {"ok": False, "reason": "no planes", "used": 0, "rejected": 0}

    # Seed = plane with best inlier_ratio
    seed = max(planes, key=lambda p: float(p.inlier_ratio))
    seed_n = np.asarray(seed.normal, dtype=np.float64)
    seed_c = np.asarray(seed.centroid, dtype=np.float64)

    used: List[PlaneModel] = []
    rejected = 0
    cos_thresh = math.cos(math.radians(float(max_normal_angle_deg)))

    for p in planes:
        n = _orient_agree(np.asarray(p.normal, dtype=np.float64), seed_n)
        c = np.asarray(p.centroid, dtype=np.float64)
        if float(np.dot(n, seed_n)) < cos_thresh:
            rejected += 1
            continue
        if float(np.linalg.norm(c - seed_c)) > float(max_centroid_dist_m):
            rejected += 1
            continue
        used.append(
            PlaneModel(
                normal=(float(n[0]), float(n[1]), float(n[2])),
                d=float(p.d),
                centroid=p.centroid,
                inlier_ratio=p.inlier_ratio,
                num_points=p.num_points,
                num_inliers=p.num_inliers,
            )
        )

    if not used:
        return None, {"ok": False, "reason": "all planes rejected by consistency", "used": 0, "rejected": rejected}

    # Weight by inlier_ratio (floor so weak views still count a bit)
    weights = np.asarray([max(0.05, float(p.inlier_ratio)) for p in used], dtype=np.float64)
    weights = weights / weights.sum()
    ns = np.stack([_orient_agree(np.asarray(p.normal), seed_n) for p in used], axis=0)
    cs = np.stack([np.asarray(p.centroid) for p in used], axis=0)
    n_star = ns.T @ weights
    n_star = n_star / max(float(np.linalg.norm(n_star)), 1e-12)
    c_star = cs.T @ weights
    d_star = -float(np.dot(n_star, c_star))
    fused = _as_plane(
        n_star,
        d_star,
        c_star,
        inlier_ratio=float(np.mean([p.inlier_ratio for p in used])),
        num_points=int(sum(p.num_points for p in used)),
        num_inliers=int(sum(p.num_inliers for p in used)),
    )
    return fused, {"ok": True, "used": len(used), "rejected": rejected, "seed_inlier": seed.inlier_ratio}


def cross_check_fused_cloud(
    plane_star: PlaneModel,
    fused_cloud: np.ndarray,
    *,
    ransac_thresh: float = 0.008,
    max_normal_angle_deg: float = 15.0,
    max_centroid_dist_m: float = 0.03,
) -> Dict[str, Any]:
    """RANSAC on P_fuse and compare to π* (plan C2 cross-validation)."""
    pts = np.asarray(fused_cloud, dtype=np.float64)
    out: Dict[str, Any] = {
        "ok": False,
        "cloud_plane": None,
        "normal_angle_deg": None,
        "centroid_dist_m": None,
        "message": "",
    }
    if pts.shape[0] < 30:
        out["message"] = f"fused cloud too small: {pts.shape[0]}"
        return out
    try:
        cloud_plane = fit_plane_ransac(pts, distance_thresh=ransac_thresh)
    except Exception as exc:  # noqa: BLE001
        out["message"] = f"cloud ransac failed: {exc}"
        return out

    n1 = np.asarray(plane_star.normal, dtype=np.float64)
    n2 = np.asarray(cloud_plane.normal, dtype=np.float64)
    n2 = _orient_agree(n2, n1)
    cos = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    ang = math.degrees(math.acos(cos))
    dist = float(np.linalg.norm(np.asarray(plane_star.centroid) - np.asarray(cloud_plane.centroid)))
    out["cloud_plane"] = cloud_plane.to_dict()
    out["normal_angle_deg"] = ang
    out["centroid_dist_m"] = dist
    ok = ang <= float(max_normal_angle_deg) and dist <= float(max_centroid_dist_m)
    out["ok"] = ok
    out["message"] = "ok" if ok else f"cross-check fail ang={ang:.1f}deg dist={dist:.4f}m"
    # Prefer cloud centroid if check passes (often denser)
    if ok:
        out["preferred_centroid"] = list(cloud_plane.centroid)
        out["preferred_normal"] = list(n2)
    else:
        out["preferred_centroid"] = list(plane_star.centroid)
        out["preferred_normal"] = list(plane_star.normal)
    return out
