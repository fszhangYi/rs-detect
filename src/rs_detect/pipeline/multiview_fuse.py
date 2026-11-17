"""M3 — multi-view capture + voxel/plane fusion → FineResult (c*, n*)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from rs_detect.camera.models import FramePacket
from rs_detect.geometry.plane_fuse import cross_check_fused_cloud, fuse_planes
from rs_detect.geometry.voxel_fuse import voxel_fuse_points
from rs_detect.motion.viewpoint import generate_viewpoints
from rs_detect.pipeline.fine_localize import FineResult
from rs_detect.pipeline.multiview_capture import (
    ViewObservation,
    build_T_base_cam,
    capture_wrist_view,
)


@dataclass
class MultiviewFuseResult:
    ok: bool
    fine: Optional[FineResult]
    views: List[ViewObservation] = field(default_factory=list)
    viewpoints: List[Dict[str, Any]] = field(default_factory=list)
    voxel_stats: Optional[Dict[str, Any]] = None
    plane_fuse_stats: Optional[Dict[str, Any]] = None
    cross_check: Optional[Dict[str, Any]] = None
    p_pregrasp: Optional[List[float]] = None
    pregrasp_z_lift_m: float = 0.05
    message: str = ""
    fallback: str = ""  # "" | "single_best_view" | "none"

    def to_dict(self) -> Dict[str, Any]:
        fine_d = self.fine.to_dict() if self.fine else None
        return {
            "ok": self.ok,
            "message": self.message,
            "fallback": self.fallback,
            "fine": fine_d,
            "c_star": fine_d.get("c_star") if fine_d else None,
            "n_star": fine_d.get("n_star") if fine_d else None,
            "p_pregrasp": self.p_pregrasp,
            "pregrasp_z_lift_m": self.pregrasp_z_lift_m,
            "viewpoints": self.viewpoints,
            "views": [v.to_dict() for v in self.views],
            "num_views_ok": sum(1 for v in self.views if v.ok),
            "num_views": len(self.views),
            "voxel_stats": self.voxel_stats,
            "plane_fuse_stats": self.plane_fuse_stats,
            "cross_check": self.cross_check,
            "has_viz": bool(self.fine and self.fine.viz_jpeg),
        }


def run_multiview_fuse(
    *,
    center_xyz: Sequence[float],
    grab_packet: Callable[[], FramePacket],
    move_xyz: Callable[[Sequence[float]], Any],
    get_T_base_flange: Callable[[], Optional[List[List[float]]]],
    T_flange_d405: Sequence[Sequence[float]],
    method: str = "auto",
    weights: Optional[Path] = None,
    conf: float = 0.25,
    class_filter: Optional[List[str]] = None,
    is_mock: bool = False,
    depth_scale: float = 0.001,
    ransac_thresh: float = 0.008,
    max_points: int = 20000,
    n_views: int = 5,
    radius_m: float = 0.03,
    settle_s: float = 0.35,
    voxel_m: float = 0.004,
    min_views_ok: int = 2,
    min_points_total: int = 80,
    pregrasp_z_lift_m: float = 0.05,
    dry_run_translate: bool = False,
    serial: str = "",
    device: Optional[str] = None,
    imgsz: int = 640,
) -> MultiviewFuseResult:
    """Visit N viewpoints → fuse clouds/planes → c*, n* in base (+ p_pregrasp)."""
    vps = generate_viewpoints(center_xyz, n=n_views, radius_m=radius_m, include_center=True)
    ref = list(vps[0]["xyz"])
    views: List[ViewObservation] = []
    last_viz: Optional[bytes] = None

    for vp in vps:
        xyz = list(vp["xyz"])
        try:
            move_xyz(xyz)
        except Exception as exc:  # noqa: BLE001
            views.append(
                ViewObservation(
                    ok=False,
                    name=str(vp.get("name") or "vp"),
                    xyz_cmd=xyz,
                    T_base_cam=None,
                    points_base=None,
                    plane_base=None,
                    message=f"move failed: {exc}",
                )
            )
            continue
        if settle_s > 0:
            time.sleep(float(settle_s))
        try:
            packet = grab_packet()
        except Exception as exc:  # noqa: BLE001
            views.append(
                ViewObservation(
                    ok=False,
                    name=str(vp.get("name") or "vp"),
                    xyz_cmd=xyz,
                    T_base_cam=None,
                    points_base=None,
                    plane_base=None,
                    message=f"grab failed: {exc}",
                )
            )
            continue

        T_bf = get_T_base_flange()
        T_bc = build_T_base_cam(
            T_flange_d405=T_flange_d405,
            T_base_flange=T_bf,
            viewpoint_xyz=xyz,
            ref_xyz=ref,
            dry_run_translate=dry_run_translate,
        )
        obs = capture_wrist_view(
            packet,
            name=str(vp.get("name") or "vp"),
            xyz_cmd=xyz,
            T_base_cam=T_bc,
            method=method,
            weights=weights,
            conf=conf,
            class_filter=class_filter,
            is_mock=is_mock,
            depth_scale=depth_scale,
            ransac_thresh=ransac_thresh,
            max_points=max_points,
            make_viz=True,
            device=device,
            imgsz=imgsz,
        )
        if obs.viz_jpeg:
            last_viz = obs.viz_jpeg
        views.append(obs)

    ok_views = [v for v in views if v.ok and v.points_base is not None and v.plane_base is not None]
    lift = float(pregrasp_z_lift_m)

    def _fail(msg: str, fallback: str = "") -> MultiviewFuseResult:
        return MultiviewFuseResult(
            ok=False,
            fine=None,
            views=views,
            viewpoints=vps,
            message=msg,
            fallback=fallback,
            pregrasp_z_lift_m=lift,
        )

    if len(ok_views) < int(min_views_ok):
        # M3.6 fallback: best single view if any
        if ok_views:
            best = max(ok_views, key=lambda v: float(v.plane_base.inlier_ratio if v.plane_base else 0))
            c = list(best.plane_base.centroid)  # type: ignore[union-attr]
            n = list(best.plane_base.normal)  # type: ignore[union-attr]
            fine = FineResult(
                ok=True,
                c_star=c,
                n_star=n,
                c_cam=None,
                n_cam=None,
                inlier_ratio=float(best.plane_base.inlier_ratio),  # type: ignore[union-attr]
                num_points=best.num_points,
                fuse_quality=float(best.plane_base.inlier_ratio),  # type: ignore[union-attr]
                serial=serial or (best.detection or {}).get("serial") or "",
                timestamp_ms=best.timestamp_ms,
                detection=best.detection,
                frame="base",
                message="fallback single best view",
                viz_jpeg=last_viz or best.viz_jpeg,
            )
            pre = [c[0], c[1], c[2] + lift]
            return MultiviewFuseResult(
                ok=True,
                fine=fine,
                views=views,
                viewpoints=vps,
                p_pregrasp=pre,
                pregrasp_z_lift_m=lift,
                message=f"only {len(ok_views)} ok views (<{min_views_ok}); used best single view",
                fallback="single_best_view",
            )
        return _fail(f"insufficient ok views: {len(ok_views)}/{len(views)} (need ≥{min_views_ok})")

    clouds = [v.points_base for v in ok_views if v.points_base is not None]
    fused, vox_stats = voxel_fuse_points(clouds, voxel_m=voxel_m)
    if fused.shape[0] < int(min_points_total):
        return _fail(f"fused cloud too small: {fused.shape[0]} (<{min_points_total})", fallback="none")

    planes = [v.plane_base for v in ok_views if v.plane_base is not None]
    plane_star, plane_stats = fuse_planes(planes)
    if plane_star is None:
        return _fail(f"plane fuse failed: {plane_stats.get('reason')}", fallback="none")

    xcheck = cross_check_fused_cloud(plane_star, fused, ransac_thresh=ransac_thresh)
    c_star = list(xcheck.get("preferred_centroid") or plane_star.centroid)
    n_star = list(xcheck.get("preferred_normal") or plane_star.normal)
    quality = float(plane_star.inlier_ratio)
    if xcheck.get("ok"):
        quality = min(1.0, quality + 0.05)

    fine = FineResult(
        ok=True,
        c_star=c_star,
        n_star=n_star,
        c_cam=None,
        n_cam=None,
        inlier_ratio=float(plane_star.inlier_ratio),
        num_points=int(fused.shape[0]),
        fuse_quality=quality,
        serial=serial,
        timestamp_ms=float(ok_views[-1].timestamp_ms),
        detection={"multiview": True, "num_views_ok": len(ok_views)},
        frame="base",
        message="ok" if xcheck.get("ok") else f"fused with weak cross-check: {xcheck.get('message')}",
        viz_jpeg=last_viz,
    )
    pre = [c_star[0], c_star[1], c_star[2] + lift]
    return MultiviewFuseResult(
        ok=True,
        fine=fine,
        views=views,
        viewpoints=vps,
        voxel_stats=vox_stats,
        plane_fuse_stats=plane_stats,
        cross_check=dict(xcheck),
        p_pregrasp=pre,
        pregrasp_z_lift_m=lift,
        message=fine.message,
        fallback="",
    )
