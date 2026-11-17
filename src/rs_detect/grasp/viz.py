"""Client-side overlays for grasp-service hole / S0 results on the uploaded scene frame."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _b64_to_gray(raw: Any) -> Optional[np.ndarray]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        buf = base64.b64decode(raw)
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _pick_mask(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in (
        "mask_b64",
        "sam_mask_b64",
        "object_mask_b64",
        "material_mask_b64",
        "vis_mask_b64",
    ):
        m = _b64_to_gray(obj.get(key))
        if m is not None:
            return m
    nested = obj.get("mask") if isinstance(obj.get("mask"), dict) else None
    if nested:
        for key in ("b64", "mask_b64", "png_b64"):
            m = _b64_to_gray(nested.get(key))
            if m is not None:
                return m
    return None


def _pick_server_vis(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("vis_b64", "overlay_b64", "debug_image_b64", "annotated_b64", "preview_b64"):
        raw = obj.get(key)
        if not raw or not isinstance(raw, str):
            continue
        try:
            buf = base64.b64decode(raw)
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:  # noqa: BLE001
            continue
    return None


def project_cam_to_uv(
    point_cam: Sequence[float],
    intr: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    if point_cam is None or len(point_cam) < 3:
        return None
    if any(v is None for v in list(point_cam)[:3]):
        return None
    x, y, z = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
    if z <= 1e-6:
        return None
    fx = float(intr.get("fx") or 0.0)
    fy = float(intr.get("fy") or 0.0)
    ppx = float(intr.get("ppx") or 0.0)
    ppy = float(intr.get("ppy") or 0.0)
    if fx <= 0 or fy <= 0:
        return None
    u = int(round(fx * x / z + ppx))
    v = int(round(fy * y / z + ppy))
    return u, v


def _blend_mask(vis: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    h, w = vis.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    m = mask > 0
    if not np.any(m):
        return vis
    overlay = vis.copy()
    overlay[m] = color
    out = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return out


def _draw_point(
    vis: np.ndarray,
    uv: Optional[Tuple[int, int]],
    *,
    color: Tuple[int, int, int],
    label: str,
    marker=cv2.MARKER_CROSS,
) -> None:
    if uv is None:
        return
    u, v = uv
    h, w = vis.shape[:2]
    if u < 0 or v < 0 or u >= w or v >= h:
        return
    cv2.drawMarker(vis, (u, v), color, markerType=marker, markerSize=22, thickness=2)
    cv2.circle(vis, (u, v), 8, color, 2, lineType=cv2.LINE_AA)
    cv2.putText(
        vis,
        label,
        (min(w - 8, u + 10), max(18, v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_normal_arrow(
    vis: np.ndarray,
    center: Sequence[float],
    normal: Sequence[float],
    intr: Dict[str, Any],
    *,
    length_m: float = 0.04,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> None:
    if center is None or normal is None or len(center) < 3 or len(normal) < 3:
        return
    if any(v is None for v in list(center)[:3] + list(normal)[:3]):
        return
    c = np.asarray(center[:3], dtype=float)
    n = np.asarray(normal[:3], dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-9:
        return
    n = n / n_norm
    tip = c + n * float(length_m)
    uv0 = project_cam_to_uv(c, intr)
    uv1 = project_cam_to_uv(tip, intr)
    if uv0 is None or uv1 is None:
        return
    cv2.arrowedLine(vis, uv0, uv1, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(vis, "n", (uv1[0] + 4, uv1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _banner(vis: np.ndarray, lines: List[str], color: Tuple[int, int, int] = (240, 240, 240)) -> None:
    y = 22
    for line in lines:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 20


def render_materials_viz(
    color_bgr: np.ndarray,
    materials: Sequence[Dict[str, Any]],
    *,
    selected_id: Optional[int] = None,
    raw_response: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Overlay numbered material bboxes for interactive pick."""
    raw = raw_response if isinstance(raw_response, dict) else {}
    server_vis = _pick_server_vis(raw)
    vis = server_vis.copy() if server_vis is not None else color_bgr.copy()
    h, w = vis.shape[:2]
    palette = [
        (40, 180, 40),
        (0, 165, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
        (80, 80, 255),
    ]
    items = [m for m in (materials or []) if isinstance(m, dict)]
    for i, m in enumerate(items):
        color = palette[i % len(palette)]
        try:
            mid = int(m.get("id", i))
        except (TypeError, ValueError):
            mid = i
        selected = selected_id is not None and int(selected_id) == mid
        thickness = 3 if selected else 2
        bbox = m.get("bbox_xyxy_px")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0, y0, x1, y1 = [int(round(float(v))) for v in list(bbox)[:4]]
            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w - 1, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h - 1, y1))
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
            label = f"#{mid}"
            score = m.get("score")
            if score is not None:
                try:
                    label += f" {float(score):.2f}"
                except (TypeError, ValueError):
                    pass
            cv2.putText(
                vis,
                label,
                (x0 + 4, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        # polygon overlay (normalized 0..1)
        poly = m.get("polygon_normalized")
        if isinstance(poly, (list, tuple)) and len(poly) >= 3:
            pts = []
            for p in poly:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([int(round(float(p[0]) * w)), int(round(float(p[1]) * h))])
            if len(pts) >= 3:
                arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [arr], True, color, 1, lineType=cv2.LINE_AA)
                if selected:
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, [arr], color)
                    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
    _banner(
        vis,
        [
            "MATERIALS detect",
            f"count={len(items)}  selected={selected_id if selected_id is not None else '—'}",
            "click a candidate in UI → hole detect on that crop",
        ],
    )
    return vis


def render_hole_viz(
    color_bgr: np.ndarray,
    intrinsics: Dict[str, Any],
    hole: Optional[Dict[str, Any]],
    *,
    raw_response: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    hole = hole if isinstance(hole, dict) else {}
    raw = raw_response if isinstance(raw_response, dict) else {}
    server_vis = _pick_server_vis(hole) or _pick_server_vis(raw)
    if server_vis is not None:
        vis = server_vis.copy()
    else:
        vis = color_bgr.copy()

    mask = _pick_mask(hole) or _pick_mask(raw)
    if mask is not None:
        vis = _blend_mask(vis, mask, (40, 180, 40), alpha=0.32)

    center = hole.get("center_camera_m")
    pre = hole.get("pregrasp_camera_m")
    normal = hole.get("plane_normal_camera") or hole.get("normal_camera")
    _draw_point(vis, project_cam_to_uv(center, intrinsics), color=(0, 0, 255), label="center")
    _draw_point(
        vis,
        project_cam_to_uv(pre, intrinsics),
        color=(255, 180, 0),
        label="pregrasp",
        marker=cv2.MARKER_TILTED_CROSS,
    )
    _draw_normal_arrow(vis, center or [], normal or [], intrinsics)

    # Optional pixel fields from server
    for key, label, color in (
        ("center_uv", "uv", (0, 0, 255)),
        ("u_v", "uv", (0, 0, 255)),
    ):
        uv = hole.get(key)
        if isinstance(uv, (list, tuple)) and len(uv) >= 2 and uv[0] is not None:
            _draw_point(vis, (int(uv[0]), int(uv[1])), color=color, label=label)

    tcp = hole.get("tcp_target") if isinstance(hole.get("tcp_target"), dict) else {}
    pos = tcp.get("position_m")
    tcp_ok = isinstance(pos, (list, tuple)) and len(pos) >= 3 and all(v is not None for v in list(pos)[:3])
    _banner(
        vis,
        [
            "HOLE detect",
            f"calibrated={hole.get('calibrated')}  tcp={'OK' if tcp_ok else 'null'}",
            "red=center  orange=pregrasp  yellow=n  green=mask",
        ],
    )
    return vis


def render_s0_viz(
    color_bgr: np.ndarray,
    intrinsics: Dict[str, Any],
    s0: Optional[Dict[str, Any]],
) -> np.ndarray:
    s0 = s0 if isinstance(s0, dict) else {}
    server_vis = _pick_server_vis(s0)
    vis = server_vis.copy() if server_vis is not None else color_bgr.copy()

    mask = _pick_mask(s0)
    qc = s0.get("mask_qc") if isinstance(s0.get("mask_qc"), dict) else {}
    if mask is None:
        mask = _pick_mask(qc)
    if mask is not None:
        vis = _blend_mask(vis, mask, (255, 128, 0), alpha=0.30)

    anchor = (
        s0.get("servo_anchor_scene_m")
        or (s0.get("motion") or {}).get("servo_anchor_scene_m")
        or s0.get("anchor_camera_m")
    )
    _draw_point(
        vis,
        project_cam_to_uv(anchor, intrinsics) if isinstance(anchor, (list, tuple)) else None,
        color=(255, 0, 255),
        label="servo_anchor",
        marker=cv2.MARKER_DIAMOND,
    )

    # Optional GraspNet camera points
    for g in (s0.get("grasps") or s0.get("candidates") or [])[:5]:
        if not isinstance(g, dict):
            continue
        pt = g.get("position_camera_m") or g.get("center_camera_m") or g.get("xyz_camera_m")
        _draw_point(
            vis,
            project_cam_to_uv(pt, intrinsics) if isinstance(pt, (list, tuple)) else None,
            color=(0, 255, 255),
            label="grasp",
            marker=cv2.MARKER_STAR,
        )

    grip = s0.get("gripper") if isinstance(s0.get("gripper"), dict) else {}
    sid = str(s0.get("session_id") or "")[:8]
    _banner(
        vis,
        [
            "S0 plan",
            f"session={sid or '—'}  calibrated={s0.get('calibrated')}",
            f"gripper.width_m={grip.get('width_m')}  action={grip.get('action')}",
            "magenta=servo_anchor  orange=SAM mask",
        ],
    )
    return vis


def encode_jpeg(vis: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()
