"""Build grasp-service ``POST /v1/site/configure`` payload from rs-detect station data."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from rs_detect.calib.extrinsics import IDENTITY_4X4


def _mat4(raw: Any) -> List[List[float]]:
    if raw is None:
        return [row[:] for row in IDENTITY_4X4]
    arr = np.asarray(raw, dtype=float).reshape(4, 4)
    return [[float(arr[i, j]) for j in range(4)] for i in range(4)]


def _is_identity(mat: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    eye = np.eye(4)
    return bool(np.allclose(np.asarray(mat, dtype=float), eye, atol=tol))


def _xyz_rpy_to_mat4(xyz: Sequence[float], rpy: Sequence[float]) -> List[List[float]]:
    R = Rotation.from_euler("xyz", [float(v) for v in list(rpy)[:3]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = float(xyz[0])
    T[1, 3] = float(xyz[1])
    T[2, 3] = float(xyz[2])
    return [[float(T[i, j]) for j in range(4)] for i in range(4)]


def mat4_to_xyz_rpy(raw: Any) -> Dict[str, List[float]]:
    """4×4 → xyz_m + rpy_rad/deg (scipy euler ``xyz``, same as ``_xyz_rpy_to_mat4``)."""
    T = np.asarray(_mat4(raw), dtype=float)
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    xyz = [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])]
    rpy_rad = [float(v) for v in rpy]
    rpy_deg = [float(math.degrees(v)) for v in rpy_rad]
    return {"xyz_m": xyz, "rpy_rad": rpy_rad, "rpy_deg": rpy_deg}


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    C = A @ B
    return [[float(C[i, j]) for j in range(4)] for i in range(4)]


def _pose_or_mat_to_mat4(
    block: Dict[str, Any],
    *,
    mat_key: str,
    xyz_key: str = "xyz_m",
    rpy_key: str = "rpy_rad",
    allow_null: bool = False,
) -> Optional[List[List[float]]]:
    """Prefer xyz+rpy (UI) over raw 4×4 when rebuilding transforms."""
    xyz = block.get(xyz_key)
    rpy = block.get(rpy_key)
    if isinstance(xyz, (list, tuple)) and isinstance(rpy, (list, tuple)) and len(xyz) >= 3 and len(rpy) >= 3:
        return _xyz_rpy_to_mat4(xyz, rpy)
    if mat_key in block:
        if block.get(mat_key) is None and allow_null:
            return None
        return _mat4(block.get(mat_key))
    if allow_null:
        return None
    return _mat4(None)


def normalize_site_configure_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived 4×4 fields from xyz+rpy (preferred) before POST."""
    out = dict(payload or {})
    tcp = dict(out.get("tcp") or {})
    xyz = tcp.get("tcp_xyz_m")
    rpy = tcp.get("gripper_rpy_rad")
    if isinstance(xyz, (list, tuple)) and isinstance(rpy, (list, tuple)) and len(xyz) >= 3 and len(rpy) >= 3:
        tcp["transform_flange_tcp_m"] = _xyz_rpy_to_mat4(xyz, rpy)
        tcp["tcp_xyz_m"] = [float(x) for x in list(xyz)[:3]]
        tcp["gripper_rpy_rad"] = [float(x) for x in list(rpy)[:3]]
        out["tcp"] = tcp

    he = dict(out.get("hand_eye") or {})
    T_scene = _pose_or_mat_to_mat4(
        he,
        mat_key="transform_base_from_scene_camera_m",
        xyz_key="xyz_m",
        rpy_key="rpy_rad",
    )
    if T_scene is not None:
        he["transform_base_from_scene_camera_m"] = T_scene
        pose = mat4_to_xyz_rpy(T_scene)
        he["xyz_m"] = pose["xyz_m"]
        he["rpy_rad"] = pose["rpy_rad"]
        out["hand_eye"] = he

    wh = dict(out.get("wrist_hand_eye") or {})
    if wh:
        T_wf = _pose_or_mat_to_mat4(
            wh,
            mat_key="transform_flange_from_wrist_camera_m",
            xyz_key="flange_xyz_m",
            rpy_key="flange_rpy_rad",
        )
        if T_wf is None and (wh.get("xyz_m") is not None or wh.get("rpy_rad") is not None):
            # allow alternate keys if UI sent flat xyz/rpy for flange
            T_wf = _pose_or_mat_to_mat4(
                wh,
                mat_key="transform_flange_from_wrist_camera_m",
                xyz_key="xyz_m",
                rpy_key="rpy_rad",
            )
        if T_wf is not None:
            wh["transform_flange_from_wrist_camera_m"] = T_wf
            pose_f = mat4_to_xyz_rpy(T_wf)
            wh["flange_xyz_m"] = pose_f["xyz_m"]
            wh["flange_rpy_rad"] = pose_f["rpy_rad"]

        has_base_pose = (
            wh.get("base_xyz_m") is not None
            or wh.get("base_rpy_rad") is not None
            or "transform_base_from_wrist_camera_m" in wh
        )
        if has_base_pose:
            T_wb = _pose_or_mat_to_mat4(
                {
                    "xyz_m": wh.get("base_xyz_m"),
                    "rpy_rad": wh.get("base_rpy_rad"),
                    "transform_base_from_wrist_camera_m": wh.get("transform_base_from_wrist_camera_m"),
                },
                mat_key="transform_base_from_wrist_camera_m",
                allow_null=True,
            )
            wh["transform_base_from_wrist_camera_m"] = T_wb
            if T_wb is not None:
                pose_b = mat4_to_xyz_rpy(T_wb)
                wh["base_xyz_m"] = pose_b["xyz_m"]
                wh["base_rpy_rad"] = pose_b["rpy_rad"]
        out["wrist_hand_eye"] = wh
    return out


def build_site_configure_payload(
    *,
    extrinsics: Dict[str, Any],
    arm_cfg: Dict[str, Any],
    grasp_cfg: Dict[str, Any],
    selection: Optional[Dict[str, Any]] = None,
    T_base_flange: Optional[Sequence[Sequence[float]]] = None,
    pipeline_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble site configure body from local station configs.

    Mapping (rs-detect → grasp-service):
    - hand_eye.transform_base_from_scene_camera_m ← extrinsics.d435.T (T_base_cam)
    - wrist: prefer T_base_wrist = T_base_flange @ T_flange_cam when FK/base flange known;
      always include transform_flange_from_wrist_camera_m ← extrinsics.d405.T
    - tcp ← robot_cell_c arm.tcp_xyz + gripper_rpy (+ gripper widths from grasp_cfg if any)
    """
    selection = selection or {}
    pipe = pipeline_cfg or {}
    d435 = extrinsics.get("d435") or {}
    d405 = extrinsics.get("d405") or {}
    T_scene = _mat4(d435.get("T"))
    T_flange_wrist = _mat4(d405.get("T"))

    scene_calibrated = (not _is_identity(T_scene)) and bool(d435.get("serial") or selection.get("scene"))
    hand_eye = {
        "calibrated": bool(scene_calibrated),
        "transform_base_from_scene_camera_m": T_scene,
        "frame_from": d435.get("frame_from") or "base",
        "frame_to": d435.get("frame_to") or "scene_camera",
        "serial": d435.get("serial") or selection.get("scene") or selection.get("d435_serial") or "",
    }

    T_bf = _mat4(T_base_flange) if T_base_flange is not None else None
    if T_bf is None and pipe.get("assume_flange_identity", True):
        T_bf = _mat4(IDENTITY_4X4)
    T_base_wrist = None
    wrist_source = "flange_only"
    if T_bf is not None:
        T_base_wrist = _matmul(T_bf, T_flange_wrist)
        wrist_source = "compose_base_flange_x_flange_cam"
    wrist_hand_eye = {
        "transform_base_from_wrist_camera_m": T_base_wrist,
        "transform_flange_from_wrist_camera_m": T_flange_wrist,
        "source": wrist_source,
        "serial": d405.get("serial") or selection.get("wrist") or selection.get("d405_serial") or "",
        "note": (
            "rs-detect D405 is eye-in-hand T_flange_cam; base_from_wrist composed when T_base_flange available. "
            "grasp-service HTTP S2 does not use wrist→base — only wrist color_intrinsics + image."
        ),
    }

    tcp_xyz = list(arm_cfg.get("tcp_xyz") or [0.0, 0.0, 0.117])
    gripper_rpy = list(arm_cfg.get("gripper_rpy") or [0.0, 0.0, math.pi])
    T_flange_tcp = _xyz_rpy_to_mat4(tcp_xyz, gripper_rpy)
    tcp = {
        "transform_flange_tcp_m": T_flange_tcp,
        "tcp_xyz_m": [float(x) for x in tcp_xyz[:3]],
        "gripper_rpy_rad": [float(x) for x in gripper_rpy[:3]],
        "gripper": {
            "action_default": "set_width_at_pregrasp",
            "width_m_open": float(grasp_cfg.get("gripper_width_m_open") or 0.08),
            "width_m_closed": float(grasp_cfg.get("gripper_width_m_closed") or 0.0),
        },
    }

    sam = {
        "material_text": str(grasp_cfg.get("sam_text") or "white plastic part"),
        "hole_sam_text": str(grasp_cfg.get("hole_sam_text") or "large rectangular hole opening"),
    }

    motion = {
        "retreat_m": float(grasp_cfg.get("retreat_m") or 0.05),
    }

    final_approach = {
        "approach": {
            "insert_total_m": float(grasp_cfg.get("insert_total_m") or 0.012),
            "step_m": float(grasp_cfg.get("step_m") or 0.002),
            "linear_speed_m_s": float(grasp_cfg.get("linear_speed_m_s") or 0.004),
            "direction_sign": int(grasp_cfg.get("direction_sign") or 1),
            "pause_ms_between_steps": int(grasp_cfg.get("pause_ms_between_steps") or 80),
        },
        "gripper": {
            "action_after_approach": "close",
            "width_m_command": float(grasp_cfg.get("gripper_width_m_closed") or 0.0),
        },
        "policy": {
            # Server yaml key (final_approach*.yaml); request field remains force_without_s2_exit.
            "allow_force_without_s2_exit": bool(grasp_cfg.get("force_without_s2_exit") or False),
        },
    }

    meta = {
        "client": "rs-detect",
        "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "robot_id": arm_cfg.get("id") or "arm-c",
        "robot_ip": arm_cfg.get("robot_ip"),
        "selection": {
            "scene": selection.get("scene") or selection.get("d435_serial") or "",
            "wrist": selection.get("wrist") or selection.get("d405_serial") or "",
        },
        "extrinsics_notes": extrinsics.get("notes") or "",
        "d435_serial": d435.get("serial") or "",
        "d405_serial": d405.get("serial") or "",
    }

    return {
        "hand_eye": hand_eye,
        "wrist_hand_eye": wrist_hand_eye,
        "tcp": tcp,
        "sam": sam,
        "motion": motion,
        "final_approach": final_approach,
        "meta": meta,
    }
