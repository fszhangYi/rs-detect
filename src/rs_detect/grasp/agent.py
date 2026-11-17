"""Local grasp Agent: capture RGB-D → grasp-service → optional Cell-C motion."""

from __future__ import annotations

import base64
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from rs_detect.grasp.client import GraspServiceClient, GraspServiceError
from rs_detect.grasp.frame_pack import (
    STOCK_MOCK_ID,
    list_capture_frames,
    load_mock_packet,
    mock_asset_paths,
    normalize_mock_source_id,
    pack_frame_upload,
)
from rs_detect.grasp.site_pack import build_site_configure_payload, normalize_site_configure_payload
from rs_detect.grasp.viz import encode_jpeg, render_hole_viz, render_materials_viz, render_s0_viz
from rs_detect.paths import GRASP_CLIENT_CONFIG_PATH

logger = logging.getLogger(__name__)

DEFAULT_CFG: Dict[str, Any] = {
    "base_url": "http://127.0.0.1:6010",
    "timeout_s": 60.0,
    "sam_text": "white plastic part",
    "hole_sam_text": "large rectangular hole opening",
    "offset_along_normal_mm": 5.0,
    "normal_direction": 1,
    # S0 = hole_center_base - s0_along_normal_m * plane_normal_base
    # (stand-off on the opposite side of the reported plane normal)
    "s0_along_normal_m": 0.07,
    "depth_scale": 1000.0,
    "include_s1": False,
    "prefer_hole_tcp": True,
    "execute_motion": False,
    "force_without_s2_exit": False,
    "use_mock_frames": False,
    # Mock RGB-D source: __stock__ = data/mock; else data/capture/<id> stem.
    "mock_scene_id": STOCK_MOCK_ID,
    "mock_wrist_id": STOCK_MOCK_ID,
    "s2_hz": 10.0,
    "s2_max_steps": 600,
    "insert_total_m": 0.012,
    "step_m": 0.002,
    "linear_speed_m_s": 0.004,
    "direction_sign": 1,
    "pause_ms_between_steps": 80,
    "retreat_m": 0.05,
    "gripper_width_m_open": 0.08,
    "gripper_width_m_closed": 0.0,
    # S0 GraspNet flange Z often opposite Elite tip (+Z). true → prefer Rx180 so tip points down.
    "s0_flip_approach": True,
    # S0 pre-grasp: force wrist J6 (rad). Default ≈ 4π (720° wrap).
    "s0_j6_rad": 12.56,
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quat_xyzw_to_rpy(q: Sequence[float]) -> List[float]:
    """Ultralytics/grasp quaternion xyzw → roll,pitch,yaw (xyz extrinsic rad)."""
    x, y, z, w = [float(v) for v in list(q)[:4]]
    return [float(v) for v in Rotation.from_quat([x, y, z, w]).as_euler("xyz")]


def _rpy_to_R(rpy: Sequence[float]) -> np.ndarray:
    return Rotation.from_euler("xyz", [float(v) for v in list(rpy)[:3]]).as_matrix()


def _R_to_quat_xyzw(R: np.ndarray) -> List[float]:
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


class GraspAgent:
    """One Agent instance shared by the web Grasp tab."""

    def __init__(self, service: Any = None) -> None:
        self._svc = service  # DetectService (set later if None)
        self._lock = threading.RLock()
        self._cfg = self._load_cfg()
        self._client = GraspServiceClient(
            base_url=str(self._cfg.get("base_url") or DEFAULT_CFG["base_url"]),
            timeout_s=float(self._cfg.get("timeout_s") or DEFAULT_CFG["timeout_s"]),
        )
        self._stage = "idle"
        self._session_id: Optional[str] = None
        self._error = ""
        self._last: Dict[str, Any] = {}
        self._log: Deque[str] = deque(maxlen=400)
        self._s2_thread: Optional[threading.Thread] = None
        self._s2_stop = threading.Event()
        self._s2_running = False
        self._s2_stats: Dict[str, Any] = {}
        self._hole: Optional[Dict[str, Any]] = None
        self._materials: List[Dict[str, Any]] = []
        self._material_selected_id: Optional[int] = None
        self._s0: Optional[Dict[str, Any]] = None
        self._final: Optional[Dict[str, Any]] = None
        self._pregrasp_target: Optional[Dict[str, Any]] = None
        self._s2_latched = False
        self._site: Optional[Dict[str, Any]] = None
        self._frame_cache: Dict[str, Dict[str, Any]] = {}
        self._viz_jpeg: Dict[str, bytes] = {}
        self._viz_meta: Dict[str, Any] = {}

    # ---- wiring ----
    def bind_service(self, service: Any) -> None:
        self._svc = service

    def _load_cfg(self) -> Dict[str, Any]:
        cfg = dict(DEFAULT_CFG)
        path = GRASP_CLIENT_CONFIG_PATH
        if path.is_file():
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if isinstance(raw, dict):
                    cfg.update(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("grasp_client.yaml load failed: %s", exc)
        return cfg

    def save_cfg(self, updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if updates:
                for k, v in updates.items():
                    if k in DEFAULT_CFG or k in self._cfg:
                        self._cfg[k] = v
            GRASP_CLIENT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            GRASP_CLIENT_CONFIG_PATH.write_text(
                yaml.safe_dump(self._cfg, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self._client.set_base_url(str(self._cfg.get("base_url") or DEFAULT_CFG["base_url"]))
            self._client.timeout_s = float(self._cfg.get("timeout_s") or DEFAULT_CFG["timeout_s"])
            return dict(self._cfg)

    def _log_line(self, msg: str) -> None:
        line = f"[{_utc_iso()}] {msg}"
        with self._lock:
            self._log.append(line)
        logger.info("grasp-agent %s", msg)

    # ---- frames ----
    def _require_svc(self) -> Any:
        if self._svc is None:
            raise RuntimeError("DetectService not bound")
        return self._svc

    def _serial_for_role(self, role: str) -> str:
        svc = self._require_svc()
        sel = svc.get_selection()
        if role == "wrist":
            serial = sel.get("wrist") or sel.get("d405_serial") or ""
        else:
            serial = sel.get("scene") or sel.get("d435_serial") or ""
        if not serial:
            raise RuntimeError(f"未选定{'腕部' if role == 'wrist' else '场景'}相机（先在 M0 出流）")
        return str(serial)

    def capture_frame(self, role: str = "scene", *, require_depth: bool = True) -> Dict[str, Any]:
        depth_scale = float(self._cfg.get("depth_scale") or 1000.0)
        use_mock = bool(self._cfg.get("use_mock_frames"))
        if use_mock:
            src_key = "mock_wrist_id" if str(role).lower() in {"wrist", "d405"} else "mock_scene_id"
            src_id = normalize_mock_source_id(self._cfg.get(src_key))
            pkt, info = load_mock_packet(role, capture_id=src_id)
            packed = pack_frame_upload(pkt, depth_scale=depth_scale, require_depth=require_depth)
            packed["_meta"] = {
                "role": role,
                "serial": pkt.serial,
                "source": "mock",
                "source_id": info.get("source_id") or src_id,
                "color": info.get("color"),
                "depth": info.get("depth"),
            }
        else:
            svc = self._require_svc()
            serial = self._serial_for_role(role)
            st = svc.streams.get(serial)
            if st is None:
                raise RuntimeError(f"相机未出流: {serial}（role={role}）")
            pkt = st.get_latest()
            if pkt is None:
                raise RuntimeError(f"尚无帧: {serial}")
            packed = pack_frame_upload(pkt, depth_scale=depth_scale, require_depth=require_depth)
            packed["_meta"] = {"role": role, "serial": serial, "source": "live"}

        # Cache raw color/depth for Hole/S0 overlays + material-crop hole detect.
        depth = None
        if getattr(pkt, "depth_u16", None) is not None:
            depth = np.asarray(pkt.depth_u16).copy()
        with self._lock:
            self._frame_cache[role] = {
                "color_bgr": np.asarray(pkt.color_bgr).copy(),
                "depth_u16": depth,
                "intrinsics": dict(packed.get("color_intrinsics") or {}),
                "depth_scale": float(packed.get("depth_scale") or depth_scale),
                "serial": packed.get("_meta", {}).get("serial") or pkt.serial,
                "source": packed.get("_meta", {}).get("source"),
                "captured_at": _utc_iso(),
            }
        return packed

    def _frame_body(self, role: str) -> Dict[str, Any]:
        packed = self.capture_frame(role)
        # Strip local meta before POST
        return {k: v for k, v in packed.items() if not k.startswith("_")}

    def _build_viz(self, kind: str) -> Optional[str]:
        """Render and cache jpeg for materials|hole|s0. Returns URL path or None."""
        kind = str(kind or "").lower()
        cache = self._frame_cache.get("scene")
        if not cache or cache.get("color_bgr") is None:
            return None
        color = cache["color_bgr"]
        intr = cache.get("intrinsics") or {}
        try:
            if kind == "materials":
                vis = render_materials_viz(
                    color,
                    self._materials,
                    selected_id=self._material_selected_id,
                    raw_response=((self._last or {}).get("materials_detect") or {}),
                )
            elif kind == "hole":
                vis = render_hole_viz(
                    color,
                    intr,
                    self._hole,
                    raw_response=((self._last or {}).get("hole_detect") or {}),
                )
            elif kind == "s0":
                vis = render_s0_viz(color, intr, self._s0)
            else:
                return None
            jpeg = encode_jpeg(vis)
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"viz/{kind} FAIL {exc}")
            return None
        with self._lock:
            self._viz_jpeg[kind] = jpeg
            self._viz_meta[kind] = {
                "at": _utc_iso(),
                "serial": cache.get("serial"),
                "source": cache.get("source"),
                "bytes": len(jpeg),
            }
        return f"/api/grasp/viz/{kind}.jpg"

    def get_viz_jpeg(self, kind: str) -> Optional[bytes]:
        kind = str(kind or "").lower()
        with self._lock:
            data = self._viz_jpeg.get(kind)
            return bytes(data) if data else None

    def set_mock_frames(
        self,
        enabled: Optional[bool] = None,
        *,
        scene_id: Optional[str] = None,
        wrist_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Toggle mock RGB-D and/or select data/capture source ids."""
        updates: Dict[str, Any] = {}
        if enabled is not None:
            updates["use_mock_frames"] = bool(enabled)
        if scene_id is not None:
            updates["mock_scene_id"] = normalize_mock_source_id(scene_id)
        if wrist_id is not None:
            updates["mock_wrist_id"] = normalize_mock_source_id(wrist_id)
        if updates:
            self.save_cfg(updates)

        on = bool(self._cfg.get("use_mock_frames"))
        scene = mock_asset_paths(
            "scene", capture_id=normalize_mock_source_id(self._cfg.get("mock_scene_id"))
        )
        wrist = mock_asset_paths(
            "wrist", capture_id=normalize_mock_source_id(self._cfg.get("mock_wrist_id"))
        )
        if on and not (scene.get("ready") and wrist.get("ready")):
            raise FileNotFoundError(
                f"mock 图不完整：scene={scene.get('color')} wrist={wrist.get('color')}"
            )
        self._log_line(
            f"use_mock_frames={on} scene_id={scene.get('source_id')} "
            f"wrist_id={wrist.get('source_id')} "
            f"scene={scene.get('color')} wrist={wrist.get('color')}"
        )
        return self.status()

    def mock_info(self) -> Dict[str, Any]:
        scene_id = normalize_mock_source_id(self._cfg.get("mock_scene_id"))
        wrist_id = normalize_mock_source_id(self._cfg.get("mock_wrist_id"))
        scene = mock_asset_paths("scene", capture_id=scene_id)
        wrist = mock_asset_paths("wrist", capture_id=wrist_id)
        captures = list_capture_frames()
        sources = [
            {
                "id": STOCK_MOCK_ID,
                "label": "stock · data/mock",
                "source": "stock",
                "ready": True,
            },
            *captures,
        ]
        # Cache-bust previews when source changes.
        q_scene = f"&id={scene_id}"
        q_wrist = f"&id={wrist_id}"
        return {
            "enabled": bool(self._cfg.get("use_mock_frames")),
            "ready": bool(scene.get("ready") and wrist.get("ready")),
            "scene_id": scene_id,
            "wrist_id": wrist_id,
            "sources": sources,
            "scene": {
                k: scene[k]
                for k in (
                    "role",
                    "camera_key",
                    "color",
                    "depth",
                    "meta",
                    "ready",
                    "source",
                    "source_id",
                )
                if k in scene
            },
            "wrist": {
                k: wrist[k]
                for k in (
                    "role",
                    "camera_key",
                    "color",
                    "depth",
                    "meta",
                    "ready",
                    "source",
                    "source_id",
                )
                if k in wrist
            },
            "preview": {
                "scene": {
                    "color": f"/api/grasp/mock/preview.jpg?role=scene&which=color{q_scene}",
                    "depth": f"/api/grasp/mock/preview.jpg?role=scene&which=depth{q_scene}",
                },
                "wrist": {
                    "color": f"/api/grasp/mock/preview.jpg?role=wrist&which=color{q_wrist}",
                    "depth": f"/api/grasp/mock/preview.jpg?role=wrist&which=depth{q_wrist}",
                },
            },
        }

    # ---- status ----
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "stage": self._stage,
                "session_id": self._session_id,
                "error": self._error,
                "config": dict(self._cfg),
                "mock": self.mock_info(),
                "s2_running": self._s2_running,
                "s2_latched": self._s2_latched,
                "s2_stats": dict(self._s2_stats),
                "hole": self._hole,
                "materials": {
                    "items": list(self._materials),
                    "selected_id": self._material_selected_id,
                    "count": len(self._materials),
                },
                "s0": self._s0,
                "final": self._final,
                "site": self._site,
                "pregrasp_target": self._pregrasp_target,
                "viz": {
                    "hole": "/api/grasp/viz/hole.jpg" if "hole" in self._viz_jpeg else None,
                    "materials": (
                        "/api/grasp/viz/materials.jpg"
                        if "materials" in self._viz_jpeg
                        else None
                    ),
                    "s0": "/api/grasp/viz/s0.jpg" if "s0" in self._viz_jpeg else None,
                    "meta": dict(self._viz_meta),
                },
                "last": self._last,
                "log_tail": list(self._log)[-80:],
                "selection": self._svc.get_selection() if self._svc else {},
            }

    def reset(self) -> Dict[str, Any]:
        self.stop_s2()
        with self._lock:
            self._stage = "idle"
            self._session_id = None
            self._error = ""
            self._last = {}
            self._hole = None
            self._materials = []
            self._material_selected_id = None
            self._s0 = None
            self._final = None
            self._site = None
            self._pregrasp_target = None
            self._s2_latched = False
            self._s2_stats = {}
            self._frame_cache = {}
            self._viz_jpeg = {}
            self._viz_meta = {}
            self._log.append(f"[{_utc_iso()}] reset")
        return self.status()

    # ---- remote steps ----
    def _resolve_T_base_flange(self) -> Optional[List[List[float]]]:
        """Prefer live FK→flange; else extrinsics robot.T_base_flange / assume identity."""
        svc = self._require_svc()
        pipe = svc._load_pipeline_cfg() if hasattr(svc, "_load_pipeline_cfg") else {}
        try:
            pose = svc.arm.fk() if getattr(svc, "arm", None) else None
            if pose and len(pose) >= 6:
                T_base_tcp = np.eye(4)
                T_base_tcp[:3, :3] = _rpy_to_R(pose[3:6])
                T_base_tcp[0, 3] = float(pose[0])
                T_base_tcp[1, 3] = float(pose[1])
                T_base_tcp[2, 3] = float(pose[2])
                tcp_xyz = list((svc.arm.arm_cfg or {}).get("tcp_xyz") or [0.0, 0.0, 0.117])
                T_flange_tcp = np.eye(4)
                T_flange_tcp[0, 3] = float(tcp_xyz[0])
                T_flange_tcp[1, 3] = float(tcp_xyz[1])
                T_flange_tcp[2, 3] = float(tcp_xyz[2])
                T_bf = T_base_tcp @ np.linalg.inv(T_flange_tcp)
                return [[float(T_bf[i, j]) for j in range(4)] for i in range(4)]
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"WARN live T_base_flange from FK failed: {exc}")
        if hasattr(svc, "_T_base_flange"):
            return svc._T_base_flange(pipe)
        return None

    def build_site_payload(self) -> Dict[str, Any]:
        svc = self._require_svc()
        extrinsics = svc.get_extrinsics() if hasattr(svc, "get_extrinsics") else {}
        arm_cfg = dict(getattr(getattr(svc, "arm", None), "arm_cfg", None) or {})
        selection = svc.get_selection() if hasattr(svc, "get_selection") else {}
        pipe = svc._load_pipeline_cfg() if hasattr(svc, "_load_pipeline_cfg") else {}
        return build_site_configure_payload(
            extrinsics=extrinsics if isinstance(extrinsics, dict) else {},
            arm_cfg=arm_cfg,
            grasp_cfg=dict(self._cfg),
            selection=selection if isinstance(selection, dict) else {},
            T_base_flange=self._resolve_T_base_flange(),
            pipeline_cfg=pipe if isinstance(pipe, dict) else {},
        )

    def site_config(self) -> Dict[str, Any]:
        """Fetch remote site schema / effective config (does not change local stage alone)."""
        try:
            data = self._client.site_config()
            with self._lock:
                self._last = {**self._last, "site_config": data}
                self._error = ""
            self._log_line("site/config ok")
            return {**self.status(), "ok": True, "site_config": data}
        except GraspServiceError as exc:
            with self._lock:
                self._error = str(exc)
            self._log_line(f"site/config FAIL {exc}")
            return {**self.status(), "ok": False, "error": str(exc)}

    def _persist_site_to_local_cfg(self, payload: Dict[str, Any]) -> None:
        """Mirror editable site fields back into grasp_client.yaml."""
        sam = payload.get("sam") if isinstance(payload.get("sam"), dict) else {}
        motion = payload.get("motion") if isinstance(payload.get("motion"), dict) else {}
        fa = payload.get("final_approach") if isinstance(payload.get("final_approach"), dict) else {}
        approach = fa.get("approach") if isinstance(fa.get("approach"), dict) else {}
        policy = fa.get("policy") if isinstance(fa.get("policy"), dict) else {}
        tcp = payload.get("tcp") if isinstance(payload.get("tcp"), dict) else {}
        gripper = tcp.get("gripper") if isinstance(tcp.get("gripper"), dict) else {}
        updates: Dict[str, Any] = {}
        if sam.get("material_text") is not None:
            updates["sam_text"] = str(sam.get("material_text"))
        if sam.get("hole_sam_text") is not None:
            updates["hole_sam_text"] = str(sam.get("hole_sam_text"))
        if motion.get("retreat_m") is not None:
            updates["retreat_m"] = float(motion.get("retreat_m"))
        if approach.get("insert_total_m") is not None:
            updates["insert_total_m"] = float(approach.get("insert_total_m"))
        if approach.get("step_m") is not None:
            updates["step_m"] = float(approach.get("step_m"))
        if approach.get("linear_speed_m_s") is not None:
            updates["linear_speed_m_s"] = float(approach.get("linear_speed_m_s"))
        if approach.get("direction_sign") is not None:
            updates["direction_sign"] = int(approach.get("direction_sign"))
        if approach.get("pause_ms_between_steps") is not None:
            updates["pause_ms_between_steps"] = int(approach.get("pause_ms_between_steps"))
        if policy.get("allow_force_without_s2_exit") is not None:
            updates["force_without_s2_exit"] = bool(policy.get("allow_force_without_s2_exit"))
        elif policy.get("force_without_s2_exit") is not None:
            # legacy alias from older client drafts
            updates["force_without_s2_exit"] = bool(policy.get("force_without_s2_exit"))
        if gripper.get("width_m_open") is not None:
            updates["gripper_width_m_open"] = float(gripper.get("width_m_open"))
        if gripper.get("width_m_closed") is not None:
            updates["gripper_width_m_closed"] = float(gripper.get("width_m_closed"))
        if updates:
            self.save_cfg(updates)

    def site_sync(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Step 1: push site payload to grasp-service (UI draft or auto-built from station)."""
        try:
            if payload is None:
                payload = self.build_site_payload()
            else:
                if not isinstance(payload, dict):
                    raise RuntimeError("site payload must be a JSON object")
                payload = normalize_site_configure_payload(dict(payload))
                meta = dict(payload.get("meta") or {})
                meta["client"] = meta.get("client") or "rs-detect"
                meta["synced_at"] = _utc_iso()
                payload["meta"] = meta
                self._persist_site_to_local_cfg(payload)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stage = "site_bad"
                self._error = str(exc)
            self._log_line(f"site payload build FAIL {exc}")
            return {**self.status(), "ok": False, "error": str(exc)}

        remote_before: Optional[Dict[str, Any]] = None
        try:
            remote_before = self._client.site_config()
        except GraspServiceError as exc:
            self._log_line(f"WARN site/config before sync: {exc}")

        try:
            data = self._client.site_configure(payload)
        except GraspServiceError as exc:
            with self._lock:
                self._stage = "site_bad"
                self._error = str(exc)
                self._last = {"site_sync": {"payload": payload, "error": str(exc)}}
            self._log_line(f"site/configure FAIL {exc}")
            return {**self.status(), "ok": False, "error": str(exc), "payload": payload}

        remote_after: Optional[Dict[str, Any]] = None
        try:
            remote_after = self._client.site_config()
        except GraspServiceError:
            remote_after = None

        summary = {
            "hand_eye_calibrated": ((payload.get("hand_eye") or {}).get("calibrated")),
            "sam": payload.get("sam"),
            "retreat_m": ((payload.get("motion") or {}).get("retreat_m")),
            "robot_id": ((payload.get("meta") or {}).get("robot_id")),
            "wrist_source": ((payload.get("wrist_hand_eye") or {}).get("source")),
        }
        with self._lock:
            self._site = {
                "synced": True,
                "summary": summary,
                "configure_response": data,
                "remote_before": remote_before,
                "remote_after": remote_after,
                "last_payload": payload,
            }
            self._stage = "site_ok"
            self._last = {
                "site_sync": {
                    "payload": payload,
                    "response": data,
                    "remote_after": remote_after,
                }
            }
            self._error = ""
        self._log_line(
            f"site/configure ok calibrated={summary.get('hand_eye_calibrated')} "
            f"retreat_m={summary.get('retreat_m')} robot={summary.get('robot_id')}"
        )
        return {
            **self.status(),
            "ok": True,
            "payload": payload,
            "response": data,
            "site_config": remote_after,
        }

    def health(self) -> Dict[str, Any]:
        """Optional connectivity probe (not pipeline step 1)."""
        try:
            data = self._client.health()
            with self._lock:
                self._last = {**self._last, "health": data}
                self._error = ""
            self._log_line(f"health ok={data.get('ok')} stages={data.get('stages')}")
            return {**self.status(), "ok": True, "health": data}
        except GraspServiceError as exc:
            with self._lock:
                self._error = str(exc)
            self._log_line(f"health FAIL {exc}")
            return {**self.status(), "ok": False, "error": str(exc)}

    @staticmethod
    def _position_usable(obj: Optional[Dict[str, Any]]) -> bool:
        """True when pose dict has a non-null position_m (or xyz fallback)."""
        if not isinstance(obj, dict):
            return False
        pos = obj.get("position_m")
        if pos is None:
            pos = obj.get("xyz_m") or obj.get("position")
        if pos is None:
            return False
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return all(v is not None for v in list(pos)[:3])
        return False

    @staticmethod
    def _sanitize_material(item: Dict[str, Any]) -> Dict[str, Any]:
        """Keep UI-friendly material fields (drop huge blobs)."""
        out: Dict[str, Any] = {}
        for k in (
            "id",
            "cls",
            "sam_text",
            "score",
            "area_px",
            "bbox_xyxy_px",
            "center_pixel",
            "polygon_normalized",
        ):
            if k in item:
                out[k] = item[k]
        try:
            out["id"] = int(out.get("id", 0))
        except (TypeError, ValueError):
            out["id"] = 0
        return out

    def materials_detect(self) -> Dict[str, Any]:
        """Detect all material instances (SAM); user selects one before hole/S0."""
        frame = self._frame_body("scene")
        # materials/detect only needs color (+ optional sam_text).
        color_body = {
            "color_b64": frame.get("color_b64"),
            "sam_text": self._cfg.get("sam_text"),
        }
        try:
            data = self._client.materials_detect(color_body)
        except GraspServiceError as exc:
            with self._lock:
                self._error = str(exc)
                self._stage = "error"
            self._log_line(f"materials/detect FAIL {exc}")
            raise
        raw_items = data.get("materials") if isinstance(data, dict) else None
        if not isinstance(raw_items, list):
            raw_items = []
        items = [
            self._sanitize_material(m)
            for m in raw_items
            if isinstance(m, dict)
        ]
        items.sort(key=lambda m: (-float(m.get("score") or 0.0), -float(m.get("area_px") or 0.0)))
        with self._lock:
            self._materials = items
            self._material_selected_id = None
            self._hole = None  # must re-select before S0
            self._last = {"materials_detect": data}
            self._stage = "materials"
            self._error = ""
        viz_url = self._build_viz("materials")
        self._log_line(
            f"materials/detect ok count={len(items)} sam_text={self._cfg.get('sam_text')!r} "
            f"viz={bool(viz_url)}"
        )
        return {
            **self.status(),
            "ok": True,
            "count": len(items),
            "result": {"ok": bool(data.get("ok", True)), "count": len(items), "materials": items},
            "viz_url": viz_url,
        }

    def select_material(self, material_id: int) -> Dict[str, Any]:
        """Select a materials/detect instance → crop hole/detect → set active hole."""
        try:
            mid = int(material_id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid material_id: {material_id}") from exc
        with self._lock:
            items = list(self._materials)
        match = next((m for m in items if int(m.get("id", -1)) == mid), None)
        if match is None:
            raise RuntimeError(f"物料 id={mid} 不在候选列表（先跑「检测全部物料」）")

        cache = self._frame_cache.get("scene") or {}
        color = cache.get("color_bgr")
        depth = cache.get("depth_u16")
        intr = dict(cache.get("intrinsics") or {})
        if color is None:
            # Re-capture if cache missing (e.g. after restart).
            self._frame_body("scene")
            cache = self._frame_cache.get("scene") or {}
            color = cache.get("color_bgr")
            depth = cache.get("depth_u16")
            intr = dict(cache.get("intrinsics") or {})
        if color is None or depth is None:
            raise RuntimeError("无场景 RGB-D 缓存 — 先出流或启用 mock 后再检测物料")

        bbox = match.get("bbox_xyxy_px") or []
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            raise RuntimeError(f"物料 id={mid} 缺少 bbox_xyxy_px")
        x0, y0, x1, y1 = [int(round(float(v))) for v in list(bbox)[:4]]
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(f"物料 id={mid} bbox 无效: {bbox}")
        pad = 24
        h, w = color.shape[:2]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)
        crop_color = color[y0:y1, x0:x1]
        crop_depth = depth[y0:y1, x0:x1]
        if crop_color.size == 0 or crop_depth.size == 0:
            raise RuntimeError("裁剪区域为空")

        ok_c, buf_c = cv2.imencode(".png", crop_color)
        ok_d, buf_d = cv2.imencode(".png", crop_depth)
        if not ok_c or not ok_d:
            raise RuntimeError("裁剪帧 PNG 编码失败")

        crop_intr = {
            "fx": float(intr.get("fx") or 0.0),
            "fy": float(intr.get("fy") or 0.0),
            "ppx": float(intr.get("ppx") or 0.0) - float(x0),
            "ppy": float(intr.get("ppy") or 0.0) - float(y0),
            "width": int(crop_color.shape[1]),
            "height": int(crop_color.shape[0]),
        }
        frame = {
            "color_b64": base64.b64encode(buf_c.tobytes()).decode("ascii"),
            "depth_b64": base64.b64encode(buf_d.tobytes()).decode("ascii"),
            "color_intrinsics": crop_intr,
            "depth_scale": float(cache.get("depth_scale") or self._cfg.get("depth_scale") or 1000.0),
            "sam_text": self._cfg.get("sam_text") or match.get("sam_text"),
            "offset_along_normal_mm": self._cfg.get("offset_along_normal_mm"),
            "normal_direction": self._cfg.get("normal_direction"),
        }
        try:
            data = self._client.hole_detect(frame)
        except GraspServiceError as exc:
            with self._lock:
                self._error = str(exc)
                self._stage = "error"
            self._log_line(f"hole/detect(crop mat={mid}) FAIL {exc}")
            raise

        hole = data.get("hole") if isinstance(data.get("hole"), dict) else data
        if not isinstance(hole, dict):
            raise RuntimeError("hole/detect 未返回 hole 对象")
        hole = dict(hole)
        # Remap crop pixels → full-frame for viz / UI.
        cp = hole.get("center_pixel")
        if isinstance(cp, dict) and cp.get("u") is not None and cp.get("v") is not None:
            hole["center_pixel"] = {
                "u": float(cp["u"]) + float(x0),
                "v": float(cp["v"]) + float(y0),
            }
        elif isinstance(cp, (list, tuple)) and len(cp) >= 2:
            hole["center_pixel"] = [float(cp[0]) + float(x0), float(cp[1]) + float(y0)]
        hole["material_id"] = mid
        hole["material"] = dict(match)
        hole["crop_bbox_xyxy_px"] = [x0, y0, x1, y1]

        try:
            geom = self._hole_geometry_base(hole)
            hole["center_base_m"] = geom["center_base_m"]
            hole["plane_normal_base"] = geom["normal_base"]
            hole["s0_pregrasp_base_m"] = geom["s0_position_m"]
            hole["s0_along_normal_m"] = geom["offset_m"]
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"WARN hole→base geometry (mat={mid}): {exc}")

        data = dict(data) if isinstance(data, dict) else {"hole": hole}
        data["hole"] = hole
        data["material_id"] = mid

        tcp = hole.get("tcp_target") if isinstance(hole.get("tcp_target"), dict) else None
        tcp_ready = self._position_usable(tcp if isinstance(tcp, dict) else None)
        with self._lock:
            self._material_selected_id = mid
            self._hole = hole
            self._last = {"hole_detect": data, "material_select": {"id": mid}}
            self._stage = "hole"
            self._error = ""
        viz_m = self._build_viz("materials")
        viz_h = self._build_viz("hole")
        self._log_line(
            f"select material id={mid} → hole ok tcp_usable={tcp_ready} "
            f"center={hole.get('center_camera_m')} s0={hole.get('s0_pregrasp_base_m')} "
            f"viz_m={bool(viz_m)} viz_h={bool(viz_h)}"
        )
        return {
            **self.status(),
            "ok": True,
            "material_id": mid,
            "result": data,
            "viz_url": viz_h,
        }

    def hole_detect(self) -> Dict[str, Any]:
        """Legacy single-shot hole/detect on full frame (no material pick)."""
        frame = self._frame_body("scene")
        try:
            # HoleDetectRequest: sam_text = material SAM mask text (NOT hole_sam_text).
            # Hole SAM text lives in server sam_stl_map.yaml; override only via S2 hole_sam_text.
            data = self._client.hole_detect(
                frame,
                offset_along_normal_mm=self._cfg.get("offset_along_normal_mm"),
                normal_direction=self._cfg.get("normal_direction"),
                sam_text=self._cfg.get("sam_text"),
            )
        except GraspServiceError as exc:
            with self._lock:
                self._error = str(exc)
                self._stage = "error"
            self._log_line(f"hole/detect FAIL {exc}")
            raise
        hole = data.get("hole") if isinstance(data.get("hole"), dict) else data
        if isinstance(hole, dict):
            try:
                geom = self._hole_geometry_base(hole)
                hole = dict(hole)
                hole["center_base_m"] = geom["center_base_m"]
                hole["plane_normal_base"] = geom["normal_base"]
                hole["s0_pregrasp_base_m"] = geom["s0_position_m"]
                hole["s0_along_normal_m"] = geom["offset_m"]
                data = dict(data)
                data["hole"] = hole
            except Exception as exc:  # noqa: BLE001
                self._log_line(f"WARN hole→base geometry: {exc}")
        tcp = (hole or {}).get("tcp_target") if isinstance(hole, dict) else None
        tcp_ready = self._position_usable(tcp if isinstance(tcp, dict) else None)
        with self._lock:
            self._hole = hole
            self._material_selected_id = None
            self._last = {"hole_detect": data}
            self._stage = "hole"
            self._error = ""
        viz_url = self._build_viz("hole")
        self._log_line(
            f"hole/detect ok tcp_usable={tcp_ready} calibrated={(hole or {}).get('calibrated')} "
            f"center={((hole or {}).get('center_camera_m'))} "
            f"s0={((hole or {}).get('s0_pregrasp_base_m'))} viz={bool(viz_url)}"
        )
        return {**self.status(), "ok": True, "result": data, "viz_url": viz_url}

    def _hole_geometry_base(self, hole: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Hole center + plane normal in base; S0 = center + offset·normal."""
        hole = hole if isinstance(hole, dict) else (self._hole if isinstance(self._hole, dict) else {})
        if not hole:
            raise RuntimeError("无 hole 结果 — 先跑场景孔检测")
        T_bc = self._T_base_cam()
        if T_bc is None:
            raise RuntimeError("无 T_base_cam — 检查 extrinsics.d435")
        center_cam = hole.get("center_camera_m")
        if not (
            isinstance(center_cam, (list, tuple))
            and len(center_cam) >= 3
            and all(v is not None for v in list(center_cam)[:3])
        ):
            raise RuntimeError("hole.center_camera_m 不可用")
        normal_cam = hole.get("plane_normal_camera") or hole.get("normal_camera")
        tcp = hole.get("tcp_target") if isinstance(hole.get("tcp_target"), dict) else {}
        if not (
            isinstance(normal_cam, (list, tuple))
            and len(normal_cam) >= 3
            and all(v is not None for v in list(normal_cam)[:3])
        ):
            normal_cam = tcp.get("approach_unit_camera")
        if not (
            isinstance(normal_cam, (list, tuple))
            and len(normal_cam) >= 3
            and all(v is not None for v in list(normal_cam)[:3])
        ):
            raise RuntimeError("hole.plane_normal_camera 不可用")

        sign = float(self._cfg.get("normal_direction") or 1)
        offset_m = float(self._cfg.get("s0_along_normal_m") or 0.07)
        center = self._cam_point_to_base(T_bc, center_cam)
        n = self._cam_dir_to_base(T_bc, normal_cam)
        if not any(abs(v) > 1e-9 for v in n):
            raise RuntimeError("法向变换后为零向量")
        n = [sign * float(v) for v in n]
        nn = float(np.linalg.norm(n))
        n = [float(v) / nn for v in n]
        # Stand-off on the opposite side of the reported plane normal.
        s0_pos = [
            float(center[0] - offset_m * n[0]),
            float(center[1] - offset_m * n[1]),
            float(center[2] - offset_m * n[2]),
        ]
        return {
            "center_base_m": center,
            "normal_base": n,
            "s0_position_m": s0_pos,
            "offset_m": offset_m,
            "calibrated": bool(hole.get("calibrated")),
        }

    def build_s0_from_hole(self) -> Dict[str, Any]:
        """S0 TCP = hole_center − s0_along_normal_m · n; tool+Z tip-down ‖ normal."""
        geom = self._hole_geometry_base()
        tip_down = bool(self._cfg.get("s0_flip_approach", True))
        tool_z = self._choose_tool_z_along_normal(
            geom["normal_base"], prefer_tip_down=tip_down
        )
        R = self._R_align_tool_z(tool_z)
        rpy = [float(v) for v in Rotation.from_matrix(R).as_euler("xyz")]
        pos = list(geom["s0_position_m"])
        xyzrpy = pos + rpy
        quat = _R_to_quat_xyzw(R)
        T_ft = self._T_flange_tcp()
        flange_xyzrpy = self._T_to_xyzrpy(self._xyzrpy_to_T(xyzrpy) @ np.linalg.inv(T_ft))
        target = {
            "command_type": "move_tcp",
            "frame": "base",
            "role": "tcp",
            "position_m": pos,
            "quaternion_xyzw": quat,
            "xyzrpy": xyzrpy,
        }
        return {
            "ok": True,
            "source": "hole_center - s0_along_normal_m · n",
            "offset_m": geom["offset_m"],
            "hole_center_base_m": geom["center_base_m"],
            "normal_base": geom["normal_base"],
            "position_m": pos,
            "tool_z_base": [float(tool_z[0]), float(tool_z[1]), float(tool_z[2])],
            "xyzrpy": xyzrpy,
            "tcp_xyzrpy": xyzrpy,
            "flange_xyzrpy": flange_xyzrpy,
            "quaternion_xyzw": quat,
            "target": target,
            "tip_down": tip_down,
            "calibrated": geom["calibrated"],
        }

    def s0_plan(self) -> Dict[str, Any]:
        """Remote S0 for session/gripper; motion pose = hole + 7cm along normal."""
        if not isinstance(self._hole, dict) or not self._hole:
            raise RuntimeError("S0 依赖孔检测：请先跑 Hole detect")
        local = self.build_s0_from_hole()
        frame = self._frame_body("scene")
        data: Dict[str, Any] = {}
        remote_err = ""
        try:
            data = self._client.s0_plan(frame, sam_text=self._cfg.get("sam_text"))
            if not isinstance(data, dict):
                data = {"raw": data}
        except GraspServiceError as exc:
            # Still allow local geometry S0 for canvas / pre-grasp; session may be missing.
            remote_err = str(exc)
            self._log_line(f"WARN s0/plan remote FAIL {exc} — 使用本地 hole+{local['offset_m']}m")
            data = {
                "calibrated": bool(local.get("calibrated")),
                "session_id": self._session_id,
                "gripper": {},
                "remote_error": remote_err,
            }

        motion = dict(data.get("motion") or {})
        motion["plan_backend"] = "hole_along_normal"
        motion["s0_along_normal_m"] = local["offset_m"]
        motion["hole_center_base_m"] = local["hole_center_base_m"]
        motion["plane_normal_base"] = local["normal_base"]
        motion["pregrasp_flange"] = dict(local["target"])
        motion["tcp_target"] = dict(local["target"])
        data["motion"] = motion
        data["client_s0"] = {
            k: local[k]
            for k in (
                "source",
                "offset_m",
                "hole_center_base_m",
                "normal_base",
                "position_m",
                "tool_z_base",
                "xyzrpy",
                "tip_down",
            )
        }
        data["calibrated"] = bool(data.get("calibrated") or local.get("calibrated"))

        sid = data.get("session_id")
        with self._lock:
            if sid:
                self._session_id = str(sid)
            self._s0 = data
            self._last = {"s0_plan": data}
            self._stage = "s0"
            self._error = remote_err
        viz_url = self._build_viz("s0")
        self._log_line(
            f"s0/plan session={sid} backend=hole+{local['offset_m']}m "
            f"pos={local['position_m']} tip_down={local['tip_down']} "
            f"gripper={((data.get('gripper') or {}).get('width_m'))} "
            f"remote={'ok' if not remote_err else 'fail'} viz={bool(viz_url)}"
        )
        return {**self.status(), "ok": True, "result": data, "viz_url": viz_url, "local_s0": local}

    def _T_flange_tcp(self) -> np.ndarray:
        """Match sensors_arm FK/IK: translation-only flange→TCP (gripper_rpy is viz mesh only)."""
        svc = self._require_svc()
        arm_cfg = dict(getattr(getattr(svc, "arm", None), "arm_cfg", None) or {})
        tcp_xyz = list(arm_cfg.get("tcp_xyz") or [0.0, 0.0, 0.117])
        T = np.eye(4)
        T[0, 3] = float(tcp_xyz[0])
        T[1, 3] = float(tcp_xyz[1])
        T[2, 3] = float(tcp_xyz[2])
        return T

    def _xyzrpy_to_T(self, xyzrpy: Sequence[float]) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = _rpy_to_R(list(xyzrpy)[3:6])
        T[0, 3] = float(xyzrpy[0])
        T[1, 3] = float(xyzrpy[1])
        T[2, 3] = float(xyzrpy[2])
        return T

    def _T_to_xyzrpy(self, T: np.ndarray) -> List[float]:
        rpy = [float(v) for v in Rotation.from_matrix(T[:3, :3]).as_euler("xyz")]
        return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])] + rpy

    def _tool_z_world(self, tcp_xyzrpy: Sequence[float]) -> np.ndarray:
        return _rpy_to_R(list(tcp_xyzrpy)[3:6])[:, 2].astype(float)

    @staticmethod
    def _unit(v: Sequence[float]) -> np.ndarray:
        arr = np.asarray(list(v)[:3], dtype=float)
        n = float(np.linalg.norm(arr))
        if n < 1e-9:
            return np.zeros(3, dtype=float)
        return arr / n

    def _R_align_tool_z(
        self,
        tool_z: Sequence[float],
        *,
        seed_R: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Rotation with column-2 = tool +Z (flange→TCP). Twist from seed_R x-axis if given."""
        z = self._unit(tool_z)
        if float(np.linalg.norm(z)) < 1e-9:
            return np.eye(3)
        ref = None
        if seed_R is not None and seed_R.shape == (3, 3):
            ref = seed_R[:, 0].astype(float)
        if ref is None or abs(float(np.dot(ref, z))) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(ref, z))) > 0.95:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
        x = ref - z * float(np.dot(ref, z))
        x = self._unit(x)
        y = np.cross(z, x)
        y = self._unit(y)
        # re-orthogonalize x in case of numeric drift
        x = np.cross(y, z)
        x = self._unit(x)
        return np.column_stack([x, y, z])

    def _choose_tool_z_along_normal(
        self,
        normal_base: Sequence[float],
        *,
        tcp_pos: Optional[Sequence[float]] = None,
        toward_pos: Optional[Sequence[float]] = None,
        prefer_tip_down: bool = True,
    ) -> np.ndarray:
        """Pick ±normal so flange→TCP (= tool +Z) is collinear with the plane normal.

        Default: tip-down (tool +Z has world Z≤0) so flange sits above the tip —
        "toward hole center" alone often flips into a bottom-up grasp when the
        plane normal already points downward.
        """
        n = self._unit(normal_base)
        if float(np.linalg.norm(n)) < 1e-9:
            return np.array([0.0, 0.0, -1.0], dtype=float)
        if prefer_tip_down:
            # Clear vertical: force tip toward world −Z (top-down).
            if abs(float(n[2])) > 0.25:
                return n if float(n[2]) <= 0.0 else -n
        if (not prefer_tip_down) and tcp_pos is not None and toward_pos is not None:
            toward = self._unit(
                [float(toward_pos[i]) - float(tcp_pos[i]) for i in range(3)]
            )
            if float(np.linalg.norm(toward)) > 1e-9:
                return n if float(np.dot(n, toward)) >= 0.0 else -n
        return n if float(n[2]) <= 0.0 else -n

    def _s0_j6_rad(self) -> float:
        """Configured S0 wrist J6 (rad); default 12.56 ≈ 4π."""
        try:
            return float(self._cfg.get("s0_j6_rad", 12.56))
        except (TypeError, ValueError):
            return 12.56

    def _s0_ik_seed(self, arm: Any) -> List[float]:
        """Current joints with J6 forced to ``s0_j6_rad`` for S0 IK seed."""
        try:
            seed = list(arm.get_joints() or [])
        except Exception:  # noqa: BLE001
            seed = []
        while len(seed) < 6:
            seed.append(0.0)
        seed = [float(v) for v in seed[:6]]
        seed[5] = self._s0_j6_rad()
        return seed

    def _force_s0_j6(self, ik: Dict[str, Any], arm: Optional[Any] = None) -> Dict[str, Any]:
        """Overwrite IK joints[5] with configured S0 J6; refresh FK if possible."""
        joints = ik.get("joints_rad")
        if not isinstance(joints, (list, tuple)) or len(joints) < 6:
            return ik
        out = dict(ik)
        q = [float(v) for v in list(joints)[:6]]
        j6 = self._s0_j6_rad()
        q[5] = j6
        out["joints_rad"] = q
        out["s0_j6_rad"] = j6
        out["s0_j6_forced"] = True
        if arm is not None:
            try:
                fk = arm.fk(q)
                if isinstance(fk, (list, tuple)) and len(fk) >= 6:
                    out["achieved_xyzrpy"] = [float(v) for v in list(fk)[:6]]
                    out["tcp_xyzrpy"] = [float(v) for v in list(fk)[:6]]
            except Exception:  # noqa: BLE001
                pass
        return out

    def _ik_tcp_preview(self, tcp_xyzrpy: Sequence[float]) -> Dict[str, Any]:
        """Loose TCP IK for Three.js ghost: try ori flips, then soft-accept best."""
        svc = self._require_svc()
        arm = getattr(svc, "arm", None)
        if arm is None:
            return {"ok": False, "joints_rad": None, "error": "arm unavailable"}

        seed_q = self._s0_ik_seed(arm)
        T0 = self._xyzrpy_to_T(list(tcp_xyzrpy)[:6])
        Rx = self._xyzrpy_to_T([0.0, 0.0, 0.0, math.pi, 0.0, 0.0])
        Ry = self._xyzrpy_to_T([0.0, 0.0, 0.0, 0.0, math.pi, 0.0])
        Rz = self._xyzrpy_to_T([0.0, 0.0, 0.0, 0.0, 0.0, math.pi])

        def _flip(T: np.ndarray, Rloc: np.ndarray) -> List[float]:
            out = T.copy()
            out[:3, :3] = T[:3, :3] @ Rloc[:3, :3]
            return self._T_to_xyzrpy(out)

        candidates: List[Tuple[str, List[float]]] = [
            ("nominal", [float(v) for v in list(tcp_xyzrpy)[:6]]),
            ("ori_Rx", _flip(T0, Rx)),
            ("ori_Ry", _flip(T0, Ry)),
            ("ori_Rz", _flip(T0, Rz)),
        ]

        def _run(pose: Sequence[float], **extra: Any) -> Dict[str, Any]:
            try:
                ik = dict(arm.ik(list(pose)[:6], seed=seed_q, preview=True, **extra) or {})
            except Exception as exc:  # noqa: BLE001
                ik = {"ok": False, "joints_rad": None, "error": str(exc)}
            achieved = ik.get("achieved_xyzrpy")
            if isinstance(achieved, (list, tuple)) and len(achieved) >= 6:
                ik["tcp_xyzrpy"] = [float(v) for v in list(achieved)[:6]]
            else:
                ik["tcp_xyzrpy"] = [float(v) for v in list(pose)[:6]]
            return self._force_s0_j6(ik, arm)

        attempts: List[Dict[str, Any]] = []
        ok_ranked: List[Tuple[float, float, Dict[str, Any]]] = []
        soft_best: Optional[Dict[str, Any]] = None
        soft_score = (-1e9, 1e9)  # (tip_down, pos_err) — maximize tip_down, minimize err

        def _score(ik: Dict[str, Any]) -> Tuple[float, float]:
            tcp = ik.get("tcp_xyzrpy") or tcp_xyzrpy
            tip_down = float(-self._tool_z_world(tcp)[2])
            err = float(ik.get("pos_err_m") if ik.get("pos_err_m") is not None else 1e3)
            return (tip_down, -err)

        for name, pose in candidates:
            ik = _run(pose)
            ik["strategy"] = name
            tip_down = float(-self._tool_z_world(ik.get("tcp_xyzrpy") or pose)[2])
            ik["tip_down"] = tip_down
            attempts.append(
                {
                    "strategy": name,
                    "ok": ik.get("ok"),
                    "pos_err_m": ik.get("pos_err_m"),
                    "tip_down": tip_down,
                    "error": ik.get("error"),
                }
            )
            joints = ik.get("joints_rad")
            if ik.get("ok") and joints:
                td, neg_err = _score(ik)
                ok_ranked.append((td, neg_err, ik))
            err = ik.get("pos_err_m")
            if joints and err is not None:
                sc = (tip_down, -float(err))
                if sc > soft_score:
                    soft_best = ik
                    soft_score = sc

        if ok_ranked:
            ok_ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
            out = dict(ok_ranked[0][2])
            out["attempts"] = attempts
            return out

        # Ultra-loose pass so joints survive pose-check for ghost preview.
        for name, pose in candidates:
            ik = _run(
                pose,
                position_tolerance_m=8.0e-2,
                accept_residual=2.5e-1,
                max_nfev=2000,
            )
            ik["strategy"] = f"{name}_relaxed"
            tip_down = float(-self._tool_z_world(ik.get("tcp_xyzrpy") or pose)[2])
            ik["tip_down"] = tip_down
            attempts.append(
                {
                    "strategy": ik["strategy"],
                    "ok": ik.get("ok"),
                    "pos_err_m": ik.get("pos_err_m"),
                    "tip_down": tip_down,
                    "error": ik.get("error"),
                }
            )
            joints = ik.get("joints_rad")
            if ik.get("ok") and joints:
                td, neg_err = _score(ik)
                ok_ranked.append((td, neg_err, ik))
            err = ik.get("pos_err_m")
            if joints and err is not None:
                sc = (tip_down, -float(err))
                if sc > soft_score:
                    soft_best = ik
                    soft_score = sc

        if ok_ranked:
            ok_ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
            out = dict(ok_ranked[0][2])
            out["preview_relaxed"] = True
            out["attempts"] = attempts
            return out

        if soft_best and soft_best.get("joints_rad"):
            out = dict(soft_best)
            out["ok"] = True
            out["preview_relaxed"] = True
            out["attempts"] = attempts
            return out

        return {
            "ok": False,
            "joints_rad": None,
            "error": (attempts[-1].get("error") if attempts else "IK failed"),
            "tcp_xyzrpy": [float(v) for v in list(tcp_xyzrpy)[:6]],
            "attempts": attempts,
        }

    def _ik_s0_tcp_for_canvas(self, flange_xyzrpy: Sequence[float]) -> Dict[str, Any]:
        """S0 pregrasp_flange → TCP IK for top-down pick viz.

        When ``s0_flip_approach`` (default true), apply Rx180 on the flange frame so
        Elite tip (+Z) points world-down — GraspNet flange Z is often opposite.
        Still evaluates nominal / Ry / Rz as fallbacks ranked by tip-down + IK error.
        """
        svc = self._require_svc()
        arm = getattr(svc, "arm", None)
        if arm is None:
            return {"ok": False, "joints_rad": None, "error": "arm unavailable"}

        flip_pref = bool(self._cfg.get("s0_flip_approach", True))
        seed_q = self._s0_ik_seed(arm)
        T_bf = self._xyzrpy_to_T(flange_xyzrpy)
        T_ft = self._T_flange_tcp()
        T_tcp_nom = T_bf @ T_ft
        Rx = self._xyzrpy_to_T([0.0, 0.0, 0.0, math.pi, 0.0, 0.0])
        Ry = self._xyzrpy_to_T([0.0, 0.0, 0.0, 0.0, math.pi, 0.0])
        Rz = self._xyzrpy_to_T([0.0, 0.0, 0.0, 0.0, 0.0, math.pi])

        def _ori_flip(T_tcp: np.ndarray, T_rot: np.ndarray) -> np.ndarray:
            """Flip TCP orientation in place (keep tip position, reverse approach)."""
            out = T_tcp.copy()
            out[:3, :3] = T_tcp[:3, :3] @ T_rot[:3, :3]
            return out

        # Orientation-only flips keep TCP xyz; flange@Rx moves tip to the other side.
        candidates: List[Tuple[str, List[float]]] = []
        pose_nom = self._T_to_xyzrpy(T_tcp_nom)
        pose_rx = self._T_to_xyzrpy(_ori_flip(T_tcp_nom, Rx))
        pose_ry = self._T_to_xyzrpy(_ori_flip(T_tcp_nom, Ry))
        pose_rz = self._T_to_xyzrpy(_ori_flip(T_tcp_nom, Rz))
        if flip_pref:
            # Default: reverse approach — GraspNet Z often opposite Elite tip (+Z down).
            candidates.extend(
                [
                    ("approach_flip_Rx", pose_rx),
                    ("approach_flip_Ry", pose_ry),
                    ("nominal", pose_nom),
                    ("approach_flip_Rz", pose_rz),
                ]
            )
        else:
            candidates.extend(
                [
                    ("nominal", pose_nom),
                    ("approach_flip_Rx", pose_rx),
                    ("approach_flip_Ry", pose_ry),
                    ("approach_flip_Rz", pose_rz),
                ]
            )

        attempts: List[Dict[str, Any]] = []
        first_ok: Optional[Dict[str, Any]] = None
        soft_best: Optional[Dict[str, Any]] = None
        soft_err = 1e9
        for flip_name, pose in candidates:
            tool_z = self._tool_z_world(pose)
            tip_down = float(-tool_z[2])
            wrist_above = float(flange_xyzrpy[2] - float(pose[2]))
            try:
                ik = dict(arm.ik(pose, seed=seed_q, preview=True) or {})
            except Exception as exc:  # noqa: BLE001
                ik = {"ok": False, "joints_rad": None, "error": str(exc)}
            ik = self._force_s0_j6(ik, arm)
            ik["strategy"] = flip_name
            ik["tip_down"] = tip_down
            ik["wrist_above_tip"] = wrist_above
            ik["s0_flip_approach"] = flip_pref
            ik["target_xyzrpy"] = [float(v) for v in pose[:6]]
            joints = ik.get("joints_rad")
            achieved = ik.get("achieved_xyzrpy")
            if isinstance(achieved, (list, tuple)) and len(achieved) >= 6:
                ik["tcp_xyzrpy"] = [float(v) for v in list(achieved)[:6]]
            else:
                ik["tcp_xyzrpy"] = [float(v) for v in pose[:6]]
            tool_z_a = self._tool_z_world(ik["tcp_xyzrpy"])
            ik["tip_down_achieved"] = float(-tool_z_a[2])
            attempts.append(
                {
                    k: ik.get(k)
                    for k in (
                        "strategy",
                        "ok",
                        "pos_err_m",
                        "tip_down",
                        "wrist_above_tip",
                        "error",
                    )
                }
            )
            # Respect candidate order (approach_flip_Rx first when flip enabled).
            if ik.get("ok") and joints and first_ok is None:
                first_ok = ik
            if joints:
                err = float(ik.get("pos_err_m") or soft_err)
                if err < soft_err:
                    soft_best = ik
                    soft_err = err

        if first_ok:
            first_ok = dict(first_ok)
            first_ok["attempts"] = attempts
            return first_ok
        if soft_best and soft_best.get("joints_rad"):
            soft_best = dict(soft_best)
            soft_best["ok"] = True
            soft_best["preview_relaxed"] = True
            soft_best["attempts"] = attempts
            return soft_best

        return {
            "ok": False,
            "joints_rad": None,
            "error": (attempts[-1].get("error") if attempts else "IK failed"),
            "tcp_xyzrpy": list(candidates[0][1]),
            "attempts": attempts,
            "s0_j6_rad": self._s0_j6_rad(),
        }

    def _T_base_cam(self) -> Optional[np.ndarray]:
        """Scene camera T_base_cam from extrinsics.d435.T (4×4)."""
        svc = self._svc
        if svc is None or not hasattr(svc, "get_extrinsics"):
            return None
        try:
            ext = svc.get_extrinsics() or {}
        except Exception:  # noqa: BLE001
            return None
        T = (ext.get("d435") or {}).get("T")
        if T is None:
            return None
        try:
            arr = np.asarray(T, dtype=float).reshape(4, 4)
        except Exception:  # noqa: BLE001
            return None
        return arr

    @staticmethod
    def _cam_point_to_base(T_bc: np.ndarray, p_cam: Sequence[float]) -> List[float]:
        p = np.asarray(list(p_cam)[:3], dtype=float)
        out = T_bc[:3, :3] @ p + T_bc[:3, 3]
        return [float(out[0]), float(out[1]), float(out[2])]

    @staticmethod
    def _cam_dir_to_base(T_bc: np.ndarray, n_cam: Sequence[float]) -> List[float]:
        n = np.asarray(list(n_cam)[:3], dtype=float)
        out = T_bc[:3, :3] @ n
        norm = float(np.linalg.norm(out))
        if norm < 1e-9:
            return [0.0, 0.0, 0.0]
        out = out / norm
        return [float(out[0]), float(out[1]), float(out[2])]

    def hole_canvas_pose(self) -> Dict[str, Any]:
        """Hole detect point (center) + plane normal in base — no arm pose."""
        geom = self._hole_geometry_base()
        position = list(geom["center_base_m"])
        normal_base = list(geom["normal_base"])
        arrow_m = float(geom["offset_m"])  # draw arrow to S0 offset (7cm)
        tip = list(geom["s0_position_m"])
        self._log_line(
            f"hole/canvas_pose ball={position} n={normal_base} "
            f"s0={tip} offset={arrow_m} calibrated={geom.get('calibrated')}"
        )
        return {
            "ok": True,
            "source": "center_camera_m→base",
            "frame": "base",
            "position_m": position,
            "normal_base": normal_base,
            "normal_source": "plane_normal_camera→base",
            "normal_length_m": arrow_m,
            "normal_tip_m": tip,
            "center_base_m": position,
            "s0_pregrasp_base_m": tip,
            "s0_along_normal_m": arrow_m,
            "calibrated": bool(geom.get("calibrated")),
        }

    def _s0_approach_unit(self, R_raw: np.ndarray) -> Tuple[np.ndarray, str]:
        """GraspNet pregrasp often stores approach on +X; Elite tip uses +Z.

        Prefer the column most aligned with hole plane normal when available.
        """
        hole_n = self._hole_normal_base()
        best_axis = 0
        best_score = -1.0
        best_signed = R_raw[:, 0].astype(float)
        for i in range(3):
            col = R_raw[:, i].astype(float)
            nn = float(np.linalg.norm(col))
            if nn < 1e-9:
                continue
            col = col / nn
            if hole_n is not None:
                score = abs(float(np.dot(col, hole_n)))
            else:
                # Default GraspNet: approach = +X
                score = 1.0 if i == 0 else 0.0
            if score > best_score:
                best_score = score
                best_axis = i
                best_signed = col
        label = f"R_raw[:,{best_axis}]"
        if hole_n is not None and best_score >= 0.5:
            if float(np.dot(best_signed, hole_n)) < 0.0:
                best_signed = -best_signed
                label += " (flip to hole n)"
            else:
                label += " (‖ hole n)"
        return best_signed, label

    def s0_canvas_pose(self) -> Dict[str, Any]:
        """S0 = hole_center + 7cm·n → TCP marker + approach arrow + IK ghost."""
        local = self.build_s0_from_hole()
        tcp_xyzrpy = list(local["tcp_xyzrpy"])
        flange_xyzrpy = list(local["flange_xyzrpy"])
        tool_z_list = list(local["tool_z_base"])
        tip_down = bool(local.get("tip_down"))
        arrow_dir = list(tool_z_list)
        try:
            ik = self._ik_tcp_preview(tcp_xyzrpy)
            ik = dict(ik)
            ik["strategy"] = f"hole+{local['offset_m']}m·tip_down={tip_down}"
            ik["tip_down"] = float(-tool_z_list[2])
            ik["wrist_above_tip"] = float(flange_xyzrpy[2] - tcp_xyzrpy[2])
        except Exception as exc:  # noqa: BLE001
            ik = {
                "ok": False,
                "joints_rad": None,
                "error": str(exc),
                "tcp_xyzrpy": tcp_xyzrpy,
                "strategy": f"hole+{local['offset_m']}m·tip_down={tip_down}",
            }

        j6 = self._s0_j6_rad()
        self._log_line(
            f"s0/canvas_pose tcp={tcp_xyzrpy[:3]} flange={flange_xyzrpy[:3]} "
            f"from_hole={local['hole_center_base_m']} offset={local['offset_m']} "
            f"tool_z={tool_z_list} tip_down={-tool_z_list[2]:.3f} "
            f"wrist_above={flange_xyzrpy[2]-tcp_xyzrpy[2]:.3f} "
            f"j6={j6:.4f} ik_ok={ik.get('ok')}"
        )
        return {
            "ok": True,
            "source": local["source"],
            "frame": "tcp",
            "xyzrpy": list(tcp_xyzrpy[:6]),
            "position_m": list(tcp_xyzrpy[:3]),
            "rpy_rad": list(tcp_xyzrpy[3:6]),
            "flange_xyzrpy": flange_xyzrpy,
            "tcp_xyzrpy": list(tcp_xyzrpy[:6]),
            "normal_base": arrow_dir,
            "normal_length_m": float(local["offset_m"]),
            "tool_z_base": tool_z_list,
            "hole_center_base_m": local["hole_center_base_m"],
            "offset_m": local["offset_m"],
            "s0_j6_rad": j6,
            "arrow_source": "tool+Z",
            "calibrated": bool(local.get("calibrated")),
            "ik": ik,
            "session_id": self._session_id,
        }

    def _pick_pregrasp(self) -> Dict[str, Any]:
        """Prefer S0 = hole + s0_along_normal_m; fall back to server poses."""
        hole = self._hole or {}
        s0 = self._s0 or {}
        motion = s0.get("motion") or {}
        calibrated = bool(s0.get("calibrated") or hole.get("calibrated"))

        # 1) Local rule: hole center + N cm along normal
        try:
            local = self.build_s0_from_hole()
            return {
                "source": "s0.hole_along_normal",
                "target": dict(local["target"]),
                "calibrated": calibrated or bool(local.get("calibrated")),
                "position_usable": True,
                "local_s0": local,
            }
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"WARN build_s0_from_hole: {exc}")

        tcp = hole.get("tcp_target") if isinstance(hole.get("tcp_target"), dict) else None
        flange = motion.get("pregrasp_flange") if isinstance(motion.get("pregrasp_flange"), dict) else None
        prefer_hole = bool(self._cfg.get("prefer_hole_tcp", True))
        if prefer_hole and self._position_usable(tcp):
            source = "hole.tcp_target"
            target = dict(tcp or {})
        elif self._position_usable(flange):
            source = "s0.motion.pregrasp_flange"
            target = dict(flange or {})
        elif self._position_usable(tcp):
            source = "hole.tcp_target"
            target = dict(tcp or {})
        else:
            raise RuntimeError(
                "无可用 pre-grasp：需先孔检测（S0=孔心沿法向 "
                f"{float(self._cfg.get('s0_along_normal_m') or 0.07)}m）"
            )
        if not calibrated:
            self._log_line("WARN calibrated=false")
        return {
            "source": source,
            "target": target,
            "calibrated": calibrated,
            "position_usable": True,
        }

    def _hole_normal_base(self) -> Optional[List[float]]:
        """Plane / approach normal in base from last hole detect."""
        hole = self._hole if isinstance(self._hole, dict) else {}
        if not hole:
            return None
        T_bc = self._T_base_cam()
        if T_bc is None:
            return None
        sign = float(self._cfg.get("normal_direction") or 1)
        tcp = hole.get("tcp_target") if isinstance(hole.get("tcp_target"), dict) else {}
        for key, label in (
            ("approach_unit_camera", "approach"),
            ("plane_normal_camera", "plane"),
            ("normal_camera", "plane"),
        ):
            raw = tcp.get(key) if key == "approach_unit_camera" else hole.get(key)
            if isinstance(raw, (list, tuple)) and len(raw) >= 3 and all(
                v is not None for v in list(raw)[:3]
            ):
                n = self._cam_dir_to_base(T_bc, raw)
                if any(abs(v) > 1e-9 for v in n):
                    out = [sign * float(v) for v in n]
                    nn = float(np.linalg.norm(out))
                    if nn > 1e-9:
                        return [float(v) / nn for v in out]
        return None

    def _align_xyzrpy_tool_z_to_normal(
        self,
        xyzrpy: Sequence[float],
        normal_base: Sequence[float],
        *,
        prefer_tip_down: bool = True,
    ) -> List[float]:
        """Keep TCP position; rebuild RPY so tool +Z (flange→TCP) ‖ normal."""
        pos = [float(v) for v in list(xyzrpy)[:3]]
        seed_R = _rpy_to_R(list(xyzrpy)[3:6]) if len(list(xyzrpy)) >= 6 else None
        tool_z = self._choose_tool_z_along_normal(
            normal_base, prefer_tip_down=prefer_tip_down
        )
        R = self._R_align_tool_z(tool_z, seed_R=seed_R)
        rpy = [float(v) for v in Rotation.from_matrix(R).as_euler("xyz")]
        return pos + rpy

    def move_pregrasp(self) -> Dict[str, Any]:
        pick = self._pick_pregrasp()
        target = pick["target"]
        if not self._position_usable(target):
            raise RuntimeError(
                "拒绝 move_tcp：position_m 为 null（未标定 tcp_target 不是假位姿，而是空位姿）"
            )
        with self._lock:
            self._pregrasp_target = pick
        exec_motion = bool(self._cfg.get("execute_motion"))
        xyzrpy = self._target_to_xyzrpy(target)
        aligned = pick.get("source") == "s0.hole_along_normal"
        if not aligned and pick.get("source") == "hole.tcp_target":
            n_base = self._hole_normal_base()
            if n_base is not None:
                tip_down = bool(self._cfg.get("s0_flip_approach", True))
                xyzrpy = self._align_xyzrpy_tool_z_to_normal(
                    xyzrpy, n_base, prefer_tip_down=tip_down
                )
                aligned = True
        gripper = ((self._s0 or {}).get("gripper") or {})
        j6 = self._s0_j6_rad()
        # Resolve joints with forced J6 so LIVE / ghost share the same wrist wrap.
        ik_for_move: Optional[Dict[str, Any]] = None
        try:
            ik_for_move = self._ik_tcp_preview(xyzrpy)
        except Exception as exc:  # noqa: BLE001
            self._log_line(f"WARN move_pregrasp IK preview: {exc}")
        joints = None
        if isinstance(ik_for_move, dict) and ik_for_move.get("joints_rad"):
            joints = [float(v) for v in list(ik_for_move["joints_rad"])[:6]]
            joints[5] = j6
        note = {
            "execute_motion": exec_motion,
            "xyzrpy": xyzrpy,
            "joints_rad": joints,
            "s0_j6_rad": j6,
            "gripper_width_m": gripper.get("width_m"),
            "gripper_action": gripper.get("action"),
            "source": pick["source"],
            "calibrated": pick["calibrated"],
            "command_type": target.get("command_type"),
            "frame": target.get("frame"),
            "aligned_tool_z_to_normal": aligned,
            "s0_along_normal_m": float(self._cfg.get("s0_along_normal_m") or 0.07),
            "ik_ok": bool(ik_for_move and ik_for_move.get("ok") and joints),
        }
        if not exec_motion:
            self._log_line(f"move_pregrasp DRY-RUN {note}")
            with self._lock:
                self._stage = "moved"
                self._last = {"move_pregrasp": {"dry_run": True, **note}}
            return {**self.status(), "ok": True, "dry_run": True, **note}

        svc = self._require_svc()
        svc.arm.ensure_ready()
        if joints is not None:
            moved = svc.arm.move_joints(joints)
            if isinstance(moved, dict):
                moved = dict(moved)
                moved["ik"] = {
                    k: (ik_for_move or {}).get(k)
                    for k in ("ok", "pos_err_m", "strategy", "s0_j6_rad", "s0_j6_forced")
                }
                moved["commanded_joints_rad"] = list(joints)
                moved["commanded_xyzrpy"] = list(xyzrpy[:6])
        else:
            # Fallback: Cartesian IK then force J6 on the solved joints.
            arm = svc.arm
            seed = self._s0_ik_seed(arm)
            ik = dict(arm.ik(xyzrpy, seed=seed) or {})
            if not ik.get("ok") or not ik.get("joints_rad"):
                raise RuntimeError(f"IK failed: {ik.get('error')}")
            q = [float(v) for v in list(ik["joints_rad"])[:6]]
            q[5] = j6
            moved = arm.move_joints(q)
            if isinstance(moved, dict):
                moved = dict(moved)
                moved["ik"] = {k: v for k, v in ik.items() if k != "debug"}
                moved["commanded_joints_rad"] = list(q)
                moved["s0_j6_rad"] = j6
        self._log_line(
            f"move_pregrasp LIVE ok source={pick['source']} xyzrpy={xyzrpy} "
            f"j6={j6:.4f} joints={joints} "
            f"gripper.width_m={gripper.get('width_m')} action={gripper.get('action')}"
        )
        with self._lock:
            self._stage = "moved"
            self._last = {"move_pregrasp": {"dry_run": False, "moved": moved, **note}}
        return {**self.status(), "ok": True, "dry_run": False, "moved": moved, **note}

    def _target_to_xyzrpy(self, target: Dict[str, Any]) -> List[float]:
        """Accept tcp_target / pregrasp_flange shapes → [x,y,z,r,p,y]."""
        if "xyzrpy" in target and isinstance(target["xyzrpy"], (list, tuple)):
            return [float(v) for v in list(target["xyzrpy"])[:6]]
        pos = target.get("position_m")
        if pos is None:
            pos = target.get("xyz_m") or target.get("position")
        if pos is None and all(k in target for k in ("x", "y", "z")):
            pos = [target["x"], target["y"], target["z"]]
        if pos is None or any(v is None for v in list(pos)[:3]):
            raise RuntimeError(
                f"target.position_m is null — cannot move_tcp (keys={list(target)})"
            )
        pos = [float(v) for v in list(pos)[:3]]
        quat = target.get("quaternion_xyzw")
        if quat is not None and any(v is None for v in list(quat)[:4]):
            quat = None
        if quat is not None:
            rpy = _quat_xyzw_to_rpy(quat)
        elif "rpy_rad" in target:
            rpy = [float(v) for v in list(target["rpy_rad"])[:3]]
        elif "xyzrpy" in target:
            rpy = [float(v) for v in list(target["xyzrpy"])[3:6]]
        else:
            svc = self._require_svc()
            cur = svc.arm.fk() or [0, 0, 0, 0, 0, math.pi]
            rpy = [float(cur[3]), float(cur[4]), float(cur[5])]
        return pos + rpy

    def s1_latch(self) -> Dict[str, Any]:
        sid = self._require_session()
        frame = self._frame_body("scene")
        data = self._client.s1_latch(sid, frame)
        motion = data.get("motion") or {}
        with self._lock:
            self._stage = "s1"
            self._last = {"s1_latch": data}
        # Server: {"command_type": "none", "note": "latched"}
        self._log_line(f"s1/latch ok motion={motion.get('command_type')} note={motion.get('note')}")
        return {**self.status(), "ok": True, "result": data}

    def s1_step(self) -> Dict[str, Any]:
        sid = self._require_session()
        frame = self._frame_body("scene")
        data = self._client.s1_step(sid, frame)
        motion = data.get("motion") or {}
        # Protocol: never command non-zero velocity in S1 (monitor_only)
        with self._lock:
            self._last = {"s1_step": data}
        self._log_line(
            f"s1/step cmd={motion.get('command_type')} exit={motion.get('exit_s1_servo')} "
            f"proceed_to_s2={motion.get('proceed_to_s2')} hint={motion.get('hint_handoff_wrist_cam')}"
        )
        return {**self.status(), "ok": True, "result": data}

    def s2_latch(self) -> Dict[str, Any]:
        sid = self._require_session()
        frame = self._frame_body("wrist")
        data = self._client.s2_latch(
            sid,
            frame,
            hole_sam_text=self._cfg.get("hole_sam_text"),
        )
        with self._lock:
            self._stage = "s2_latch"
            self._s2_latched = True
            self._s2_stats = {
                "steps": 0,
                "mode": None,
                "last_exit": False,
                "latched_at": _utc_iso(),
            }
            self._last = {"s2_latch": data}
        self._log_line("s2/latch ok (session s2_targets required before final/approach)")
        return {**self.status(), "ok": True, "result": data}

    def s2_step_once(self, *, from_loop: bool = False) -> Dict[str, Any]:
        """One S2 frame: capture wrist → POST /v1/s2/step → optional motion.

        Manual single-step and Loop both call this until ``exit_s2_servo``.
        """
        sid = self._require_session()
        if not self._s2_latched:
            raise RuntimeError("请先 POST /v1/s2/latch（服务端 session 需有 s2_targets）")
        if self._s2_running and not from_loop:
            raise RuntimeError("S2 Loop 运行中，请先停止 Loop 再单步")
        frame = self._frame_body("wrist")
        # Production tip: without mask_b64 server re-runs hole+SAM every frame (slow).
        # Tracking mask upload not implemented yet — loop is "可跑" not "可控".
        data = self._client.s2_step(sid, frame)
        motion = data.get("motion") or {}
        exited = bool(motion.get("exit_s2_servo"))
        applied = None
        if bool(self._cfg.get("execute_motion")) and motion.get("command_type") == "cartesian_velocity":
            applied = self._apply_tool_velocity(motion, dt=1.0 / max(1.0, float(self._cfg.get("s2_hz") or 10)))
        with self._lock:
            prev = dict(self._s2_stats or {})
            steps = int(prev.get("steps") or 0) + 1
            mode = str(prev.get("mode") or ("loop" if from_loop else "manual"))
            if not from_loop:
                mode = "manual"
            self._s2_stats = {
                **prev,
                "steps": steps,
                "last_exit": exited,
                "mode": mode,
                "last_at": _utc_iso(),
            }
            self._last = {"s2_step": data, "applied": applied, "mode": mode}
            if exited:
                self._stage = "s2_exit"
            elif not from_loop:
                self._stage = "s2_step"
        self._log_line(
            f"s2/step#{int((self._s2_stats or {}).get('steps') or 0)} "
            f"mode={'loop' if from_loop else 'manual'} exit={exited} "
            f"vel={motion.get('velocity')} applied={applied is not None}"
        )
        return {
            **self.status(),
            "ok": True,
            "result": data,
            "applied": applied,
            "exit_s2_servo": exited,
            "steps": (self._s2_stats or {}).get("steps"),
        }

    def start_s2_loop(self) -> Dict[str, Any]:
        if self._s2_running:
            return {**self.status(), "ok": True, "already": True}
        self._require_session()
        if not self._s2_latched:
            raise RuntimeError("请先 S2 latch，再启动 loop")
        if self._stage == "s2_exit" and (self._s2_stats or {}).get("last_exit"):
            raise RuntimeError("服务端已 exit_s2_servo；请重新 S2 latch 后再 Loop / 单步")
        self._log_line(
            "WARN S2 loop 未传 mask_b64：服务端每帧重跑孔检测+SAM，难以稳定 10–30Hz"
        )
        self._s2_stop.clear()
        self._s2_running = True
        with self._lock:
            self._s2_stats = {
                "steps": 0,
                "started_at": _utc_iso(),
                "mode": "loop",
                "last_exit": False,
            }
            self._stage = "s2_loop"
        t = threading.Thread(target=self._s2_worker, daemon=True, name="grasp-s2-loop")
        self._s2_thread = t
        t.start()
        self._log_line(f"s2 loop start hz={self._cfg.get('s2_hz')}")
        return self.status()

    def stop_s2(self) -> Dict[str, Any]:
        self._s2_stop.set()
        th = self._s2_thread
        if th and th.is_alive():
            th.join(timeout=2.0)
        self._s2_running = False
        return self.status()

    def _s2_worker(self) -> None:
        hz = max(1.0, float(self._cfg.get("s2_hz") or 10))
        period = 1.0 / hz
        max_steps = int(self._cfg.get("s2_max_steps") or 600)
        steps = 0
        try:
            while not self._s2_stop.is_set() and steps < max_steps:
                t0 = time.time()
                try:
                    out = self.s2_step_once(from_loop=True)
                    steps = int((self._s2_stats or {}).get("steps") or steps + 1)
                    if out.get("exit_s2_servo") or ((out.get("result") or {}).get("motion") or {}).get("exit_s2_servo"):
                        self._log_line(f"s2 loop exit_s2_servo after {steps} steps")
                        break
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._error = str(exc)
                        self._stage = "error"
                    self._log_line(f"s2 loop error: {exc}")
                    break
                elapsed = time.time() - t0
                time.sleep(max(0.0, period - elapsed))
        finally:
            self._s2_running = False
            with self._lock:
                if self._stage == "s2_loop":
                    self._stage = "s2_exit" if (self._s2_stats or {}).get("last_exit") else "s2_stopped"
            self._log_line(f"s2 loop stopped steps={steps}")

    def _apply_tool_velocity(self, motion: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Integrate tool-frame cartesian velocity for one tick → move_pose."""
        vel = motion.get("velocity") or {}
        vx = float(vel.get("vx_m_s") or 0.0)
        vy = float(vel.get("vy_m_s") or 0.0)
        vz = float(vel.get("vz_m_s") or 0.0)
        wz = float(vel.get("wz_rad_s") or 0.0)
        if abs(vx) + abs(vy) + abs(vz) + abs(wz) < 1e-9:
            return {"skipped": True, "reason": "zero_velocity"}
        svc = self._require_svc()
        cur = svc.arm.fk()
        if not cur or len(cur) < 6:
            raise RuntimeError("arm FK unavailable for servo tick")
        R = _rpy_to_R(cur[3:6])
        d_tool = np.array([vx, vy, vz], dtype=float) * float(dt)
        d_base = R @ d_tool
        new_xyz = [float(cur[0] + d_base[0]), float(cur[1] + d_base[1]), float(cur[2] + d_base[2])]
        new_rpy = [float(cur[3]), float(cur[4]), float(cur[5] + wz * dt)]
        xyzrpy = new_xyz + new_rpy
        moved = svc.arm.move_pose(xyzrpy, wait=True)
        return {"xyzrpy": xyzrpy, "moved": moved.get("ok", True)}

    def final_approach(self) -> Dict[str, Any]:
        sid = self._require_session()
        force = bool(self._cfg.get("force_without_s2_exit"))
        # Server requires successful s2/latch (s2_targets) AND normally exit_s2_servo.
        # force_without_s2_exit also needs server policy.allow_force_without_s2_exit (site yaml).
        if not self._s2_latched and not force:
            raise RuntimeError("请先 POST /v1/s2/latch（否则 Final 会 400：请先 POST /v1/s2/latch）")
        if force:
            self._log_line(
                "WARN force_without_s2_exit=true：还需服务端 "
                "final_approach.policy.allow_force_without_s2_exit=true（可用 Site 推送）"
            )
        kwargs = {
            "insert_total_m": self._cfg.get("insert_total_m"),
            "step_m": self._cfg.get("step_m"),
            "linear_speed_m_s": self._cfg.get("linear_speed_m_s"),
            "direction_sign": self._cfg.get("direction_sign"),
            "force_without_s2_exit": force,
        }
        try:
            data = self._client.final_approach(sid, **kwargs)
        except GraspServiceError as exc:
            with self._lock:
                self._error = str(exc)
                self._stage = "error"
            self._log_line(f"final/approach FAIL {exc}")
            raise
        with self._lock:
            self._final = data
            self._last = {"final_approach": data}
            self._stage = "final"
        self._log_line(f"final/approach ok segments={((data.get('motion') or {}).get('segment_count'))}")
        return {**self.status(), "ok": True, "result": data}

    def execute_final(self) -> Dict[str, Any]:
        plan = self._final
        if not plan:
            raise RuntimeError("无 Final 计划；先调用 final/approach")
        motion = plan.get("motion") or {}
        gripper = plan.get("gripper") or {}
        n = int(motion.get("segment_count") or 0)
        seg = float(motion.get("segment_m") or 0.0)
        sign = float(motion.get("direction_sign") or self._cfg.get("direction_sign") or 1)
        pause_ms = int(motion.get("pause_ms_between_steps") or self._cfg.get("pause_ms_between_steps") or 80)
        axis = str(motion.get("axis") or "x").lower()
        exec_motion = bool(self._cfg.get("execute_motion"))
        steps_done = []
        if not exec_motion:
            self._log_line(f"execute_final DRY-RUN n={n} seg={seg} axis={axis} gripper={gripper}")
            with self._lock:
                self._stage = "done"
                self._last = {"execute_final": {"dry_run": True, "segment_count": n, "gripper": gripper}}
            return {**self.status(), "ok": True, "dry_run": True, "segment_count": n, "gripper": gripper}

        svc = self._require_svc()
        svc.arm.ensure_ready()
        for i in range(n):
            cur = svc.arm.fk()
            if not cur:
                raise RuntimeError("FK failed during final segments")
            R = _rpy_to_R(cur[3:6])
            d_tool = np.zeros(3, dtype=float)
            idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            d_tool[idx] = seg * sign
            d_base = R @ d_tool
            xyzrpy = [
                float(cur[0] + d_base[0]),
                float(cur[1] + d_base[1]),
                float(cur[2] + d_base[2]),
                float(cur[3]),
                float(cur[4]),
                float(cur[5]),
            ]
            moved = svc.arm.move_pose(xyzrpy)
            steps_done.append({"i": i + 1, "xyzrpy": xyzrpy, "ok": moved.get("ok", True)})
            time.sleep(max(0.0, pause_ms / 1000.0))
        self._log_line(
            f"execute_final LIVE done n={n}; gripper close intent={gripper.get('width_m_command')} "
            f"when={gripper.get('when')}"
        )
        with self._lock:
            self._stage = "done"
            self._last = {
                "execute_final": {
                    "dry_run": False,
                    "steps": steps_done,
                    "gripper": gripper,
                }
            }
        return {**self.status(), "ok": True, "dry_run": False, "steps": steps_done, "gripper": gripper}

    def run_hole_pipeline(self, *, include_s1: Optional[bool] = None) -> Dict[str, Any]:
        """Hole path via grasp-service only (not Detect ICP/M3).

        site sync → hole/detect → s0 → move → (opt S1) → s2 latch/loop → final → exec.
        Does **not** call rs-detect pipeline ICP / multiview ``p_pregrasp`` — avoid double pre-grasp.
        """
        use_s1 = self._cfg.get("include_s1") if include_s1 is None else include_s1
        results: Dict[str, Any] = {}
        results["site"] = self.site_sync()
        if not results["site"].get("ok"):
            return {**self.status(), "ok": False, "results": results}
        results["hole"] = self.hole_detect()
        results["s0"] = self.s0_plan()
        results["move"] = self.move_pregrasp()
        if use_s1:
            results["s1_latch"] = self.s1_latch()
            # a few monitor steps
            for _ in range(3):
                step = self.s1_step()
                results.setdefault("s1_steps", []).append(step.get("result"))
                if ((step.get("result") or {}).get("motion") or {}).get("exit_s1_servo"):
                    break
                time.sleep(0.15)
        results["s2_latch"] = self.s2_latch()
        results["s2_loop"] = self.start_s2_loop()
        # Wait for loop (bounded)
        deadline = time.time() + max(5.0, float(self._cfg.get("s2_max_steps") or 600) / max(1.0, float(self._cfg.get("s2_hz") or 10)))
        while self._s2_running and time.time() < deadline:
            time.sleep(0.2)
        self.stop_s2()
        results["final"] = self.final_approach()
        results["execute"] = self.execute_final()
        return {**self.status(), "ok": True, "results": {k: (v.get("ok") if isinstance(v, dict) else True) for k, v in results.items()}, "detail": results}

    def _require_session(self) -> str:
        with self._lock:
            sid = self._session_id
        if not sid:
            raise RuntimeError("无 session_id；请先 POST s0/plan")
        return sid


_AGENT: Optional[GraspAgent] = None
_AGENT_LOCK = threading.Lock()


def get_grasp_agent(service: Any = None) -> GraspAgent:
    global _AGENT
    with _AGENT_LOCK:
        if _AGENT is None:
            _AGENT = GraspAgent(service)
        elif service is not None:
            _AGENT.bind_service(service)
        return _AGENT
