from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

import yaml

from rs_detect.calib.extrinsics import ExtrinsicsStore
from rs_detect.calib.intrinsics import fetch_intrinsics_from_sdk, fetch_intrinsics_from_streamer
from rs_detect.camera.recorder import SnapshotWriter, VideoRecorder, colorize_depth
from rs_detect.camera.scanner import scan_devices, split_by_role
from rs_detect.camera.streamer import StreamManager
from rs_detect.calib.extrinsics import IDENTITY_4X4
from rs_detect.detect.pipeline import run_single_frame
from rs_detect.motion.arm_adapter import CellCArmAdapter
from rs_detect.paths import (
    CAPTURES_DIR,
    EXTRINSICS_PATH,
    PIPELINE_CONFIG_PATH,
    RECORDINGS_DIR,
    ROBOT_CONFIG_PATH,
    ROOT,
    SAFE_POINTS_PATH,
    STORAGE_CONFIG_PATH,
    YOLO_CONFIG_PATH,
    ensure_dirs,
)
from rs_detect.pipeline.coarse_localize import coarse_localize
from rs_detect.pipeline.fine_localize import fine_localize_single
from rs_detect.pipeline.m2_pipeline import run_m2_coarse_then_fine
from rs_detect.pipeline.multiview_fuse import run_multiview_fuse
from rs_detect.pipeline.servo_loop import run_d405_servo_loop
from rs_detect.grasp.agent import get_grasp_agent
from rs_detect.train.yolo_trainer import get_trainer

logger = logging.getLogger(__name__)


class DetectService:
    """Shared backend for desktop UI and web API."""

    def __init__(self):
        ensure_dirs()
        self.streams = StreamManager()
        self.extrinsics = ExtrinsicsStore(EXTRINSICS_PATH)
        self._recorders: Dict[str, VideoRecorder] = {}
        self._last_detect: Dict[str, Any] = {}
        self._last_viz: Dict[str, bytes] = {}
        self._last_pipeline: Dict[str, Any] = {}
        self._workflow: Dict[str, Any] = {}  # M2 Steps state: coarse / p_safe / …
        self._suction_on = False  # placeholder IO state
        storage = self._load_storage_cfg()
        self._captures_dir = self._resolve_dir(storage.get("captures_dir") or CAPTURES_DIR)
        self._recordings_dir = self._resolve_dir(storage.get("recordings_dir") or RECORDINGS_DIR)
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = SnapshotWriter(self._captures_dir)
        # M0→M1 binding: selected serials (wrist=D405, scene=D435)
        ext = self.extrinsics.data
        self._selection = {
            "wrist": (ext.get("d405") or {}).get("serial") or None,
            "scene": (ext.get("d435") or {}).get("serial") or None,
        }
        self._yolo_cfg = self._load_yolo_cfg()
        self._pipeline_cfg = self._load_pipeline_cfg()
        self.arm = CellCArmAdapter(ROBOT_CONFIG_PATH)
        self.yolo_train = get_trainer()
        self.grasp = get_grasp_agent(self)

    def _load_yolo_cfg(self) -> Dict[str, Any]:
        if YOLO_CONFIG_PATH.exists():
            return yaml.safe_load(YOLO_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return {
            "method": "auto",
            "weights": "models/cardboard_yolo.pt",
            "conf": 0.25,
            "class_filter": [],
            "depth_scale": 0.001,
            "ransac_thresh_m": 0.008,
            "max_points": 20000,
        }

    def _load_pipeline_cfg(self) -> Dict[str, Any]:
        if PIPELINE_CONFIG_PATH.exists():
            return yaml.safe_load(PIPELINE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return {
            "work_distance_m": 0.25,
            "safe_z_lift_m": 0.08,
            "pregrasp_z_lift_m": 0.05,
            "fine_max_iters": 5,
            "fine_pos_tol_m": 0.005,
            "fine_settle_s": 0.35,
            "max_gate_distance_m": 0.08,
            "max_gate_normal_angle_deg": 25,
            "skip_fine_if_coarse_fails": True,
            "assume_flange_identity": True,
        }

    def get_detect_config(self) -> Dict[str, Any]:
        self._yolo_cfg = self._load_yolo_cfg()
        self._pipeline_cfg = self._load_pipeline_cfg()
        out = dict(self._yolo_cfg)
        out["pipeline"] = dict(self._pipeline_cfg)
        wpath = self._weights_path(self._yolo_cfg)
        out["weights_path"] = str(wpath)
        out["weights_exists"] = wpath.is_file()
        out["weights_ready"] = wpath.is_file()
        try:
            winfo = self.yolo_train.weights_info()
            out["train"] = {
                "m2_ready": bool(winfo.get("m2_ready")),
                "installed": winfo.get("installed"),
                "bootstrap_exists": bool(winfo.get("bootstrap_exists")),
            }
        except Exception:  # noqa: BLE001
            out["train"] = {"m2_ready": wpath.is_file()}
        return out

    @staticmethod
    def _resolve_dir(path_like: Any) -> Path:
        p = Path(str(path_like)).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        else:
            p = p.resolve()
        return p

    def _load_storage_cfg(self) -> Dict[str, Any]:
        if STORAGE_CONFIG_PATH.exists():
            data = yaml.safe_load(STORAGE_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return data
        return {
            "captures_dir": str(CAPTURES_DIR.relative_to(ROOT)),
            "recordings_dir": str(RECORDINGS_DIR.relative_to(ROOT)),
        }

    def get_storage_settings(self) -> dict:
        return {
            "captures_dir": str(self._captures_dir),
            "recordings_dir": str(self._recordings_dir),
            "captures_dir_display": self._display_path(self._captures_dir),
            "recordings_dir_display": self._display_path(self._recordings_dir),
            "defaults": {
                "captures_dir": str(CAPTURES_DIR),
                "recordings_dir": str(RECORDINGS_DIR),
            },
            "config_path": str(STORAGE_CONFIG_PATH),
        }

    @staticmethod
    def _display_path(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    def set_storage_settings(
        self,
        *,
        captures_dir: Optional[str] = None,
        recordings_dir: Optional[str] = None,
    ) -> dict:
        """Update photo/video save roots; persists to config/storage.yaml."""
        if any(r.active for r in self._recorders.values()):
            raise RuntimeError("stop active recordings before changing save paths")
        cap = self._resolve_dir(captures_dir) if captures_dir else self._captures_dir
        rec = self._resolve_dir(recordings_dir) if recordings_dir else self._recordings_dir
        cap.mkdir(parents=True, exist_ok=True)
        rec.mkdir(parents=True, exist_ok=True)
        # Prefer repo-relative paths in yaml when under ROOT
        payload = {
            "captures_dir": self._display_path(cap),
            "recordings_dir": self._display_path(rec),
        }
        STORAGE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORAGE_CONFIG_PATH.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._captures_dir = cap
        self._recordings_dir = rec
        self.snapshots = SnapshotWriter(cap)
        return self.get_storage_settings()

    def _weights_path(self, cfg: Dict[str, Any]) -> Path:
        weights = cfg.get("weights") or ""
        weights_path = Path(weights)
        if not weights_path.is_absolute():
            weights_path = ROOT / weights_path
        return weights_path

    def _require_stream_packet(self, serial: str):
        st = self.streams.get(serial)
        if st is None or not st.running:
            raise RuntimeError(f"stream not running for {serial}; start M0 stream first")
        pkt = st.get_latest()
        if pkt is None:
            raise RuntimeError(f"no frame yet for {serial}")
        return st, pkt

    def _T_base_flange(self, pipe_cfg: Dict[str, Any]):
        ext = self.extrinsics.data
        custom = (ext.get("robot") or {}).get("T_base_flange")
        if custom:
            return custom
        if pipe_cfg.get("assume_flange_identity", True):
            return [row[:] for row in IDENTITY_4X4]
        return None

    # ---- M1↔M2 selection binding ----
    def get_selection(self) -> Dict[str, Any]:
        return {
            "wrist": self._selection.get("wrist") or "",
            "scene": self._selection.get("scene") or "",
            "d405_serial": self._selection.get("wrist") or "",
            "d435_serial": self._selection.get("scene") or "",
        }

    def bind_selection(self, role: str, serial: str, persist: bool = True) -> Dict[str, Any]:
        """Bind M1 chosen camera to M2 serial fields."""
        if role not in ("wrist", "scene"):
            raise ValueError("role must be wrist|scene")
        serial = (serial or "").strip()
        self._selection[role] = serial or None
        if persist and serial:
            if role == "wrist":
                self.extrinsics.update(d405_serial=serial)
            else:
                self.extrinsics.update(d435_serial=serial)
        return self.get_selection()

    # ---- M1 scan / stream ----
    @staticmethod
    def _pick_scan_serial(candidates: list, preferred: Optional[str]) -> Optional[str]:
        """Prefer real hardware over mock; keep preferred only if still valid."""
        if not candidates:
            return None
        serials = {d.serial for d in candidates}
        has_real = any(not d.is_mock for d in candidates)
        if preferred and preferred in serials:
            pref = next(d for d in candidates if d.serial == preferred)
            if pref.is_mock and has_real:
                return next(d.serial for d in candidates if not d.is_mock)
            return preferred
        real = next((d for d in candidates if not d.is_mock), None)
        return (real or candidates[0]).serial

    def scan(self) -> Dict[str, Any]:
        devices = scan_devices(include_mock_if_empty=True)
        buckets = split_by_role(devices)
        # Unknown RealSense units still selectable (append after role-matched)
        unknowns = list(buckets["unknown"])
        wrist_list = list(buckets["wrist_candidates"]) + unknowns
        scene_list = list(buckets["scene_candidates"]) + unknowns

        # Reconcile selection against this scan (drop stale MOCK when HW appears)
        for role, cands in (("wrist", wrist_list), ("scene", scene_list)):
            picked = self._pick_scan_serial(cands, self._selection.get(role))
            if picked and picked != self._selection.get(role):
                self.bind_selection(role, picked, persist=True)
            elif not picked and self._selection.get(role):
                # Stale serial no longer present
                self._selection[role] = None

        return {
            "devices": [d.to_dict() for d in devices],
            "wrist_candidates": [d.to_dict() for d in wrist_list],
            "scene_candidates": [d.to_dict() for d in scene_list],
            "unknown": [d.to_dict() for d in unknowns],
            "selection": self.get_selection(),
            "hardware_count": sum(1 for d in devices if not d.is_mock),
            "mock_fallback": bool(devices) and all(d.is_mock for d in devices),
        }

    def start_stream(self, serial: str, role: str, width: int = 640, height: int = 480, fps: int = 30) -> dict:
        if role not in ("wrist", "scene"):
            raise ValueError("role must be wrist|scene")
        st = self.streams.start(serial, role=role, width=width, height=height, fps=fps, enable_depth=True)
        selection = self.bind_selection(role, serial, persist=True)
        return {
            "serial": serial,
            "role": role,
            "running": st.running,
            "is_mock": st.is_mock,
            "intrinsics": fetch_intrinsics_from_streamer(st),
            "selection": selection,
        }

    def stop_stream(self, serial: str) -> dict:
        if serial in self._recorders and self._recorders[serial].active:
            self._recorders[serial].stop()
            self._recorders.pop(serial, None)
        self.streams.stop(serial)
        return {"serial": serial, "running": False}

    def stop_all(self) -> None:
        for sn in list(self._recorders.keys()):
            try:
                self._recorders[sn].stop()
            except Exception:
                pass
        self._recorders.clear()
        self.streams.stop_all()

    def shutdown_graceful(self) -> dict:
        """Stop recordings/streams and disarm arm (best-effort) before process exit."""
        notes = []
        try:
            self.stop_all()
            notes.append("streams_stopped")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"streams_error:{exc}")
        try:
            self.arm.stop()
            notes.append("arm_stop")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"arm_stop_error:{exc}")
        try:
            self.arm.set_armed(False)
            notes.append("arm_disarmed")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"arm_disarm_error:{exc}")
        try:
            self.arm.disconnect()
            notes.append("arm_disconnected")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"arm_disconnect_error:{exc}")
        return {"ok": True, "notes": notes}

    def stream_status(self) -> dict:
        return self.streams.list_streams()

    def get_jpeg(self, serial: str, which: str = "color", quality: int = 80) -> Optional[bytes]:
        st = self.streams.get(serial)
        if st is None:
            return None
        pkt = st.get_latest()
        if pkt is None:
            return None
        if which == "depth":
            if pkt.depth_u16 is None:
                return None
            img = colorize_depth(pkt.depth_u16)
        else:
            img = pkt.color_bgr
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return buf.tobytes()

    def snapshot(self, serial: str) -> dict:
        st = self.streams.get(serial)
        if st is None:
            raise RuntimeError(f"stream not running: {serial}")
        pkt = st.get_latest()
        if pkt is None:
            raise RuntimeError("no frame yet")
        paths = self.snapshots.save(pkt)
        return paths

    def start_recording(self, serial: str) -> dict:
        st = self.streams.get(serial)
        if st is None:
            raise RuntimeError(f"stream not running: {serial}")
        if serial in self._recorders and self._recorders[serial].active:
            return {"path": str(self._recorders[serial].out_path), "already": True}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        out = self._recordings_dir / f"{serial}_{ts}.mp4"
        rec = VideoRecorder(st, out)
        rec.start()
        self._recorders[serial] = rec
        return {"path": str(out), "recording": True}

    def stop_recording(self, serial: str) -> dict:
        rec = self._recorders.pop(serial, None)
        if rec is None:
            return {"recording": False}
        return {**rec.stop(), "recording": False}

    # ---- M2 calib ----
    def get_intrinsics(self, serial: str, from_live: bool = True) -> dict:
        st = self.streams.get(serial)
        if from_live and st is not None and st.running:
            return {"serial": serial, "source": "live_sdk", "intrinsics": fetch_intrinsics_from_streamer(st)}
        is_mock = serial.startswith("MOCK_")
        return {
            "serial": serial,
            "source": "sdk_probe",
            "intrinsics": fetch_intrinsics_from_sdk(serial, is_mock=is_mock),
        }

    def get_extrinsics(self) -> dict:
        return self.extrinsics.reload()

    def set_extrinsics(self, payload: dict) -> dict:
        return self.extrinsics.update(
            d405_T=payload.get("d405_T"),
            d435_T=payload.get("d435_T"),
            d405_serial=payload.get("d405_serial"),
            d435_serial=payload.get("d435_serial"),
            notes=payload.get("notes"),
        )

    # ---- M2: D435 coarse → D405 fine → gate (plan flow) ----
    def run_detect(
        self,
        serial: Optional[str] = None,
        role: Optional[str] = None,
        method: Optional[str] = None,
    ) -> dict:
        """Debug: single-camera frame detect (prefer run_m2_pipeline for plan flow)."""
        if not serial:
            if role in ("wrist", "scene"):
                serial = self._selection.get(role)
            elif role == "d405":
                serial = self._selection.get("wrist")
            elif role == "d435":
                serial = self._selection.get("scene")
        if not serial:
            serial = self._selection.get("scene") or self._selection.get("wrist")
        if not serial:
            raise RuntimeError("no camera selected; choose in M0 first")

        st, pkt = self._require_stream_packet(serial)
        cfg = self._load_yolo_cfg()
        weights_path = self._weights_path(cfg)
        result = run_single_frame(
            pkt,
            method=method or cfg.get("method") or "auto",
            weights=weights_path,
            conf=float(cfg.get("conf") or 0.25),
            class_filter=cfg.get("class_filter") or None,
            is_mock=st.is_mock,
            depth_scale=float(cfg.get("depth_scale") or 0.001),
            ransac_thresh=float(cfg.get("ransac_thresh_m") or 0.008),
            max_points=int(cfg.get("max_points") or 20000),
            make_viz=True,
            device=cfg.get("device"),
            imgsz=int(cfg.get("imgsz") or 640),
        )
        payload = result.to_dict()
        payload["role"] = role or ("scene" if serial == self._selection.get("scene") else "wrist")
        self._last_detect[serial] = payload
        if result.viz_jpeg:
            self._last_viz[serial] = result.viz_jpeg
        return payload

    def run_coarse(
        self,
        method: Optional[str] = None,
        *,
        z_lift_m: Optional[float] = None,
        take_snapshot: bool = True,
    ) -> dict:
        """Step1: D435 snapshot + coarse → p_coarse; p_safe = p_coarse + Z lift."""
        serial = self._selection.get("scene")
        if not serial:
            raise RuntimeError("no scene (D435) selected; choose in M0 first")
        st, pkt = self._require_stream_packet(serial)
        cfg = self._load_yolo_cfg()
        pipe = self._load_pipeline_cfg()
        ext = self.extrinsics.data
        T_base_d435 = (ext.get("d435") or {}).get("T") or IDENTITY_4X4
        lift = float(z_lift_m if z_lift_m is not None else (pipe.get("safe_z_lift_m") or 0.08))
        result = coarse_localize(
            pkt,
            T_base_d435,
            method=method or cfg.get("method") or "auto",
            weights=self._weights_path(cfg),
            conf=float(cfg.get("conf") or 0.25),
            class_filter=cfg.get("class_filter") or None,
            is_mock=st.is_mock,
            depth_scale=float(cfg.get("depth_scale") or 0.001),
            ransac_thresh=float(cfg.get("ransac_thresh_m") or 0.008),
            max_points=int(cfg.get("max_points") or 20000),
            work_distance_m=float(pipe.get("work_distance_m") or 0.25),
            make_viz=True,
            device=cfg.get("device"),
            imgsz=int(cfg.get("imgsz") or 640),
        )
        payload = result.to_dict()
        payload["z_lift_m"] = lift
        payload["p_safe"] = None
        if result.ok and result.p_coarse is not None:
            # Safety hover: same XY, Z raised in base frame
            px, py, pz = result.p_coarse
            payload["p_safe"] = [float(px), float(py), float(pz) + lift]
        if take_snapshot:
            try:
                payload["snapshot"] = self.snapshots.save(pkt, prefix="coarse_")
            except Exception as exc:  # noqa: BLE001
                payload["snapshot_error"] = str(exc)
        self._last_detect[serial] = payload
        if result.viz_jpeg:
            self._last_viz[serial] = result.viz_jpeg
        # Persist for subsequent Steps
        self._workflow["step"] = 0
        self._workflow["coarse"] = payload
        self._workflow["p_coarse"] = payload.get("p_coarse")
        self._workflow["n_coarse"] = payload.get("n_coarse")
        self._workflow["p_safe"] = payload.get("p_safe")
        self._workflow["z_lift_m"] = lift
        self._workflow["at_safe"] = False
        self._workflow.pop("move_safe", None)
        self._workflow.pop("fine_servo", None)
        self._workflow.pop("grasp", None)
        self._workflow.pop("safe_run", None)
        self._workflow.pop("suction_release", None)
        # Anchor actual TCP so dry-run relative moves can report perception-frame TCP
        try:
            fk = self.arm.fk()
            self._workflow["tcp_anchor"] = list(fk[:3]) if fk else None
        except Exception:  # noqa: BLE001
            self._workflow["tcp_anchor"] = None
        return payload

    def get_workflow(self) -> dict:
        return dict(self._workflow)

    def set_workflow_safe_z(self, z_lift_m: float) -> dict:
        """Recompute p_safe from last p_coarse when user edits Z lift."""
        lift = float(z_lift_m)
        coarse = self._workflow.get("p_coarse")
        if not coarse or len(coarse) < 3:
            raise RuntimeError("no p_coarse yet; run step1 coarse first")
        p_safe = [float(coarse[0]), float(coarse[1]), float(coarse[2]) + lift]
        self._workflow["z_lift_m"] = lift
        self._workflow["p_safe"] = p_safe
        if isinstance(self._workflow.get("coarse"), dict):
            self._workflow["coarse"]["z_lift_m"] = lift
            self._workflow["coarse"]["p_safe"] = p_safe
        return self.get_workflow()

    def _move_xyz_smart(self, xyz_base: list) -> dict:
        """Absolute TCP move; dry-run fallback maps perception Δ relative to current TCP.

        Mock cameras + identity extrinsics put p_coarse near camera origin, which is
        outside the arm workspace — relative fallback keeps the Steps demo runnable.
        """
        try:
            moved = self.arm.move_xyz_keep_rpy(xyz_base, wait=True)
            moved["motion_mode"] = "absolute"
            return moved
        except RuntimeError as exc:
            if not self.arm.dry_run:
                raise
            cur = self.arm.fk()
            if not cur:
                raise
            ref = self._workflow.get("p_coarse") or list(cur[:3])
            cmd = [
                float(cur[0]) + (float(xyz_base[0]) - float(ref[0])),
                float(cur[1]) + (float(xyz_base[1]) - float(ref[1])),
                float(cur[2]) + (float(xyz_base[2]) - float(ref[2])),
            ]
            moved = self.arm.move_xyz_keep_rpy(cmd, wait=True)
            moved["motion_mode"] = "relative_dry_run"
            moved["ik_error"] = str(exc)
            moved["xyz_requested"] = list(xyz_base)
            moved["xyz_commanded"] = cmd
            return moved

    def move_to_safe(self) -> dict:
        """Step2: arm → p_safe (keep current TCP orientation)."""
        p_safe = self._workflow.get("p_safe")
        if not p_safe or len(p_safe) < 3:
            raise RuntimeError("no p_safe; finish step1 coarse first")
        arm_st = self.arm.ensure_ready()
        moved = self._move_xyz_smart(list(p_safe))
        pose = moved.get("cartesian_xyzrpy") or self.arm.fk()
        mode = moved.get("motion_mode") or "absolute"
        out = {
            "ok": True,
            "p_safe": list(p_safe),
            "motion_mode": mode,
            "arm": {
                "connected": arm_st.get("connected"),
                "armed": True,
                "dry_run": self.arm.dry_run,
                "joints_rad": moved.get("joints_rad"),
                "cartesian_xyzrpy": pose,
                "commanded_xyzrpy": moved.get("commanded_xyzrpy"),
                "ik": moved.get("ik"),
                "ik_error": moved.get("ik_error"),
                "xyz_commanded": moved.get("xyz_commanded"),
            },
            "message": (
                "moved to p_safe"
                if mode == "absolute"
                else "IK unreachable for absolute p_safe; dry-run used relative Δ from p_coarse"
            ),
        }
        self._workflow["step"] = 1
        self._workflow["at_safe"] = True
        self._workflow["move_safe"] = out
        return out

    def run_multiview_fine(
        self,
        method: Optional[str] = None,
        *,
        pregrasp_z_lift_m: Optional[float] = None,
        n_views: Optional[int] = None,
        radius_m: Optional[float] = None,
        move_pregrasp: Optional[bool] = None,
    ) -> dict:
        """Step3 / M3: N wrist viewpoints → voxel+plane fuse → c* / p_pregrasp."""
        if not self._workflow.get("p_safe") and not self._workflow.get("p_coarse"):
            raise RuntimeError("no p_safe/p_coarse; finish step1 first")
        serial = self._selection.get("wrist")
        if not serial:
            raise RuntimeError("no wrist (D405) selected; choose in M0 first")
        self.arm.ensure_ready()
        if not self._workflow.get("at_safe") and self._workflow.get("p_safe"):
            self.move_to_safe()

        st = self.streams.get(serial)
        if st is None or not st.running:
            raise RuntimeError(f"wrist stream not running: {serial}")

        cfg = self._load_yolo_cfg()
        pipe = self._load_pipeline_cfg()
        ext = self.extrinsics.data
        T_flange = (ext.get("d405") or {}).get("T") or IDENTITY_4X4
        lift = float(
            pregrasp_z_lift_m
            if pregrasp_z_lift_m is not None
            else (pipe.get("pregrasp_z_lift_m") or 0.05)
        )
        n_v = int(n_views if n_views is not None else (pipe.get("multiview_n") or 5))
        rad = float(radius_m if radius_m is not None else (pipe.get("multiview_radius_m") or 0.03))
        settle = float(pipe.get("multiview_settle_s") or pipe.get("fine_settle_s") or 0.35)
        voxel = float(pipe.get("multiview_voxel_m") or 0.004)
        min_ok = int(pipe.get("multiview_min_views_ok") or 2)
        min_pts = int(pipe.get("multiview_min_points") or 80)
        do_pre = (
            bool(pipe.get("multiview_move_pregrasp", True))
            if move_pregrasp is None
            else bool(move_pregrasp)
        )

        # Observe around coarse XY at safety Z when available
        p_safe = self._workflow.get("p_safe")
        p_coarse = self._workflow.get("p_coarse")
        if p_safe and len(p_safe) >= 3:
            center = [float(p_safe[0]), float(p_safe[1]), float(p_safe[2])]
        else:
            z_lift = float(self._workflow.get("z_lift_m") or pipe.get("safe_z_lift_m") or 0.08)
            center = [float(p_coarse[0]), float(p_coarse[1]), float(p_coarse[2]) + z_lift]

        dry_xlate = bool(self.arm.dry_run or pipe.get("assume_flange_identity", True))

        def _grab():
            pkt = st.get_latest()
            if pkt is None:
                raise RuntimeError("no wrist frame")
            return pkt

        def _move(xyz):
            return self._move_xyz_smart(list(xyz))

        def _T_bf():
            return self._T_base_flange(pipe)

        result = run_multiview_fuse(
            center_xyz=center,
            grab_packet=_grab,
            move_xyz=_move,
            get_T_base_flange=_T_bf,
            T_flange_d405=T_flange,
            method=method or cfg.get("method") or "auto",
            weights=self._weights_path(cfg),
            conf=float(cfg.get("conf") or 0.25),
            class_filter=cfg.get("class_filter") or None,
            is_mock=st.is_mock,
            depth_scale=float(cfg.get("depth_scale") or 0.001),
            ransac_thresh=float(cfg.get("ransac_thresh_m") or 0.008),
            max_points=int(cfg.get("max_points") or 20000),
            n_views=n_v,
            radius_m=rad,
            settle_s=settle,
            voxel_m=voxel,
            min_views_ok=min_ok,
            min_points_total=min_pts,
            pregrasp_z_lift_m=lift,
            dry_run_translate=dry_xlate,
            serial=serial,
            device=cfg.get("device"),
            imgsz=int(cfg.get("imgsz") or 640),
        )
        payload = result.to_dict()
        # Align with fine_servo fields so Step4/5 keep working
        payload["final_c_star"] = payload.get("c_star")
        payload["final_p_pregrasp"] = payload.get("p_pregrasp")
        payload["mode"] = "multiview"
        payload["steps"] = payload.get("views")
        if result.fine and result.fine.viz_jpeg:
            self._last_viz[serial] = result.fine.viz_jpeg
            self._last_detect[serial] = result.fine.to_dict()

        moved_pre = None
        if result.ok and do_pre and result.p_pregrasp:
            try:
                moved_pre = self._move_xyz_smart(list(result.p_pregrasp))
                payload["pregrasp_move"] = {
                    "ok": True,
                    "motion_mode": moved_pre.get("motion_mode"),
                    "cartesian_xyzrpy": moved_pre.get("cartesian_xyzrpy") or self.arm.fk(),
                }
            except Exception as exc:  # noqa: BLE001
                payload["pregrasp_move"] = {"ok": False, "message": str(exc)}

        payload["arm"] = self.arm.status()
        self._workflow["step"] = 2
        self._workflow["fine_servo"] = payload  # reuse slot for Steps restore
        self._workflow["multiview"] = payload
        self._workflow["c_star"] = payload.get("final_c_star")
        self._workflow["p_pregrasp"] = payload.get("final_p_pregrasp")
        self._workflow["pregrasp_z_lift_m"] = lift
        self._workflow.pop("grasp", None)
        return payload

    def run_fine_servo(
        self,
        method: Optional[str] = None,
        *,
        pregrasp_z_lift_m: Optional[float] = None,
        max_iters: Optional[int] = None,
    ) -> dict:
        """Step3 alt: D405 single-frame servo loop (M2-style, not multiview)."""
        if not self._workflow.get("p_safe"):
            raise RuntimeError("no p_safe; finish step1 first")
        self.arm.ensure_ready()
        # Prefer starting from safety hover
        if not self._workflow.get("at_safe"):
            self.move_to_safe()

        pipe = self._load_pipeline_cfg()
        lift = float(
            pregrasp_z_lift_m
            if pregrasp_z_lift_m is not None
            else (pipe.get("pregrasp_z_lift_m") or 0.05)
        )
        max_it = int(max_iters if max_iters is not None else (pipe.get("fine_max_iters") or 5))
        tol = float(pipe.get("fine_pos_tol_m") or 0.005)
        settle = float(pipe.get("fine_settle_s") or 0.35)

        def _detect():
            return self.run_fine(method=method)

        def _move(xyz):
            return self._move_xyz_smart(list(xyz))

        def _tcp():
            pose = self.arm.fk()
            if not pose:
                return None
            # Dry-run + relative mapping: expose TCP in perception base so XY tol works
            anchor = self._workflow.get("tcp_anchor")
            pc = self._workflow.get("p_coarse")
            if self.arm.dry_run and anchor and pc and len(anchor) >= 3 and len(pc) >= 3:
                return [
                    float(pc[0]) + (float(pose[0]) - float(anchor[0])),
                    float(pc[1]) + (float(pose[1]) - float(anchor[1])),
                    float(pc[2]) + (float(pose[2]) - float(anchor[2])),
                ]
            return list(pose[:3])

        result = run_d405_servo_loop(
            detect_fine=_detect,
            move_xyz=_move,
            get_tcp_xyz=_tcp,
            pregrasp_z_lift_m=lift,
            max_iters=max_it,
            pos_tol_m=tol,
            settle_s=settle,
        )
        payload = result.to_dict()
        payload["arm"] = self.arm.status()
        self._workflow["step"] = 2
        self._workflow["fine_servo"] = payload
        self._workflow["c_star"] = payload.get("final_c_star")
        self._workflow["p_pregrasp"] = payload.get("final_p_pregrasp")
        self._workflow["pregrasp_z_lift_m"] = lift
        self._workflow.pop("grasp", None)
        return payload

    def run_grasp_placeholder(self) -> dict:
        """Step4: grasp / suction on (placeholder IO)."""
        c_star = self._workflow.get("c_star") or (self._workflow.get("fine_servo") or {}).get("final_c_star")
        p_pre = self._workflow.get("p_pregrasp") or (self._workflow.get("fine_servo") or {}).get(
            "final_p_pregrasp"
        )
        if not c_star and not p_pre:
            raise RuntimeError("no fine result; finish step3 fine servo first")
        self._suction_on = True
        out = {
            "ok": True,
            "placeholder": True,
            "action": "grasp",
            "c_star": list(c_star) if c_star else None,
            "p_pregrasp": list(p_pre) if p_pre else None,
            "suction": {
                "on": True,
                "commanded": True,
                "hardware": False,
                "note": "suction ON (placeholder — IO not wired)",
            },
            "descend": {
                "enabled": False,
                "commanded": False,
                "note": "descend to c* not wired yet",
            },
            "message": "抓取占位：吸盘已标记为开启（无硬件 IO），可进入安全点列表下发",
        }
        self._workflow["step"] = 3
        self._workflow["grasp"] = out
        self._workflow.pop("safe_run", None)
        self._workflow.pop("suction_release", None)
        return out

    def get_suction_state(self) -> dict:
        return {
            "on": bool(self._suction_on),
            "hardware": False,
            "note": "placeholder IO — no physical vacuum yet",
        }

    def release_suction(self) -> dict:
        """Final step: release / open suction (placeholder)."""
        was_on = bool(self._suction_on)
        self._suction_on = False
        out = {
            "ok": True,
            "placeholder": True,
            "action": "release",
            "was_on": was_on,
            "suction": self.get_suction_state(),
            "message": (
                "已放开吸盘（占位）"
                if was_on
                else "吸盘本已关闭；已再次确认放开（占位）"
            ),
        }
        self._workflow["step"] = 5
        self._workflow["suction_release"] = out
        return out

    def get_safe_points(self) -> dict:
        data = self._load_safe_points()
        return {
            "path": str(SAFE_POINTS_PATH),
            "version": data.get("version", 1),
            "points": list(data.get("points") or []),
        }

    def set_safe_points(self, points: list) -> dict:
        cleaned = [self._normalize_safe_point(p, i) for i, p in enumerate(points or [])]
        payload = {"version": 1, "points": cleaned}
        SAFE_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAFE_POINTS_PATH.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._workflow["safe_points"] = cleaned
        return self.get_safe_points()

    def capture_safe_point(self, name: Optional[str] = None) -> dict:
        """Append current TCP XYZ (keep-rpy style) into the safe-points list and save."""
        self.arm.ensure_ready()
        pose = self.arm.fk()
        if not pose or len(pose) < 3:
            raise RuntimeError("FK unavailable; connect arm first")
        pts = list(self.get_safe_points()["points"])
        label = (name or f"tcp_{len(pts) + 1}").strip() or f"tcp_{len(pts) + 1}"
        pts.append({"name": label, "xyz": [float(pose[0]), float(pose[1]), float(pose[2])]})
        return self.set_safe_points(pts)

    def run_safe_points_sequence(self, points: Optional[list] = None) -> dict:
        """Step5: dispatch safe-point list — arm moves through them in order."""
        if points is not None:
            saved = self.set_safe_points(points)
            pts = list(saved["points"])
        else:
            pts = list(self.get_safe_points()["points"])
        if not pts:
            raise RuntimeError("safe points list is empty")
        self.arm.ensure_ready()
        steps = []
        ok = True
        for i, pt in enumerate(pts):
            name = pt.get("name") or f"p{i}"
            try:
                moved = self._dispatch_safe_point(pt)
                pose = moved.get("cartesian_xyzrpy") or self.arm.fk()
                steps.append(
                    {
                        "index": i,
                        "name": name,
                        "ok": True,
                        "point": pt,
                        "motion_mode": moved.get("motion_mode"),
                        "cartesian_xyzrpy": pose,
                        "joints_rad": moved.get("joints_rad"),
                        "message": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                ok = False
                steps.append(
                    {
                        "index": i,
                        "name": name,
                        "ok": False,
                        "point": pt,
                        "message": str(exc),
                    }
                )
                break
        out = {
            "ok": ok,
            "count": len(pts),
            "completed": sum(1 for s in steps if s.get("ok")),
            "steps": steps,
            "suction": self.get_suction_state(),
            "message": (
                f"安全点下发完成 {len(steps)}/{len(pts)}"
                if ok
                else f"安全点下发中断于第 {len(steps)}/{len(pts)} 点"
            ),
        }
        self._workflow["step"] = 4
        self._workflow["safe_run"] = out
        return out

    def _load_safe_points(self) -> Dict[str, Any]:
        if SAFE_POINTS_PATH.exists():
            data = yaml.safe_load(SAFE_POINTS_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return data
        return {"version": 1, "points": []}

    @staticmethod
    def _normalize_safe_point(raw: Any, index: int) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError(f"safe point[{index}] must be an object")
        name = str(raw.get("name") or f"p{index}").strip() or f"p{index}"
        if raw.get("home"):
            return {"name": name, "home": True}
        joints = raw.get("joints_rad")
        if joints is not None:
            q = [float(x) for x in list(joints)[:6]]
            if len(q) < 6:
                raise ValueError(f"safe point[{index}] joints_rad needs 6 values")
            return {"name": name, "joints_rad": q}
        xyz = raw.get("xyz")
        if xyz is not None:
            v = [float(x) for x in list(xyz)[:3]]
            if len(v) < 3:
                raise ValueError(f"safe point[{index}] xyz needs 3 values")
            return {"name": name, "xyz": v}
        xyzrpy = raw.get("xyzrpy")
        if xyzrpy is not None:
            v = [float(x) for x in list(xyzrpy)[:6]]
            if len(v) < 6:
                raise ValueError(f"safe point[{index}] xyzrpy needs 6 values")
            return {"name": name, "xyzrpy": v}
        raise ValueError(f"safe point[{index}] needs home|xyz|joints_rad|xyzrpy")

    def _dispatch_safe_point(self, pt: Dict[str, Any]) -> dict:
        if pt.get("home"):
            moved = self.arm.go_home()
            moved["motion_mode"] = "home"
            return moved
        if pt.get("joints_rad") is not None:
            moved = self.arm.move_joints(pt["joints_rad"], wait=True)
            moved["motion_mode"] = "joints"
            return moved
        if pt.get("xyzrpy") is not None:
            try:
                moved = self.arm.move_pose(pt["xyzrpy"], wait=True)
                moved["motion_mode"] = "absolute"
                return moved
            except RuntimeError:
                if not self.arm.dry_run:
                    raise
                # fall through to relative xyz of first 3
                return self._move_xyz_smart(list(pt["xyzrpy"][:3]))
        if pt.get("xyz") is not None:
            return self._move_xyz_smart(list(pt["xyz"]))
        raise ValueError(f"invalid safe point: {pt}")

    def run_fine(self, method: Optional[str] = None) -> dict:
        """Stage B (M2 single-frame): D405 fine suction in cam + base (if T available)."""
        serial = self._selection.get("wrist")
        if not serial:
            raise RuntimeError("no wrist (D405) selected; choose in M0 first")
        st, pkt = self._require_stream_packet(serial)
        cfg = self._load_yolo_cfg()
        pipe = self._load_pipeline_cfg()
        ext = self.extrinsics.data
        T_flange = (ext.get("d405") or {}).get("T") or IDENTITY_4X4
        result = fine_localize_single(
            pkt,
            T_flange_d405=T_flange,
            T_base_flange=self._T_base_flange(pipe),
            method=method or cfg.get("method") or "auto",
            weights=self._weights_path(cfg),
            conf=float(cfg.get("conf") or 0.25),
            class_filter=cfg.get("class_filter") or None,
            is_mock=st.is_mock,
            depth_scale=float(cfg.get("depth_scale") or 0.001),
            ransac_thresh=float(cfg.get("ransac_thresh_m") or 0.008),
            max_points=int(cfg.get("max_points") or 20000),
            make_viz=True,
            device=cfg.get("device"),
            imgsz=int(cfg.get("imgsz") or 640),
        )
        payload = result.to_dict()
        self._last_detect[serial] = payload
        if result.viz_jpeg:
            self._last_viz[serial] = result.viz_jpeg
        return payload

    def run_m2_pipeline(self, method: Optional[str] = None) -> dict:
        """Canonical M2: D435 coarse reference → D405 fine → consistency gate."""
        scene = self._selection.get("scene")
        wrist = self._selection.get("wrist")
        if not scene:
            raise RuntimeError("no scene (D435) selected; choose in M0 first")
        if not wrist:
            raise RuntimeError("no wrist (D405) selected; choose in M0 first")
        st_s, pkt_s = self._require_stream_packet(scene)
        st_w, pkt_w = self._require_stream_packet(wrist)

        cfg = self._load_yolo_cfg()
        pipe = self._load_pipeline_cfg()
        ext = self.extrinsics.data
        T_base_d435 = (ext.get("d435") or {}).get("T") or IDENTITY_4X4
        T_flange = (ext.get("d405") or {}).get("T") or IDENTITY_4X4
        T_base_flange = self._T_base_flange(pipe)
        if T_base_flange is None:
            raise RuntimeError(
                "T_base_flange unavailable; set robot.T_base_flange in extrinsics "
                "or pipeline.assume_flange_identity: true for mock"
            )

        result = run_m2_coarse_then_fine(
            pkt_s,
            pkt_w,
            T_base_d435=T_base_d435,
            T_flange_d405=T_flange,
            T_base_flange=T_base_flange,
            method=method or cfg.get("method") or "auto",
            weights=self._weights_path(cfg),
            conf=float(cfg.get("conf") or 0.25),
            class_filter=cfg.get("class_filter") or None,
            scene_is_mock=st_s.is_mock,
            wrist_is_mock=st_w.is_mock,
            depth_scale=float(cfg.get("depth_scale") or 0.001),
            ransac_thresh=float(cfg.get("ransac_thresh_m") or 0.008),
            max_points=int(cfg.get("max_points") or 20000),
            work_distance_m=float(pipe.get("work_distance_m") or 0.25),
            max_gate_distance_m=float(pipe.get("max_gate_distance_m") or 0.08),
            max_gate_normal_angle_deg=float(pipe.get("max_gate_normal_angle_deg") or 25),
            skip_fine_if_coarse_fails=bool(pipe.get("skip_fine_if_coarse_fails", True)),
            make_viz=True,
        )
        payload = result.to_dict()
        self._last_pipeline = payload
        if result.coarse.viz_jpeg:
            self._last_viz[scene] = result.coarse.viz_jpeg
            self._last_detect[scene] = result.coarse.to_dict()
        if result.fine is not None:
            if result.fine.viz_jpeg:
                self._last_viz[wrist] = result.fine.viz_jpeg
            self._last_detect[wrist] = result.fine.to_dict()
        return payload

    def get_detect_viz_jpeg(self, serial: str) -> Optional[bytes]:
        return self._last_viz.get(serial)

    def get_last_detect(self, serial: Optional[str] = None) -> dict:
        if serial:
            return self._last_detect.get(serial) or {}
        return dict(self._last_detect)

    def get_last_pipeline(self) -> dict:
        return dict(self._last_pipeline)


# process-wide singleton for API + optional desktop sharing
_SERVICE: Optional[DetectService] = None


def get_service() -> DetectService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = DetectService()
    return _SERVICE
