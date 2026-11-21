"""Cell C CS66 arm adapter — dry-run simulator + live via hik-sensors ``arm_cs_write``.

Core driver lives in **sensors** (``EliteCsArmWriteSensor``). This module only:
  - loads ``config/robot_cell_c.yaml``
  - dry-run simulates joints for Three.js
  - LIVE: open/arm/disarm/write through sensors, with the same gates as sensors-arm Cell C
    (RTSI measured joints required before Arm / motion; dense ``servoj_once`` ramp)

Do not fork CS protocol here. Vendor package ``elite_cs_sdk`` is imported by sensors
``_cs_sdk.py`` at LIVE ``open()`` time.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

logger = logging.getLogger(__name__)

try:
    from sensors_arm.arm_pose import joints_rad_to_xyzrpy, xyzrpy_to_joints_rad

    _HAS_ARM_POSE = True
except Exception as exc:  # noqa: BLE001
    joints_rad_to_xyzrpy = None  # type: ignore
    xyzrpy_to_joints_rad = None  # type: ignore
    _HAS_ARM_POSE = False
    logger.warning("sensors_arm.arm_pose unavailable (%s) — FK/IK disabled", exc)

try:
    from sensors.core.base import SensorContext
    from sensors.drivers.arm.cs_write import EliteCsArmWriteSensor

    _HAS_CS_WRITE = True
except Exception as exc:  # noqa: BLE001
    SensorContext = None  # type: ignore
    EliteCsArmWriteSensor = None  # type: ignore
    _HAS_CS_WRITE = False
    logger.warning("arm_cs_write unavailable (%s) — live connect disabled", exc)


DEFAULT_HOME = [2.8695, -1.9539, 1.8019, -1.6618, 4.7475, -0.0275]
DEFAULT_TCP = (0.0, 0.0, 0.192)  # Cell C EFG fingertip (body tip/指根≈0.117)
# Fallback only when profile / YAML soft limits are unavailable (rad).
_DEFAULT_SOFT_LO_RAD = [-math.pi] * 6
_DEFAULT_SOFT_HI_RAD = [math.pi] * 6

DEFAULT_EOAT_CATALOG: Dict[str, Dict[str, Any]] = {
    "cellc": {
        "id": "cellc",
        "label": "cellc · EFG-C65 + D405",
        "urdf": "/assets/models/cs66/cs66_cell_c.urdf?v=cellc-1",
        "tcp_xyz": [0.0, 0.0, 0.192],
        "notes": "Current Cell C EOAT (meshes/)",
    },
    "cellc_m": {
        "id": "cellc_m",
        "label": "cellc_m · digital-model/cellc_m",
        "urdf": "/assets/models/cs66/cs66_cell_c_m.urdf?v=cellc-m-2",
        "tcp_xyz": [0.0, 0.0, 0.337],
        "notes": "Parallel EOAT from digital-model/cellc_m (crossed tips; body-only base)",
    },
}


def load_robot_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _as_len6_floats(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 6:
        return None
    try:
        return [float(raw[i]) for i in range(6)]
    except (TypeError, ValueError):
        return None


def _soft_limits_from_mapping(block: Any, *, degrees: bool) -> Optional[tuple[List[float], List[float]]]:
    if not isinstance(block, dict):
        return None
    lo = _as_len6_floats(block.get("min"))
    hi = _as_len6_floats(block.get("max"))
    if lo is None or hi is None:
        return None
    if degrees:
        lo = [math.radians(x) for x in lo]
        hi = [math.radians(x) for x in hi]
    return lo, hi


def _resolve_path(raw: str, *, base: Path) -> str:
    """Resolve recipe / script paths relative to config dir or sibling sensors-arm."""
    text = str(raw or "").strip()
    if not text:
        return ""
    p = Path(text).expanduser()
    if p.is_file():
        return str(p.resolve())
    candidates = [
        base / text,
        base.parent / text,
        Path.cwd() / text,
        base.parent.parent / "sensors-arm" / text,
    ]
    if text.startswith("configs/"):
        candidates.append(base.parent.parent / "sensors-arm" / text)
    elif text.startswith("rtsi/"):
        candidates.append(base.parent.parent / "sensors-arm" / "configs" / text)
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand.resolve())
        except OSError:
            continue
    return str((base / text).resolve())


class CellCArmAdapter:
    """Dry-run joint simulator; LIVE proxies hik-sensors ``arm_cs_write`` (Cell C)."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.cfg = load_robot_config(self.config_path)
        self.arm_cfg: Dict[str, Any] = dict(self.cfg.get("arm") or {})
        self.ramp_cfg: Dict[str, Any] = dict(self.cfg.get("arm_abs_ramp") or {})
        self.dry_run = bool(self.cfg.get("dry_run", True))
        self._lock = threading.RLock()
        self._connected = False
        self._armed = False
        home = self.arm_cfg.get("home_joints_rad") or DEFAULT_HOME
        self._joints = [float(x) for x in home[:6]]
        self._target = list(self._joints)
        self._ramp_thread: Optional[threading.Thread] = None
        self._stop_ramp = threading.Event()
        self._last_error = ""
        self._sensor = None  # EliteCsArmWriteSensor | None
        self._live_backend = "dry_run" if self.dry_run else "none"
        self._joints_source = "dry" if self.dry_run else "none"
        self._measured = bool(self.dry_run)
        self._last_cmd: Optional[List[float]] = None

    def soft_limits_rad(self) -> tuple[List[float], List[float]]:
        """Per-joint soft limits (rad): YAML override → kinematics profile → ±π."""
        a = self.arm_cfg
        for key, degrees in (("soft_limits_rad", False), ("soft_limits_deg", True)):
            parsed = _soft_limits_from_mapping(a.get(key), degrees=degrees)
            if parsed is not None:
                return parsed

        kin = str(a.get("kinematics") or "cs66_kin_g26562").strip()
        prof = str(a.get("kinematics_profile") or "elite_cs66").strip() or None
        try:
            if kin == "cs66_kin_g26562":
                from sensors.cs66_kin_g26562.teach import teach_joint_soft_limits_deg

                lo_deg, hi_deg = teach_joint_soft_limits_deg(prof)
            elif kin == "cs66_kinematics":
                from sensors.cs66_kinematics.teach import teach_joint_soft_limits_deg

                lo_deg, hi_deg = teach_joint_soft_limits_deg(prof)
            else:
                from sensors.kinematics.teach import teach_joint_soft_limits_deg

                lo_deg, hi_deg = teach_joint_soft_limits_deg()
            lo = [math.radians(float(x)) for x in np.asarray(lo_deg, dtype=float).reshape(6)]
            hi = [math.radians(float(x)) for x in np.asarray(hi_deg, dtype=float).reshape(6)]
            return lo, hi
        except Exception as exc:  # noqa: BLE001
            logger.warning("soft_limits_rad fallback (±π): %s", exc)
            return list(_DEFAULT_SOFT_LO_RAD), list(_DEFAULT_SOFT_HI_RAD)

    def eoat_catalog(self) -> Dict[str, Dict[str, Any]]:
        raw = self.cfg.get("eoat_catalog")
        out = {k: dict(v) for k, v in DEFAULT_EOAT_CATALOG.items()}
        if isinstance(raw, dict):
            for key, val in raw.items():
                if isinstance(val, dict):
                    cur = dict(out.get(str(key)) or {"id": str(key)})
                    cur.update(val)
                    cur["id"] = str(val.get("id") or key)
                    out[str(key)] = cur
        return out

    def current_eoat_id(self) -> str:
        a = self.arm_cfg
        eid = str(a.get("eoat_id") or "").strip()
        catalog = self.eoat_catalog()
        if eid and eid in catalog:
            return eid
        urdf = str(a.get("urdf") or "")
        for key, entry in catalog.items():
            if str(entry.get("urdf") or "").split("?")[0] == urdf.split("?")[0]:
                return key
        return "cellc" if "cellc" in catalog else next(iter(catalog), "cellc")

    def set_eoat(self, eoat_id: str, *, persist: bool = True) -> Dict[str, Any]:
        """Switch EOAT URDF package (cellc | cellc_m). Viz-only; reconnect not required."""
        catalog = self.eoat_catalog()
        key = str(eoat_id or "").strip()
        if key not in catalog:
            raise RuntimeError(f"unknown eoat_id={eoat_id!r}; choose from {sorted(catalog)}")
        entry = catalog[key]
        with self._lock:
            self.arm_cfg["eoat_id"] = key
            self.arm_cfg["urdf"] = str(entry.get("urdf") or self.arm_cfg.get("urdf"))
            tcp = entry.get("tcp_xyz")
            if isinstance(tcp, (list, tuple)) and len(tcp) >= 3:
                self.arm_cfg["tcp_xyz"] = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
            self.cfg["arm"] = dict(self.arm_cfg)
            if "eoat_catalog" not in self.cfg:
                self.cfg["eoat_catalog"] = catalog
        if persist:
            self._persist_cfg()
        return self.eoat_status()

    def eoat_status(self) -> Dict[str, Any]:
        catalog = self.eoat_catalog()
        eid = self.current_eoat_id()
        return {
            "ok": True,
            "eoat_id": eid,
            "eoat": catalog.get(eid),
            "catalog": list(catalog.values()),
            "model": self.model_descriptor(),
        }

    # ---- catalog for Three.js ----
    def model_descriptor(self) -> Dict[str, Any]:
        a = self.arm_cfg
        lo_rad, hi_rad = self.soft_limits_rad()
        eid = self.current_eoat_id()
        return {
            "id": a.get("id") or "arm-c",
            "label": a.get("label") or "Cell C · CS66 G26562",
            "model": a.get("model") or "elite_cs66_g26562",
            "brand": a.get("brand") or "Elite",
            "product": a.get("product") or "CS66",
            "eoat_id": eid,
            "urdf": a.get("urdf") or "/assets/models/cs66/cs66_cell_c.urdf",
            "joint_names": list(a.get("joint_names") or [f"joint{i}" for i in range(1, 7)]),
            "joint_offset_rad": list(a.get("joint_offset_rad") or [0.0] * 6),
            "joint_signs": list(a.get("joint_signs") or [1.0] * 6),
            "mirror_y": bool(a.get("mirror_y", False)),
            "gripper_rpy": list(a.get("gripper_rpy") or [0.0, 0.0, math.pi]),
            "tcp_xyz": list(a.get("tcp_xyz") or list(DEFAULT_TCP)),
            "home_joints_rad": list(a.get("home_joints_rad") or DEFAULT_HOME),
            "soft_limits_rad": {"min": list(lo_rad), "max": list(hi_rad)},
            "soft_limits_deg": {
                "min": [math.degrees(x) for x in lo_rad],
                "max": [math.degrees(x) for x in hi_rad],
            },
            "kinematics": a.get("kinematics") or "cs66_kin_g26562",
            "kinematics_profile": a.get("kinematics_profile") or "elite_cs66",
            "robot_ip": a.get("robot_ip"),
            "dry_run": self.dry_run,
            "has_fk": _HAS_ARM_POSE,
            "has_cs_write": _HAS_CS_WRITE,
            "live_backend": self._live_backend,
            "kind": a.get("kind") or "arm_cs_write",
            "require_rtsi": bool(a.get("require_rtsi", True)),
        }

    def _tcp(self) -> tuple:
        t = self.arm_cfg.get("tcp_xyz") or DEFAULT_TCP
        return (float(t[0]), float(t[1]), float(t[2]))

    def _kin_kwargs(self) -> Dict[str, Any]:
        return {
            "tcp_xyz": self._tcp(),
            "kinematics": self.arm_cfg.get("kinematics") or "cs66_kin_g26562",
            "kinematics_profile": self.arm_cfg.get("kinematics_profile") or "elite_cs66",
        }

    def fk(self, joints: Optional[Sequence[float]] = None) -> Optional[List[float]]:
        if not _HAS_ARM_POSE:
            return None
        q = list(joints) if joints is not None else self.get_joints()
        return joints_rad_to_xyzrpy(q, **self._kin_kwargs())

    def ik(
        self,
        xyzrpy: Sequence[float],
        seed: Optional[Sequence[float]] = None,
        *,
        tcp_xyz: Optional[Sequence[float]] = None,
        preview: bool = False,
        **ik_kwargs: Any,
    ) -> Dict[str, Any]:
        """TCP (default) or flange (``tcp_xyz=(0,0,0)``) IK.

        ``preview=True`` loosens pose-check / residual for Three.js ghost sync.
        """
        if not _HAS_ARM_POSE:
            return {
                "ok": False,
                "error": "sensors_arm.arm_pose not importable (need Python≥3.10)",
                "joints_rad": None,
            }
        seed_q = list(seed) if seed is not None else self.get_joints()
        kw = dict(self._kin_kwargs())
        if tcp_xyz is not None:
            kw["tcp_xyz"] = [float(v) for v in list(tcp_xyz)[:3]]
        if preview:
            # Canvas ghost only: arm_pose hard-fails when
            # pos_err > max(tol*5, 2.5cm). tol=1cm → 5cm ceiling; S0 often ~5cm.
            kw.setdefault("position_tolerance_m", 3.0e-2)  # ceiling 15cm
            kw.setdefault("accept_residual", 1.2e-1)
            kw.setdefault("max_nfev", 1600)
            kw.setdefault("multi_seed", True)
        kw.update(ik_kwargs)
        return xyzrpy_to_joints_rad(list(xyzrpy), q_seed_rad=seed_q, **kw)

    def _read_sample(self) -> Dict[str, Any]:
        """Read joints from sensors driver (LIVE) or local sim (dry-run)."""
        if self.dry_run or self._sensor is None:
            with self._lock:
                q = list(self._joints)
            return {
                "joints_rad": q,
                "joints_source": "dry",
                "measured": True,
                "armed": self._armed,
                "dry_run": True,
            }
        sample = dict(self._sensor.read() or {})
        q = sample.get("joints_rad")
        source = str(sample.get("joints_source") or "none")
        measured = bool(sample.get("measured"))
        # Align with sensors-arm: live measured == RTSI (or dry).
        if source == "rtsi":
            measured = True
        with self._lock:
            if q and len(q) >= 6:
                self._joints = [float(x) for x in q[:6]]
            self._joints_source = source
            self._measured = measured
            if sample.get("error"):
                self._last_error = str(sample["error"])
            # Keep armed flag in sync with driver (keepalive may auto-disarm).
            if "armed" in sample:
                self._armed = bool(sample["armed"])
        return sample

    def get_joints(self) -> List[float]:
        sample = self._read_sample()
        q = sample.get("joints_rad")
        if q and len(q) >= 6:
            return [float(x) for x in q[:6]]
        with self._lock:
            return list(self._joints)

    def latest_joints(self, *, require_measured: bool = False) -> Optional[List[float]]:
        """Like sensors-arm Orchestrator.latest_joints — refuse synth/command when live."""
        sample = self._read_sample()
        q = sample.get("joints_rad")
        if not q or len(q) < 6:
            return None
        source = str(sample.get("joints_source") or "none")
        measured = bool(sample.get("measured")) or source == "rtsi"
        if self.dry_run:
            return [float(x) for x in q[:6]]
        if require_measured and not measured:
            return None
        if require_measured and source not in ("rtsi",):
            # Live Cell C: only RTSI counts as measured (sensors-arm gate).
            return None
        return [float(x) for x in q[:6]]

    def status(self) -> Dict[str, Any]:
        sample = self._read_sample()
        q = sample.get("joints_rad")
        if not q or len(q) < 6:
            with self._lock:
                q = list(self._joints)
        else:
            q = [float(x) for x in q[:6]]
        with self._lock:
            connected = self._connected
            armed = self._armed
            err = self._last_error
            backend = self._live_backend
            source = self._joints_source
            measured = self._measured
        pose = self.fk(q)
        home = [float(x) for x in (self.arm_cfg.get("home_joints_rad") or DEFAULT_HOME)[:6]]
        can_arm = bool(connected and (self.dry_run or (measured and source == "rtsi")))
        return {
            "ok": True,
            "dry_run": self.dry_run,
            "connected": connected,
            "armed": armed,
            "joints_rad": q,
            "cartesian_xyzrpy": pose,
            "home_joints_rad": home,
            "error": err,
            "live_backend": backend,
            "joints_source": source if connected else ("dry" if self.dry_run else "none"),
            "measured": bool(measured) if connected else bool(self.dry_run),
            "can_arm": can_arm if connected else False,
            "model": self.model_descriptor(),
        }

    def _sensor_params(self) -> Dict[str, Any]:
        """Params aligned with sensors-arm ``sensors_multi.yaml`` arm-c / arm_cs_write defaults."""
        a = self.arm_cfg
        base = self.config_path.parent
        rtsi_out = _resolve_path(
            str(a.get("rtsi_output_recipe") or "rtsi/output_recipe.txt"),
            base=base,
        )
        rtsi_in = _resolve_path(
            str(a.get("rtsi_input_recipe") or "rtsi/input_recipe.txt"),
            base=base,
        )
        if not Path(rtsi_out).is_file():
            raise RuntimeError(
                f"RTSI output recipe missing: {rtsi_out} "
                "(need config/rtsi/output_recipe.txt; align sensors-arm Cell C)"
            )
        params: Dict[str, Any] = {
            "robot_ip": a.get("robot_ip") or a.get("endpoint") or "192.168.1.200",
            "primary_port": int(a.get("primary_port") or a.get("port") or 30001),
            "num_joints": int(a.get("num_joints") or 6),
            "max_delta_deg": float(a.get("max_delta_deg") or 2.0),
            "enforce_soft_limits": bool(a.get("enforce_soft_limits", True)),
            "headless_mode": bool(a.get("headless_mode", True)),
            "require_rtsi": bool(a.get("require_rtsi", True)),
            "stream_max_rad_s": float(a.get("stream_max_rad_s") or 0.5),
            "cmd_continuity_rad": float(a.get("cmd_continuity_rad") or 0.10),
            "servoj_time": float(a.get("servoj_time") or 0.01),
            "servoj_lookahead_time": float(a.get("servoj_lookahead_time") or 0.25),
            "servoj_gain": int(a.get("servoj_gain") or 300),
            "servoj_timeout_ms": int(a.get("servoj_timeout_ms") or 2000),
            "idle_timeout_ms": int(a.get("idle_timeout_ms") or 30000),
            "keepalive_s": float(a.get("keepalive_s") or 2.0),
            "kinematics": a.get("kinematics") or "cs66_kin_g26562",
            "kinematics_profile": a.get("kinematics_profile") or "elite_cs66",
            "rtsi_output_recipe": rtsi_out,
            "rtsi_input_recipe": rtsi_in if Path(rtsi_in).is_file() else "",
            "rtsi_frequency": float(a.get("rtsi_frequency") or 125.0),
            "ensure_power": bool(a.get("ensure_power", True)),
        }
        if a.get("local_ip"):
            params["local_ip"] = str(a["local_ip"])
        if a.get("script_file_path"):
            params["script_file_path"] = _resolve_path(str(a["script_file_path"]), base=base)
        return params

    def _build_live_sensor(self):
        if not _HAS_CS_WRITE or EliteCsArmWriteSensor is None or SensorContext is None:
            raise RuntimeError(
                "hik-sensors arm_cs_write not importable; "
                "ensure PYTHONPATH includes sensors/src (or pip install -e sensors)"
            )
        # Vendor binding is required by sensors._cs_sdk at LIVE open — surface early.
        try:
            from sensors.drivers.arm._cs_sdk import import_elite_cs_sdk

            import_elite_cs_sdk()
        except ImportError as exc:
            raise RuntimeError(
                "sensors LIVE open needs vendor package elite_cs_sdk in this venv "
                "(not PyPI elirobots). Core driver remains hik-sensors arm_cs_write. "
                "See sensors-arm/docs/elite-cs-sdk-no-pypi-workaround.md"
            ) from exc
        arm_id = str(self.arm_cfg.get("id") or "arm-c")
        params = self._sensor_params()
        ctx = SensorContext(dry_run=False)
        return EliteCsArmWriteSensor(arm_id, params, ctx)

    def connect(self) -> Dict[str, Any]:
        self._last_error = ""
        if self.dry_run:
            with self._lock:
                self._connected = True
                self._armed = False
                self._live_backend = "dry_run"
                self._joints_source = "dry"
                self._measured = True
                home = self.arm_cfg.get("home_joints_rad") or DEFAULT_HOME
                self._joints = [float(x) for x in home[:6]]
                self._target = list(self._joints)
                self._last_cmd = list(self._joints)
                self._sensor = None
            logger.info("arm connect dry_run=true ip=%s", self.arm_cfg.get("robot_ip"))
            return self.status()

        with self._lock:
            if self._connected and self._sensor is not None:
                return self.status()

        try:
            sensor = self._build_live_sensor()
            sensor.open()  # sensors: EliteDriver + RTSI
            with self._lock:
                self._sensor = sensor
                self._connected = True
                self._armed = False
                self._live_backend = "arm_cs_write"
            # Prefer RTSI sample after open (may take a tick).
            q = None
            source = "none"
            for _ in range(10):
                sample = self._read_sample()
                q = sample.get("joints_rad")
                source = str(sample.get("joints_source") or "none")
                if q and len(q) >= 6 and source == "rtsi":
                    break
                time.sleep(0.05)
            with self._lock:
                if q and len(q) >= 6:
                    self._joints = [float(x) for x in q[:6]]
                    self._target = list(self._joints)
                    self._last_cmd = list(self._joints)
            logger.info(
                "arm connect LIVE ip=%s backend=arm_cs_write source=%s",
                self.arm_cfg.get("robot_ip"),
                source,
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._connected = False
                self._sensor = None
                self._live_backend = "none"
                self._joints_source = "none"
                self._measured = False
                self._last_error = str(exc)
            # ImportError / open failures are expected until vendor SDK is installed.
            logger.error("arm live connect failed: %s", exc)
            raise
        return self.status()

    def disconnect(self) -> Dict[str, Any]:
        self._stop_ramp.set()
        with self._lock:
            sensor = self._sensor
            self._sensor = None
            self._armed = False
            self._connected = False
            self._last_cmd = None
            if not self.dry_run:
                self._live_backend = "none"
                self._joints_source = "none"
                self._measured = False
            else:
                self._live_backend = "dry_run"
                self._joints_source = "dry"
                self._measured = True
        if sensor is not None:
            try:
                sensor.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("arm sensor close: %s", exc)
        return self.status()

    def set_armed(self, armed: bool) -> Dict[str, Any]:
        with self._lock:
            if not self._connected:
                raise RuntimeError("not connected")
            if self.dry_run or self._sensor is None:
                self._armed = bool(armed)
                if armed:
                    self._last_cmd = list(self._joints)
                return self.status()
            sensor = self._sensor

        if armed:
            # sensors-arm gate: refuse Arm without measured RTSI joints.
            q = self.latest_joints(require_measured=True)
            if q is None:
                raise RuntimeError(
                    "no measured RTSI joints — wait until joints_source=rtsi "
                    "matches teach pendant (CS66 checklist)"
                )
            result = sensor.arm()
            if not result.get("ok", True):
                with self._lock:
                    self._armed = False
                    self._last_error = str(result.get("error") or "arm() failed")
                raise RuntimeError(self._last_error)
            seeded = result.get("joints_rad") or q
            with self._lock:
                self._armed = True
                self._last_error = ""
                if seeded and len(seeded) >= 6:
                    self._joints = [float(x) for x in seeded[:6]]
                    self._last_cmd = list(self._joints)
                src = result.get("joints_source")
                if src:
                    self._joints_source = str(src)
                    self._measured = str(src) == "rtsi"
        else:
            self._stop_ramp.set()
            try:
                sensor.disarm(stop=True)
            finally:
                with self._lock:
                    self._armed = False
        return self.status()

    def stop(self) -> Dict[str, Any]:
        self._stop_ramp.set()
        with self._lock:
            self._target = list(self._joints)
            sensor = self._sensor if not self.dry_run else None
        if sensor is not None:
            try:
                sensor.write({"stop": True})
            except Exception as exc:  # noqa: BLE001
                logger.warning("arm stop write: %s", exc)
            with self._lock:
                self._armed = False
        return self.status()

    def go_home(self) -> Dict[str, Any]:
        home = [float(x) for x in (self.arm_cfg.get("home_joints_rad") or DEFAULT_HOME)[:6]]
        return self.move_joints(home)

    def get_home_joints(self) -> List[float]:
        return [float(x) for x in (self.arm_cfg.get("home_joints_rad") or DEFAULT_HOME)[:6]]

    def set_home_from_current(self, joints_rad: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        if joints_rad is not None:
            q = [float(x) for x in list(joints_rad)[:6]]
        else:
            # Prefer measured when live.
            q = self.latest_joints(require_measured=not self.dry_run) or self.get_joints()
        if len(q) < 6 or any(not math.isfinite(x) for x in q):
            raise ValueError("need 6 finite joints to set home")
        self.arm_cfg["home_joints_rad"] = q
        if not isinstance(self.cfg.get("arm"), dict):
            self.cfg["arm"] = dict(self.arm_cfg)
        else:
            self.cfg["arm"]["home_joints_rad"] = q
        self._persist_cfg()
        st = self.status()
        st["home_joints_rad"] = q
        st["message"] = "home set from current joints"
        return st

    def _persist_cfg(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(self.cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def set_dry_run(self, dry_run: bool, *, persist: bool = True) -> Dict[str, Any]:
        """Switch dry-run / LIVE. Must disconnect first (same as sensors-arm start-time flag)."""
        dry_run = bool(dry_run)
        with self._lock:
            if self._connected:
                raise RuntimeError("disconnect arm before switching dry_run / LIVE")
            self.dry_run = dry_run
            self.cfg["dry_run"] = dry_run
            self._live_backend = "dry_run" if dry_run else "none"
            self._joints_source = "dry" if dry_run else "none"
            self._measured = bool(dry_run)
            self._last_error = ""
        if persist:
            self._persist_cfg()
        st = self.status()
        st["message"] = (
            "dry_run=true (simulator)"
            if dry_run
            else "dry_run=false (LIVE via sensors arm_cs_write)"
        )
        return st

    def move_joints(
        self,
        joints_rad: Sequence[float],
        *,
        immediate: bool = False,
        wait: bool = True,
        wait_timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        q = [float(x) for x in list(joints_rad)[:6]]
        if len(q) < 6 or any(not math.isfinite(x) for x in q):
            raise ValueError("joints_rad must be 6 finite floats")
        with self._lock:
            if not self._connected:
                raise RuntimeError("not connected")
            if not self._armed:
                raise RuntimeError("not armed — call Arm first")
        if not self.dry_run:
            # Live: require measured RTSI before absolute send (sensors-arm).
            if self.latest_joints(require_measured=True) is None:
                raise RuntimeError(
                    "no measured RTSI joints; refuse motion (CS66 live gate)"
                )
        if immediate and self.dry_run:
            with self._lock:
                self._joints = q
                self._target = q
                self._last_cmd = list(q)
            return self.status()
        self._start_ramp(q)
        if wait:
            timeout = wait_timeout_s
            if timeout is None:
                timeout = float(self.ramp_cfg.get("t_max_s") or 8.0) + 2.0
            if self._ramp_thread is not None:
                self._ramp_thread.join(timeout=timeout)
            with self._lock:
                if self._last_error and not self.dry_run:
                    raise RuntimeError(self._last_error)
        return self.status()

    def move_pose(
        self,
        xyzrpy: Sequence[float],
        *,
        wait: bool = True,
        wait_timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        ik = self.ik(xyzrpy)
        if not ik.get("ok") or not ik.get("joints_rad"):
            raise RuntimeError(f"IK failed: {ik.get('error')}")
        out = self.move_joints(
            ik["joints_rad"],
            wait=wait,
            wait_timeout_s=wait_timeout_s,
        )
        out["ik"] = {k: v for k, v in ik.items() if k != "debug"}
        out["commanded_xyzrpy"] = [float(x) for x in list(xyzrpy)[:6]]
        return out

    def move_xyz_keep_rpy(
        self,
        xyz: Sequence[float],
        *,
        wait: bool = True,
    ) -> Dict[str, Any]:
        if len(xyz) < 3:
            raise ValueError("xyz must have 3 elements")
        cur = self.fk() or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xyzrpy = [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(cur[3]),
            float(cur[4]),
            float(cur[5]),
        ]
        return self.move_pose(xyzrpy, wait=wait)

    def ensure_ready(self) -> Dict[str, Any]:
        st = self.status()
        if not st["connected"]:
            st = self.connect()
        if not st["armed"]:
            st = self.set_armed(True)
        return st

    def _start_ramp(self, target: List[float]) -> None:
        self._stop_ramp.set()
        if self._ramp_thread and self._ramp_thread.is_alive():
            self._ramp_thread.join(timeout=1.0)
        self._stop_ramp.clear()
        # Prefer last_cmd / measured start (not lagged during prior motion).
        start = self.latest_joints(require_measured=not self.dry_run)
        with self._lock:
            if start is None:
                start = list(self._last_cmd or self._joints)
            self._target = list(target)
            self._last_error = ""
        t = threading.Thread(
            target=self._ramp_worker,
            args=(list(start), list(target)),
            daemon=True,
            name="cell-c-abs-ramp",
        )
        self._ramp_thread = t
        t.start()

    def _write_live_servoj(
        self,
        q: List[float],
        *,
        prev_cmd: List[float],
        idle_after: bool = False,
    ) -> Dict[str, Any]:
        """Dense CS write — mirrors sensors-arm AbsRampController cs_dense ticks."""
        sensor = self._sensor
        if sensor is None:
            return {"ok": False, "error": "no live sensor"}
        body: Dict[str, Any] = {
            "joints_rad": list(q),
            "reference_joints_rad": list(prev_cmd),
            "servoj_once": True,
            "relative_to_last": True,
        }
        if idle_after:
            body["idle_after"] = True
        return dict(sensor.write(body) or {})

    def _ramp_worker(self, start: List[float], target: List[float]) -> None:
        rc = self.ramp_cfg
        v = float(rc.get("v_norm_rad_s") or 0.35)
        t_min = float(rc.get("t_min_s") or 0.1)
        t_max = float(rc.get("t_max_s") or 8.0)
        hz = float(rc.get("ramp_hz") or 30.0)
        live = not self.dry_run
        if live:
            # sensors-arm / sensors-dcs: CS dense ~50 Hz
            hz = max(hz, float(rc.get("cs_ramp_hz") or 50.0))
        dist = float(np.linalg.norm(np.asarray(target, dtype=float) - np.asarray(start, dtype=float)))
        dur = max(t_min, min(t_max, dist / max(v, 1e-6)))
        n = max(2, int(round(dur * hz)))
        max_delta_rad = abs(math.radians(float(self.arm_cfg.get("max_delta_deg") or 2.0)))
        if live and max_delta_rad > 0 and dist / n > max_delta_rad:
            n = max(n, int(math.ceil(dist / max_delta_rad)))
        period = 1.0 / hz
        prev_cmd = list(start)
        try:
            next_t = time.perf_counter()
            for i in range(1, n + 1):
                if self._stop_ramp.is_set():
                    break
                s = i / float(n)
                alpha = s * s * (3.0 - 2.0 * s)
                q = [(1.0 - alpha) * a + alpha * b for a, b in zip(start, target)]
                if live:
                    result = self._write_live_servoj(q, prev_cmd=prev_cmd, idle_after=False)
                    if not result.get("ok", True):
                        raise RuntimeError(
                            result.get("error") or f"servoj_once failed at {i}/{n}"
                        )
                    prev_cmd = list(q)
                with self._lock:
                    self._joints = q
                    self._last_cmd = list(q)
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    if self._stop_ramp.wait(delay):
                        break
                else:
                    next_t = time.perf_counter()
            if not self._stop_ramp.is_set():
                if live:
                    result = self._write_live_servoj(
                        list(target), prev_cmd=prev_cmd, idle_after=True
                    )
                    if not result.get("ok", True):
                        raise RuntimeError(result.get("error") or "final servoj_once failed")
                with self._lock:
                    self._joints = list(target)
                    self._last_cmd = list(target)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = str(exc)
                self._armed = False
            logger.exception("arm ramp failed")

    def reload_config(self) -> Dict[str, Any]:
        if self._connected:
            raise RuntimeError("disconnect before reload_config")
        self.cfg = load_robot_config(self.config_path)
        self.arm_cfg = dict(self.cfg.get("arm") or {})
        self.ramp_cfg = dict(self.cfg.get("arm_abs_ramp") or {})
        self.dry_run = bool(self.cfg.get("dry_run", True))
        self._live_backend = "dry_run" if self.dry_run else "none"
        self._joints_source = "dry" if self.dry_run else "none"
        self._measured = bool(self.dry_run)
        return self.model_descriptor()
