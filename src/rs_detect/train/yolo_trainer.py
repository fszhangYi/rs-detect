"""Background Ultralytics YOLO trainer (subprocess) + progress/metrics parsing.

Mirrors ``scripts/train_yolo_cardboard.sh`` but exposes status for the web Train tab.
Defaults target the ``data/yolo_seg`` instance-seg dataset with ``yolo11n-seg``.
"""
from __future__ import annotations

import csv
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import yaml

from rs_detect.paths import ROOT, YOLO_CONFIG_PATH

logger = logging.getLogger(__name__)

DEFAULT_TASK = "segment"
DEFAULT_MODEL = "models/yolo11n-seg.pt"
DEFAULT_DATA = "data/yolo_seg/dataset.yaml"
DEFAULT_PROJECT = "runs/segment"
DEFAULT_NAME = "cardboard_seg"
DEFAULT_INSTALL = "models/cardboard_yolo.pt"
VERIFY_DIR = "runs/verify"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARTIFACT_NAMES = (
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "MaskPR_curve.png",
    "MaskF1_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "labels.jpg",
    "train_batch0.jpg",
    "val_batch0_pred.jpg",
    "val_batch0_labels.jpg",
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _count_labels(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*.txt") if p.is_file() and p.name != "classes.txt")


def _resolve_under_root(raw: str, *, must_exist: bool = False) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("path is empty")
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay under project root: {text}") from exc
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _parse_results_csv(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"rows": [], "latest": None, "epochs_done": 0}
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not raw:
                continue
            cleaned: Dict[str, Any] = {}
            for k, v in raw.items():
                key = (k or "").strip()
                if not key:
                    continue
                val = (v or "").strip()
                if val == "":
                    cleaned[key] = None
                    continue
                try:
                    cleaned[key] = float(val) if ("." in val or "e" in val.lower()) else int(val)
                except ValueError:
                    cleaned[key] = val
            rows.append(cleaned)
    latest = rows[-1] if rows else None
    epochs_done = 0
    if latest is not None and latest.get("epoch") is not None:
        try:
            epochs_done = int(float(latest["epoch"]))
        except (TypeError, ValueError):
            epochs_done = len(rows)
    elif rows:
        epochs_done = len(rows)
    return {"rows": rows, "latest": latest, "epochs_done": epochs_done}


class YoloTrainManager:
    """One-at-a-time YOLO segment train job (instance seg → M2 cardboard weights)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._log: Deque[str] = deque(maxlen=4000)
        self._state = "idle"  # idle | running | succeeded | failed | stopped
        self._error = ""
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._params: Dict[str, Any] = {}
        self._returncode: Optional[int] = None
        self._installed_to: Optional[str] = None

    # ---- catalog ----
    def defaults(self) -> Dict[str, Any]:
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "0"
        except Exception:  # noqa: BLE001
            pass
        return {
            "task": DEFAULT_TASK,
            "epochs": 50,
            "imgsz": 640,
            "batch": 8,
            "device": device,
            "model": DEFAULT_MODEL,
            "data": DEFAULT_DATA,
            "project": DEFAULT_PROJECT,
            "name": DEFAULT_NAME,
            "install_to": DEFAULT_INSTALL,
            "exist_ok": True,
        }

    def dataset_info(self) -> Dict[str, Any]:
        data_yaml = ROOT / DEFAULT_DATA
        img_train = ROOT / "data" / "yolo_seg" / "images" / "train"
        img_val = ROOT / "data" / "yolo_seg" / "images" / "val"
        lab_train = ROOT / "data" / "yolo_seg" / "labels" / "train"
        lab_val = ROOT / "data" / "yolo_seg" / "labels" / "val"
        n_train = _count_images(img_train)
        n_val = _count_images(img_val)
        names: Dict[Any, Any] = {}
        task = DEFAULT_TASK
        if data_yaml.is_file():
            try:
                raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
                if isinstance(raw, dict):
                    if raw.get("names") is not None:
                        names = raw.get("names") or {}
                    if raw.get("task"):
                        task = str(raw["task"])
            except Exception:  # noqa: BLE001
                pass
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        return {
            "data_yaml": _rel(data_yaml) if data_yaml.exists() else DEFAULT_DATA,
            "data_yaml_exists": data_yaml.is_file(),
            "task": task,
            "names": names,
            "train_images": n_train,
            "val_images": n_val,
            "train_labels": _count_labels(lab_train),
            "val_labels": _count_labels(lab_val),
            "ready": n_train >= 1 and data_yaml.is_file(),
            "hint": (
                None
                if n_train >= 1
                else "请先在 data/yolo_seg/images/train 放入分割标注图（fast-label 导出 polygon）"
            ),
        }

    def weights_info(self) -> Dict[str, Any]:
        bootstrap = ROOT / DEFAULT_MODEL
        installed = ROOT / DEFAULT_INSTALL
        cfg_weights = DEFAULT_INSTALL
        if YOLO_CONFIG_PATH.is_file():
            try:
                import yaml

                cfg = yaml.safe_load(YOLO_CONFIG_PATH.read_text(encoding="utf-8")) or {}
                if isinstance(cfg, dict) and cfg.get("weights"):
                    cfg_weights = str(cfg["weights"])
            except Exception:  # noqa: BLE001
                pass
        cfg_path = _resolve_under_root(cfg_weights) if cfg_weights else installed
        return {
            "bootstrap": _rel(bootstrap),
            "bootstrap_exists": bootstrap.is_file(),
            "bootstrap_bytes": bootstrap.stat().st_size if bootstrap.is_file() else 0,
            "installed": _rel(installed),
            "installed_exists": installed.is_file(),
            "installed_bytes": installed.stat().st_size if installed.is_file() else 0,
            "config_weights": cfg_weights,
            "config_weights_exists": cfg_path.is_file(),
            "m2_ready": cfg_path.is_file(),
        }

    def run_dir(self, project: Optional[str] = None, name: Optional[str] = None) -> Path:
        with self._lock:
            proj = project or (self._params.get("project") if self._params else None) or DEFAULT_PROJECT
            nm = name or (self._params.get("name") if self._params else None) or DEFAULT_NAME
        return _resolve_under_root(f"{proj}/{nm}")

    # ---- job control ----
    def start(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("训练已在进行中")
            ds = self.dataset_info()
            if not ds["ready"]:
                raise RuntimeError(ds.get("hint") or "数据集未就绪")
            base = self.defaults()
            raw = dict(params or {})
            task = str(raw.get("task", base["task"]) or base["task"]).strip().lower()
            if task not in {"segment", "detect"}:
                raise ValueError("task 仅支持 segment 或 detect")
            merged: Dict[str, Any] = {
                "task": task,
                "epochs": int(raw.get("epochs", base["epochs"])),
                "imgsz": int(raw.get("imgsz", base["imgsz"])),
                "batch": int(raw.get("batch", base["batch"])),
                "device": str(raw.get("device", base["device"]) or base["device"]),
                "model": str(raw.get("model", base["model"]) or base["model"]),
                "data": str(raw.get("data", base["data"]) or base["data"]),
                "project": str(raw.get("project", base["project"]) or base["project"]),
                "name": str(raw.get("name", base["name"]) or base["name"]),
                "install_to": str(raw.get("install_to", base["install_to"]) or base["install_to"]),
                "exist_ok": bool(raw.get("exist_ok", True)),
            }
            if merged["epochs"] < 1 or merged["epochs"] > 1000:
                raise ValueError("epochs 应在 1–1000")
            if merged["imgsz"] < 32 or merged["imgsz"] > 2048:
                raise ValueError("imgsz 应在 32–2048")
            if merged["batch"] < 1 or merged["batch"] > 256:
                raise ValueError("batch 应在 1–256")

            model_path = self._resolve_model(merged["model"])
            data_src = _resolve_under_root(merged["data"], must_exist=True)
            project_path = _resolve_under_root(merged["project"])
            project_path.mkdir(parents=True, exist_ok=True)
            install_to = _resolve_under_root(merged["install_to"])
            install_to.parent.mkdir(parents=True, exist_ok=True)

            # Ultralytics resolves path: . against CWD when data= is absolute — materialize
            # a run-local yaml with absolute dataset root (+ fall back val→train if empty).
            data_path = self._materialize_data_yaml(
                data_src, project_path / "_ui_dataset.yaml", task=merged["task"]
            )

            # Prefer project venv yolo CLI (same as scripts/train_yolo_cardboard.sh).
            py = Path(sys.executable)
            yolo_bin = py.parent / "yolo"
            train_args = [
                merged["task"],
                "train",
                f"data={data_path}",
                f"model={model_path}",
                f"epochs={merged['epochs']}",
                f"imgsz={merged['imgsz']}",
                f"batch={merged['batch']}",
                f"device={merged['device']}",
                f"project={project_path}",
                f"name={merged['name']}",
                f"exist_ok={'True' if merged['exist_ok'] else 'False'}",
            ]
            if yolo_bin.is_file():
                cmd = [str(yolo_bin), *train_args]
            else:
                cmd = [str(py), "-m", "ultralytics", *train_args]

            self._log.clear()
            self._error = ""
            self._returncode = None
            self._installed_to = None
            self._started_at = _utc_iso()
            self._finished_at = None
            model_for_cmd = str(model_path)
            try:
                model_rel = _rel(Path(model_path)) if Path(model_path).is_file() else model_for_cmd
            except Exception:  # noqa: BLE001
                model_rel = model_for_cmd
            self._params = {
                **merged,
                "model": model_rel,
                "data": _rel(data_src),
                "data_runtime": _rel(data_path),
                "project": _rel(project_path),
                "install_to": _rel(install_to),
                "cmd": [str(x) for x in cmd],
            }
            self._state = "running"
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Avoid Ultralytics auto-open plots in headless servers.
            env.setdefault("YOLO_VERBOSE", "True")
            launch = list(cmd)
            # Line-buffer YOLO CLI output when stdbuf is available (Linux).
            if Path("/usr/bin/stdbuf").is_file():
                launch = ["/usr/bin/stdbuf", "-oL", "-eL", *cmd]
            try:
                self._proc = subprocess.Popen(
                    launch,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._state = "failed"
                self._error = str(exc)
                self._finished_at = _utc_iso()
                raise

            self._log.append(f"[{_utc_iso()}] start: {' '.join(cmd)}")
            self._reader = threading.Thread(
                target=self._pump_stdout,
                name="yolo-train-log",
                daemon=True,
            )
            self._reader.start()
            return self.status()

    def _pump_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                text = line.rstrip("\n")
                with self._lock:
                    self._log.append(text)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._log.append(f"[log-reader] {exc}")
        finally:
            code = proc.wait()
            with self._lock:
                self._returncode = int(code)
                self._finished_at = _utc_iso()
                if self._state == "stopped":
                    pass
                elif code == 0:
                    self._state = "succeeded"
                    try:
                        installed = self._install_best_unlocked()
                        self._log.append(f"[{_utc_iso()}] installed → {installed}")
                    except Exception as exc:  # noqa: BLE001
                        self._error = f"train ok but install failed: {exc}"
                        self._log.append(f"[{_utc_iso()}] {self._error}")
                else:
                    self._state = "failed"
                    self._error = self._error or f"train exited with code {code}"
                    self._log.append(f"[{_utc_iso()}] failed rc={code}")
                self._proc = None

    @staticmethod
    def _resolve_model(raw: str) -> Path:
        """Resolve local weights under repo; allow Ultralytics hub names as last resort."""
        text = str(raw or "").strip()
        if not text:
            raise ValueError("model is empty")
        try:
            return _resolve_under_root(text, must_exist=True)
        except (ValueError, FileNotFoundError):
            pass
        # Absolute path outside root (rare) or bare hub stem like yolo11n-seg.pt
        p = Path(text).expanduser()
        if p.is_file():
            return p.resolve()
        # Hub / cache name — leave as-is for the YOLO CLI (it will download).
        return Path(text)

    @staticmethod
    def _materialize_data_yaml(src: Path, dest: Path, *, task: str = DEFAULT_TASK) -> Path:
        """Write a train-time dataset yaml with absolute ``path`` (and val fallback)."""
        raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"invalid dataset yaml: {src}")
        root_hint = raw.get("path") or "."
        root_path = Path(str(root_hint)).expanduser()
        if not root_path.is_absolute():
            root_path = (src.parent / root_path).resolve()
        train_rel = str(raw.get("train") or "images/train")
        val_rel = str(raw.get("val") or "images/val")
        train_dir = (root_path / train_rel).resolve() if not Path(train_rel).is_absolute() else Path(train_rel)
        val_dir = (root_path / val_rel).resolve() if not Path(val_rel).is_absolute() else Path(val_rel)
        if _count_images(train_dir) < 1:
            raise RuntimeError(f"no train images under {train_dir}")
        if _count_images(val_dir) < 1:
            # Demo / early labeling: reuse train split so Ultralytics can start.
            val_rel = train_rel
            logger.warning("val images empty under %s — using train split for val", val_dir)
        names = raw.get("names") or {0: "cardboard"}
        payload = {
            "task": str(raw.get("task") or task or DEFAULT_TASK),
            "path": str(root_path),
            "train": train_rel,
            "val": val_rel,
            "names": names,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return dest

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                if self._state == "running":
                    self._state = "stopped"
                return self.status()
            self._state = "stopped"
            self._log.append(f"[{_utc_iso()}] stop requested")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
        # Wait briefly outside lock
        try:
            proc.wait(timeout=8)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        with self._lock:
            self._finished_at = self._finished_at or _utc_iso()
            return self.status()

    def _install_best_unlocked(self) -> str:
        params = self._params or {}
        run_dir = self.run_dir(params.get("project"), params.get("name"))
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        src = best if best.is_file() else last
        if not src.is_file():
            raise FileNotFoundError(f"no weights under {run_dir / 'weights'}")
        dest = _resolve_under_root(params.get("install_to") or DEFAULT_INSTALL)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        self._installed_to = _rel(dest)
        return self._installed_to

    def install_weights(self) -> Dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("训练进行中，请等结束后再安装权重")
            path = self._install_best_unlocked()
            return {"ok": True, "installed_to": path, **self.status()}

    # ---- status / artifacts ----
    def metrics(self) -> Dict[str, Any]:
        run_dir = self.run_dir()
        parsed = _parse_results_csv(run_dir / "results.csv")
        with self._lock:
            epochs_total = int((self._params or {}).get("epochs") or 0)
        epochs_done = int(parsed.get("epochs_done") or 0)
        pct = 0.0
        if epochs_total > 0:
            pct = max(0.0, min(100.0, 100.0 * epochs_done / epochs_total))
        latest = parsed.get("latest") or {}
        if not isinstance(latest, dict):
            latest = {}

        def _pick(*keys: str) -> Any:
            for k in keys:
                if latest.get(k) is not None:
                    return latest.get(k)
            return None

        # Prefer mask (M) metrics for segment runs; fall back to box (B).
        return {
            "run_dir": _rel(run_dir) if run_dir.exists() else _rel(run_dir),
            "results_csv": _rel(run_dir / "results.csv") if (run_dir / "results.csv").is_file() else None,
            "epochs_done": epochs_done,
            "epochs_total": epochs_total,
            "percent": round(pct, 1),
            "latest": latest,
            "history": parsed.get("rows") or [],
            "map50": _pick("metrics/mAP50(M)", "metrics/mAP50(B)"),
            "map50_95": _pick("metrics/mAP50-95(M)", "metrics/mAP50-95(B)"),
            "map50_box": latest.get("metrics/mAP50(B)"),
            "map50_mask": latest.get("metrics/mAP50(M)"),
            "precision": _pick("metrics/precision(M)", "metrics/precision(B)"),
            "recall": _pick("metrics/recall(M)", "metrics/recall(B)"),
            "box_loss": latest.get("train/box_loss"),
            "seg_loss": latest.get("train/seg_loss"),
        }

    def artifacts(self) -> List[Dict[str, Any]]:
        run_dir = self.run_dir()
        out: List[Dict[str, Any]] = []
        if not run_dir.is_dir():
            return out
        for name in ARTIFACT_NAMES:
            p = run_dir / name
            if p.is_file():
                out.append(
                    {
                        "name": name,
                        "path": _rel(p),
                        "url": f"/api/train/yolo/artifact?path={_rel(p)}",
                        "bytes": p.stat().st_size,
                        "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    }
                )
        weights_dir = run_dir / "weights"
        for wname in ("best.pt", "last.pt"):
            p = weights_dir / wname
            if p.is_file():
                out.append(
                    {
                        "name": f"weights/{wname}",
                        "path": _rel(p),
                        "url": None,
                        "bytes": p.stat().st_size,
                        "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    }
                )
        return out

    def resolve_artifact(self, rel_path: str) -> Path:
        p = _resolve_under_root(rel_path, must_exist=True)
        runs_root = (ROOT / "runs").resolve()
        try:
            p.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError("artifact must be under runs/") from exc
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".csv", ".txt"}:
            raise ValueError(f"unsupported artifact type: {p.suffix}")
        return p

    def _load_yolo_infer_defaults(self) -> Dict[str, Any]:
        """Defaults for manual verify (aligned with config/yolo.yaml when present)."""
        conf = 0.15
        imgsz = 640
        device = str(self.defaults().get("device") or "cpu")
        class_filter: List[str] = []
        if YOLO_CONFIG_PATH.is_file():
            try:
                cfg = yaml.safe_load(YOLO_CONFIG_PATH.read_text(encoding="utf-8")) or {}
                if isinstance(cfg, dict):
                    if cfg.get("conf") is not None:
                        conf = float(cfg["conf"])
                    if cfg.get("imgsz") is not None:
                        imgsz = int(cfg["imgsz"])
                    if cfg.get("device") is not None and str(cfg["device"]).strip():
                        device = str(cfg["device"]).strip()
                    raw_cf = cfg.get("class_filter")
                    if isinstance(raw_cf, list):
                        class_filter = [str(x) for x in raw_cf if str(x).strip()]
                    elif isinstance(raw_cf, str) and raw_cf.strip():
                        class_filter = [x.strip() for x in raw_cf.split(",") if x.strip()]
            except Exception:  # noqa: BLE001
                pass
        return {"conf": conf, "imgsz": imgsz, "device": device, "class_filter": class_filter}

    def verify_weight_choices(self) -> List[Dict[str, Any]]:
        """Available weight sources for the Train-tab manual verify UI."""
        w = self.weights_info()
        run_dir = self.run_dir()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        choices = [
            {
                "value": "installed",
                "label": "M2 已安装 (cardboard_yolo.pt)",
                "path": w.get("installed"),
                "exists": bool(w.get("installed_exists")),
            },
            {
                "value": "config",
                "label": f"配置权重 ({Path(str(w.get('config_weights') or DEFAULT_INSTALL)).name})",
                "path": w.get("config_weights"),
                "exists": bool(w.get("config_weights_exists")),
            },
            {
                "value": "best",
                "label": "本次训练 best.pt",
                "path": _rel(best),
                "exists": best.is_file(),
            },
            {
                "value": "last",
                "label": "本次训练 last.pt",
                "path": _rel(last),
                "exists": last.is_file(),
            },
            {
                "value": "bootstrap",
                "label": "基座 yolo11n-seg.pt",
                "path": w.get("bootstrap"),
                "exists": bool(w.get("bootstrap_exists")),
            },
        ]
        return choices

    def resolve_verify_weights(self, source: str = "installed", weights: Optional[str] = None) -> Path:
        src = str(source or "installed").strip().lower()
        custom = str(weights or "").strip()
        if custom:
            return self._resolve_model(custom)
        w = self.weights_info()
        run_dir = self.run_dir()
        mapping = {
            "installed": w.get("installed") or DEFAULT_INSTALL,
            "m2": w.get("installed") or DEFAULT_INSTALL,
            "config": w.get("config_weights") or DEFAULT_INSTALL,
            "best": _rel(run_dir / "weights" / "best.pt"),
            "last": _rel(run_dir / "weights" / "last.pt"),
            "bootstrap": w.get("bootstrap") or DEFAULT_MODEL,
        }
        if src not in mapping:
            raise ValueError(f"未知权重来源: {source}（可用: {', '.join(mapping)} 或自定义 weights 路径）")
        path = self._resolve_model(str(mapping[src]))
        if not Path(path).is_file():
            raise FileNotFoundError(f"权重不存在: {path}（source={src}）")
        return Path(path)

    def predict_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "upload.jpg",
        weights_source: str = "installed",
        weights: Optional[str] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        class_filter: Optional[Sequence[str]] = None,
        keep_top_k: int = 0,
        apply_class_filter: bool = False,
    ) -> Dict[str, Any]:
        """Run YOLO-seg on an uploaded image; write viz under ``runs/verify/``."""
        import cv2
        import numpy as np

        from rs_detect.detect.detector import detect_yolo
        from rs_detect.detect.pipeline import draw_detection_viz

        if not image_bytes:
            raise ValueError("空图片")
        defaults = self._load_yolo_infer_defaults()
        use_conf = float(defaults["conf"] if conf is None else conf)
        use_imgsz = int(defaults["imgsz"] if imgsz is None else imgsz)
        use_device = str(device if device is not None else defaults["device"] or "cpu").strip() or "cpu"
        if use_conf <= 0 or use_conf > 1:
            raise ValueError("conf 应在 (0, 1]")
        if use_imgsz < 32 or use_imgsz > 2048:
            raise ValueError("imgsz 应在 32–2048")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"无法解码图片: {filename or 'upload'}")

        weight_path = self.resolve_verify_weights(weights_source, weights)
        filter_list: Optional[List[str]] = None
        if apply_class_filter:
            if class_filter is not None:
                filter_list = [str(x).strip() for x in class_filter if str(x).strip()]
            else:
                filter_list = list(defaults.get("class_filter") or [])
            if not filter_list:
                filter_list = None

        # keep_top_k<=0 → keep all hits (manual QA usually wants every box above conf)
        k = int(keep_top_k)
        if k <= 0:
            k = 10_000

        det = detect_yolo(
            bgr,
            weights=weight_path,
            conf=use_conf,
            imgsz=use_imgsz,
            class_filter=filter_list,
            device=use_device,
            keep_top_k=k,
        )
        viz = draw_detection_viz(bgr, det)
        out_dir = _resolve_under_root(VERIFY_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "upload.jpg").suffix.lower()
        if suffix not in IMAGE_EXTS:
            suffix = ".jpg"
        # Always persist JPEG for stable browser preview URLs.
        input_path = out_dir / "last_input.jpg"
        viz_path = out_dir / "last_viz.jpg"
        ok1, buf1 = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        ok2, buf2 = cv2.imencode(".jpg", viz, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok1 or not ok2:
            raise RuntimeError("编码验证结果失败")
        input_path.write_bytes(buf1.tobytes())
        viz_path.write_bytes(buf2.tobytes())
        mtime = datetime.fromtimestamp(viz_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        detections = []
        polys = list(det.polygons or [])
        for i, box in enumerate(det.boxes):
            poly = polys[i] if i < len(polys) else []
            detections.append(
                {
                    "box": [float(v) for v in box],
                    "score": float(det.scores[i]) if i < len(det.scores) else None,
                    "class_id": int(det.class_ids[i]) if i < len(det.class_ids) else None,
                    "class_name": str(det.class_names[i]) if i < len(det.class_names) else None,
                    "polygon_points": len(poly),
                    "has_polygon": len(poly) >= 3,
                }
            )
        return {
            "ok": True,
            "weights": _rel(weight_path),
            "weights_source": str(weights_source or "installed"),
            "conf": use_conf,
            "imgsz": use_imgsz,
            "device": use_device,
            "class_filter": filter_list,
            "apply_class_filter": bool(apply_class_filter),
            "keep_top_k": int(keep_top_k),
            "filename": filename or "upload",
            "image_size": [int(bgr.shape[1]), int(bgr.shape[0])],
            "count": len(detections),
            "detections": detections,
            "message": det.message,
            "has_mask": bool(det.mask is not None and int(np.count_nonzero(det.mask)) > 0),
            "has_polygons": any(d.get("has_polygon") for d in detections),
            "mask_pixels": int(np.count_nonzero(det.mask)) if det.mask is not None else 0,
            "input_path": _rel(input_path),
            "viz_path": _rel(viz_path),
            "input_url": f"/api/train/yolo/artifact?path={_rel(input_path)}&t={mtime}",
            "viz_url": f"/api/train/yolo/artifact?path={_rel(viz_path)}&t={mtime}",
            "mtime": mtime,
        }

    def log_tail(self, n: int = 120) -> List[str]:
        with self._lock:
            lines = list(self._log)
        n = max(1, min(int(n), 2000))
        return lines[-n:]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            state = self._state
            if running:
                state = "running"
            params = dict(self._params) if self._params else self.defaults()
            payload = {
                "ok": True,
                "state": state,
                "running": running,
                "error": self._error,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "returncode": self._returncode,
                "params": params,
                "installed_to": self._installed_to,
                "pid": self._proc.pid if running and self._proc else None,
            }
        metrics = self.metrics()
        payload["metrics"] = metrics
        payload["dataset"] = self.dataset_info()
        payload["weights"] = self.weights_info()
        payload["artifacts"] = self.artifacts()
        payload["log_tail"] = self.log_tail(80)
        payload["defaults"] = self.defaults()
        payload["verify"] = {
            "defaults": self._load_yolo_infer_defaults(),
            "weight_choices": self.verify_weight_choices(),
        }
        return payload


_TRAINER: Optional[YoloTrainManager] = None
_TRAINER_LOCK = threading.Lock()


def get_trainer() -> YoloTrainManager:
    global _TRAINER
    with _TRAINER_LOCK:
        if _TRAINER is None:
            _TRAINER = YoloTrainManager()
        return _TRAINER
