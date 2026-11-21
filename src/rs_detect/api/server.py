from __future__ import annotations

import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rs_detect.paths import (
    CAMERA_OVERLAYS_PATH,
    DRAW_SHAPES_PATH,
    RESTART_FLAG_PATH,
    ROOT,
    SCENE_OVERLAY_PATH,
    WEB_ASSETS,
    WEB_DIST,
    WEB_STATIC,
    ensure_dirs,
)
from rs_detect.service import get_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rs_detect.api")

app = FastAPI(title="rs-detect", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartStreamBody(BaseModel):
    serial: str
    role: str = Field(..., description="wrist | scene")
    width: int = 640
    height: int = 480
    fps: int = 30


class BindSelectionBody(BaseModel):
    role: str = Field(..., description="wrist | scene")
    serial: str
    persist: bool = True


class ExtrinsicsBody(BaseModel):
    d405_T: Optional[List[List[float]]] = None
    d435_T: Optional[List[List[float]]] = None
    d405_serial: Optional[str] = None
    d435_serial: Optional[str] = None
    notes: Optional[str] = None


@app.get("/api/health")
def health():
    return {"ok": True, "service": "rs-detect"}


def _graceful_cleanup() -> None:
    """Stop cameras / recordings / arm before process exit."""
    try:
        svc = get_service()
    except Exception:  # noqa: BLE001
        return
    try:
        svc.stop_all()
    except Exception:  # noqa: BLE001
        logger.exception("stop_all failed during shutdown")
    try:
        svc.arm.stop()
    except Exception:  # noqa: BLE001
        pass
    try:
        svc.arm.set_armed(False)
    except Exception:  # noqa: BLE001
        pass
    try:
        svc.arm.disconnect()
    except Exception:  # noqa: BLE001
        pass


def _schedule_process_exit(*, restart: bool, delay_s: float = 0.45) -> None:
    def _worker() -> None:
        time.sleep(delay_s)
        _graceful_cleanup()
        if restart:
            try:
                RESTART_FLAG_PATH.write_text("1\n", encoding="utf-8")
            except Exception:  # noqa: BLE001
                logger.exception("failed to write restart flag")
        # Prefer SIGTERM so uvicorn can wind down; fall back to hard exit
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            os._exit(0)

    threading.Thread(target=_worker, daemon=True, name="rs-detect-exit").start()


@app.post("/api/system/restart")
def api_system_restart():
    """Stop process after cleanup; start_web.sh relaunches when restart flag is set."""
    _schedule_process_exit(restart=True)
    return {"ok": True, "action": "restart", "message": "backend restarting"}


@app.post("/api/system/shutdown")
def api_system_shutdown():
    """Safe stop: cleanup hardware then exit (no relaunch)."""
    if RESTART_FLAG_PATH.exists():
        try:
            RESTART_FLAG_PATH.unlink()
        except Exception:  # noqa: BLE001
            pass
    _schedule_process_exit(restart=False)
    return {"ok": True, "action": "shutdown", "message": "backend shutting down"}


@app.get("/api/scan")
def api_scan():
    return get_service().scan()


@app.get("/api/selection")
def api_get_selection():
    return get_service().get_selection()


@app.post("/api/selection")
def api_bind_selection(body: BindSelectionBody):
    try:
        return get_service().bind_selection(body.role, body.serial, persist=body.persist)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/streams")
def api_streams():
    return get_service().stream_status()


@app.post("/api/stream/start")
def api_stream_start(body: StartStreamBody):
    try:
        return get_service().start_stream(body.serial, body.role, body.width, body.height, body.fps)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/stream/stop")
def api_stream_stop(serial: str = Query(...)):
    return get_service().stop_stream(serial)


@app.get("/api/frame.jpg")
def api_frame(serial: str = Query(...), which: str = Query("color")):
    data = get_service().get_jpeg(serial, which=which)
    if data is None:
        raise HTTPException(404, "no frame")
    return Response(content=data, media_type="image/jpeg")


@app.post("/api/snapshot")
def api_snapshot(serial: str = Query(...)):
    try:
        return get_service().snapshot(serial)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/record/start")
def api_record_start(serial: str = Query(...)):
    try:
        return get_service().start_recording(serial)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


class StorageSettingsBody(BaseModel):
    captures_dir: Optional[str] = Field(None, description="photo / snapshot save directory")
    recordings_dir: Optional[str] = Field(None, description="video / mp4 save directory")


@app.get("/api/settings/storage")
def api_settings_storage_get():
    """M1 media save paths (photos + videos)."""
    return get_service().get_storage_settings()


@app.put("/api/settings/storage")
def api_settings_storage_put(body: StorageSettingsBody):
    try:
        return get_service().set_storage_settings(
            captures_dir=body.captures_dir,
            recordings_dir=body.recordings_dir,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/record/stop")
def api_record_stop(serial: str = Query(...)):
    return get_service().stop_recording(serial)


@app.get("/api/intrinsics")
def api_intrinsics(serial: str = Query(...), from_live: bool = True):
    try:
        return get_service().get_intrinsics(serial, from_live=from_live)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/extrinsics")
def api_get_extrinsics():
    return get_service().get_extrinsics()


@app.put("/api/extrinsics")
def api_put_extrinsics(body: ExtrinsicsBody):
    try:
        return get_service().set_extrinsics(body.dict(exclude_none=True))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


class DetectBody(BaseModel):
    serial: Optional[str] = None
    role: Optional[str] = Field(None, description="wrist|scene|d405|d435")
    method: Optional[str] = Field(None, description="auto|yolo|hsv|mock")


class PipelineBody(BaseModel):
    method: Optional[str] = Field(None, description="auto|yolo|hsv|mock")
    z_lift_m: Optional[float] = Field(None, description="safety Z lift above p_coarse (m)")
    take_snapshot: bool = True


class SafeZBody(BaseModel):
    z_lift_m: float = Field(..., description="recompute p_safe = p_coarse + Z lift")


class FineServoBody(BaseModel):
    method: Optional[str] = Field(None, description="auto|yolo|hsv|mock")
    pregrasp_z_lift_m: Optional[float] = Field(None, description="p_pregrasp = c* + Z lift")
    max_iters: Optional[int] = None


class MultiviewBody(BaseModel):
    method: Optional[str] = Field(None, description="auto|yolo|hsv|mock")
    pregrasp_z_lift_m: Optional[float] = Field(None, description="p_pregrasp = c* + Z lift")
    n_views: Optional[int] = Field(None, description="M3 viewpoints including center")
    radius_m: Optional[float] = Field(None, description="XY ring radius (m)")
    move_pregrasp: Optional[bool] = True


class YoloTrainStartBody(BaseModel):
    task: Optional[str] = Field(None, description="segment|detect")
    epochs: Optional[int] = Field(None, ge=1, le=1000)
    imgsz: Optional[int] = Field(None, ge=32, le=2048)
    batch: Optional[int] = Field(None, ge=1, le=256)
    device: Optional[str] = None
    model: Optional[str] = None
    data: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    install_to: Optional[str] = None
    exist_ok: Optional[bool] = True


@app.get("/api/train/yolo/status")
def api_train_yolo_status():
    """YOLO train job status: params / progress / metrics / artifacts / log."""
    return get_service().yolo_train.status()


@app.post("/api/train/yolo/start")
def api_train_yolo_start(body: YoloTrainStartBody = YoloTrainStartBody()):
    try:
        params = {k: v for k, v in body.dict().items() if v is not None}
        return get_service().yolo_train.start(params)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/train/yolo/stop")
def api_train_yolo_stop():
    try:
        return get_service().yolo_train.stop()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/train/yolo/install")
def api_train_yolo_install():
    """Copy runs/.../weights/best.pt → models/cardboard_yolo.pt for M2."""
    try:
        return get_service().yolo_train.install_weights()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/train/yolo/log")
def api_train_yolo_log(n: int = Query(120, ge=1, le=2000)):
    return {"ok": True, "lines": get_service().yolo_train.log_tail(n)}


@app.get("/api/train/yolo/artifact")
def api_train_yolo_artifact(path: str = Query(..., description="repo-relative under runs/")):
    try:
        file_path = get_service().yolo_train.resolve_artifact(path)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc
    suffix = file_path.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media)


@app.post("/api/train/yolo/predict")
async def api_train_yolo_predict(
    file: UploadFile = File(..., description="image to verify against a YOLO weight"),
    weights_source: str = Form("installed"),
    weights: Optional[str] = Form(None),
    conf: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    device: Optional[str] = Form(None),
    class_filter: Optional[str] = Form(None, description="comma-separated; used when apply_class_filter=1"),
    apply_class_filter: bool = Form(False),
    keep_top_k: int = Form(0, description="0 = keep all detections above conf"),
):
    """Manual verify: upload an image, run YOLO-seg, return boxes + viz URLs."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空文件")
    name = file.filename or "upload.jpg"
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        # Still try decode; reject only clearly non-image content-types when set.
        ct = (file.content_type or "").lower()
        if ct and not ct.startswith("image/"):
            raise HTTPException(400, f"仅支持图片上传，收到 content-type={ct}")
    cf_list: Optional[List[str]] = None
    if class_filter is not None and str(class_filter).strip():
        cf_list = [x.strip() for x in str(class_filter).split(",") if x.strip()]
    try:
        return get_service().yolo_train.predict_image(
            raw,
            filename=name,
            weights_source=weights_source,
            weights=weights,
            conf=conf,
            imgsz=imgsz,
            device=device,
            class_filter=cf_list,
            keep_top_k=keep_top_k,
            apply_class_filter=bool(apply_class_filter),
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


# ---- grasp-service client Agent (S0 → S2 → Final) ----

class GraspConfigBody(BaseModel):
    base_url: Optional[str] = None
    timeout_s: Optional[float] = None
    sam_text: Optional[str] = None
    hole_sam_text: Optional[str] = None
    offset_along_normal_mm: Optional[float] = None
    normal_direction: Optional[int] = None
    s0_along_normal_m: Optional[float] = None
    depth_scale: Optional[float] = None
    include_s1: Optional[bool] = None
    prefer_hole_tcp: Optional[bool] = None
    execute_motion: Optional[bool] = None
    force_without_s2_exit: Optional[bool] = None
    use_mock_frames: Optional[bool] = None
    mock_scene_id: Optional[str] = None
    mock_wrist_id: Optional[str] = None
    s0_flip_approach: Optional[bool] = None
    s0_j6_rad: Optional[float] = None
    s2_hz: Optional[float] = None
    s2_max_steps: Optional[int] = None
    insert_total_m: Optional[float] = None
    step_m: Optional[float] = None
    linear_speed_m_s: Optional[float] = None
    direction_sign: Optional[int] = None
    pause_ms_between_steps: Optional[int] = None
    retreat_m: Optional[float] = None
    gripper_width_m_open: Optional[float] = None
    gripper_width_m_closed: Optional[float] = None


class GraspRunBody(BaseModel):
    include_s1: Optional[bool] = None


class GraspMockBody(BaseModel):
    enabled: Optional[bool] = None
    scene_id: Optional[str] = None
    wrist_id: Optional[str] = None


class GraspSiteConfigureBody(BaseModel):
    """Optional client-edited site payload; omit to auto-build from station configs."""
    payload: Optional[Dict[str, Any]] = None


def _grasp_http(exc: Exception) -> HTTPException:
    from rs_detect.grasp.client import GraspServiceError

    if isinstance(exc, GraspServiceError):
        # Preserve 400/404/422/500; map transport failure to 503
        if exc.status_code == 0:
            code = 503
        elif exc.status_code in (400, 404, 422, 500):
            code = exc.status_code
        else:
            code = 400
        return HTTPException(code, exc.detail)
    return HTTPException(400, str(exc))


@app.get("/api/grasp/status")
def api_grasp_status():
    return get_service().grasp.status()


@app.get("/api/grasp/health")
def api_grasp_health():
    """Optional connectivity probe (not pipeline step 1)."""
    return get_service().grasp.health()


@app.get("/api/grasp/site/config")
def api_grasp_site_config():
    """Proxy GET /v1/site/config — schema + effective site summary."""
    try:
        return get_service().grasp.site_config()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.get("/api/grasp/site/payload")
def api_grasp_site_payload():
    """Preview local site-configure payload without POSTing."""
    try:
        payload = get_service().grasp.build_site_payload()
        return {"ok": True, "payload": payload, **get_service().grasp.status()}
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/site/configure")
def api_grasp_site_configure(body: Optional[GraspSiteConfigureBody] = None):
    """Step 1: push site draft (or auto-built station data) → POST /v1/site/configure."""
    try:
        payload = body.payload if body is not None else None
        out = get_service().grasp.site_sync(payload=payload)
        if not out.get("ok"):
            err = str(out.get("error") or "site configure failed")
            code = 503 if "connect failed" in err.lower() or "HTTP 0:" in err else 400
            raise HTTPException(code, err)
        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.put("/api/grasp/config")
def api_grasp_config_put(body: GraspConfigBody):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    cfg = get_service().grasp.save_cfg(updates)
    return {"ok": True, "config": cfg, **get_service().grasp.status()}


@app.post("/api/grasp/reset")
def api_grasp_reset():
    return get_service().grasp.reset()


@app.get("/api/grasp/mock")
def api_grasp_mock_get():
    return get_service().grasp.mock_info()


@app.post("/api/grasp/mock")
def api_grasp_mock_set(body: GraspMockBody):
    """Toggle mock RGB-D and/or select data/capture source (scene/wrist)."""
    try:
        return get_service().grasp.set_mock_frames(
            body.enabled,
            scene_id=body.scene_id,
            wrist_id=body.wrist_id,
        )
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.get("/api/grasp/mock/preview.jpg")
def api_grasp_mock_preview(
    role: str = Query("scene", description="scene|wrist"),
    which: str = Query("color", description="color|depth"),
    id: Optional[str] = Query(None, description="capture stem or __stock__"),
):
    """Serve mock color or depth preview (stock or selected data/capture)."""
    import cv2

    from rs_detect.grasp.frame_pack import depth_preview_bgr, load_mock_packet

    try:
        grasp = get_service().grasp
        cfg = getattr(grasp, "_cfg", {}) or {}
        role_l = str(role or "scene").lower()
        if id:
            src_id = id
        elif role_l in {"wrist", "d405"}:
            src_id = cfg.get("mock_wrist_id")
        else:
            src_id = cfg.get("mock_scene_id")
        pkt, _info = load_mock_packet(role, capture_id=src_id)
        which_l = str(which or "color").lower()
        if which_l in {"depth", "d"}:
            img = depth_preview_bgr(pkt.depth_u16)
            if img is None:
                raise RuntimeError("mock depth missing")
        else:
            img = pkt.color_bgr
            if img is None:
                raise RuntimeError("mock color missing")
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/grasp/viz/{kind}.jpg")
def api_grasp_viz(kind: str):
    """Materials / Hole / S0 overlay jpeg (client-side projection on last scene frame)."""
    kind_l = str(kind or "").lower().replace(".jpg", "")
    if kind_l not in {"hole", "s0", "materials"}:
        raise HTTPException(404, "kind must be materials|hole|s0")
    data = get_service().grasp.get_viz_jpeg(kind_l)
    if not data:
        raise HTTPException(404, f"no {kind_l} viz yet — run detect/plan first")
    return Response(content=data, media_type="image/jpeg")


class GraspMaterialSelectBody(BaseModel):
    material_id: int = Field(..., description="id from materials/detect candidate list")


@app.post("/api/grasp/materials/detect")
def api_grasp_materials_detect():
    """Detect all material instances; pick one before hole/S0."""
    try:
        return get_service().grasp.materials_detect()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/materials/select")
def api_grasp_materials_select(body: GraspMaterialSelectBody):
    """Select a material candidate → crop hole/detect → set active hole."""
    try:
        return get_service().grasp.select_material(body.material_id)
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/hole/detect")
def api_grasp_hole_detect():
    try:
        return get_service().grasp.hole_detect()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/hole/canvas_pose")
def api_grasp_hole_canvas_pose():
    """Hole grasp position + plane normal (base) for Three.js marker/arrow."""
    try:
        return get_service().grasp.hole_canvas_pose()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s0/plan")
def api_grasp_s0_plan():
    try:
        return get_service().grasp.s0_plan()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s0/canvas_pose")
def api_grasp_s0_canvas_pose():
    """S0 pregrasp_flange → xyzrpy (+ IK) for Three.js marker/ghost sync."""
    try:
        return get_service().grasp.s0_canvas_pose()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/move_pregrasp")
def api_grasp_move_pregrasp():
    try:
        return get_service().grasp.move_pregrasp()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s1/latch")
def api_grasp_s1_latch():
    try:
        return get_service().grasp.s1_latch()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s1/step")
def api_grasp_s1_step():
    try:
        return get_service().grasp.s1_step()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s2/latch")
def api_grasp_s2_latch():
    try:
        return get_service().grasp.s2_latch()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s2/step")
def api_grasp_s2_step():
    try:
        return get_service().grasp.s2_step_once()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s2/start")
def api_grasp_s2_start():
    try:
        return get_service().grasp.start_s2_loop()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/s2/stop")
def api_grasp_s2_stop():
    return get_service().grasp.stop_s2()


@app.post("/api/grasp/final/approach")
def api_grasp_final_approach():
    try:
        return get_service().grasp.final_approach()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/final/execute")
def api_grasp_final_execute():
    try:
        return get_service().grasp.execute_final()
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.post("/api/grasp/run/hole")
def api_grasp_run_hole(body: GraspRunBody = GraspRunBody()):
    """One-shot hole pipeline (blocks until S2 loop ends or max steps)."""
    try:
        return get_service().grasp.run_hole_pipeline(include_s1=body.include_s1)
    except Exception as exc:
        raise _grasp_http(exc) from exc


@app.get("/api/detect/config")
def api_detect_config():
    return get_service().get_detect_config()


@app.post("/api/detect/run")
def api_detect_run(body: DetectBody = DetectBody()):
    """Debug single-camera detect. Prefer /api/pipeline/m2 for plan flow."""
    try:
        return get_service().run_detect(serial=body.serial, role=body.role, method=body.method)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/pipeline/workflow")
def api_pipeline_workflow():
    """M2 Steps shared state (coarse / p_safe / …)."""
    return get_service().get_workflow()


@app.post("/api/pipeline/safe_z")
def api_pipeline_safe_z(body: SafeZBody):
    try:
        return get_service().set_workflow_safe_z(body.z_lift_m)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/coarse")
def api_pipeline_coarse(body: PipelineBody = PipelineBody()):
    """Step1: D435 snapshot + coarse → p_coarse / p_safe."""
    try:
        return get_service().run_coarse(
            method=body.method,
            z_lift_m=body.z_lift_m,
            take_snapshot=body.take_snapshot,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/move_safe")
def api_pipeline_move_safe():
    """Step2: move arm TCP to p_safe."""
    try:
        return get_service().move_to_safe()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/fine")
def api_pipeline_fine(body: PipelineBody = PipelineBody()):
    """Single-frame D405 fine (debug). Prefer /api/pipeline/fine_servo for closed loop."""
    try:
        return get_service().run_fine(method=body.method)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/fine_servo")
def api_pipeline_fine_servo(body: FineServoBody = FineServoBody()):
    """Step3 alt: D405 single-frame servo loop (legacy M2-style)."""
    try:
        return get_service().run_fine_servo(
            method=body.method,
            pregrasp_z_lift_m=body.pregrasp_z_lift_m,
            max_iters=body.max_iters,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/multiview")
def api_pipeline_multiview(body: MultiviewBody = MultiviewBody()):
    """Step3 / M3: multi-view capture + voxel/plane fusion → c* / p_pregrasp."""
    try:
        return get_service().run_multiview_fine(
            method=body.method,
            pregrasp_z_lift_m=body.pregrasp_z_lift_m,
            n_views=body.n_views,
            radius_m=body.radius_m,
            move_pregrasp=body.move_pregrasp,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/grasp")
def api_pipeline_grasp():
    """Step4: grasp / suction on (placeholder IO)."""
    try:
        return get_service().run_grasp_placeholder()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


class SafePointsBody(BaseModel):
    points: List[Dict[str, Any]] = Field(default_factory=list)


class CaptureSafePointBody(BaseModel):
    name: Optional[str] = None


@app.get("/api/pipeline/safe_points")
def api_pipeline_safe_points_get():
    return get_service().get_safe_points()


@app.put("/api/pipeline/safe_points")
def api_pipeline_safe_points_put(body: SafePointsBody):
    try:
        return get_service().set_safe_points(body.points)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/safe_points/capture")
def api_pipeline_safe_points_capture(body: CaptureSafePointBody = CaptureSafePointBody()):
    """Append current TCP XYZ into the safe-points list."""
    try:
        return get_service().capture_safe_point(name=body.name)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/safe_points/run")
def api_pipeline_safe_points_run(body: SafePointsBody = SafePointsBody()):
    """Step5: move arm through safe points in order (optional body overrides + saves)."""
    try:
        pts = body.points if body.points else None
        return get_service().run_safe_points_sequence(pts)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/pipeline/suction")
def api_pipeline_suction_state():
    return get_service().get_suction_state()


@app.post("/api/pipeline/suction/release")
def api_pipeline_suction_release():
    """Final: release / open suction cup (placeholder)."""
    try:
        return get_service().release_suction()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pipeline/m2")
def api_pipeline_m2(body: PipelineBody = PipelineBody()):
    """Canonical: D435 coarse → D405 fine → gate (no depth blending)."""
    try:
        return get_service().run_m2_pipeline(method=body.method)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/pipeline/last")
def api_pipeline_last():
    return get_service().get_last_pipeline()


@app.get("/api/detect/viz.jpg")
def api_detect_viz(serial: str = Query(...)):
    data = get_service().get_detect_viz_jpeg(serial)
    if data is None:
        raise HTTPException(404, "no detect viz; run pipeline/detect first")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/detect/last")
def api_detect_last(serial: Optional[str] = None):
    return get_service().get_last_detect(serial)


class ArmJointsBody(BaseModel):
    joints_rad: List[float]


class ArmPoseBody(BaseModel):
    xyzrpy: List[float]


def _default_scene_overlay() -> Dict[str, Any]:
    return {
        "enabled": False,
        "url": "/assets/scene/g26562-01-00_asm.stl",
        "units_of_file": "mm",
        "scale": 0.001,
        "opacity": 0.32,
        "color": "#7a93ad",
        "align": {
            "base_in_scene_mm": [0.0, 0.0, 0.0],
            "yaw_deg": 0.0,
        },
        "offset_m": [0.0, 0.0, 0.0],
        "offset_rpy_deg": [0.0, 0.0, 0.0],
    }


def _load_scene_overlay() -> Dict[str, Any]:
    import yaml

    data = _default_scene_overlay()
    if SCENE_OVERLAY_PATH.exists():
        raw = yaml.safe_load(SCENE_OVERLAY_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            data.update(raw)
    # Also expose static placement.json extras if present
    place = WEB_ASSETS / "scene" / "placement.json"
    if place.exists():
        try:
            import json

            pj = json.loads(place.read_text(encoding="utf-8"))
            data.setdefault("name", pj.get("name"))
            data.setdefault("notes", pj.get("notes"))
        except Exception:  # noqa: BLE001
            pass
    data["config_path"] = str(SCENE_OVERLAY_PATH.relative_to(ROOT))
    return data


class SceneOverlayBody(BaseModel):
    enabled: Optional[bool] = None
    url: Optional[str] = None
    opacity: Optional[float] = None
    color: Optional[str] = None
    scale: Optional[float] = None
    offset_m: Optional[List[float]] = None
    offset_rpy_deg: Optional[List[float]] = None
    align: Optional[Dict[str, Any]] = None


@app.get("/api/scene_overlay")
def api_scene_overlay_get():
    """Cell CAD scene mesh overlay (separate from URDF)."""
    return _load_scene_overlay()


@app.put("/api/scene_overlay")
def api_scene_overlay_put(body: SceneOverlayBody):
    """Persist overlay fine-tune (does not modify URDF)."""
    import yaml

    data = _load_scene_overlay()
    patch = body.model_dump(exclude_none=True)
    if "align" in patch and isinstance(patch["align"], dict):
        align = dict(data.get("align") or {})
        align.update(patch.pop("align"))
        data["align"] = align
    data.update(patch)
    # strip response-only / runtime keys so YAML stays a clean persisted config
    for _k in ("config_path", "asset_missing", "asset_path", "asset_hint"):
        data.pop(_k, None)
    SCENE_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENE_OVERLAY_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return _load_scene_overlay()


def _default_camera_overlays() -> Dict[str, Any]:
    return {
        "enabled": False,
        "notes": "Official D435/D405 meshes; poses in arm base (ROS Z-up, meters).",
        "cameras": [],
    }


def _load_camera_overlays() -> Dict[str, Any]:
    import yaml

    data = _default_camera_overlays()
    if CAMERA_OVERLAYS_PATH.exists():
        raw = yaml.safe_load(CAMERA_OVERLAYS_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            data.update(raw)
    cams = data.get("cameras")
    if not isinstance(cams, list):
        data["cameras"] = []
    data["config_path"] = str(CAMERA_OVERLAYS_PATH.relative_to(ROOT))
    return data


class CameraOverlaysBody(BaseModel):
    enabled: Optional[bool] = None
    notes: Optional[str] = None
    cameras: Optional[List[Dict[str, Any]]] = None


@app.get("/api/camera_overlays")
def api_camera_overlays_get():
    """RealSense CAD overlays (separate from URDF / scene CAD)."""
    return _load_camera_overlays()


@app.put("/api/camera_overlays")
def api_camera_overlays_put(body: CameraOverlaysBody):
    """Persist per-camera XYZ+RPY fine-tune (does not modify URDF)."""
    import yaml

    data = _load_camera_overlays()
    patch = body.model_dump(exclude_none=True)
    data.update(patch)
    data.pop("config_path", None)
    CAMERA_OVERLAYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMERA_OVERLAYS_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return _load_camera_overlays()


def _default_draw_shapes() -> Dict[str, Any]:
    return {
        "enabled": True,
        "notes": "Drawn box/sphere/cylinder primitives in arm base (ROS Z-up, meters).",
        "active_layer_id": "layer_1",
        "layers": [
            {
                "id": "layer_1",
                "label": "绘制层 1",
                "enabled": True,
                "shapes": [],
            }
        ],
    }


def _load_draw_shapes() -> Dict[str, Any]:
    import yaml

    data = _default_draw_shapes()
    if DRAW_SHAPES_PATH.exists():
        raw = yaml.safe_load(DRAW_SHAPES_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            data.update(raw)
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        data["layers"] = _default_draw_shapes()["layers"]
    if not data.get("active_layer_id"):
        data["active_layer_id"] = data["layers"][0]["id"]
    data["config_path"] = str(DRAW_SHAPES_PATH.relative_to(ROOT))
    return data


class DrawShapesBody(BaseModel):
    enabled: Optional[bool] = None
    notes: Optional[str] = None
    active_layer_id: Optional[str] = None
    layers: Optional[List[Dict[str, Any]]] = None


@app.get("/api/draw_shapes")
def api_draw_shapes_get():
    """Editable drawn primitive layers (box / sphere / cylinder)."""
    return _load_draw_shapes()


@app.put("/api/draw_shapes")
def api_draw_shapes_put(body: DrawShapesBody):
    """Persist draw layers and per-shape XYZ+RPY (does not modify URDF)."""
    import yaml

    data = _load_draw_shapes()
    patch = body.model_dump(exclude_none=True)
    data.update(patch)
    data.pop("config_path", None)
    DRAW_SHAPES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAW_SHAPES_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return _load_draw_shapes()


@app.get("/api/arm/state")
def api_arm_state():
    return get_service().arm.status()


@app.get("/api/arm/model")
def api_arm_model():
    return get_service().arm.model_descriptor()


class ArmDryRunBody(BaseModel):
    dry_run: bool = True
    persist: bool = True


@app.post("/api/arm/connect")
def api_arm_connect():
    try:
        return get_service().arm.connect()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/disconnect")
def api_arm_disconnect():
    return get_service().arm.disconnect()


@app.get("/api/arm/dry_run")
def api_arm_dry_run_get():
    arm = get_service().arm
    return {
        "dry_run": arm.dry_run,
        "connected": arm.status().get("connected"),
        "has_cs_write": bool(arm.model_descriptor().get("has_cs_write")),
        "robot_ip": arm.arm_cfg.get("robot_ip"),
        "live_backend": arm.status().get("live_backend"),
    }


@app.put("/api/arm/dry_run")
def api_arm_dry_run_put(body: ArmDryRunBody):
    """Switch simulator ↔ LIVE. Disconnect first. Persists to robot_cell_c.yaml by default."""
    try:
        return get_service().arm.set_dry_run(body.dry_run, persist=body.persist)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/reload_config")
def api_arm_reload_config():
    try:
        return get_service().arm.reload_config()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/arm")
def api_arm_arm():
    try:
        return get_service().arm.set_armed(True)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/disarm")
def api_arm_disarm():
    try:
        return get_service().arm.set_armed(False)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/stop")
def api_arm_stop():
    return get_service().arm.stop()


@app.post("/api/arm/home")
def api_arm_home():
    try:
        return get_service().arm.go_home()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/home/set")
def api_arm_home_set():
    """Set current joints as HOME and persist to robot_cell_c.yaml."""
    try:
        return get_service().arm.set_home_from_current()
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/command")
def api_arm_command(body: ArmJointsBody):
    try:
        return get_service().arm.move_joints(body.joints_rad)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/move_pose")
def api_arm_move_pose(body: ArmPoseBody):
    try:
        return get_service().arm.move_pose(body.xyzrpy)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/arm/ik")
def api_arm_ik(body: ArmPoseBody):
    return get_service().arm.ik(body.xyzrpy)


def _favicon_path() -> Path:
    for base in (WEB_STATIC, WEB_DIST):
        p = base / "favicon.svg"
        if p.exists():
            return p
    return WEB_STATIC / "favicon.svg"


@app.get("/favicon.ico")
@app.get("/favicon.svg")
def api_favicon():
    path = _favicon_path()
    if not path.exists():
        raise HTTPException(404, "favicon missing")
    return FileResponse(path, media_type="image/svg+xml")


def _mount_frontend() -> None:
    ensure_dirs()
    if WEB_ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=str(WEB_ASSETS), follow_symlink=True), name="assets")
        logger.info("Serving assets from %s", WEB_ASSETS)
    # Prefer Vite build output (web/dist). Fallback: legacy static shell if present.
    if WEB_DIST.exists() and (WEB_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
        logger.info("Serving web dist from %s", WEB_DIST)
    elif WEB_STATIC.exists() and (WEB_STATIC / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(WEB_STATIC), html=True), name="web")
        logger.info("Serving web public from %s", WEB_STATIC)
    else:
        logger.warning("No web frontend found (run: cd web && pnpm run build)")


_mount_frontend()


def main() -> None:
    uvicorn.run("rs_detect.api.server:app", host="0.0.0.0", port=8787, reload=False)


if __name__ == "__main__":
    main()
