from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rs_detect.paths import ROOT, WEB_DIST, WEB_STATIC, ensure_dirs
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


def _mount_frontend() -> None:
    ensure_dirs()
    # Prefer built Vite dist; fallback to public static (CDN Ant Design page)
    if WEB_DIST.exists() and (WEB_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
        logger.info("Serving web dist from %s", WEB_DIST)
    elif WEB_STATIC.exists() and (WEB_STATIC / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(WEB_STATIC), html=True), name="web")
        logger.info("Serving web public from %s", WEB_STATIC)


_mount_frontend()


def main() -> None:
    uvicorn.run("rs_detect.api.server:app", host="0.0.0.0", port=8787, reload=False)


if __name__ == "__main__":
    main()
