from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import get_settings
from .storage import (
    ValidationError,
    cleanup_previews,
    cleanup_uploads,
    ensure_dirs,
    get_session,
    new_session,
)
from .ws import ws_loop
from .tts_backend import build_backend
from .logging_config import configure_logging


log = logging.getLogger(__name__)


def _safe_file_response(path: Path) -> FileResponse:
    # FileResponse will stream the file; this helper keeps a single place for settings.
    return FileResponse(path=str(path), media_type="audio/wav")


def _cors_origins_from_env() -> list[str] | None:
    raw = (os.getenv("CORS_ALLOW_ORIGINS", "*") or "").strip()
    if not raw:
        return None
    if raw.lower() in {"none", "off", "false", "0"}:
        return None
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    ensure_dirs(settings)
    app.state.settings = settings
    app.state.tts_backend = build_backend(settings)

    if settings.preload_model_on_startup:
        # This blocks startup intentionally so the first request doesn't hit a cold/missing model.
        await asyncio.to_thread(app.state.tts_backend.ensure_ready)

    stop_event = asyncio.Event()

    async def previews_task() -> None:
        while not stop_event.is_set():
            removed = await cleanup_previews(settings)
            if removed:
                log.info("Cleaned previews: %s", removed)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=settings.previews_cleanup_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def uploads_task() -> None:
        while not stop_event.is_set():
            removed = await cleanup_uploads(settings)
            if removed:
                log.info("Cleaned uploads: %s", removed)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=settings.uploads_cleanup_interval_seconds)
            except asyncio.TimeoutError:
                pass

    tasks = [asyncio.create_task(previews_task()), asyncio.create_task(uploads_task())]
    try:
        yield
    finally:
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)

_cors_origins = _cors_origins_from_env()
if _cors_origins is not None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def root():
    return {
        "name": "qwen3-server",
        "endpoints": ["/health", "/upload", "/ws", "/previews/{job_id}.wav"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/health")
def health_api():
    return health()


@app.get("/healthz")
def healthz():
    return health()


@app.get("/ingress/health")
@app.get("/ingress/api/health")
@app.get("/ingress/healthz")
def health_ingress():
    return health()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    settings = app.state.settings
    session = new_session(settings)
    data = await file.read()
    # Store as source.wav (as per spec). Caller should upload WAV.
    session.source_path.write_bytes(data)
    return {"session_id": session.session_id}


@app.post("/api/upload")
async def upload_api(file: UploadFile = File(...)):
    return await upload(file=file)


@app.post("/ingress/upload")
@app.post("/ingress/api/upload")
async def upload_ingress(file: UploadFile = File(...)):
    return await upload(file=file)


@app.get("/previews/{job_id}.wav")
async def get_preview(job_id: str):
    settings = app.state.settings
    path = settings.previews_dir / f"{job_id}.wav"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return _safe_file_response(path)


@app.get("/api/previews/{job_id}.wav")
async def get_preview_api(job_id: str):
    return await get_preview(job_id=job_id)


@app.get("/ingress/previews/{job_id}.wav")
@app.get("/ingress/api/previews/{job_id}.wav")
async def get_preview_ingress(job_id: str):
    return await get_preview(job_id=job_id)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    settings = app.state.settings
    backend = app.state.tts_backend
    await ws_loop(settings, backend, websocket)


@app.websocket("/api/ws")
async def ws_endpoint_api(websocket: WebSocket):
    await ws_endpoint(websocket)


@app.websocket("/ingress/ws")
@app.websocket("/ingress/api/ws")
async def ws_endpoint_ingress(websocket: WebSocket):
    await ws_endpoint(websocket)
