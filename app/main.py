from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .config import get_settings
from .storage import (
    ValidationError,
    cleanup_previews,
    cleanup_uploads,
    ensure_dirs,
    get_session,
    list_voices,
    load_voice_transcription,
    new_session,
    voice_source_wav,
)
from .ws import ws_loop
from .tts_backend import build_backend
from .logging_config import configure_logging
from .log_context import set_correlation_id
from .log_utils import safe_preview_payload


log = logging.getLogger(__name__)


def _safe_file_response(path: Path) -> FileResponse:
    # FileResponse will stream the file; this helper keeps a single place for settings.
    return FileResponse(path=str(path), media_type="audio/wav")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, fmt=settings.log_format)
    ensure_dirs(settings)
    app.state.settings = settings
    app.state.tts_backend = build_backend(settings)

    # Torch diagnostics (helps confirm whether container has CUDA-enabled torch)
    try:
        import torch

        log.info(
            "torch runtime torch_version=%s cuda_available=%s cuda_devices=%s",
            getattr(torch, "__version__", "-"),
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
    except Exception:
        log.info("torch runtime not available")

    log.info(
        "startup settings tts_backend=%s audio_root=%s log_level=%s log_format=%s",
        settings.tts_backend,
        settings.audio_root,
        settings.log_level,
        settings.log_format,
    )

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


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        settings = request.app.state.settings
        cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:16]
        set_correlation_id(cid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            # Avoid logging request bodies here; endpoints can opt-in.
            log.info(
                "http %s %s status=%s duration_ms=%s client=%s",
                request.method,
                request.url.path,
                getattr(locals().get("response"), "status_code", "-"),
                elapsed_ms,
                request.client.host if request.client else "-",
            )
            set_correlation_id(None)


app.add_middleware(RequestLoggingMiddleware)


@app.get("/")
def root():
    return {
        "name": "qwen3-server",
        "endpoints": ["/health", "/upload", "/ws", "/previews/{job_id}.wav"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def voices_list():
    """List saved voices.

    Saved voices are created via the WebSocket `save_voice` flow.
    """

    settings = app.state.settings
    return {"voices": list_voices(settings)}



# Deprecated: /upload endpoint
# @app.post("/upload")
# async def upload(file: UploadFile = File(...)):
#     settings = app.state.settings
#     session = new_session(settings)
#     data = await file.read()
#     # Store as source.wav (as per spec). Caller should upload WAV.
#     session.source_path.write_bytes(data)
#     return {"session_id": session.session_id}

# New /preview endpoint: accepts audio, transcription, and response text, returns generated audio
from fastapi import Form
from fastapi.responses import StreamingResponse
import io

@app.post("/preview")
async def preview(
    audio: UploadFile = File(...),
    transcription: str = Form(...),
    response_text: str = Form(...),
):
    settings = app.state.settings
    backend = app.state.tts_backend
    log.info(
        "preview_request filename=%s content_type=%s transcription_len=%s response_text_len=%s",
        getattr(audio, "filename", "-"),
        getattr(audio, "content_type", "-"),
        len(transcription or ""),
        len(response_text or ""),
    )
    # Create a temp session folder
    session = new_session(settings)
    log.info("preview_session_created session_id=%s", session.session_id)
    audio_data = await audio.read()
    session.source_path.write_bytes(audio_data)
    log.info("preview_audio_saved session_id=%s bytes=%s", session.session_id, len(audio_data))
    # Save transcription alongside audio for traceability (optional)
    transcription_path = session.folder / "transcription.txt"
    transcription_path.write_text(transcription)
    # Synthesize preview using the uploaded audio and transcription
    # (Assume backend uses transcription for improved voice cloning if supported)
    out_wav = session.folder / "preview.wav"
    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(
            backend.synthesize_preview,
            text=response_text,
            source_wav=session.source_path,
            out_wav=out_wav,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "preview_synth_complete session_id=%s out_bytes=%s duration_ms=%s text=%s",
            session.session_id,
            out_wav.stat().st_size if out_wav.exists() else 0,
            elapsed_ms,
            safe_preview_payload(response_text, limit_chars=settings.log_payload_chars),
        )
    except Exception as e:
        log.exception(
            "preview_synth_failed session_id=%s text=%s",
            session.session_id,
            safe_preview_payload(response_text, limit_chars=settings.log_payload_chars),
        )
        return JSONResponse(status_code=500, content={"error": str(e)})
    # Return the generated audio file as a streaming response
    return StreamingResponse(
        io.BytesIO(out_wav.read_bytes()),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename=preview.wav"},
    )


@app.post("/preview-from-voice")
async def preview_from_voice(
    voice_id: str = Form(...),
    response_text: str = Form(...),
):
    """Generate a preview using a previously saved voice.

    This avoids re-uploading the reference audio on every call.
    """

    settings = app.state.settings
    backend = app.state.tts_backend

    try:
        source_wav = voice_source_wav(settings, voice_id)
    except ValidationError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

    saved_transcription = load_voice_transcription(settings, voice_id)

    out_dir = settings.audio_root / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    out_wav = out_dir / f"{job_id}.wav"

    log.info(
        "preview_from_voice_request voice_id=%s response_text_len=%s saved_transcription_len=%s",
        voice_id,
        len(response_text or ""),
        len(saved_transcription or ""),
    )

    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(
            backend.synthesize_preview,
            text=response_text,
            source_wav=source_wav,
            out_wav=out_wav,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "preview_from_voice_complete voice_id=%s job_id=%s out_bytes=%s duration_ms=%s text=%s",
            voice_id,
            job_id,
            out_wav.stat().st_size if out_wav.exists() else 0,
            elapsed_ms,
            safe_preview_payload(response_text, limit_chars=settings.log_payload_chars),
        )
    except Exception as e:
        log.exception(
            "preview_from_voice_failed voice_id=%s text=%s",
            voice_id,
            safe_preview_payload(response_text, limit_chars=settings.log_payload_chars),
        )
        return JSONResponse(status_code=500, content={"error": str(e)})

    return StreamingResponse(
        io.BytesIO(out_wav.read_bytes()),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=preview.wav"},
    )


# @app.get("/previews/{job_id}.wav")
# async def get_preview(job_id: str):
#     settings = app.state.settings
#     path = settings.previews_dir / f"{job_id}.wav"
#     if not path.exists():
#         return JSONResponse(status_code=404, content={"error": "not_found"})
#     return _safe_file_response(path)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    settings = app.state.settings
    backend = app.state.tts_backend
    await ws_loop(settings, backend, websocket)
