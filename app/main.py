from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import time
import uuid
from typing import Any

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
    list_voices,
    write_latest_preview,
    load_voice_transcription,
    voice_source_wav,
)
from .ws import ws_loop
from .tts_backend import build_backend
from .logging_config import configure_logging
from .log_context import set_correlation_id
from .log_utils import safe_preview_payload
from .torch_utils import cuda_assert_payload, is_cuda_device_side_assert


log = logging.getLogger(__name__)


def _torch_startup_diagnostics(settings) -> dict[str, Any]:
    """Best-effort torch/device diagnostics for startup logs.

    Keep this resilient: it must never crash startup if torch isn't importable
    or if CUDA libraries aren't present.
    """

    info: dict[str, Any] = {
        "settings_device": getattr(settings, "device", None),
        "settings_torch_dtype": getattr(settings, "torch_dtype", None),
        "settings_device_map": getattr(settings, "device_map", None),
        "disable_torch_nnpack": getattr(settings, "disable_torch_nnpack", None),
        "torch_disable_nnpack_env": os.getenv("TORCH_DISABLE_NNPACK"),
    }

    try:
        import torch

        info.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_devices": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        )

        # Resolve the effective device/dtype using the same rules as our backend.
        device = (settings.device or "auto").lower() if getattr(settings, "device", None) else "auto"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = (settings.device_map or "auto").lower() if getattr(settings, "device_map", None) else "auto"
        if device_map == "auto":
            device_map = "cuda" if device.startswith("cuda") else "cpu"

        dtype_setting = (settings.torch_dtype or "auto").lower() if getattr(settings, "torch_dtype", None) else "auto"
        if dtype_setting == "auto":
            if device.startswith("cuda"):
                bf16_supported = False
                try:
                    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
                    bf16_supported = bool(is_bf16_supported()) if callable(is_bf16_supported) else False
                except Exception:
                    bf16_supported = False
                dtype = torch.bfloat16 if bf16_supported else torch.float32
            else:
                dtype = torch.float32
        else:
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(dtype_setting, torch.float32)

        info.update(
            {
                "effective_device": device,
                "effective_device_map": device_map,
                "effective_dtype": str(dtype).replace("torch.", ""),
            }
        )
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"

    return info


def _torch_performance_tweaks() -> dict[str, Any]:
    """Best-effort performance toggles.

    These are safe-ish defaults that can improve throughput on CUDA.
    They should never crash startup.
    """

    applied: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            # TF32 can speed up float32 matmuls on Ampere+.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                applied["tf32"] = True
            except Exception:
                applied["tf32"] = "unsupported"

            # Prefer faster matmul kernels when available.
            try:
                torch.set_float32_matmul_precision("high")
                applied["matmul_precision"] = "high"
            except Exception:
                applied["matmul_precision"] = "unsupported"

            # Encourage Flash/efficient SDPA when possible (PyTorch will fall back safely).
            try:
                sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
                if callable(sdp_kernel):
                    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
                    applied["sdp_kernel"] = "flash/mem_efficient"
            except Exception:
                applied["sdp_kernel"] = "unsupported"
    except Exception as e:
        applied["error"] = f"{type(e).__name__}: {e}"

    return applied


def _safe_file_response(path: Path) -> FileResponse:
    # FileResponse will stream the file; this helper keeps a single place for settings.
    return FileResponse(path=str(path), media_type="audio/wav", filename="preview.wav")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Apply low-level torch env toggles as early as possible.
    # Must be set before importing torch for it to take effect.
    if settings.disable_torch_nnpack:
        # PyTorch recognizes TORCH_DISABLE_NNPACK=1 to skip NNPACK init.
        os.environ.setdefault("TORCH_DISABLE_NNPACK", "1")

    configure_logging(settings.log_level, fmt=settings.log_format)

    # Device / torch diagnostics (helps confirm container has the right torch wheel + CUDA visibility)
    diag = _torch_startup_diagnostics(settings)
    log.info(
        "startup torch_diagnostics settings_device=%s settings_device_map=%s settings_torch_dtype=%s effective_device=%s effective_device_map=%s effective_dtype=%s torch_version=%s cuda_available=%s cuda_device_count=%s cuda_devices=%s",
        diag.get("settings_device"),
        diag.get("settings_device_map"),
        diag.get("settings_torch_dtype"),
        diag.get("effective_device"),
        diag.get("effective_device_map"),
        diag.get("effective_dtype"),
        diag.get("torch_version"),
        diag.get("cuda_available"),
        diag.get("cuda_device_count"),
        diag.get("cuda_devices"),
    )

    tweaks = _torch_performance_tweaks()
    if tweaks:
        log.info("startup torch_perf_tweaks %s", tweaks)

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
    audio_data = await audio.read()
    log.info("preview_audio_received bytes=%s", len(audio_data))

    # Persist stable 'latest preview' artifacts.
    try:
        lp = write_latest_preview(
            settings=settings,
            source_wav_bytes=audio_data,
            transcription=transcription,
            extra_meta={"origin": "http_preview"},
        )
    except Exception as e:
        log.exception("preview_write_latest_failed")
        return JSONResponse(status_code=500, content={"error": f"Failed to save latest preview: {e}"})
    # Synthesize preview using the uploaded audio and transcription
    # (Assume backend uses transcription for improved voice cloning if supported)
    out_dir = settings.previews_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "preview-latest.wav"
    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(
            backend.synthesize_preview,
            text=response_text,
            source_wav=lp.source_path,
            out_wav=out_wav,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "preview_synth_complete session_id=%s out_bytes=%s duration_ms=%s text=%s",
            "latest",
            out_wav.stat().st_size if out_wav.exists() else 0,
            elapsed_ms,
            safe_preview_payload(response_text, limit_chars=settings.log_payload_chars),
        )
    except Exception as e:
        log.exception(
            "preview_synth_failed session_id=%s text=%s",
            "latest",
            safe_preview_payload(response_text, limit_chars=settings.log_payload_chars),
        )
        if is_cuda_device_side_assert(e):
            return JSONResponse(status_code=500, content=cuda_assert_payload(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
    return _safe_file_response(out_wav)


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
        if is_cuda_device_side_assert(e):
            return JSONResponse(status_code=500, content=cuda_assert_payload(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

    return _safe_file_response(out_wav)


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
