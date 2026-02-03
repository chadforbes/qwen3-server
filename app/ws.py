from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any

from fastapi import WebSocket

from .config import Settings
from .storage import ValidationError, get_session, preview_path, save_voice
from .tts_backend import TTSBackend, TTSError


log = logging.getLogger(__name__)


async def ws_send(websocket: WebSocket, msg_type: str, data: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps({"type": msg_type, "data": data}))


async def handle_message(settings: Settings, backend: TTSBackend, websocket: WebSocket, message: dict[str, Any]) -> None:
    msg_type = message.get("type")
    data = message.get("data") or {}
    if not isinstance(msg_type, str) or not isinstance(data, dict):
        raise ValidationError("Invalid message format")

    if msg_type == "generate_preview":
        session_id = data.get("session_id")
        text = data.get("text")
        if not isinstance(session_id, str):
            raise ValidationError("session_id is required")
        if not isinstance(text, str):
            raise ValidationError("text is required")

        session = get_session(settings, session_id)
        if not session.source_path.exists():
            raise ValidationError("Uploaded source.wav not found for session")

        job_id = secrets.token_urlsafe(9).replace("-", "_").replace("~", "_")[:12]
        out_wav = preview_path(settings, job_id)

        # Real preview generation (non-blocking)
        try:
            await asyncio.to_thread(backend.synthesize_preview, text=text, source_wav=session.source_path, out_wav=out_wav)
        except TTSError as e:
            log.exception("generate_preview failed (session_id=%s)", session_id)
            raise ValidationError(str(e))

        await ws_send(
            websocket,
            "tts_complete",
            {"job_id": job_id, "audio_url": f"/previews/{job_id}.wav", "temporary": True},
        )
        return

    if msg_type == "save_voice":
        session_id = data.get("session_id")
        name = data.get("name")
        description = data.get("description")
        if not isinstance(session_id, str):
            raise ValidationError("session_id is required")
        if not isinstance(name, str):
            raise ValidationError("name is required")
        if description is not None and not isinstance(description, str):
            raise ValidationError("description must be a string")

        result = await asyncio.to_thread(
            save_voice,
            settings=settings,
            session_id=session_id,
            name=name,
            description=description,
            embedding_provider=backend,
        )
        log.info("voice_saved voice_id=%s name=%s", result.get("voice_id"), result.get("name"))
        await ws_send(websocket, "voice_saved", result)
        return

    raise ValidationError(f"Unknown message type: {msg_type}")


async def ws_loop(settings: Settings, backend: TTSBackend, websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValidationError("Message must be a JSON object")
                await handle_message(settings, backend, websocket, message)
            except ValidationError as e:
                log.warning("ws validation error: %s", e)
                await ws_send(websocket, "error", {"message": str(e)})
            except json.JSONDecodeError:
                log.warning("ws invalid json")
                await ws_send(websocket, "error", {"message": "Invalid JSON"})
    except Exception:
        # Client disconnected or server shutting down.
        return
