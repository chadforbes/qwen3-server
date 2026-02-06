from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_ws_save_voice_flow(tmp_path: Path, monkeypatch):
    # Force app to use tmp audio root
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")

    with TestClient(app) as client:
        # Seed latest preview
        from app.config import Settings
        from app.storage import ensure_dirs, write_latest_preview

        settings = Settings(audio_root=tmp_path)
        ensure_dirs(settings)
        write_latest_preview(settings=settings, source_wav_bytes=b"RIFF....WAVE", transcription="Hello")

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "save_voice", "data": {"name": "Nova", "description": "Warm"}})
            msg = ws.receive_json()
            assert msg["type"] == "voice_saved"
            assert msg["data"]["voice_id"] == "nova"


def test_ws_save_voice_flow_after_generate_preview(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")

    # Seed latest preview
    from app.config import Settings
    from app.storage import ensure_dirs, write_latest_preview

    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)
    write_latest_preview(settings=settings, source_wav_bytes=b"RIFF....WAVE", transcription="Hello")

    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "generate_preview", "data": {"text": "Hello"}})
            msg1 = ws.receive_json()
            assert msg1["type"] == "tts_complete"

            # Now save (no session id model).
            ws.send_json({"type": "save_voice", "data": {"name": "Nova", "description": "Warm"}})
            msg2 = ws.receive_json()
            assert msg2["type"] == "voice_saved"
            assert msg2["data"]["voice_id"] == "nova"

        # Metadata exists
        vdir = tmp_path / "voices" / "nova"
        assert (vdir / "metadata.json").exists()


def test_ws_generate_preview_and_save_using_latest(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")

    # Seed the latest-preview artifacts directly.
    from app.config import Settings
    from app.storage import ensure_dirs, write_latest_preview

    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)
    write_latest_preview(settings=settings, source_wav_bytes=b"RIFF....WAVE", transcription="Hi")

    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "generate_preview", "data": {"text": "Hello"}})
            msg1 = ws.receive_json()
            assert msg1["type"] == "tts_complete"

            ws.send_json({"type": "save_voice", "data": {"name": "Nova", "description": "Warm"}})
            msg2 = ws.receive_json()
            assert msg2["type"] == "voice_saved"
            assert msg2["data"]["voice_id"] == "nova"
