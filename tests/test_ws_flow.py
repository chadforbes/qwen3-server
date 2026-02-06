from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def _make_session(tmp_path: Path, client: TestClient) -> str:
    # There's no /upload endpoint anymore. Create a session folder the same way
    # the service does, then use it with the websocket save_voice flow.
    from app.config import Settings
    from app.storage import ensure_dirs, new_session

    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)
    session = new_session(settings)
    session.source_path.write_bytes(b"RIFF....WAVE")
    return session.session_id


def test_ws_save_voice_flow(tmp_path: Path, monkeypatch):
    # Force app to use tmp audio root
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")

    with TestClient(app) as client:
        session_id = _make_session(tmp_path, client)

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "save_voice", "data": {"session_id": session_id, "name": "Nova", "description": "Warm"}})
            msg = ws.receive_json()
            assert msg["type"] == "voice_saved"
            assert msg["data"]["voice_id"] == "nova"


def test_ws_save_voice_flow_without_session_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")

    # Create a session (as if an upload/preview happened elsewhere)
    from app.config import Settings
    from app.storage import ensure_dirs, new_session

    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)
    session = new_session(settings)
    session.source_path.write_bytes(b"RIFF....WAVE")

    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # First call generate_preview to set last_session_id for this connection.
            ws.send_json({"type": "generate_preview", "data": {"session_id": session.session_id, "text": "Hello"}})
            msg1 = ws.receive_json()
            assert msg1["type"] == "tts_complete"

            # Now omit session_id when saving.
            ws.send_json({"type": "save_voice", "data": {"name": "Nova", "description": "Warm"}})
            msg2 = ws.receive_json()
            assert msg2["type"] == "voice_saved"
            assert msg2["data"]["voice_id"] == "nova"

        # Metadata exists
        vdir = tmp_path / "voices" / "nova"
        assert (vdir / "metadata.json").exists()
