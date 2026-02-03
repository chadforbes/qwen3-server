from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_ws_save_voice_flow(tmp_path: Path, monkeypatch):
    # Force app to use tmp audio root
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")

    with TestClient(app) as client:
        # Upload
        resp = client.post("/upload", files={"file": ("source.wav", b"RIFF....WAVE", "audio/wav")})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "save_voice", "data": {"session_id": session_id, "name": "Nova", "description": "Warm"}})
            msg = ws.receive_json()
            assert msg["type"] == "voice_saved"
            assert msg["data"]["voice_id"] == "nova"

        # Metadata exists
        vdir = tmp_path / "voices" / "nova"
        assert (vdir / "metadata.json").exists()
