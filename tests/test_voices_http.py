from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _make_saved_voice(tmp_path: Path, voice_id: str = "nova") -> None:
    vdir = tmp_path / "voices" / voice_id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "source.wav").write_bytes(b"RIFF....WAVE")
    (vdir / "embedding.json").write_text("{}", encoding="utf-8")
    (vdir / "transcription.txt").write_text("Hello this is my voice", encoding="utf-8")
    (vdir / "metadata.json").write_text(
        '{"voice_id": "nova", "name": "Nova", "description": "Warm", "created_at": "2026-01-01T00:00:00Z", "source_file": "source.wav", "embedding_file": "embedding.json", "transcription_file": "transcription.txt"}',
        encoding="utf-8",
    )


def test_get_voices_lists_saved_voices(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")
    monkeypatch.setenv("PRELOAD_MODEL_ON_STARTUP", "0")

    _make_saved_voice(tmp_path)

    from app.main import app

    with TestClient(app) as client:
        r = client.get("/voices")
        assert r.status_code == 200
        payload = r.json()
        assert "voices" in payload
        assert payload["voices"][0]["voice_id"] == "nova"
    assert payload["voices"][0]["transcription_file"] == "transcription.txt"


def test_preview_from_voice_uses_saved_source_wav(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")
    monkeypatch.setenv("PRELOAD_MODEL_ON_STARTUP", "0")

    _make_saved_voice(tmp_path)

    from app.main import app

    with TestClient(app) as client:
        r = client.post(
            "/preview-from-voice",
            data={"voice_id": "nova", "response_text": "Hello"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        # mock backend produces a valid WAV header
        assert r.content[:4] == b"RIFF"


def test_preview_from_voice_unknown_voice_404(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("TTS_BACKEND", "mock")
    monkeypatch.setenv("PRELOAD_MODEL_ON_STARTUP", "0")

    from app.main import app

    with TestClient(app) as client:
        r = client.post(
            "/preview-from-voice",
            data={"voice_id": "missing", "response_text": "Hello"},
        )
        assert r.status_code == 404
