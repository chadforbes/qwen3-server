from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.storage import (
    ValidationError,
    ensure_dirs,
    save_voice,
    write_latest_preview,
)


def test_save_voice_creates_voice(tmp_path: Path):
    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)

    write_latest_preview(settings=settings, source_wav_bytes=b"RIFF....WAVE", transcription="Hello")

    result = save_voice(settings=settings, name="Nova", description="Warm")
    assert result["voice_id"] == "nova"

    vdir = settings.voices_dir / "nova"
    assert (vdir / "source.wav").exists()
    assert (vdir / "embedding.json").exists()
    assert (vdir / "metadata.json").exists()

    meta = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["voice_id"] == "nova"
    assert meta["name"] == "Nova"
    assert meta["description"] == "Warm"
    assert meta["created_at"].endswith("Z")


def test_save_voice_from_latest_preview_persists_transcription(tmp_path: Path):
    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)

    write_latest_preview(settings=settings, source_wav_bytes=b"RIFF....WAVE", transcription="Hello world")

    result = save_voice(settings=settings, name="Nova", description="Warm")
    assert result["voice_id"] == "nova"

    vdir = settings.voices_dir / "nova"
    assert (vdir / "source.wav").exists()
    assert (vdir / "transcription.txt").read_text(encoding="utf-8") == "Hello world"


def test_voice_name_collision_rejected(tmp_path: Path):
    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)

    write_latest_preview(settings=settings, source_wav_bytes=b"a", transcription="")
    save_voice(settings=settings, name="Nova", description=None)

    with pytest.raises(ValidationError):
        write_latest_preview(settings=settings, source_wav_bytes=b"b", transcription="")
        save_voice(settings=settings, name="Nova", description=None)
