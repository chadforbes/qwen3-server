from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.storage import (
    ValidationError,
    ensure_dirs,
    new_session,
    save_voice,
)


def test_save_voice_creates_voice(tmp_path: Path):
    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)

    session = new_session(settings)
    session.source_path.write_bytes(b"RIFF....WAVE")

    result = save_voice(settings=settings, session_id=session.session_id, name="Nova", description="Warm")
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


def test_voice_name_collision_rejected(tmp_path: Path):
    settings = Settings(audio_root=tmp_path)
    ensure_dirs(settings)

    s1 = new_session(settings)
    s1.source_path.write_bytes(b"a")
    save_voice(settings=settings, session_id=s1.session_id, name="Nova", description=None)

    s2 = new_session(settings)
    s2.source_path.write_bytes(b"b")
    with pytest.raises(ValidationError):
        save_voice(settings=settings, session_id=s2.session_id, name="Nova", description=None)
