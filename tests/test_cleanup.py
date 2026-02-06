from __future__ import annotations

import os
import time
from pathlib import Path

from app.config import Settings
from app.storage import cleanup_previews_sync, cleanup_uploads_sync, ensure_dirs


def _touch_with_age(path: Path, age_seconds: int) -> None:
    now = time.time()
    ts = now - age_seconds
    os.utime(path, (ts, ts))


def test_cleanup_previews(tmp_path: Path):
    settings = Settings(audio_root=tmp_path, previews_retention_seconds=60)
    ensure_dirs(settings)

    keep = settings.previews_dir / "keep.wav"
    old = settings.previews_dir / "old.wav"
    keep.write_bytes(b"x")
    old.write_bytes(b"x")
    _touch_with_age(old, 61)

    removed = cleanup_previews_sync(settings)
    assert removed == 1
    assert keep.exists()
    assert not old.exists()


def test_cleanup_uploads_keeps_latest(tmp_path: Path):
    settings = Settings(audio_root=tmp_path, uploads_retention_seconds=1)
    ensure_dirs(settings)

    latest_dir = settings.uploads_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    source = latest_dir / "source.wav"
    source.write_bytes(b"x")
    _touch_with_age(source, 999)

    removed = cleanup_uploads_sync(settings)
    assert removed == 0
    assert latest_dir.exists()
    assert source.exists()
