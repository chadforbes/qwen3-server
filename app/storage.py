from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from .config import Settings


log = logging.getLogger(__name__)


_NAME_MAX_LEN = 80
_DESC_MAX_LEN = 500


class StorageError(RuntimeError):
    pass


class ValidationError(StorageError):
    pass


@dataclass(frozen=True)
class UploadSession:
    session_id: str
    folder: Path
    source_path: Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dirs(settings: Settings) -> None:
    settings.audio_root.mkdir(parents=True, exist_ok=True)
    settings.previews_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "storage_dirs_ready audio_root=%s previews=%s uploads=%s voices=%s",
        settings.audio_root,
        settings.previews_dir,
        settings.uploads_dir,
        settings.voices_dir,
    )


def new_session(settings: Settings) -> UploadSession:
    session_id = uuid.uuid4().hex
    folder = settings.uploads_dir / session_id
    source_path = folder / "source.wav"
    folder.mkdir(parents=True, exist_ok=False)
    log.info("storage_new_session session_id=%s", session_id)
    return UploadSession(session_id=session_id, folder=folder, source_path=source_path)


def get_session(settings: Settings, session_id: str) -> UploadSession:
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValidationError("Invalid session_id")
    folder = settings.uploads_dir / session_id
    source_path = folder / "source.wav"
    return UploadSession(session_id=session_id, folder=folder, source_path=source_path)


def _slugify_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValidationError("Voice name is required")
    if len(name) > _NAME_MAX_LEN:
        raise ValidationError("Voice name is too long")

    lowered = name.lower()
    # Keep alnum, convert runs of non-alnum to single '-'
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if not slug:
        raise ValidationError("Voice name must contain letters/numbers")
    if len(slug) > 64:
        slug = slug[:64].rstrip("-")
    return slug


def compute_voice_id(settings: Settings, name: str) -> str:
    voice_id = _slugify_name(name)
    voice_path = settings.voices_dir / voice_id
    if voice_path.exists():
        # Deterministic + unique: require user to pick a different name.
        raise ValidationError("Voice name already exists")
    return voice_id


def validate_description(description: Optional[str]) -> str:
    if description is None:
        return ""
    desc = description.strip()
    if len(desc) > _DESC_MAX_LEN:
        raise ValidationError("Voice description is too long")
    return desc


def voice_dir(settings: Settings, voice_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", voice_id):
        raise ValidationError("Invalid voice_id")
    return settings.voices_dir / voice_id


def preview_path(settings: Settings, job_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{6,64}", job_id):
        raise ValidationError("Invalid job_id")
    return settings.previews_dir / f"{job_id}.wav"


def write_embedding_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metadata_json(
    *,
    voice_id: str,
    name: str,
    description: str,
    created_at: datetime,
    out_path: Path,
) -> None:
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    created_at_utc = created_at.astimezone(timezone.utc).replace(microsecond=0)

    payload: dict[str, Any] = {
        "voice_id": voice_id,
        "name": name,
        "description": description,
        "created_at": created_at_utc.isoformat().replace("+00:00", "Z"),
        "source_file": "source.wav",
        "embedding_file": "embedding.json",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_voice(
    *,
    settings: Settings,
    session_id: str,
    name: str,
    description: Optional[str],
    embedding_provider: Optional[Any] = None,
) -> dict[str, str]:
    session = get_session(settings, session_id)
    if not session.source_path.exists():
        raise ValidationError("Uploaded source.wav not found for session")

    log.info(
        "storage_save_voice_begin session_id=%s name=%s",
        session_id,
        name.strip() if isinstance(name, str) else "-",
    )

    voice_id = compute_voice_id(settings, name)
    desc = validate_description(description)

    vdir = voice_dir(settings, voice_id)
    vdir.mkdir(parents=True, exist_ok=False)

    dest_source = vdir / "source.wav"
    dest_embedding = vdir / "embedding.json"
    dest_metadata = vdir / "metadata.json"

    # Move uploaded source.wav into permanent location
    shutil.move(str(session.source_path), str(dest_source))
    log.info("storage_save_voice_moved_source voice_id=%s", voice_id)

    # Remove now-empty upload folder (best-effort)
    try:
        shutil.rmtree(session.folder)
    except FileNotFoundError:
        pass

    if embedding_provider is not None:
        embedding_payload = embedding_provider.create_embedding(source_wav=dest_source)
        if not isinstance(embedding_payload, dict):
            raise StorageError("Embedding provider returned invalid payload")
    else:
        data = dest_source.read_bytes()
        embedding_payload = {"type": "placeholder_sha256", "sha256": sha256(data).hexdigest()}

    write_embedding_json(embedding_payload, dest_embedding)
    write_metadata_json(
        voice_id=voice_id,
        name=name.strip(),
        description=desc,
        created_at=utc_now(),
        out_path=dest_metadata,
    )

    log.info(
        "storage_save_voice_done voice_id=%s embedding_type=%s",
        voice_id,
        embedding_payload.get("type") if isinstance(embedding_payload, dict) else "-",
    )
    return {"voice_id": voice_id, "name": name.strip(), "description": desc}


def _iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def _is_older_than(path: Path, older_than_seconds: int, now_ts: float) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return (now_ts - mtime) > older_than_seconds


def cleanup_previews_sync(settings: Settings) -> int:
    now_ts = datetime.now(timezone.utc).timestamp()
    removed = 0
    for wav in settings.previews_dir.glob("*.wav"):
        if _is_older_than(wav, settings.previews_retention_seconds, now_ts):
            try:
                wav.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def cleanup_uploads_sync(settings: Settings) -> int:
    now_ts = datetime.now(timezone.utc).timestamp()
    removed_sessions = 0
    if not settings.uploads_dir.exists():
        return 0

    for session_dir in settings.uploads_dir.iterdir():
        if not session_dir.is_dir():
            continue
        source = session_dir / "source.wav"
        # If source.wav is missing, treat as removable.
        candidate = source if source.exists() else session_dir
        if _is_older_than(candidate, settings.uploads_retention_seconds, now_ts):
            try:
                shutil.rmtree(session_dir)
                removed_sessions += 1
            except FileNotFoundError:
                pass
    return removed_sessions


async def cleanup_previews(settings: Settings) -> int:
    return await asyncio.to_thread(cleanup_previews_sync, settings)


async def cleanup_uploads(settings: Settings) -> int:
    return await asyncio.to_thread(cleanup_uploads_sync, settings)
