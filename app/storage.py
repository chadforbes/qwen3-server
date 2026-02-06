from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
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
class LatestPreview:
    """Stable location for the most recent preview's source artifacts.

    This intentionally isn't keyed by session id, so clients can follow a simple
    'upload/generate preview, then save' flow without tracking ids.
    """

    folder: Path
    source_path: Path
    transcription_path: Path
    meta_path: Path


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


def latest_preview(settings: Settings) -> LatestPreview:
    folder = settings.uploads_dir / "latest"
    source_path = folder / "source.wav"
    transcription_path = folder / "transcription.txt"
    meta_path = folder / "meta.json"
    return LatestPreview(
        folder=folder,
        source_path=source_path,
        transcription_path=transcription_path,
        meta_path=meta_path,
    )


def write_latest_preview(
    *,
    settings: Settings,
    source_wav_bytes: bytes,
    transcription: str,
    extra_meta: Optional[dict[str, Any]] = None,
) -> LatestPreview:
    """Persist most recent preview source audio and transcript.

    Writes atomically (best-effort) so readers won't see partial state.
    """

    lp = latest_preview(settings)
    lp.folder.mkdir(parents=True, exist_ok=True)

    # Atomic-ish write: write to temp files then replace.
    tmp_source = lp.folder / "source.wav.tmp"
    tmp_trans = lp.folder / "transcription.txt.tmp"
    tmp_meta = lp.folder / "meta.json.tmp"

    tmp_source.write_bytes(source_wav_bytes)
    tmp_trans.write_text(transcription or "", encoding="utf-8")

    meta: dict[str, Any] = {
        "saved_at": utc_now().astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_file": "source.wav",
        "transcription_file": "transcription.txt",
        "source_bytes": len(source_wav_bytes),
        "transcription_len": len(transcription or ""),
    }
    if extra_meta:
        # Avoid surprising nested objects; keep it JSON-serializable.
        try:
            json.dumps(extra_meta)
            meta.update(extra_meta)
        except TypeError:
            pass
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    os.replace(tmp_source, lp.source_path)
    os.replace(tmp_trans, lp.transcription_path)
    os.replace(tmp_meta, lp.meta_path)

    return lp


def load_latest_preview(settings: Settings) -> LatestPreview:
    lp = latest_preview(settings)
    if not lp.source_path.exists():
        raise ValidationError("No latest preview source.wav found")
    # transcription may be missing for older flows; treat as optional.
    return lp


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


def load_voice_metadata(settings: Settings, voice_id: str) -> dict[str, Any]:
    """Load saved voice metadata.json for a voice.

    Raises ValidationError if the voice doesn't exist or metadata is invalid.
    """

    vdir = voice_dir(settings, voice_id)
    meta_path = vdir / "metadata.json"
    if not meta_path.exists():
        raise ValidationError("Voice not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise StorageError("Failed to read voice metadata") from e
    if not isinstance(meta, dict):
        raise StorageError("Invalid voice metadata")
    return meta


def list_voices(settings: Settings) -> list[dict[str, Any]]:
    """Return saved voices (metadata) sorted by voice_id."""

    if not settings.voices_dir.exists():
        return []

    voices: list[dict[str, Any]] = []
    for child in settings.voices_dir.iterdir():
        if not child.is_dir():
            continue
        voice_id = child.name
        try:
            meta = load_voice_metadata(settings, voice_id)
        except ValidationError:
            # Skip malformed entries.
            continue
        except StorageError:
            continue
        # Ensure id is always present
        meta.setdefault("voice_id", voice_id)
        voices.append(meta)

    voices.sort(key=lambda v: str(v.get("voice_id") or ""))
    return voices


def voice_source_wav(settings: Settings, voice_id: str) -> Path:
    """Return path to saved voice's source.wav.

    Raises ValidationError if voice doesn't exist.
    """

    vdir = voice_dir(settings, voice_id)
    source = vdir / "source.wav"
    if not source.exists():
        raise ValidationError("Voice not found")
    return source


def voice_transcription_path(settings: Settings, voice_id: str) -> Path:
    """Return path to saved voice's transcription.txt.

    May not exist for voices created before transcription was persisted.
    """

    vdir = voice_dir(settings, voice_id)
    return vdir / "transcription.txt"


def load_voice_transcription(settings: Settings, voice_id: str) -> str:
    """Load a saved voice transcription, or return empty string if missing."""

    path = voice_transcription_path(settings, voice_id)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


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
    session_id: Optional[str] = None,
    name: str,
    description: Optional[str],
    embedding_provider: Optional[Any] = None,
) -> dict[str, str]:
    if session_id is not None:
        # We're intentionally moving away from session-id based uploads.
        raise ValidationError("session_id is no longer supported; use latest preview")

    latest = load_latest_preview(settings)
    source_path = latest.source_path
    session_transcription = latest.transcription_path
    log.info("storage_save_voice_using_latest_preview")

    log.info(
        "storage_save_voice_begin session_id=%s name=%s",
        "latest",
        name.strip() if isinstance(name, str) else "-",
    )

    voice_id = compute_voice_id(settings, name)
    desc = validate_description(description)

    vdir = voice_dir(settings, voice_id)
    vdir.mkdir(parents=True, exist_ok=False)

    dest_source = vdir / "source.wav"
    dest_embedding = vdir / "embedding.json"
    dest_metadata = vdir / "metadata.json"
    dest_transcription = vdir / "transcription.txt"

    # Move uploaded source.wav into permanent location
    shutil.copy2(str(source_path), str(dest_source))
    log.info("storage_save_voice_moved_source voice_id=%s", voice_id)

    # Persist transcription if present. Best-effort: old clients/sessions may not have it.
    transcription_saved = False
    if session_transcription.exists():
        try:
            shutil.copy2(str(session_transcription), str(dest_transcription))
            transcription_saved = True
            log.info("storage_save_voice_moved_transcription voice_id=%s", voice_id)
        except Exception:
            log.exception("storage_save_voice_transcription_move_failed voice_id=%s", voice_id)

    # Do not delete the reserved latest preview folder.

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

    if transcription_saved:
        # Patch metadata.json to include transcription file reference.
        try:
            meta = json.loads(dest_metadata.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                meta["transcription_file"] = "transcription.txt"
                dest_metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            # Non-fatal; transcription is still present on disk.
            log.exception("storage_save_voice_metadata_patch_failed voice_id=%s", voice_id)

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
        # Reserved folder for 'most recent preview' artifacts.
        if session_dir.name == "latest":
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
