from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    audio_root: Path
    tts_backend: str = "qwen"  # qwen | mock
    qwen_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    preload_model_on_startup: bool = True
    log_level: str = "INFO"
    cors_allow_origins: str = "*"
    previews_retention_seconds: int = 20 * 60
    previews_cleanup_interval_seconds: int = 5 * 60
    uploads_retention_seconds: int = 12 * 60 * 60
    uploads_cleanup_interval_seconds: int = 30 * 60

    @property
    def previews_dir(self) -> Path:
        return self.audio_root / "previews"

    @property
    def uploads_dir(self) -> Path:
        return self.audio_root / "uploads"

    @property
    def voices_dir(self) -> Path:
        return self.audio_root / "voices"


def get_settings() -> Settings:
    audio_root = Path(os.getenv("AUDIO_ROOT", "./audio")).resolve()
    return Settings(
        audio_root=audio_root,
        tts_backend=os.getenv("TTS_BACKEND", "qwen").strip().lower() or "qwen",
        qwen_model_id=os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base").strip()
        or "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        preload_model_on_startup=_env_bool("PRELOAD_MODEL_ON_STARTUP", True),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").strip() or "*",
    )
