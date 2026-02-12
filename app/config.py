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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_int_optional(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class Settings:
    audio_root: Path
    tts_backend: str = "qwen"  # qwen | mock
    qwen_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    preload_model_on_startup: bool = True
    # Optional generation tuning (primarily affects speed vs length/quality).
    qwen_max_new_tokens: int | None = None
    qwen_non_streaming_mode: bool = False

    # Diagnostics
    # If enabled, log timing breakdowns for qwen-tts calls.
    log_qwen_timings: bool = False
    log_level: str = "INFO"
    log_format: str = "text"  # text | json
    log_payload_chars: int = 500

    # Torch / device selection
    # DEVICE can be: auto | cpu | cuda | cuda:0 | cuda:1 ...
    device: str = "auto"
    # Prefer float16 on CUDA by default to reduce memory; CPU stays float32.
    torch_dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    # Some libraries use device_map to shard models; for our use we keep it simple.
    # Allowed: auto | cpu | cuda
    device_map: str = "auto"
    previews_retention_seconds: int = 20 * 60
    previews_cleanup_interval_seconds: int = 5 * 60
    uploads_retention_seconds: int = 12 * 60 * 60
    uploads_cleanup_interval_seconds: int = 30 * 60

    # Server
    uvicorn_workers: int = 1

    # Torch low-level backend toggles
    # NNPACK is a CPU-only backend; on many VMs/CPUs it prints noisy warnings like
    # "Could not initialize NNPACK! Reason: Unsupported hardware.".
    # Disabling it is safe for CUDA workloads and keeps logs clean.
    disable_torch_nnpack: bool = True

    # If CUDA hits a device-side assert (or invalid multinomial probs) during generation,
    # retry once on CPU and keep subsequent synth on CPU to keep the server usable.
    cuda_fallback_to_cpu: bool = True

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
        qwen_max_new_tokens=_env_int_optional("QWEN_MAX_NEW_TOKENS"),
        qwen_non_streaming_mode=_env_bool("QWEN_NON_STREAMING_MODE", False),
        log_qwen_timings=_env_bool("LOG_QWEN_TIMINGS", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        log_format=os.getenv("LOG_FORMAT", "text").strip().lower() or "text",
        log_payload_chars=_env_int("LOG_PAYLOAD_CHARS", 500),
        device=os.getenv("DEVICE", "auto").strip().lower() or "auto",
        torch_dtype=os.getenv("TORCH_DTYPE", "auto").strip().lower() or "auto",
        device_map=os.getenv("DEVICE_MAP", "auto").strip().lower() or "auto",
        uvicorn_workers=max(1, _env_int("UVICORN_WORKERS", 1)),
        disable_torch_nnpack=_env_bool("DISABLE_TORCH_NNPACK", True),
        cuda_fallback_to_cpu=_env_bool("CUDA_FALLBACK_TO_CPU", True),
    )
