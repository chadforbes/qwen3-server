from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings
from app.preview_audio import write_sine_wav
from app.tts_backend import QwenTTSBackend


def main() -> int:
    # Keep this script tiny and dependency-free; it is meant to be used
    # to verify local Qwen voice cloning runtime wiring.
    os.environ.setdefault("TTS_BACKEND", "qwen")
    os.environ.setdefault("PRELOAD_MODEL_ON_STARTUP", "0")
    os.environ.setdefault("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    settings = get_settings()
    backend = QwenTTSBackend(settings=settings)

    out_dir = settings.audio_root
    out_dir.mkdir(parents=True, exist_ok=True)

    source = out_dir / "sox_smoke_source.wav"
    preview = out_dir / "sox_smoke_preview.wav"

    write_sine_wav(source, seconds=0.4, sample_rate=16000)

    backend.ensure_ready()
    backend.synthesize_preview(
        text="Hello from Qwen voice cloning smoke test.",
        source_wav=source,
        out_wav=preview,
    )

    print(f"OK wrote {preview} ({preview.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
