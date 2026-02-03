from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .preview_audio import write_sine_wav


class TTSError(RuntimeError):
    pass


class VoiceEmbeddingProvider(Protocol):
    def create_embedding(self, *, source_wav: Path) -> dict[str, Any]: ...


class TTSBackend(VoiceEmbeddingProvider, Protocol):
    def synthesize_preview(self, *, text: str, source_wav: Path, out_wav: Path) -> None: ...

    def ensure_ready(self) -> None: ...


@dataclass
class MockBackend:
    def synthesize_preview(self, *, text: str, source_wav: Path, out_wav: Path) -> None:
        # text/source_wav intentionally ignored for mock.
        write_sine_wav(out_wav)

    def create_embedding(self, *, source_wav: Path) -> dict[str, Any]:
        from hashlib import sha256

        return {"type": "placeholder_sha256", "sha256": sha256(source_wav.read_bytes()).hexdigest()}

    def ensure_ready(self) -> None:
        return


class QwenTTSBackend:
    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._log = logging.getLogger(__name__)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            if shutil.which("sox") is None:
                self._log.warning("SoX binary not found on PATH; qwen-tts may fail at runtime")

            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as e:
            raise TTSError(
                "Qwen TTS backend not available. Install dependencies: pip install qwen-tts torch"
            ) from e

        self._model = Qwen3TTSModel.from_pretrained(
            self._settings.qwen_model_id,
            device_map="cpu",
            dtype=torch.float32,
        )
        return self._model

    def ensure_ready(self) -> None:
        # Force model resolution/download at startup.
        self._log.info("Ensuring Qwen3 TTS model is available: %s", self._settings.qwen_model_id)
        self._load_model()
        self._log.info("Qwen3 TTS model ready")

    def synthesize_preview(self, *, text: str, source_wav: Path, out_wav: Path) -> None:
        if not text.strip():
            raise TTSError("text is required")
        if not source_wav.exists():
            raise TTSError("source.wav not found")

        model = self._load_model()

        # Voice cloning only: build a clone prompt from the uploaded reference audio.
        try:
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=str(source_wav),
                ref_text=None,
                x_vector_only_mode=True,
            )
        except TypeError:
            # Older API variants: fall back to passing ref_text if required.
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=str(source_wav),
                ref_text="",
                x_vector_only_mode=True,
            )

        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="Auto",
                voice_clone_prompt=voice_clone_prompt,
            )
        except TypeError:
            # Some versions accept ref_audio directly.
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=str(source_wav),
                ref_text=None,
            )

        if not wavs:
            raise TTSError("No audio generated")

        try:
            import soundfile as sf
        except Exception as e:
            raise TTSError("Missing dependency: soundfile") from e

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), wavs[0], sr)

    def create_embedding(self, *, source_wav: Path) -> dict[str, Any]:
        model = self._load_model()
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(source_wav),
                ref_text=None,
                x_vector_only_mode=True,
            )
        except TypeError:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(source_wav),
                ref_text="",
                x_vector_only_mode=True,
            )

        # Best-effort JSON-serializable structure.
        serializable = json.loads(json.dumps(prompt, default=str))
        return {"type": "qwen3_voice_clone_prompt", "model_id": self._settings.qwen_model_id, "prompt": serializable}


def build_backend(settings: Settings) -> TTSBackend:
    backend = settings.tts_backend
    if backend == "mock":
        return MockBackend()
    if backend == "qwen":
        return QwenTTSBackend(settings=settings)
    raise TTSError(f"Unknown TTS_BACKEND: {backend}")
