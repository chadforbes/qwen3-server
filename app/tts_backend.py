from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .preview_audio import write_sine_wav
from .torch_compat import torch_arch_mismatch_hint
from .torch_utils import is_cuda_device_side_assert


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
        self._cpu_model = None
        self._cuda_disabled = False
        self._log = logging.getLogger(__name__)

    @staticmethod
    def _is_invalid_probability_error(exc: BaseException) -> bool:
        # Common torch.multinomial CUDA assertion text.
        msg = str(exc).lower()
        if "probability tensor contains" in msg:
            return True
        if "torch.multinomial" in msg and ("nan" in msg or "inf" in msg or "< 0" in msg or "<0" in msg):
            return True
        return False

    def _load_cpu_model(self):
        if self._cpu_model is not None:
            return self._cpu_model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as e:
            raise TTSError(
                "Qwen TTS backend not available. Install dependencies: pip install qwen-tts torch"
            ) from e

        self._log.info(
            "Loading Qwen3 TTS model on CPU for fallback model_id=%s",
            self._settings.qwen_model_id,
        )
        self._cpu_model = Qwen3TTSModel.from_pretrained(
            self._settings.qwen_model_id,
            device_map="cpu",
            dtype=torch.float32,
        )
        return self._cpu_model

    def _load_model(self):
        if self._model is not None:
            return self._model

        if self._cuda_disabled:
            return self._load_cpu_model()
        try:
            if shutil.which("sox") is None:
                self._log.warning("SoX binary not found on PATH; qwen-tts may fail at runtime")

            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as e:
            raise TTSError(
                "Qwen TTS backend not available. Install dependencies: pip install qwen-tts torch"
            ) from e

        # Device selection
        device = (self._settings.device or "auto").lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # device_map: allow override, else follow device.
        device_map = (self._settings.device_map or "auto").lower()
        if device_map == "auto":
            # qwen-tts expects a string like "cpu" / "cuda" here.
            device_map = "cuda" if device.startswith("cuda") else "cpu"

        # dtype selection
        dtype_setting = (self._settings.torch_dtype or "auto").lower()
        if dtype_setting == "auto":
            # CUDA defaults to reduced precision to reduce VRAM.
            # Prefer bfloat16 when supported (often more numerically stable than float16).
            # If bf16 isn't supported, default to float32 for stability.
            # (Users can still force float16 via TORCH_DTYPE=float16.)
            if device.startswith("cuda"):
                bf16_supported = False
                try:
                    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
                    bf16_supported = bool(is_bf16_supported()) if callable(is_bf16_supported) else False
                except Exception:
                    bf16_supported = False
                dtype = torch.bfloat16 if bf16_supported else torch.float32
            else:
                dtype = torch.float32
        else:
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(dtype_setting, torch.float32)

        self._log.info(
            "Loading Qwen3 TTS model model_id=%s device=%s device_map=%s dtype=%s cuda_available=%s",
            self._settings.qwen_model_id,
            device,
            device_map,
            str(dtype).replace("torch.", ""),
            torch.cuda.is_available(),
        )

        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self._settings.qwen_model_id,
                device_map=device_map,
                dtype=dtype,
            )
        except Exception as e:
            hint = torch_arch_mismatch_hint(e)
            if hint is not None:
                raise TTSError(f"{hint.title}. {hint.suggested_action}") from e
            # If CUDA was requested but fails (missing drivers, wrong torch wheel), fall back to CPU.
            if device.startswith("cuda") or device_map == "cuda":
                self._log.exception("Failed to load model on CUDA; falling back to CPU")
                self._cuda_disabled = True
                self._model = self._load_cpu_model()
            else:
                raise
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

        def _run_generate(m):
            # Voice cloning only: build a clone prompt from the uploaded reference audio.
            try:
                voice_clone_prompt = m.create_voice_clone_prompt(
                    ref_audio=str(source_wav),
                    ref_text=None,
                    x_vector_only_mode=True,
                )
            except TypeError:
                # Older API variants: fall back to passing ref_text if required.
                voice_clone_prompt = m.create_voice_clone_prompt(
                    ref_audio=str(source_wav),
                    ref_text="",
                    x_vector_only_mode=True,
                )

            try:
                return m.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=voice_clone_prompt,
                )
            except TypeError:
                # Some versions accept ref_audio directly.
                return m.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=str(source_wav),
                    ref_text=None,
                )

        model = self._load_model()
        try:
            wavs, sr = _run_generate(model)
        except Exception as e:
            hint = torch_arch_mismatch_hint(e)
            if hint is not None:
                raise TTSError(f"{hint.title}. {hint.suggested_action}") from e

            should_fallback = (
                self._settings.cuda_fallback_to_cpu
                and not self._cuda_disabled
                and (is_cuda_device_side_assert(e) or self._is_invalid_probability_error(e))
            )
            if not should_fallback:
                raise

            # After a CUDA device-side assert, the CUDA context is typically poisoned for the
            # lifetime of the process. Switch to CPU for subsequent requests.
            self._cuda_disabled = True
            self._log.exception("Qwen TTS CUDA failure during generate; retrying on CPU")
            cpu_model = self._load_cpu_model()
            wavs, sr = _run_generate(cpu_model)

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
