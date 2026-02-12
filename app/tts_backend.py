from __future__ import annotations

import json
import logging
import shutil
import threading
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
        # Cache the most recent voice-clone prompt derived from a reference WAV.
        # Building it can be expensive; for the common workflow (same reference
        # audio, many response_text iterations) this provides a big speedup.
        self._prompt_cache_lock = threading.Lock()
        self._prompt_cache_key: tuple[int, str, int, int] | None = None
        self._prompt_cache_value: Any | None = None

    @staticmethod
    def _source_sig(source_wav: Path) -> tuple[str, int, int]:
        st = source_wav.stat()
        # Keyed by resolved path + mtime_ns + size to avoid stale cache when
        # the file is replaced in-place (e.g. uploads/latest/source.wav).
        return (str(source_wav.resolve()), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size))

    def _effective_device_for_tensors(self) -> str:
        """Best-effort runtime device selection for prompt tensors.

        Keep this conservative: if CUDA is disabled or unavailable, stay on CPU.
        """

        if self._cuda_disabled:
            return "cpu"
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"

        device = (self._settings.device or "auto").lower()
        if device == "auto":
            return "cuda"
        if device.startswith("cuda"):
            return device
        return "cpu"

    @staticmethod
    def _move_prompt_to_device(prompt: dict[str, Any], *, device: str) -> dict[str, Any]:
        try:
            import torch
        except Exception:
            return prompt

        if device == "cpu":
            # Ensure tensors are on CPU for portability.
            def to_cpu(x: Any) -> Any:
                if isinstance(x, torch.Tensor):
                    return x.detach().to("cpu")
                if isinstance(x, list):
                    return [to_cpu(v) for v in x]
                if isinstance(x, dict):
                    return {k: to_cpu(v) for k, v in x.items()}
                return x

            return to_cpu(prompt)

        def to_dev(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, list):
                return [to_dev(v) for v in x]
            if isinstance(x, dict):
                return {k: to_dev(v) for k, v in x.items()}
            return x

        return to_dev(prompt)

    def _load_prompt_from_pt(self, *, source_wav: Path) -> dict[str, Any] | None:
        """Load persisted prompt dict if available.

        We persist a torch-serialized file for saved voices so prompts can be
        reused across restarts without JSON coercion issues.
        """

        pt_path = source_wav.parent / "embedding.pt"
        if not pt_path.exists():
            return None
        try:
            import torch

            payload = torch.load(str(pt_path), map_location="cpu")
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("type") != "qwen3_voice_clone_prompt_pt_v1":
            return None
        if payload.get("model_id") != self._settings.qwen_model_id:
            return None
        prompt = payload.get("voice_clone_prompt")
        if not isinstance(prompt, dict):
            return None

        device = self._effective_device_for_tensors()
        self._log.debug("voice_clone_prompt_loaded source=pt path=%s", str(pt_path))
        return self._move_prompt_to_device(prompt, device=device)

    def _maybe_persist_prompt_pt(self, *, source_wav: Path, voice_clone_prompt: dict[str, Any]) -> None:
        """Persist prompt next to source.wav when it looks like a saved voice.

        This enables faster preview-from-voice across restarts without requiring
        the user to re-save the voice.
        """

        # Only do this for saved voices (audio/voices/<id>/source.wav), not for uploads/latest.
        if not (source_wav.parent / "metadata.json").exists():
            return
        pt_path = source_wav.parent / "embedding.pt"
        if pt_path.exists():
            return
        try:
            import torch

            payload = {
                "type": "qwen3_voice_clone_prompt_pt_v1",
                "model_id": self._settings.qwen_model_id,
                "voice_clone_prompt": self._move_prompt_to_device(voice_clone_prompt, device="cpu"),
            }
            torch.save(payload, str(pt_path))
            self._log.debug("voice_clone_prompt_persisted source=computed path=%s", str(pt_path))
        except Exception:
            # Non-fatal.
            return

    def _get_voice_clone_prompt_with_source(self, model: Any, *, source_wav: Path) -> tuple[dict[str, Any], str]:
        # Prefer persisted prompt for saved voices.
        persisted = self._load_prompt_from_pt(source_wav=source_wav)
        if persisted is not None:
            return persisted, "pt"

        source_path, mtime_ns, size = self._source_sig(source_wav)
        device = self._effective_device_for_tensors()
        cache_key = (id(model), source_path, mtime_ns, size, device)

        with self._prompt_cache_lock:
            if self._prompt_cache_key == cache_key and self._prompt_cache_value is not None:
                self._log.debug("voice_clone_prompt_loaded source=cache")
                return self._prompt_cache_value, "cache"

        # Compute outside lock to avoid blocking unrelated callers.
        try:
            items = model.create_voice_clone_prompt(
                ref_audio=str(source_wav),
                ref_text=None,
                x_vector_only_mode=True,
            )
        except TypeError:
            # Older API variants: fall back to passing ref_text if required.
            items = model.create_voice_clone_prompt(
                ref_audio=str(source_wav),
                ref_text="",
                x_vector_only_mode=True,
            )

        # Convert to the dict form `generate_voice_clone(..., voice_clone_prompt=...)` accepts.
        # This avoids needing qwen-tts's internal VoiceClonePromptItem class outside this call.
        prompt: dict[str, Any] = {
            "ref_code": [getattr(it, "ref_code", None) for it in (items or [])],
            "ref_spk_embedding": [getattr(it, "ref_spk_embedding", None) for it in (items or [])],
            "x_vector_only_mode": [bool(getattr(it, "x_vector_only_mode", True)) for it in (items or [])],
            "icl_mode": [bool(getattr(it, "icl_mode", False)) for it in (items or [])],
        }
        prompt = self._move_prompt_to_device(prompt, device=device)

        # If this is a saved voice, persist for next time.
        self._maybe_persist_prompt_pt(source_wav=source_wav, voice_clone_prompt=prompt)

        with self._prompt_cache_lock:
            self._prompt_cache_key = cache_key
            self._prompt_cache_value = prompt
        return prompt, "computed"

    def _get_voice_clone_prompt(self, model: Any, *, source_wav: Path) -> dict[str, Any]:
        prompt, _src = self._get_voice_clone_prompt_with_source(model, source_wav=source_wav)
        return prompt

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
            t_prompt0 = None
            try:
                t_prompt0 = __import__("time").perf_counter()
            except Exception:
                t_prompt0 = None

            voice_clone_prompt, prompt_source = self._get_voice_clone_prompt_with_source(m, source_wav=source_wav)

            t_prompt_ms = None
            if t_prompt0 is not None:
                try:
                    t_prompt_ms = int((__import__("time").perf_counter() - t_prompt0) * 1000)
                except Exception:
                    t_prompt_ms = None

            t_gen0 = None
            try:
                t_gen0 = __import__("time").perf_counter()
            except Exception:
                t_gen0 = None

            gen_kwargs: dict[str, Any] = {}
            if self._settings.qwen_max_new_tokens is not None and self._settings.qwen_max_new_tokens > 0:
                gen_kwargs["max_new_tokens"] = int(self._settings.qwen_max_new_tokens)
            if getattr(self._settings, "qwen_non_streaming_mode", False):
                gen_kwargs["non_streaming_mode"] = True

            try:
                out = m.generate_voice_clone(
                    text=text,
                    language="Auto",
                    voice_clone_prompt=voice_clone_prompt,
                    **gen_kwargs,
                )
            except TypeError:
                # Some versions accept ref_audio directly.
                out = m.generate_voice_clone(
                    text=text,
                    language="Auto",
                    ref_audio=str(source_wav),
                    ref_text=None,
                    **gen_kwargs,
                )

            t_gen_ms = None
            if t_gen0 is not None:
                try:
                    t_gen_ms = int((__import__("time").perf_counter() - t_gen0) * 1000)
                except Exception:
                    t_gen_ms = None

            if getattr(self._settings, "log_qwen_timings", False):
                self._log.info(
                    "qwen_tts_timing prompt_source=%s prompt_ms=%s generate_ms=%s",
                    prompt_source,
                    t_prompt_ms if t_prompt_ms is not None else "-",
                    t_gen_ms if t_gen_ms is not None else "-",
                )
            else:
                self._log.debug(
                    "qwen_tts_timing prompt_source=%s prompt_ms=%s generate_ms=%s",
                    prompt_source,
                    t_prompt_ms if t_prompt_ms is not None else "-",
                    t_gen_ms if t_gen_ms is not None else "-",
                )

            return out

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
        voice_clone_prompt = self._get_voice_clone_prompt(model, source_wav=source_wav)

        # Persist a torch-serialized prompt for reuse across restarts.
        try:
            import torch

            pt_path = source_wav.parent / "embedding.pt"
            payload = {
                "type": "qwen3_voice_clone_prompt_pt_v1",
                "model_id": self._settings.qwen_model_id,
                "voice_clone_prompt": self._move_prompt_to_device(voice_clone_prompt, device="cpu"),
            }
            torch.save(payload, str(pt_path))
        except Exception:
            # Non-fatal; prompt will still be cached in-memory and can be backfilled
            # during generation.
            pass

        # Keep embedding.json lightweight and stable.
        return {
            "type": "qwen3_voice_clone_prompt_pt_v1",
            "model_id": self._settings.qwen_model_id,
            "prompt_file": "embedding.pt",
        }


def build_backend(settings: Settings) -> TTSBackend:
    backend = settings.tts_backend
    if backend == "mock":
        return MockBackend()
    if backend == "qwen":
        return QwenTTSBackend(settings=settings)
    raise TTSError(f"Unknown TTS_BACKEND: {backend}")
