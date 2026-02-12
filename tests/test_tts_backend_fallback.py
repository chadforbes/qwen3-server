from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.config import Settings
from app.preview_audio import write_sine_wav
from app.tts_backend import QwenTTSBackend


class _FailingCudaModel:
    def create_voice_clone_prompt(self, **kwargs):
        return {"prompt": "ok"}

    def generate_voice_clone(self, **kwargs):
        raise RuntimeError("CUDA error: device-side assert triggered")


class _InvalidProbModel:
    def create_voice_clone_prompt(self, **kwargs):
        return {"prompt": "ok"}

    def generate_voice_clone(self, **kwargs):
        raise RuntimeError(
            "probability tensor contains either `inf`, `nan` or element < 0"
        )


class _CpuOkModel:
    def create_voice_clone_prompt(self, **kwargs):
        return {"prompt": "ok"}

    def generate_voice_clone(self, **kwargs):
        # The backend writes via soundfile; we stub soundfile in the test.
        return ([[0.0, 0.1, 0.0]], 22050)


@pytest.fixture(autouse=True)
def _stub_soundfile(monkeypatch, tmp_path: Path):
    class _SF:
        @staticmethod
        def write(path: str, data, sr: int):
            # Create a placeholder output so the backend succeeds.
            Path(path).write_bytes(b"RIFFxxxxWAVE")

    monkeypatch.setitem(sys.modules, "soundfile", _SF)


@pytest.mark.parametrize("failing_model", [_FailingCudaModel(), _InvalidProbModel()])
def test_synthesize_preview_falls_back_to_cpu(monkeypatch, tmp_path: Path, failing_model):
    settings = Settings(audio_root=tmp_path, cuda_fallback_to_cpu=True)
    backend = QwenTTSBackend(settings=settings)

    # Force first attempt to fail as if on CUDA.
    monkeypatch.setattr(backend, "_load_model", lambda: failing_model)

    cpu_model = _CpuOkModel()
    cpu_loaded = {"count": 0}

    def _load_cpu_model():
        cpu_loaded["count"] += 1
        return cpu_model

    monkeypatch.setattr(backend, "_load_cpu_model", _load_cpu_model)

    source = tmp_path / "source.wav"
    out = tmp_path / "out.wav"
    write_sine_wav(source)

    backend.synthesize_preview(text="hello", source_wav=source, out_wav=out)

    assert out.exists()
    assert cpu_loaded["count"] == 1
    assert backend._cuda_disabled is True
