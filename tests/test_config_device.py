from __future__ import annotations


from app.config import get_settings


def test_device_env_parsing(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("DEVICE_MAP", "cuda")
    monkeypatch.setenv("TORCH_DTYPE", "float16")

    s = get_settings()
    assert s.device == "cuda"
    assert s.device_map == "cuda"
    assert s.torch_dtype == "float16"
