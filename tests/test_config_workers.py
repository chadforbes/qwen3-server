from __future__ import annotations


from app.config import get_settings


def test_uvicorn_workers_default(monkeypatch):
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    s = get_settings()
    assert s.uvicorn_workers == 1


def test_uvicorn_workers_env(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "3")
    s = get_settings()
    assert s.uvicorn_workers == 3


def test_uvicorn_workers_minimum_1(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "0")
    s = get_settings()
    assert s.uvicorn_workers == 1
