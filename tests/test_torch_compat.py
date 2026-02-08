from __future__ import annotations

from app.torch_compat import torch_arch_mismatch_hint


def test_torch_arch_mismatch_hint_detects_common_error() -> None:
    e = RuntimeError("CUDA error: no kernel image is available for execution on the device")
    hint = torch_arch_mismatch_hint(e)
    assert hint is not None
    assert "GPU not supported" in hint.title
    assert "cu128" in hint.suggested_action


def test_torch_arch_mismatch_hint_ignores_unrelated_errors() -> None:
    e = RuntimeError("something else")
    assert torch_arch_mismatch_hint(e) is None
