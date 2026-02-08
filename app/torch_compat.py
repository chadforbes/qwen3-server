from __future__ import annotations

"""Torch compatibility helpers.

We want to turn the most common GPU incompatibility failures (e.g. RTX 50-series
sm_120 on a torch wheel that only includes up to sm_90) into actionable messages.

These errors frequently show up as:
- RuntimeError: CUDA error: no kernel image is available for execution on the device
- RuntimeError: no kernel image is available for execution on the device
- messages mentioning 'sm_120' / 'compute capability'

The exact wording varies by PyTorch/CUDA version, so we match a few patterns.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional


_ARCH_MISMATCH_RE = re.compile(
    r"(no kernel image is available for execution on the device|"
    r"sm_\d+|compute capability|"
    r"the provided PTX was compiled with an unsupported toolchain)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TorchCompatHint:
    title: str
    message: str
    suggested_action: str


def torch_arch_mismatch_hint(exc: BaseException) -> Optional[TorchCompatHint]:
    msg = str(exc)
    if not msg:
        return None

    if not _ARCH_MISMATCH_RE.search(msg):
        return None

    # Users can override these in their own builds; keep them as envs.
    suggested_tag = os.getenv("SUGGESTED_CUDA_TAG", "cu128")

    return TorchCompatHint(
        title="GPU not supported by this PyTorch build",
        message=(
            "Your NVIDIA GPU is newer than the CUDA kernels packaged in this image's PyTorch wheel. "
            "This commonly happens on RTX 50-series (sm_120) when using older CUDA wheels (e.g. cu121)."
        ),
        suggested_action=(
            "Use an image built with PyTorch CUDA 12.8+ wheels (e.g. tag ':{tag}'), "
            "or run on CPU by setting DEVICE=cpu."
        ).format(tag=suggested_tag),
    )
