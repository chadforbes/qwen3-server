from __future__ import annotations


def is_cuda_device_side_assert(exc: BaseException) -> bool:
    """Detect common CUDA device-side assert failures from exception text.

    These errors often leave the CUDA context in a bad state for the process,
    so the operational advice is typically to restart the container.

    We intentionally avoid importing torch here to keep this helper safe to
    import in CPU-only environments.
    """

    msg = str(exc).lower()
    if "device-side assert" in msg:
        return True
    if "cudaerrorassert" in msg:
        return True
    if type(exc).__name__ == "AcceleratorError" and "cuda" in msg:
        return True
    return False


def cuda_assert_payload(exc: BaseException) -> dict[str, str]:
    return {
        "error": "CUDA device-side assert triggered",
        "hint": "This usually requires restarting the server container. Try TORCH_DTYPE=bfloat16 (or float32) and ensure your CUDA torch wheel matches the GPU/driver. For debugging, set CUDA_LAUNCH_BLOCKING=1.",
        "details": str(exc),
    }
