from __future__ import annotations

import re
from typing import Any, Mapping


_AUTH_RE = re.compile(r"\b(bearer)\s+[^\s]+", re.IGNORECASE)


def truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def redact_auth(value: str) -> str:
    # Redact bearer tokens if they appear inside loggable strings.
    return _AUTH_RE.sub(r"\\1 ***", value)


def safe_preview_payload(payload: Any, *, limit_chars: int) -> str:
    """Human-readable, redacted, truncated representation for logs."""

    try:
        if isinstance(payload, str):
            return truncate(redact_auth(payload), limit_chars)
        if isinstance(payload, bytes):
            return f"<bytes len={len(payload)}>"
        if isinstance(payload, Mapping):
            # Avoid dumping huge nested structures; keep top-level keys.
            keys = list(payload.keys())
            return truncate(f"<dict keys={keys}>", limit_chars)
        return truncate(redact_auth(str(payload)), limit_chars)
    except Exception:
        return "<unloggable>"
