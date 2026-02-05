from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from .log_context import get_correlation_id


class _UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        # Ensure every log record has a correlation id for easier tracing.
        record.correlation_id = get_correlation_id() or "-"
        return True


class _JsonFormatter(_UTCFormatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        # Include exception info if present.
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        import json

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, fmt: str = "text") -> None:
    root = logging.getLogger()

    # Avoid duplicating handlers on reload.
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if (fmt or "text").lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            _UTCFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s [cid=%(correlation_id)s]: %(message)s",
            )
        )

    root.setLevel(level)
    root.addHandler(handler)

    # Keep common noisy loggers at or above root.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)

