from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class _UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()

    # Avoid duplicating handlers on reload.
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _UTCFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    )

    root.setLevel(level)
    root.addHandler(handler)

    # Keep common noisy loggers at or above root.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)
