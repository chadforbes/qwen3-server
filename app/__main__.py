from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=max(1, int(settings.uvicorn_workers or 1)),
    )


if __name__ == "__main__":
    main()
