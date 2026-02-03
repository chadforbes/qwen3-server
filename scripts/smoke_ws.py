from __future__ import annotations

import asyncio
import json

import httpx
import websockets


async def main() -> None:
    # Run server separately (see README). This script assumes localhost:8000.
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        r = await client.post("/upload", files={"file": ("source.wav", b"RIFF....WAVE", "audio/wav")})
        r.raise_for_status()
        session_id = r.json()["session_id"]

    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        await ws.send(json.dumps({"type": "generate_preview", "data": {"session_id": session_id, "text": "Hello"}}))
        print(await ws.recv())

        await ws.send(json.dumps({"type": "save_voice", "data": {"session_id": session_id, "name": "Nova", "description": "Warm"}}))
        print(await ws.recv())


if __name__ == "__main__":
    asyncio.run(main())
