from __future__ import annotations

import asyncio
import json

import requests
import websockets


async def main():
    # No-session flow:
    # 1) POST /preview to seed uploads/latest (reference audio + transcription)
    # 2) WS generate_preview uses uploads/latest
    # 3) WS save_voice persists from uploads/latest

    with open("./audio/uploads/test.wav", "rb") as f:
        r = requests.post(
            "http://localhost:8000/preview",
            files={"audio": ("test.wav", f, "audio/wav")},
            data={"transcription": "Hello", "response_text": "Hello"},
        )
    r.raise_for_status()
    # Response is audio/wav bytes
    print(f"/preview -> status={r.status_code} bytes={len(r.content)}")

    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({"type": "generate_preview", "data": {"text": "Hello"}}))
        msg = await ws.recv()
        print(msg)

        await ws.send(json.dumps({"type": "save_voice", "data": {"name": "Nova", "description": "Warm"}}))
        msg = await ws.recv()
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())
