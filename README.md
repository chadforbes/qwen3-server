# Qwen TTS Server (Voice Lifecycle)

Implements the voice creation + preview lifecycle:

- Upload reference audio (temporary, 12h retention unless saved)
- Generate preview audio (temporary, 20m retention)
- Save voice (moves upload to permanent voice folder + writes metadata/embedding)
- Background asyncio cleanup tasks

## Run (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Qwen3 TTS backend

This server runs Qwen3 TTS via `qwen-tts` (default backend).

This implementation uses Qwen in **voice cloning** mode only:

- The uploaded `audio/uploads/<session_id>/source.wav` is the reference voice.
- `generate_preview` synthesizes speech in the cloned voice.
- `save_voice` persists a Qwen voice-clone prompt into `audio/voices/<voice_id>/embedding.json`.

Environment variables:

- `TTS_BACKEND` = `qwen` (default) or `mock`
- `QWEN_TTS_MODEL_ID` = `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (default)
- `PRELOAD_MODEL_ON_STARTUP` = `1` (default) to download/load model during startup
- `LOG_LEVEL` = `INFO` (default)
- `CORS_ALLOW_ORIGINS` = `*` (default). Comma-separated origins (e.g. `http://homeassistant.local:8123`). Set to `none` to disable.

Notes:

- First run will download the model from Hugging Face.
- `embedding.json` stores a Qwen "voice clone prompt" when using `qwen`.
- On Windows, `qwen-tts` may require the SoX binary (`sox`) on your `PATH`.

If you run into Torch install issues on CPU-only machines, install the CPU wheel explicitly:

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Run (Docker)

```powershell
docker build -t qwen-tts-server .
docker run --rm -p 8000:8000 -v ${PWD}\audio:/app/audio qwen-tts-server
```

## Endpoints

- `POST /upload` (multipart) → creates `session_id` and stores `source.wav` under `audio/uploads/<session_id>/source.wav`
- `GET /previews/{job_id}.wav` → serves preview WAV if not expired
- `GET /health` → ok
- `WS /ws` → JSON messages (see below)

Compatibility aliases (same behavior):

- `GET /api/health`
- `POST /api/upload`
- `GET /api/previews/{job_id}.wav`
- `WS /api/ws`

## WebSocket messages

### Generate preview

Send:

```json
{ "type": "generate_preview", "data": { "session_id": "...", "text": "Hello" } }
```

Receive:

```json
{ "type": "tts_complete", "data": { "job_id": "abc123", "audio_url": "/previews/abc123.wav", "temporary": true } }
```

### Save voice

Send:

```json
{ "type": "save_voice", "data": { "session_id": "...", "name": "Nova", "description": "Warm" } }
```

Receive:

```json
{ "type": "voice_saved", "data": { "voice_id": "nova", "name": "Nova", "description": "Warm" } }
```

## Tests

```powershell
pytest -q
```

Clean test artifacts:

```powershell
./scripts/clean_test_artifacts.ps1
```
