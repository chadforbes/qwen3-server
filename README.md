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

To run with multiple workers (note: `--reload` doesn’t combine well with multi-worker), prefer:

```powershell
$env:UVICORN_WORKERS="2"
python -m app
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
- `LOG_FORMAT` = `text` (default) or `json`
- `LOG_PAYLOAD_CHARS` = `500` (default)
- `UVICORN_WORKERS` = `1` (default)

GPU / Torch selection:

- `DEVICE` = `auto` (default) | `cpu` | `cuda` | `cuda:0`
- `DEVICE_MAP` = `auto` (default) | `cpu` | `cuda`
- `TORCH_DTYPE` = `auto` (default) | `float32` | `float16` | `bfloat16`

Notes:

- With `DEVICE=auto`, the server will use `cuda` if `torch.cuda.is_available()` is true, otherwise CPU.
- With CUDA, `TORCH_DTYPE=auto` will prefer `float16` to reduce VRAM usage.

- First run will download the model from Hugging Face.
- `embedding.json` stores a Qwen "voice clone prompt" when using `qwen`.
- On Windows, `qwen-tts` may require the SoX binary (`sox`) on your `PATH`.

If you run into Torch install issues on CPU-only machines, install the CPU wheel explicitly:

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Run (Docker)

```powershell
docker build -t qwen3-server .
docker run --rm -p 8000:8000 -v ${PWD}\audio:/app/audio qwen3-server
```

### Run (Docker + NVIDIA GPU)

You need:

- NVIDIA GPU drivers installed on the host
- NVIDIA Container Toolkit installed (so Docker can expose the GPU)
- A CUDA-enabled PyTorch wheel inside the image (see note below)

Example:

```powershell
docker run --rm -p 8000:8000 -v ${PWD}\audio:/app/audio --gpus all -e DEVICE=cuda -e TORCH_DTYPE=float16 qwen3-server
```

> Note: the current `Dockerfile` is CPU-oriented (`python:3.11-slim`). For best GPU support, we should add a separate CUDA base image (e.g., `nvidia/cuda:*`) or a `Dockerfile.gpu` variant and install the matching CUDA PyTorch wheel.

## Endpoints

- `POST /preview` (multipart) → Upload a source audio file, its transcription, and the desired response text. Returns a synthesized preview audio file in the cloned voice. (Replaces `/upload`)
- `GET /previews/{job_id}.wav` → (optional) serve preview WAV by job id (only if enabled in code)
- `GET /health` → ok
- `WS /ws` → JSON messages (see below)

### `/preview` endpoint usage

**Request:**

`POST /preview`

Form-data (multipart):
- `audio`: WAV file (reference voice)
- `transcription`: Text transcription of the audio
- `response_text`: The text to synthesize in the cloned voice

**Response:**
- Returns a WAV audio file containing the synthesized response in the cloned voice.

**Example (using curl):**

```sh
curl -X POST http://localhost:8000/preview \
	-F "audio=@source.wav" \
	-F "transcription=Hello, this is my voice." \
	-F "response_text=How can I help you today?" \
	--output preview.wav
```

---

> **Note:** The old `/upload` endpoint is now deprecated. Use `/preview` for uploading audio and generating previews in a single step.

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
python -m pytest -q
```

Clean test artifacts:

```powershell
./scripts/clean_test_artifacts.ps1
```
