# Qwen TTS Server (voice cloning + saved voices)

This is a small FastAPI server that wraps Qwen3 TTS for a simple workflow:

1) Upload reference audio + transcript and generate a preview
2) If you like it, save that voice for later
3) Generate future previews from the saved voice without re-uploading

> Current model: **single-user / no sessions**. The server always operates on the
> most recently uploaded reference audio under `audio/uploads/latest/`.

## Quickstart (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch first (pick CPU or CUDA):
pip install --index-url https://download.pytorch.org/whl/cpu torch
# For CUDA (example):
# pip install --index-url https://download.pytorch.org/whl/cu121 torch

pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

To run with multiple workers (note: `--reload` doesn’t combine well with multi-worker), prefer:

```powershell
$env:UVICORN_WORKERS="2"
python -m app
```

> Note: with `UVICORN_WORKERS>1`, the server is still usable, but the **no-session
> “latest upload wins”** model can behave unexpectedly under concurrent users.
> This repo intentionally optimizes for a single real user right now.

## Backend: Qwen3 TTS

This server runs Qwen3 TTS via `qwen-tts` (default backend).

This implementation uses Qwen in **voice cloning** mode only.

- Latest reference audio:
	- `audio/uploads/latest/source.wav`
	- `audio/uploads/latest/transcription.txt`
- Generating previews uses the latest reference
- Saving a voice persists the reference under `audio/voices/<voice_id>/`

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

If you run into Torch install issues, install the appropriate wheel explicitly (CPU or CUDA) as shown above.

## Quickstart (Docker)

```powershell
docker build -t qwen3-server .
docker run --rm -p 8000:8000 -v ${PWD}\audio:/app/audio qwen3-server
```

### Docker build: CPU vs GPU (single Dockerfile)

This repo uses a **single** `Dockerfile`.

- Default build installs **CPU** PyTorch.
- To build a CUDA-enabled image, pass build args to install the CUDA PyTorch wheel.

CPU (default):

```powershell
docker build -t qwen3-server .
```

CUDA (example: cu121):

```powershell
docker build -t qwen3-server:cu121 --build-arg TORCH_VARIANT=cuda --build-arg TORCH_CUDA=cu121 .
```

### Run (Docker + NVIDIA GPU)

You need:

- NVIDIA GPU drivers installed on the host
- NVIDIA Container Toolkit installed (so Docker can expose the GPU)
- A CUDA-enabled PyTorch wheel inside the image (see note below)

Example:

```powershell
docker run --rm -p 8000:8000 -v ${PWD}\audio:/app/audio --gpus all -e DEVICE=cuda -e TORCH_DTYPE=float16 qwen3-server:cu121
```

> Note: The base image is still `python:3.11-slim`. CUDA-enabled PyTorch inside the container still requires:
> - NVIDIA drivers on the host
> - NVIDIA Container Toolkit
> - `docker run --gpus all ...`

## Endpoints

### HTTP

- `GET /health` → `{ "status": "ok" }`
- `GET /voices` → list saved voices (metadata)
- `POST /preview` (multipart) → upload reference audio + transcript and generate a preview WAV
- `POST /preview-from-voice` (multipart form) → generate a preview from a saved `voice_id`

### WebSocket

- `WS /ws` → JSON messages (`generate_preview`, `save_voice`)

## End-to-end workflow

### 1) Upload + preview (HTTP)

`POST /preview` multipart form-data:

- `audio`: WAV file (reference voice)
- `transcription`: text transcript of the audio
- `response_text`: the text to synthesize in the cloned voice

Response:

- HTTP 200
- `Content-Type: audio/wav`
- Body is the preview WAV bytes

Example:

```sh
curl -X POST http://localhost:8000/preview \
	-F "audio=@source.wav" \
	-F "transcription=Hello, this is my voice." \
	-F "response_text=How can I help you today?" \
	--output preview.wav
```


### 2) Save voice (WebSocket)

Once you like what you heard, save the current latest reference voice:

Send:

```json
{ "type": "save_voice", "data": { "name": "Nova", "description": "Warm" } }
```

Receive:

```json
{ "type": "voice_saved", "data": { "voice_id": "nova", "name": "Nova", "description": "Warm" } }
```

Notes:

- The server always saves from the most recent upload: `audio/uploads/latest/*`.
- If there is no latest upload yet, call `POST /preview` first.

After saving, you can:

- list via `GET /voices`
- preview via `POST /preview-from-voice`

When a voice is saved, the server will also persist `transcription.txt` into the voice folder if it exists in `uploads/latest/`.

### 3) Preview from a saved voice (HTTP)

`POST /preview-from-voice` multipart form-data:

- `voice_id`: the saved voice id (e.g., `nova`)
- `response_text`: the text to synthesize

Example using a saved voice:

```sh
curl -X POST http://localhost:8000/preview-from-voice \
  -F "voice_id=nova" \
  -F "response_text=How can I help you today?" \
  --output preview.wav
```
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

## WebSocket messages

### Generate preview

Send:

```json
{ "type": "generate_preview", "data": { "text": "Hello" } }
```

Receive:

```json
{ "type": "tts_complete", "data": { "job_id": "abc123", "audio_url": "/previews/abc123.wav", "temporary": true } }
```

### Save voice

See the workflow section above.

## Files on disk

The server writes under `AUDIO_ROOT` (default `./audio`):

- `audio/uploads/latest/`
	- `source.wav` (most recent reference audio)
	- `transcription.txt` (most recent transcript)
	- `meta.json` (timestamps + sizes)
- `audio/previews/`
	- `preview-latest.wav` (last HTTP `/preview` output)
	- `<job_id>.wav` (WS `generate_preview` outputs, cleaned up periodically)
- `audio/voices/<voice_id>/`
	- `source.wav`
	- `transcription.txt` (if available)
	- `metadata.json`
	- `embedding.json`

## Tests

```powershell
python -m pytest -q
```

Clean test artifacts:

```powershell
./scripts/clean_test_artifacts.ps1
```
