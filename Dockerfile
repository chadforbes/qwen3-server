FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUDIO_ROOT=/app/audio

COPY requirements.txt /app/requirements.txt

# Torch install strategy (single Dockerfile):
# - Default: CPU wheel
# - Optional: CUDA wheel via build args
#
# Examples:
#   docker build -t qwen3-server .
#   docker build -t qwen3-server:cu121 --build-arg TORCH_VARIANT=cuda --build-arg TORCH_CUDA=cu121 .
ARG TORCH_VARIANT=cpu
ARG TORCH_CUDA=cu121
ARG QWEN_TTS_VERSION=0.0.5

# qwen-tts uses a number of optional deps; our server backend requires these at
# import/runtime. We intentionally avoid installing gradio (UI) here.
ARG QWEN_TTS_EXTRAS="einops librosa soundfile sox onnxruntime"

# Install torch based on build args.
# NOTE: CUDA builds require an NVIDIA runtime + compatible host driver.
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/$TORCH_CUDA torch torchaudio; \
    else \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio; \
    fi

# Install qwen-tts without its optional heavy deps (we install torch/torchaudio
# explicitly above so the wheels match the selected torch variant), then install
# the rest of the pinned requirements.
RUN pip install --no-cache-dir --no-deps qwen-tts==${QWEN_TTS_VERSION} \
 && pip install --no-cache-dir ${QWEN_TTS_EXTRAS} \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000

CMD ["python", "-m", "app"]
