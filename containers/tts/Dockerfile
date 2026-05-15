# Qwen3-TTS voice cloning container for NVIDIA GB10 (aarch64, Blackwell)
#
# Uses faster-qwen3-tts with CUDA graph optimization for low-latency synthesis.
# OpenAI-compatible /v1/audio/speech endpoint with voice cloning support.
#
# Base image starts from NVIDIA 25.01, then pins the PyTorch/TorchVision
# wheel stack that passed the 2026-05-09 GB10 saved-meeting TTS benchmark.
#
# transformers is pinned <5 because qwen-tts 0.1.1 (a transitive dep
# of faster-qwen3-tts) has Requires-Dist: transformers==4.57.3.
# Revisit when qwen-tts ships a v5-compatible release, or when we
# drop the qwen-tts fallback path.
#
# Build:
#   docker build -t scribe-tts:latest containers/tts/
#
# Run:
#   docker run -d --gpus all --network host --name scribe-tts \
#     -e TTS_PORT=8002 \
#     -e TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
#     -v /data/huggingface:/root/.cache/huggingface \
#     scribe-tts:latest

FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

# Fix torchvision circular import and move TTS onto a GB10/Blackwell-capable
# PyTorch wheel stack. Keep these pins explicit; loose torchvision pulled a
# new CUDA stack once and should not float again on future rebuilds.
RUN pip install --no-cache-dir --force-reinstall \
    'numpy==1.26.4' \
    'torch==2.11.0' \
    'torchvision==0.26.0' \
    'torchaudio==2.11.0'

# Install faster-qwen3-tts and server dependencies with explicit pins.
RUN pip install --no-cache-dir \
    'transformers>=4.57,<5' \
    'faster-qwen3-tts==0.2.6' \
    'qwen-tts==0.1.1' \
    'fastapi==0.135.3' \
    'uvicorn==0.44.0' \
    'soundfile==0.13.1' \
    'httpx==0.28.1'

COPY server.py voice_male.wav voice_female.wav /app/

ENV TTS_PORT=8002
ENV TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base

EXPOSE 8002

CMD ["python", "server.py"]
