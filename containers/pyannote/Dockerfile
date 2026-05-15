# pyannote speaker diarization container for NVIDIA GB10 (aarch64, Blackwell)
#
# Uses pyannote.audio 4.x with Blackwell SM_121 compatibility patches.
# Based on NVIDIA PyTorch container (ARM64 native).
#
# Base image pinned at 25.01-py3 (CUDA 12.8) because the GB10 driver
# (580.126.09) reports CUDA 13.0 as max runtime. NGC 26.03-py3 bundles
# CUDA 13.2 which may not work on this driver. Revisit after driver
# upgrade.
#
# Build:
#   docker build -t pyannote-diarize:latest containers/pyannote/
#
# Run:
#   docker run -d --gpus all --network host --name scribe-diarization \
#     -e HF_TOKEN=$HF_TOKEN \
#     -e DIARIZE_MAX_SPEAKERS=4 \
#     -v /data/huggingface:/root/.cache/huggingface \
#     pyannote-diarize:latest

FROM nvcr.io/nvidia/pytorch:25.01-py3

# pyannote 4.x dropped sox/soundfile audio backends; only ffmpeg/torchcodec
# work. NGC PyTorch 25.01 may not ship ffmpeg by default.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix torchvision circular import (NVIDIA PyTorch 25.01 issue)
RUN pip install --no-cache-dir --force-reinstall torchvision

# Install pyannote.audio 4.x and server dependencies. pyannote 4.x pulls
# transformers as a transitive dep; we don't upper-bound it here because
# pyannote 4 is transformers-v5-compatible upstream.
RUN pip install --no-cache-dir \
    'pyannote.audio==4.0.4' \
    'fastapi==0.135.3' \
    'uvicorn==0.44.0' \
    'soundfile==0.13.1'

COPY server.py /app/server.py

ENV DIARIZE_PORT=8001
ENV DIARIZE_MAX_SPEAKERS=4

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
