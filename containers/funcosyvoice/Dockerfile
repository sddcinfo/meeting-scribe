# Fun-CosyVoice3.5 (FunAudioLLM/CosyVoice3.5) sidecar for the
# 2026-Q2 model-challenger bench.  See plans/stateful-marinating-whistle.md
# Track B.  Apache 2.0; not a production replacement until the bench
# clears L-TTFA + Q-MOS + Q-CLONE gates.
#
# Base image follows the qwen3-tts container's GB10 / CUDA-12.8 /
# driver-580 compatibility constraint (NGC PyTorch 25.01).  Pinned to
# its sha256 digest at build time per S3.
#
# Build:
#   docker build \
#     --build-arg COSY_REVISION=<pinned-hf-sha> \
#     -t funcosyvoice:bench containers/funcosyvoice/
#
# Run (under the bench compose profile):
#   docker compose --profile bench up -d funcosyvoice

FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --force-reinstall torchvision

# CosyVoice 3.5 ships its own openai-compatible /v1/audio/speech shim
# (see FunAudioLLM/CosyVoice repo, openai_server.py).  We pull it via
# pip; the model itself is downloaded on first start unless the HF
# cache is mounted.  Pin the upstream package version conservatively
# until upstream's own release cadence stabilises.
RUN pip install --no-cache-dir \
    'transformers>=4.57,<5' \
    'fun-cosyvoice==3.5.0' \
    'fastapi==0.135.3' \
    'uvicorn==0.44.0' \
    'soundfile==0.13.1' \
    'numpy==2.4.4'

# Pinned per S3.  Override at build time:
#   docker build --build-arg COSY_REVISION=<sha> ...
ARG COSY_REVISION=""
ENV COSY_REVISION=${COSY_REVISION}
ENV COSY_MODEL_ID=FunAudioLLM/CosyVoice3.5
ENV TTS_PORT=8024

EXPOSE 8024

# fun-cosyvoice ships an openai-compatible server entry point.  If the
# upstream module path changes, switch to a custom server.py adapter.
CMD ["python", "-m", "fun_cosyvoice.openai_server", "--host", "0.0.0.0", "--port", "8024"]
