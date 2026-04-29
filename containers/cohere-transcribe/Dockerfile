# Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026) sidecar
# for the 2026-Q2 model-challenger bench.  See plans/stateful-marinating-whistle.md
# Track A.  Apache 2.0; not a production replacement until the bench
# clears Q-JA-corpus and L-live gates.
#
# Base image follows the pyannote container's GB10 / CUDA-12.8 /
# driver-580 compatibility constraint (NGC PyTorch 25.01).  Pinned to
# its sha256 digest at build time per S3.
#
# Build:
#   docker build \
#     --build-arg COHERE_REVISION=<pinned-hf-sha> \
#     -t cohere-transcribe:bench containers/cohere-transcribe/
#
# Run (under the bench compose profile):
#   docker compose --profile bench up -d cohere-transcribe

FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --force-reinstall torchvision

# Transformers loads the model; FastAPI fronts it with the same
# /v1/chat/completions shape that benchmarks/asr_accuracy_latency.py
# already calls (zero harness change).
RUN pip install --no-cache-dir \
    'transformers>=5.4,<6' \
    'accelerate>=0.30' \
    'fastapi==0.135.3' \
    'uvicorn==0.44.0' \
    'soundfile==0.13.1' \
    'librosa==0.11.0' \
    'numpy==2.4.4' \
    'sentencepiece' \
    'protobuf'

COPY server.py /app/server.py

# Pinned per S3.  Override at build time:
#   docker build --build-arg COHERE_REVISION=<sha> ...
ARG COHERE_REVISION=""
ENV COHERE_REVISION=${COHERE_REVISION}
ENV COHERE_MODEL_ID=CohereLabs/cohere-transcribe-03-2026
ENV ASR_PORT=8013

EXPOSE 8013

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8013"]
