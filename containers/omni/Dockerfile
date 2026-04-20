# Qwen3-Omni-30B-A3B unified container (ASR + TTS + text in one model).
#
# Consolidation target per plan `vLLM Consolidation + Omni Spike`. Mirrors
# the TTS spike Dockerfile's supply-chain pattern; production cutover
# replaces this with a wheelhouse+lockfile consumer after the first
# successful boot.
#
# Build (HF_MODEL_REVISION is required — no default — so reproducibility is
# enforced at build time):
#   docker build --build-arg HF_MODEL_REVISION=<sha> -t scribe-omni:spike \
#       containers/omni/

ARG VLLM_BASE_TAG=v0.18.0
FROM vllm/vllm-openai:${VLLM_BASE_TAG}

# Required — no default. `docker build` fails if the caller doesn't pass it.
ARG HF_MODEL_REVISION
ARG VLLM_OMNI_SHA=f55ea28005f889aa5f10d4ceb694573c564da127
ARG HF_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct

RUN test -n "$HF_MODEL_REVISION" || (echo "HF_MODEL_REVISION build arg required" && exit 1)

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg git sox libsox-fmt-all jq curl zstd ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth=1 --branch v0.18.0 https://github.com/vllm-project/vllm-omni.git /workspace/vllm-omni && \
    cd /workspace/vllm-omni && \
    git rev-parse HEAD > /workspace/vllm-omni-sha.txt && \
    sed -i '/fa3-fwd/d' requirements/cuda.txt

RUN cd /workspace/vllm-omni && uv pip install --system --no-cache-dir ".[dev]"

RUN python3 -c "import vllm_flash_attn; print('vllm_flash_attn OK:', vllm_flash_attn.__version__)" || \
    (pip install --no-cache-dir packaging ninja && \
     pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1)

# Prefetch the pinned HF model revision into the image. No runtime hub access.
RUN pip install --no-cache-dir 'huggingface_hub[cli]>=0.26' && \
    huggingface-cli download ${HF_MODEL_ID} \
        --revision ${HF_MODEL_REVISION} \
        --local-dir /models/qwen3-omni-30b-a3b

# Vendor the stage config at a known path (no dependency on vllm-omni source drift).
RUN mkdir -p /opt/stage_configs && \
    cp /workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml \
       /opt/stage_configs/qwen3_omni_moe.yaml

EXPOSE 8032

# Unified Omni instance serves /v1/audio/speech (TTS), /v1/audio/transcriptions
# (ASR), /v1/chat/completions (text + translate).
CMD ["vllm", "serve", "/models/qwen3-omni-30b-a3b", \
     "--stage-configs-path", "/opt/stage_configs/qwen3_omni_moe.yaml", \
     "--omni", "--port", "8032", \
     "--trust-remote-code", "--enforce-eager", \
     "--gpu-memory-utilization", "0.30"]
