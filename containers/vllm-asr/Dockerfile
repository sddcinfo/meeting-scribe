# scribe-vllm-asr — stock vLLM + audio extras for Qwen3-ASR.
#
# vllm/vllm-openai does not ship vllm[audio] (soundfile, librosa, av) —
# Qwen3-ASR's input_audio messages return "Please install vllm[audio]
# for audio support" without them. This image is a thin layer on top
# of the upstream image that adds those extras.
#
# Translation runs on stock vllm/vllm-openai (no audio path). Only the
# ASR side needs this derivative.
#
# Pin contract — see containers/vllm-asr/README.md for the full
# version-pin rationale and bump procedure. Bumping this line
# requires re-running ASR end-to-end smoke and confirming
# `gpu_memory_utilization=0.04` in recipes/qwen3-asr-vllm.yaml still
# boots model + KV cache + encoder cache against the new vllm version.

# Pinned: v0.20.0 aarch64+cu130 (validated 2026-04-30 on GB10 against
# the production-tested gpu_memory_utilization=0.10). Bumping this pin
# requires re-running ASR end-to-end smoke and the procedure documented
# in containers/vllm-asr/README.md.
FROM vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404@sha256:f81415d5682a03566d92dc46e883d73a6ac001e6ef1fc4293512269f1372b8b9

# vllm[audio]          — soundfile/librosa/av for Qwen3-ASR's input_audio messages.
# vllm[fastsafetensors] — required by `--load-format fastsafetensors` in the
#                         compose command. Defensive: even though v0.20.0
#                         bundles this, an explicit install protects against
#                         future base-image changes that drop the dependency.
#                         Memory note: fastsafetensors is the ~19x model-load
#                         speedup on GB10's NVMe and must not be dropped.
RUN pip install --no-cache-dir 'vllm[audio]' 'vllm[fastsafetensors]'
