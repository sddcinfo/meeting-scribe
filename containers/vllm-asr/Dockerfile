# scribe-vllm-asr — stock vLLM + audio extras for Qwen3-ASR.
#
# vllm/vllm-openai:latest does not ship vllm[audio] (soundfile, librosa,
# av) — Qwen3-ASR's input_audio messages return "Please install
# vllm[audio] for audio support" without them. This image is a thin
# layer on top of the upstream image that adds those extras.
#
# Translation runs on stock vllm/vllm-openai (no audio path). Only the
# ASR side needs this derivative.

FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir 'vllm[audio]'
