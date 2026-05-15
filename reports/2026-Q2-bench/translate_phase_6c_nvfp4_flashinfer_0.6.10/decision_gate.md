# Track B.1 — NVFP4 + FlashInfer 0.6.10 (highest leverage, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

This is the highest-leverage track in the 2026-Q2 bench cycle.
FlashInfer 0.6.10 reportedly extends FP4 GEMM to SM_121 — the
Phase 6.B failure mode that forced the Marlin fallback and cost
~8 BLEU. If FlashInfer 0.6.10's NVFP4 path lands cleanly, this
recipe becomes the new translation primary.

## Pins (to fill at recipe-write time)

| Field | Value |
|---|---|
| `challenger_model` | `RedHatAI/Qwen3.6-35B-A3B-NVFP4 @ <40-char-sha>` |
| `baseline_model` | `Qwen/Qwen3.6-35B-A3B-FP8` (production) |
| `challenger_image` | `autosre-vllm-fi0.6.10:bench` (FROM `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404@sha256:<digest>`, `pip install flashinfer==0.6.10`) |
| `baseline_image` | autosre production (digest captured at bench-time) |
| `corpus` | EN→JA + JA→EN translation pairs from the existing benchmarks/fixtures/ |
| `bench_window` | TBD (set `MEETING_SCRIBE_BENCH_WINDOW=1` for the run) |

## Recipe (to land in `auto-sre/autosre/backends/recipes/`)

`qwen3.6-35b-a3b-nvfp4-fi0.6.10.yaml` — clone of `qwen3.6-35b-a3b-fp8.yaml` with:

* `model_id: RedHatAI/Qwen3.6-35B-A3B-NVFP4`
* `quantization: nvfp4`
* `--load-format=fastsafetensors`
* port 8015 (staging — never overlap production 8010)
* same env hardening (`VLLM_ALLOW_LONG_MAX_MODEL_LEN`,
  `CUBLAS_WORKSPACE_CONFIG`, `VLLM_MARLIN_USE_ATOMIC_ADD`)

## Dockerfile (`auto-sre/Dockerfile.vllm-fi0.6.10`)

`FROM vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404@sha256:<digest>`
(exact pinned digest matching meeting-scribe's `containers/vllm-asr/Dockerfile`),
then `RUN pip install flashinfer==0.6.10`. Recorded in this
decision_gate's pins table.

## Run script (`meeting-scribe/scripts/bench/run_phase_6c.sh`)

* `sddc hf download RedHatAI/Qwen3.6-35B-A3B-NVFP4 --revision <sha>`
* Start challenger on 127.0.0.1:8015 (loopback-only binding)
* Run `translation_benchmark.py` against 8015 and 8010
* Score-only mode → markdown report under this directory

## Gates

* **EN→JA sacreBLEU ≥ 66.97 − 1.0** with bootstrap 95% CI
  excluding −1.0 → PROMOTE
* **JA→EN sacreBLEU within ±1.0 of 54.63**
* **COMET delta within −0.02**
* **TTFT p99 ≤ 290 ms**
* **Zero `flashinfer_mm_fp4 Error Internal`** in the run log
* **SLO probe zero ABORTs** during the bench window

## Result

(populated post-run)
