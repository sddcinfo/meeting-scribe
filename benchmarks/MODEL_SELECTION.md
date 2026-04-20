# Model Selection — Meeting Scribe

Analysis of model choices for each component.  Updated when new models
are evaluated.  Last refreshed 2026-04-19 after the Qwen3.6-FP8
production migration.

## Current Stack

| Component | Model | Params | Active | VRAM | Why Chosen |
|---|---|---|---|---|---|
| **ASR** | Qwen3-ASR-1.7B | 1.7B | 1.7B (dense) | ~5 GB | Best open-source multilingual ASR, 52 languages, 97.9% lang-ID |
| **Translation** | Qwen/Qwen3.6-35B-A3B-FP8 | 35B | ~3B (MoE) | ~35 GB weights / ~89 GB allocation | +11.4 BLEU EN→JA over 3.5-INT4 (production primary direction); native FP8, vLLM-native auto-detect, 262 144 native context |
| **TTS** | Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 0.6B (dense) | ~5 GB total (×2 replicas) | 3-second voice cloning, 10 languages, 97ms first-packet |
| **Diarization** | pyannote.audio 3.1 | ~5M | 5M | ~2 GB | Best open-source diarization, ~11% DER, GPU-accelerated |

## Component Analysis

---

### ASR: Qwen3-ASR-1.7B ✅ (Keep)

**Status**: Best available for our use case.

| Model | Params | Languages | WER (EN) | JA Quality | vLLM | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | 1.7B | 52 | Competitive | **Best open-source** | ✅ | Our current model |
| Qwen3-ASR-0.6B | 0.6B | 52 | Higher | Good | ✅ | Smaller but lower accuracy |
| Cohere Transcribe | 2B | 14 | **5.42%** (#1 HF) | Not specialized | ❌ | API-only, not self-hosted |
| NVIDIA Canary Qwen | 2.5B | ? | 5.63% | Unknown | ❌ | Not open-source weights |
| Whisper Large V3 Turbo | 809M | 99 | ~5.8% | Good | ❌ (faster-whisper) | Older, surpassed by Qwen3 |

**Verdict**: Qwen3-ASR-1.7B is the right choice. Superior Japanese quality, 52 languages, vLLM-native. The 0.6B variant trades accuracy for speed — not worth it given our VRAM headroom.

---

### Translation: Qwen/Qwen3.6-35B-A3B-FP8 ✅ (Production since 2026-04-18)

**Status**: Production primary.  Migrated from Qwen3.5-INT4 after the
Phase 3 quality A/B showed +11.4 BLEU EN→JA on the 72-pair meeting
corpus (sacreBLEU with the MeCab tokenizer for Japanese output).

**2026-04-19 quality numbers — 72-pair `en_ja_meeting_v2` corpus:**

| Direction | Qwen3.5-INT4 (prior baseline) | Qwen3.6-FP8 (production) | Δ      |
|-----------|------------------------------:|-------------------------:|-------:|
| EN → JA   | 55.57                         | **66.97**                | **+11.40** — major |
| JA → EN   | 55.44                         | 54.63                    | −0.81 (within ±1.0 gate) |

EN → JA is the primary live direction (English speakers → Japanese
viewers), so the +11.4 BLEU is the headline reason 3.6 went to
production despite the +13 GB weight delta vs INT4.

**2026-04-19 perf numbers — single-resident perf bench
(`reports/phase5/decision_gate_2026-04-18.md`):**

| Workload | Phase | TTFT p50 | TTFT p99 | TPS p50 | Errors |
|---|---|---:|---:|---:|---:|
| translation | isolated   | 116 ms | 122 ms | 51.56 | 0 |
| translation | contention | 158 ms | 195 ms | 31.87 | 0 |
| coding      | isolated   | 270 ms | 275 ms | 42.53 | 0 |
| coding      | contention | 398 ms | 422 ms | 27.29 | 0 |

Translation TTFT p99 195 ms under contention with the coding agent
holds the ~400 ms live-path SLO with comfortable margin.

**Considered alternatives (rejected):**

| Model                                            | Why not                                                                                       |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `Qwen/Qwen3.5-35B-A3B-FP8`                       | EN→JA quality measured below INT4 baseline; 3.6-FP8 wins on quality and unlocks unified VL.   |
| `Intel/Qwen3.5-35B-A3B-int4-AutoRound`           | The prior production model; superseded.  Recipe + weights kept on disk as a one-commit-revert rollback. |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` / `QuantTrio/...-AWQ` | NVIDIA DGX Spark forum testing rates AWQ inferior to FP8 on GB10 — *"cannot be recommended even for single Spark"*. |
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`           | Third-party, no GB10 validation, no upside vs FP8 by forum consensus.                         |
| `caiovicentino1/Qwen3.6-35B-A3B-HLWQ-CT-INT4`    | Requires custom vLLM fork — not a drop-in.                                                    |
| `unsloth/...-GGUF` / `bartowski/...-GGUF`        | llama.cpp only, not vLLM-native.                                                              |
| Qwen3.5-9B / Qwen3.5-4B                          | Smaller models traded VRAM for quality — irrelevant once 3.6-FP8 fit on the box.              |
| Qwen3-MT (API)                                   | API-only, no self-hostable weights.                                                           |

**Production hardening (required on GB10 / SM121):**

The MoE router gate's BF16 GEMM hit `CUBLAS_STATUS_INTERNAL_ERROR`
under burst concurrency on 2026-04-18 23:45 — see
`reports/reliability/cublas_crash_2026-04-18.md`.  Two env vars in the
production recipe (`repos/auto-sre/autosre/backends/recipes/qwen3.6-fp8-nightly.yaml`)
cover this failure mode:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` — pre-sizes 8 × 4 MiB cuBLAS workspaces.
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` — NVIDIA DGX Spark thread 366822 recommendation.

`--attention-backend=flashinfer` is also required: flipping it brought
coding TTFT p99 from 751 ms → 279 ms on this hardware.  The flag is
now flagged perf-sensitive in `recipe_guard.py`.

**Refinement-side quality uplift** (production default
`refinement_context_window_segments=4`): folds the last 4
already-refined `(source, translation)` tuples into the system prompt
for the trailing refinement worker, dropping JA → EN fragment-
hallucination at +49 ms p50 cost — see
`reports/context_window_sweep/2026-04-19/summary.md`.

---

### TTS: Qwen3-TTS-12Hz-0.6B-Base ✅ (Keep)

**Status**: Best available for voice cloning.

| Model | Params | Languages | Voice Cloning | Latency | Self-hosted | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-0.6B-Base** | 0.6B | 10 | **3s reference** | 97ms TTFA | ✅ | Our current model |
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | 10 | 3s reference | ~150ms | ✅ | Higher quality, more VRAM |
| Fish Speech S2 Pro | ? | ? | Reference audio | ? | ✅ (vLLM-Omni) | Lower WER on benchmarks |
| Voxtral TTS (Mistral) | ? | 9 | Preset only | **70ms** | ✅ (vLLM-Omni) | No zero-shot cloning |
| NVIDIA MagpieTTS v2 | ? | 9 | ? | Low | ❌ (Riva) | Proprietary |

**Verdict**: The 0.6B Base model is the right choice for voice cloning. The 1.7B variant offers marginally better quality but at 2.5x VRAM cost — not worth it. Voxtral TTS is faster but can't clone voices. Fish Speech S2 Pro is interesting but requires vLLM-Omni (no ARM64 build).

---

### Diarization: pyannote.audio 3.1 ✅ (Keep, watch Falcon)

**Status**: Best established option, but alternatives emerging.

| Model | DER | Speed | GPU | Real-time | Notes |
|---|---|---|---|---|---|
| **pyannote 3.1** | ~11% | 2.5% RTF (GPU) | ✅ | ✅ | Our current model |
| NVIDIA Sortformer | ~12% | Real-time | ✅ | ✅ | Streaming-native, NeMo/Riva |
| Falcon (Picovoice) | ~11% | **221x less compute** | ❌ (CPU) | ✅ | 15x less memory, comparable accuracy |
| Diart | ~15% | 500ms buffer | ✅ | ✅ | Python framework, uses pyannote |
| "diarize" (new) | ? | **7x faster than pyannote** | ❌ (CPU) | ✅ | New library, less battle-tested |

**Verdict**: pyannote 3.1 remains the best GPU-accelerated option. Falcon is worth watching — comparable accuracy at 221x less compute, but CPU-only. For GB10 with GPU headroom, pyannote is correct.

---

## Decision Log

| Date | Component | Decision | Rationale |
|---|---|---|---|
| 2026-04-08 | ASR | Keep Qwen3-ASR-1.7B | Best multilingual, superior Japanese, vLLM-native |
| 2026-04-08 | TTS | Switch to Qwen3-TTS-12Hz-0.6B-Base | Voice cloning, faster-qwen3-tts container (was non-functional Qwen/Qwen3-TTS) |
| 2026-04-08 | Diarization | Keep pyannote 3.1 | Best GPU-accelerated, reliable, well-tested |
| 2026-04-08 | Driver | Stay on 580.x | 590 has deadlock bug, 595 is beta |
| 2026-04-09 | Translation | Benchmarked Qwen3.5-9B-FP8 | 89% JA→EN, 77% EN→JA, 55 GB savings — viable fallback if pressure increases |
| 2026-04-18 | Translation | **Migrate to Qwen/Qwen3.6-35B-A3B-FP8** | Phase 3 A/B: +11.4 BLEU EN→JA on real meeting corpus; native FP8, vLLM auto-detect, 262 144 native context; unified multimodal capability |
| 2026-04-18 | vLLM config | `--attention-backend=flashinfer` | Coding TTFT p99 751 ms → 279 ms (3.2× improvement); now perf-sensitive in `recipe_guard.py` |
| 2026-04-18 | vLLM image | `vllm/vllm-openai:nightly` | The custom `vllm-qwen35-v2:latest` image predated the Qwen3.6 release.  Nightly is the working version per NVIDIA forum thread 366822. |
| 2026-04-19 | Translation hardening | Add `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `VLLM_MARLIN_USE_ATOMIC_ADD=1` to recipe env | Covers the MoE router gate cuBLAS internal-error class observed on 2026-04-18 23:45.  See `reports/reliability/cublas_crash_2026-04-18.md`. |
| 2026-04-19 | Refinement quality | Add rolling meeting-context window (`refinement_context_window_segments=4`) | 2026-04-19 sweep: JA→EN fragment hallucination drops materially at +49 ms p50 cost.  See `reports/context_window_sweep/2026-04-19/summary.md`. |
| 2026-04-19 | Production defaults | Flip `VllmBackend.default_model` and CI baseline pointer to 3.6-FP8 + flashinfer baseline | 3.6-FP8 has been live for a day, survived the incident, and the hardening + context-window sweep both landed clean.  3.5-INT4 recipe + weights kept on disk as one-commit-revert rollback. |

## Sources

- [Best Open Source Translation Models 2026](https://www.siliconflow.com/articles/en/best-open-source-models-for-translation)
- [Best Open Source LLM for Japanese](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Japanese)
- [Qwen-MT Translation Model](https://qwenlm.github.io/blog/qwen-mt/)
- [Qwen3-ASR vs Whisper](https://www.blog.brightcoding.dev/2026/04/07/qwen3-asr-the-revolutionary-speech-tool-for-52-languages)
- [Best STT Models 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [Speaker Diarization Models Compared](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)
- [State of Speaker Diarization: pyannote vs Falcon](https://picovoice.ai/blog/state-of-speaker-diarization/)
- [Qwen3.5-9B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B)
