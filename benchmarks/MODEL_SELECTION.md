# Model Selection — Meeting Scribe

Analysis of model choices for each component.  Updated when new models
are evaluated.  Last refreshed 2026-04-28 with:

* The March-April 2026 market survey (MiMo-V2.5, OmniVoice,
  Fun-CosyVoice3.5, VibeVoice, Higgs-Audio V2.5, pyannote 4.0 /
  community-1, Cohere Transcribe).
* The 2026-Q2 model-challenger bench results
  (`reports/2026-Q2-bench/{asr_cohere,tts_funcosyvoice,diarize_community1}/decision_gate.md`).
* **The pyannote 4.0 / community-1 production swap** (this entry).
  Track C of the bench cleared every gate; community-1 is now the
  production diarize pipeline and the 3.1 code paths have been removed
  in the same change set.

## Current Stack

| Component | Model | Params | Active | VRAM | Why Chosen |
|---|---|---|---|---|---|
| **ASR** | Qwen3-ASR-1.7B | 1.7B | 1.7B (dense) | ~5 GB | Best open-source multilingual ASR, 52 languages, 97.9% lang-ID |
| **Translation** | Qwen/Qwen3.6-35B-A3B-FP8 | 35B | ~3B (MoE) | ~35 GB weights / ~89 GB allocation | +11.4 BLEU EN→JA over 3.5-INT4 (production primary direction); native FP8, vLLM-native auto-detect, 262 144 native context |
| **TTS** | Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 0.6B (dense) | ~5 GB total (×2 replicas) | 3-second voice cloning, 10 languages, 97ms first-packet |
| **Diarization** | pyannote.audio 4.0.4 + `pyannote/speaker-diarization-community-1` | ~5M | 5M | ~2 GB | Production since 2026-04-28. Adds an `exclusive_speaker_diarization` single-speaker timeline that resolves 100 % of 3.1's overlapping seconds to single-speaker assignment (Track C bench, 40-min 4-speaker production meeting). +2.1 % wall-clock vs 3.1 baseline. |

## Component Analysis

---

### ASR: Qwen3-ASR-1.7B ✅ (Keep)

**Status**: Best available for our use case.

| Model | Params | Languages | WER (EN) | JA Quality | vLLM | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B** | 1.7B | 52 | Competitive | **Best open-source** | ✅ | Our current model |
| Qwen3-ASR-0.6B | 0.6B | 52 | Higher | Good | ✅ | Smaller but lower accuracy |
| Cohere Transcribe (03-2026) | 2B Conformer | 14 (incl. JA) | **5.42%** (#1 HF Open ASR) | Untested on JA business corpus | ❌ (transformers only) | Open-sourced Apache 2.0 on 2026-03-26 — was API-only at our last refresh, now self-hostable.  Worth a JA A/B vs Qwen3-ASR-1.7B. |
| **MiMo-V2.5-ASR** (Xiaomi) | 8B | Mandarin + 4 dialects + EN code-switch | SOTA on Open ASR English | **No JA support** | ❌ (transformers only) | Released 2026-04-22, MIT.  Beats Qwen3-ASR / Seed-ASR 2.0 / Whisper-large-v3 on CN+EN.  Hard pass for our EN→JA workflow but worth bookmarking if we add CN. |
| NVIDIA Canary Qwen | 2.5B | ? | 5.63% | Unknown | ❌ | Not open-source weights |
| Whisper Large V3 Turbo | 809M | 99 | ~5.8% | Good | ❌ (faster-whisper) | Older, surpassed by Qwen3 |
| Meta Omnilingual ASR | 300M / 7B | **1,600+** | Mid-tier on top languages | Untested | ❌ (fairseq2) | 2025-11-10 release — wide tail-language coverage but no streaming, no vLLM, no head-to-head win on top languages.  Skip for prod. |

**Verdict**: Qwen3-ASR-1.7B is the right choice. Superior Japanese quality, 52 languages, vLLM-native. The 0.6B variant trades accuracy for speed — not worth it given our VRAM headroom.

**2026-04-28 market check.**  Two notable post-baseline releases:

- **Cohere Transcribe** flipped Apache 2.0 on 2026-03-26 (was API-only at our last refresh).  Now self-hostable, includes Japanese in its 14-language set, sits #1 on the HF Open ASR Leaderboard for English (5.42% WER).  Worth running our `en_ja_meeting_v2` corpus against it — if JA WER beats Qwen3-ASR-1.7B by a meaningful margin, it becomes a candidate replacement; if not, the leaderboard win is English-only and Qwen3-ASR keeps its seat.
- **MiMo-V2.5-ASR** (Xiaomi, 2026-04-22, 8B, MIT) is the headline April release the user flagged.  It outperforms Qwen3-ASR / Seed-ASR 2.0 / Whisper-large-v3 / Gemini-3.1-Pro on Mandarin + English + Chinese dialects + lyrics, but **the model card omits Japanese entirely** — the bilingual Mandarin/English framing is by design.  Hard pass for the EN→JA primary workflow; bookmark for any future Mandarin product.

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
| Fish Speech S2 Pro | ? | 80+ | Reference audio | ~100ms TTFA (H200) | ✅ (vLLM-Omni) | Lower WER on benchmarks |
| Voxtral TTS (Mistral) | ? | 9 | Preset only | **70ms** | ✅ (vLLM-Omni) | No zero-shot cloning |
| NVIDIA MagpieTTS v2 | ? | 9 | ? | Low | ❌ (Riva) | Proprietary |
| **Fun-CosyVoice3.5** (Tongyi) | ? | **13 incl. JA** | 10–20s ref | ~63ms TTFA (35% drop vs 3.0) | ✅ (FunAudioLLM) | 2026-03-02 release.  DiffRO+GRPO RL, rare-char err 15.2% → 5.3%, FreeStyle natural-language voice control. **Strongest JA-capable challenger.** |
| **OmniVoice** (k2-fsa) | ~0.6B (Qwen3-0.6B base) | **600+ zero-shot** | 3–10s ref | RTF **0.025** (40× RT) | ✅ (custom `omnivoice` lib) | 2026-03-31, Apache 2.0.  3.7K stars / 460K HF dl in 3 weeks.  No streaming surface confirmed; license for commercial use unclear (card says "academic research only"). |
| **VibeVoice-Realtime-0.5B** (MS) | 0.5B | ? | Yes (with audible AI disclaimer + watermark) | Streaming | ✅ | MIT, but Microsoft explicitly says **research-only, not for prod**.  Family also includes 1.5B long-form (90 min, 4 spk) and `VibeVoice-ASR` (60 min, 50+ langs, 7.77% WER). |
| **Higgs-Audio V2.5** (Boson) | 1B (down from 3B in V2) | Multi (incl. JA) | 3–10s ref | n/a (V2.5 is non-streaming) | ✅ | Aug 2025 V2 → V2.5 condensation.  V3 streaming TTS planned Q2 2026 — would be the actual disruption.  Watch the V3 release. |
| MiMo-V2.5-TTS (Xiaomi) | n/a | CN + EN | Yes (VoiceClone variant) | n/a | ❌ **Closed source** | 2026-04-22 — only available via Xiaomi MiMo Studio / API.  Not deployable on GB10. |

**Verdict**: Qwen3-TTS-12Hz-0.6B-Base remains the right choice **today** — 97 ms TTFA, vLLM-Omni-resident, JA in the 10-language set, no licensing ambiguity.  But the field tightened materially in March-April 2026:

- **Fun-CosyVoice3.5** is the credible production challenger.  It covers Japanese, ships voice cloning at 10–20 s reference, and the FreeStyle natural-language voice control would directly improve our slide-narration UX.  The 35 % first-packet-latency drop they report is the headline number worth measuring on GB10.  Action: add to `benchmarks/tts_v_qwen3_baseline_2026Q2.md` plan.
- **OmniVoice** is fascinating from a research angle — 600+ languages, RTF 0.025 — but the "academic research only" license tag and missing streaming story make it a poor swap for a 24/7 meeting box.  Keep on monitor.
- **MiMo-V2.5-TTS** is closed source and Chinese/English only — not a fit.
- **VibeVoice** carries Microsoft's own "do not use in production" advisory.  No path to prod.
- **Higgs-Audio V2.5** is interesting but its V3 (Q2 2026, streaming) is the version that would actually compete; today's V2.5 doesn't beat Qwen3-TTS-12Hz on our latency budget.

---

### Diarization: pyannote.audio 4.0.4 + `pyannote/speaker-diarization-community-1` ✅ (Production since 2026-04-28)

**Status**: Production primary.  Promoted from 3.1 → community-1 after Track C of the 2026-Q2 model-challenger bench cleared every gate.

| Model | DER | Speed | GPU | Real-time | Notes |
|---|---|---|---|---|---|
| **community-1** (production) | n/a (no labeled DER on our corpus) | 97.8 s wall-clock on 40-min 4-speaker meeting | ✅ | ✅ | +2.1 % vs 3.1 baseline.  Resolves 100 % of 3.1's overlapping seconds to single-speaker assignment via `exclusive_speaker_diarization` (Track C measurement). |
| pyannote 3.1 (retired) | ~11% | 2.5% RTF (GPU) | ✅ | ✅ | Removed from the codebase 2026-04-28.  Library was already on 4.0.4; we kept loading the older pretrained pipeline until community-1 cleared its bench. |
| NVIDIA Streaming Sortformer v2/v2.1 | **7.0% on ALI** | RTF **214×** | ✅ | ✅ (0.32 s chunk) | Aug 2025, English + Mandarin only, ≤4 speakers.  Faster but no JA support. |
| pyannoteAI Precision-2 (commercial) | 11.2% (best in benchmark) | n/a | n/a | ✅ | Cloud API only — disqualified by privacy posture. |
| Falcon (Picovoice) | ~11% | **221x less compute** | ❌ (CPU) | ✅ | 15x less memory, comparable accuracy |
| Diart | ~15% | 500ms buffer | ✅ | ✅ | Python framework, uses pyannote |

**Headline result (Track C, `4cee0e9b` 40-min 4-speaker production meeting):**

| Metric | pyannote 3.1 | community-1 | Δ |
|---|---:|---:|---:|
| Standard segments | 997 | 1048 | +5.1 % (slightly finer-grained) |
| Total overlapping seconds in standard output | 85.3 s | 85.3 s | 0 (unchanged) |
| Resolved by `exclusive_speaker_diarization` | n/a | **85.3 s / 85.3 s = 100 %** | new feature |
| Wall-clock | 95.8 s | 97.8 s | **+2.1 %** |
| Detected speakers | 4 | 4 | unchanged |

The 100 % overlap-resolution is the load-bearing finding: every second of two-speaker overlap in the standard segmentation is mapped by community-1's exclusive output to exactly one speaker per frame.  That's exactly the input shape `meeting_scribe.pipeline.speaker_attach._attach_speakers_to_events` needs for clean STT-timestamp reconciliation in the refinement worker.  See `reports/2026-Q2-bench/diarize_community1/decision_gate.md` for the full bench writeup.

**Production embed (the "deeply embed, no legacy" change set, 2026-04-28):**

| File | Change |
|---|---|
| `containers/pyannote/server.py` | Default `DIARIZE_PIPELINE_ID` flipped to community-1.  3.x `Annotation` isinstance fallback **deleted** (pyannote 4.x always returns `DiarizeOutput`).  `exclusive_segments` now part of every response — no opt-in `X-Include-Exclusive` header. |
| `src/meeting_scribe/pipeline/diarize.py` | `_diarize_single_call` returns `(standard, exclusive)` tuple.  `_merge_clusters_via_embeddings` projects the merge map onto exclusive segments so cluster ids line up across both arrays.  `_diarize_full_audio` returns a `DiarizeResult` dataclass (`segments` + `exclusive_segments`). |
| `src/meeting_scribe/pipeline/speaker_attach.py` | Primary speaker per ASR event is now picked from `exclusive_segments` (clean — every frame has exactly one speaker).  Cross-talk detection still uses the standard `segments` array.  Minority-speaker rescue cross-references the union of both. |
| `src/meeting_scribe/routes/meeting_lifecycle.py` | Two finalize callsites + the stop-time path updated to unpack `DiarizeResult` and pass both arrays to `_attach_speakers_to_events`. |
| `src/meeting_scribe/reprocess.py` | Same unpacking; reprocess uses standard segments to *shape* ASR chunks (one chunk inherits one cluster id), exclusive carried through for downstream consumers. |
| `src/meeting_scribe/recipes/sortformer-4spk.yaml` | `model_id` flipped to `pyannote/speaker-diarization-community-1`. |
| `docker-compose.gb10.yml` | Production `pyannote-diarize` explicitly pins community-1 via `DIARIZE_PIPELINE_ID`.  Bench-profile `pyannote-diarize-c1` + `pyannote-diarize-c1-scratch` sidecars **removed** — production IS community-1 now. |

**What we learned, in detail:**

1. **The HF attribute name is `exclusive_speaker_diarization`, not `exclusive_diarization`.**  The HF model card uses the bare term informally; the actual `DiarizeOutput` object exposes the longer name.  Our first server change emitted a no-op opt-in header because `getattr(raw_output, "exclusive_diarization", None)` returned `None` on every request.  Fixed by docker-execing into the live container and dumping `dir(out)` — five-minute fix once the misname was found.
2. **The pyannote.audio library was already on 4.0.4 (the latest stable, PyPI 2026-02-07).**  What was old was the pretrained *pipeline* (3.1), not the library version.  Many "we're on an old pyannote" reflexes confused the two.  This is a recurring shape: a model artifact has its own release cadence, separate from the library that loads it.  Our new `scripts/bench/check_stale_pins.py` workflow only catches library drift, not model drift — see "Future tooling" below.
3. **The Blackwell SM_121 patches in `containers/pyannote/server.py` (the `one_hot` CPU fallback + the `get_device_capability` spoof) covered community-1 unchanged.**  Both pipelines hit the same code path, so the patches generalised correctly.  No new compatibility work was needed.
4. **The exclusive output ships without embeddings.**  Standard segments carry per-speaker WeSpeaker embeddings (used for cross-chunk cluster stitching); exclusive segments don't.  This shaped the design: embedding-based merge happens on standard, then the merge map is projected onto exclusive so both arrays end up with the same global cluster ids.
5. **Cross-talk still requires the standard segmentation.**  The exclusive output, by construction, collapses two-speaker windows to a single owner.  That's exactly what we want for primary assignment but NOT what we want for "did A and B speak at the same time?".  Both arrays in the response, both used in `speaker_attach`.  Don't conflate the two.
6. **The production server change was additive first, breaking second.**  In the bench window we shipped the `X-Include-Exclusive: true` opt-in header so production traffic stayed on the 3.1-shaped response while the bench sidecar exercised the new path.  In the deep-embed change set we removed the header (community-1 is now production, opt-in no longer makes sense).  Two-step shape: opt-in for safety, then unconditional once the new shape is the one we want everywhere.
7. **`reprocess.py` and `meeting_lifecycle.py` use the diarize output for very different things.**  Reprocess uses it to shape ASR chunks — one chunk inherits one cluster id, no overlap math needed.  Lifecycle uses it for post-hoc speaker attribution — needs the overlap math.  The new `DiarizeResult` dataclass surfaces both arrays; each caller picks what it needs.  Don't try to flatten the two use cases into one return shape.
8. **DER as a hard gate is a trap when you don't have RTTMs.**  We deliberately omitted DER from Track C's pass criteria because we don't have hand-labeled speaker references for the eval meeting.  The plan's earlier rev had DER as a soft gate, computed only when labels exist.  This was the right call — a "DER number" computed against a Qwen3-ASR transcript would have been defensible-looking and meaningless.

---

## March-April 2026 Market Survey — Action Summary

Triggered by user-flagged MiMo-V2.5-ASR release; broadened to a full sweep of ASR / TTS / diarization releases since the project's 2026-04-09 baseline.  Production stack stays as-is; three pieces of follow-up work fall out:

| Component | Action | Driver | Owner / Next step |
|---|---|---|---|
| ASR | **A/B Cohere Transcribe (03-2026) on `en_ja_meeting_v2`** | Was API-only at our last refresh; now Apache 2.0 self-hostable, includes JA, #1 EN on HF leaderboard.  Cheap test, high ceiling if JA WER beats Qwen3-ASR-1.7B. | `benchmarks/asr_v_qwen3_baseline_2026Q2.md` |
| TTS | **Pilot Fun-CosyVoice3.5 against Qwen3-TTS-12Hz-0.6B-Base** | Tongyi 2026-03-02; 13 langs incl. JA, 35% first-packet-latency drop vs CosyVoice3, FreeStyle natural-language voice control would directly upgrade slide narration. | `benchmarks/tts_v_qwen3_baseline_2026Q2.md` |
| Diarization | **Stage pyannote 4.0 / community-1 on a recorded meeting** | 2026-04-22 release; *exclusive single-speaker mode* targets the overlap-conflict failure mode that hurts STT-timestamp reconciliation in our refinement worker.  Direct in-place upgrade. | Aligns with the pre-existing "community-1 upgrade path" memory item. |

Hard passes (recorded so we don't re-evaluate):

- **MiMo-V2.5-ASR** (Xiaomi) — 8B, MIT, SOTA on CN+EN+dialects+lyrics, but the model card omits Japanese.  Not deployable for our EN→JA primary direction.  Bookmark for any future Mandarin product.
- **MiMo-V2.5-TTS** (Xiaomi) — closed source, Xiaomi-Studio-only, CN+EN.  Not self-hostable.
- **Meta Omnilingual ASR** (Nov 2025, 1,600+ langs) — wide tail-language coverage, but no streaming, no vLLM, doesn't beat Qwen3-ASR on top languages.
- **OmniVoice** (k2-fsa, 600+ langs, RTF 0.025) — research-only license tag and no streaming surface; great research demo, wrong fit for a 24/7 meeting box.
- **VibeVoice** (Microsoft) — explicit "research only, do not use in production" disclaimer from the publisher.
- **Higgs-Audio V2.5** (Boson) — wait for V3 (Q2 2026, streaming TTS); today's V2.5 doesn't beat Qwen3-TTS on our latency budget.
- **NVIDIA Streaming Sortformer v2** — fastest diarization in the benchmarks (RTF 214×, 7.0% DER on ALI), but English + Mandarin only, ≤4 speakers — disqualified for the EN→JA workflow.

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
| 2026-04-28 | ASR market scan | **Hard pass on MiMo-V2.5-ASR (Xiaomi, 8B, MIT, 2026-04-22)** | SOTA on Mandarin/English/dialects/lyrics, but the model card omits Japanese — disqualifies for the EN→JA primary workflow. Bookmark for any future Mandarin product. |
| 2026-04-28 | ASR market scan | **Plan A/B for Cohere Transcribe (03-2026)** | Apache 2.0 since 2026-03-26 (was API-only at our last refresh), 14 langs incl. JA, #1 EN on HF Open ASR Leaderboard (5.42 % WER).  Cheap to test, high ceiling if JA WER beats Qwen3-ASR-1.7B on `en_ja_meeting_v2`. |
| 2026-04-28 | TTS market scan | **Plan pilot for Fun-CosyVoice3.5 (Tongyi, 2026-03-02)** | Strongest JA-capable challenger to Qwen3-TTS-12Hz-0.6B-Base.  13 langs incl. JA, voice cloning at 10–20 s ref, claimed 35 % first-packet-latency drop vs CosyVoice 3, FreeStyle natural-language voice control directly upgrades slide narration UX. |
| 2026-04-28 | TTS market scan | Hard pass on OmniVoice / VibeVoice / MiMo-V2.5-TTS / Higgs-Audio V2.5 | OmniVoice license tagged "academic research only" with no streaming surface; VibeVoice carries Microsoft's own "do not use in production" advisory; MiMo-V2.5-TTS is closed source; Higgs-Audio V2.5 is interesting but the disruptive version is V3 streaming (Q2 2026 — re-evaluate then). |
| 2026-04-28 | Diarization market scan | **Plan stage of pyannote 4.0 / community-1 (2026-04-22)** | Direct in-place upgrade to our pyannote 3.1.  *Exclusive single-speaker mode* specifically targets the overlap-conflict failure mode that's the #1 source of refinement-side mis-attribution in our pipeline.  Aligns with pre-existing "community-1 upgrade path" follow-up. |
| 2026-04-28 | Diarization market scan | Hold on NVIDIA Streaming Sortformer v2 | Fastest in benchmark (RTF 214×, 7.0 % DER on ALI), but English + Mandarin only, ≤4 speakers — disqualified for EN→JA. |
| 2026-04-28 | Diarization | **Promote pyannote 4.0 / community-1** (bench result) | Overlap-time gate cleared at 100 % on a 40-min 4-speaker production meeting (vs ≥ 30 % required). Latency within +2.1 % of 3.1. SM_121 patches cover community-1 cleanly (`exclusive_speaker_diarization` field present). Server-side change is additive (`X-Include-Exclusive: true` header). Follow-up plan stages prod swap. See `reports/2026-Q2-bench/diarize_community1/decision_gate.md`. |
| 2026-04-28 | ASR | **Defer-lean-promote on Cohere Transcribe (03-2026)** (bench result) | Char-level corpus WER 5.59 % vs Qwen3-ASR-1.7B 9.23 % on 36 JA TTS-synthesized utterances (+3.64 pp point estimate, +3.27 × latency, +2.49 × p95 latency). Bootstrap 95 % CI on (Qwen − Cohere) = [−2.22, +10.10] pp — straddles zero on this small ceiling-bound corpus, so formal Q-JA-corpus gate fails ⇒ DEFER. Recommend re-run on real public-domain JA audio + pilot Cohere as refinement-side ASR (async path). See `reports/2026-Q2-bench/asr_cohere/decision_gate.md`. |
| 2026-04-28 | TTS | **Hard pass on Fun-CosyVoice 3.5** — does NOT exist as open weights | Original 2026-04-28 survey listed it as Apache 2.0 based on news coverage (gaga.art, pandaily). Upstream check during the bench window: `fun-cosyvoice` returns HTTP 404 on PyPI; `FunAudioLLM/CosyVoice` GitHub only ships through 3.0; issue #1840 has community asking when 3.5 will open-source. Reclassify alongside MiMo-V2.5-TTS as Tongyi-platform-only / closed-source. Production stays on Qwen3-TTS-12Hz-0.6B-Base. See `reports/2026-Q2-bench/tts_funcosyvoice/decision_gate.md`. |
| 2026-04-28 | Diarization | **Production swap to pyannote 4.0 / community-1; deep embed; 3.1 code paths deleted** | Container `DIARIZE_PIPELINE_ID` defaults to community-1; 3.x `Annotation` fallback removed from `containers/pyannote/server.py`; opt-in `X-Include-Exclusive` header removed (exclusive_segments unconditional); `_diarize_full_audio` now returns `DiarizeResult(segments, exclusive_segments)` (breaking shape change, callers updated in `reprocess.py` + `meeting_lifecycle.py`); `speaker_attach._attach_speakers_to_events` picks primary from `exclusive_segments`, cross-talk from standard; `recipes/sortformer-4spk.yaml` flipped; bench-profile `pyannote-diarize-c1*` sidecars removed (redundant). |
| 2026-04-28 | ASR | **Stay on Qwen3-ASR-1.7B for now** | Track A bench was DEFER (lean-promote): point estimate -3.64 pp WER and 3× faster latency favored Cohere, but bootstrap 95 % CI on the small TTS-synth corpus straddled zero. No production change today. Re-run on real public-domain JA audio + pilot Cohere on the refinement-side ASR path is the path to a clean PROMOTE. |
| 2026-04-28 | ASR | **Track A closes — REJECT Cohere Transcribe for production replacement** (Phase B1 real-corpus re-run) | Per-language: JA char WER 4.89 % vs Qwen3-ASR 5.95 % (Δ +1.06 pp Cohere-better, CI [-0.87, +3.00] includes zero — DEFER); EN char WER 11.07 % vs Qwen3-ASR 6.23 % (Δ -4.84 pp Cohere-WORSE, CI [-8.42, -1.58] excludes zero — Q-EN-corpus FAIL). Cohere 1.5-2× faster on p95 but the EN regression is disqualifying for live replacement. Production stays on Qwen3-ASR-1.7B. Phase C1 (refinement-side Cohere staging) skipped — gated by B1=PROMOTE. Lessons: (1) TTS-synth audio invalid as ASR-bench proxy; (2) aggregate WER hides per-language asymmetry; (3) Q-EN-corpus gate earned its place. See `reports/2026-Q2-bench/asr_cohere/decision_gate.md`. |
| 2026-04-28 | Tooling | **Phase A1+A2+D1 of 2026-Q3 plan landed** | A1: Fleurs JA + EN public-domain corpus on disk (200 utterances, sha256-verified, manifest-cited at revision `d7c758a6...`). A2: `scripts/bench/check_model_drift.py` + `watchlist.yaml` companion to the existing PyPI freshness checker — polls HF Hub for newer siblings of pinned model ids and tracks closed-source TTS / Omni candidates. Wired into the existing weekly `dependency-freshness` GitHub Action. D1: `meeting-scribe bench {start,status,stop}` CLI subcommand — orchestrates `preflight.py` + `slo_probe.py` + state file lifecycle so future bench windows don't drop the SLO probe. SLO probe also fixed (was 404'ing because `auto` model id was rejected by vLLM; now discovers `/v1/models` at startup). |
| 2026-04-29 | ASR | **Real-world deep bench reinforces REJECT on Cohere** | 440 chunks across 4 production meetings (~7900 s real audio); Cohere errored on 392/440 (89.1%), cascading CUDA-wedge from one Chinese-audio chunk in `fe77b412` chunk 0. Of 47 successful Cohere responses, char-level disagreement vs Qwen3-ASR was 100% on virtually every chunk — Cohere hallucinates on near-silence + language mismatches where Qwen3 correctly returns empty / detects internally. Three architectural mismatches surfaced beyond B1's WER finding: (1) one wrong-language call takes the whole container down until restart; (2) `CohereAsrProcessor` requires language as a *positional* arg, no auto-detect path; (3) production meetings have mid-chunk JA↔EN↔Chinese code-switching that Cohere fundamentally cannot handle. Production stays on Qwen3-ASR-1.7B. No retry. See `reports/2026-Q2-bench/asr_realworld/decision_gate.md`. |
| 2026-04-29 | Tooling | **Phase C2 landed — `exclusive_segments` persisted in artifacts** | Finalize + reprocess paths now write `<meeting_dir>/speaker_lanes_exclusive.json` (frame-level community-1 single-speaker timeline) alongside the existing `speaker_lanes.json` (event-aligned). `_generate_timeline` reads the sidecar and emits `exclusive_segments` at the top level of `timeline.json` when present. Additive: UI consumers ignore unknown fields, older meetings without the sidecar load unchanged. Persists data; UI rendering of the cleaner single-speaker timeline is a follow-up. |

## Sources

- [Best Open Source Translation Models 2026](https://www.siliconflow.com/articles/en/best-open-source-models-for-translation)
- [Best Open Source LLM for Japanese](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Japanese)
- [Qwen-MT Translation Model](https://qwenlm.github.io/blog/qwen-mt/)
- [Qwen3-ASR vs Whisper](https://www.blog.brightcoding.dev/2026/04/07/qwen3-asr-the-revolutionary-speech-tool-for-52-languages)
- [Best STT Models 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [Speaker Diarization Models Compared](https://brasstranscripts.com/blog/speaker-diarization-models-comparison)
- [State of Speaker Diarization: pyannote vs Falcon](https://picovoice.ai/blog/state-of-speaker-diarization/)
- [Qwen3.5-9B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B)

### 2026-04-28 Market Survey Sources

- [MiMo-V2.5-ASR on HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR)
- [MiMo-V2.5-ASR landing page (Xiaomi MiMo)](https://mimo.xiaomi.com/mimo-v2-5-asr)
- [Xiaomi introduces MiMo-V2.5 TTS and ASR — Gizmochina (2026-04-24)](https://www.gizmochina.com/2026/04/24/xiaomi-introduces-mimo-v2-5-tts-and-asr-as-a-full-voice-pipeline-for-the-agent-era/)
- [Cohere Transcribe open-source release blog (2026-03)](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)
- [Cohere Transcribe coverage — TechCrunch (2026-03-26)](https://techcrunch.com/2026/03/26/cohere-launches-an-open-source-voice-model-specifically-for-transcription/)
- [Meta Omnilingual ASR blog (2025-11-10)](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/)
- [OmniVoice on HuggingFace (k2-fsa, 2026-03-31)](https://huggingface.co/k2-fsa/OmniVoice)
- [OmniVoice GitHub repo](https://github.com/k2-fsa/OmniVoice/)
- [Fun-CosyVoice3.5 launch coverage — Pandaily (2026-03-02)](https://pandaily.com/alibaba-tongyi-unveils-fun-cosy-voice3-5-and-fun-audio-gen-vd-with-free-style-voice-generation)
- [Fun-CosyVoice3.5 deep dive — Gaga AI](https://gaga.art/blog/fun-cosyvoice3-5-and-fun-audiogen-vd/)
- [Microsoft VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [Higgs-Audio V2.5 release blog — Boson AI](https://www.boson.ai/blog/higgs-audio-v2.5)
- [pyannote.audio 4.0 / community-1 changelog (2026-04-22)](https://www.pyannote.ai/changelog/community-1-now-available)
- [pyannote community-1 release blog](https://www.pyannote.ai/blog/community-1)
- [NVIDIA Streaming Sortformer v2.1 model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [Benchmarking Diarization Models (arXiv 2509.26177)](https://arxiv.org/html/2509.26177v1)
