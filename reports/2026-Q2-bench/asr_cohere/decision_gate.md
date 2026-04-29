# Track A — ASR decision gate (Phase B1 real-corpus re-run, 2026-04-28)

This is the final close-out of Track A.  The original Track A bench
landed at DEFER (lean-promote) on a small TTS-synth corpus.  Phase B1
of the 2026-Q3 follow-up plan re-ran the bench against 200 utterances
from the public-domain Fleurs corpus (100 JA + 100 EN).  The result
**reverses the lean-promote recommendation**: Cohere's English WER is
significantly worse, which fails the plan's `Q-EN-corpus` gate ("we
don't lose English").  Production stays on `Qwen3-ASR-1.7B`.  Track A
closes.

## Pins (S3, plan)

| Field | Value |
|---|---|
| `challenger_model` | `CohereLabs/cohere-transcribe-03-2026 @ 0a928bea9c35ac5fa6c03d732311e7ba75acd3be` |
| `baseline_model` | `Qwen/Qwen3-ASR-1.7B` (production vLLM) |
| `challenger_image` | `cohere-transcribe:bench` (built 2026-04-28; transformers ≥ 5.4 + accelerate; `CohereAsrForConditionalGeneration` loader) |
| `baseline_image` | `scribe-asr` (production vLLM) |
| `corpus` | `google/fleurs` @ revision `d7c758a6dceecd54a98cac43404d3d576e721f07` — `ja_jp/test` first 100 + `en_us/test` first 100 |
| `wer_method` | char-level NFKC-normalized, punctuation-stripped Levenshtein (uniform across JA + EN) |
| `bench_window` | 2026-04-28; `MEETING_SCRIBE_BENCH_WINDOW=1`; SLO probe live throughout |

## Headline numbers

### Per-language corpus WER (token-weighted)

| Language | Qwen3-ASR-1.7B | Cohere Transcribe (03-2026) | Δ (Qwen − Cohere) | Bootstrap 95 % CI on Δ | Per-lang gate |
|---|---:|---:|---:|---|:---:|
| **ja** (n=100) | **5.95 %** | **4.89 %** | **+1.06 pp** (Cohere-better) | [−0.87, +3.00] (includes zero) | ❌ Q-JA-corpus DEFER |
| **en** (n=100) | **6.23 %** | **11.07 %** | **−4.84 pp** (Cohere-WORSE) | [−8.42, −1.58] (excludes zero) | ❌ Q-EN-corpus FAIL |

### Latency

| Language | Qwen3-ASR p50 / p95 (ms) | Cohere p50 / p95 (ms) | p95 ratio (Cohere / Qwen) |
|---|---:|---:|---:|
| ja | 749 / 1578 | 466 / 721 | **0.46×** |
| en | 689 / 992 | 524 / 555 | **0.56×** |

Cohere is **1.5–2× faster** on p95 across both languages.  Latency was never a question; quality is.

## Gate evaluation

| Gate | Threshold | Result | Outcome |
|---|---|---|:---:|
| **Q-JA-corpus** (effect size) | Δ ≥ +0.5 pp | +1.06 pp | ✅ |
| **Q-JA-corpus** (significance) | bootstrap 95 % CI excludes zero | CI = [−0.87, +3.00] (includes 0) | ❌ |
| **Q-EN-corpus** | Cohere within +0.5 pp of Qwen3-ASR (we don't lose English) | Cohere is +4.84 pp worse | ❌ **FAIL** |
| **L-live** | Cohere p95 ≤ Qwen3-ASR p95 + 50 ms | Cohere is 273–857 ms FASTER | ✅ |
| **SLO** | Live-path probe held throughout | Held; no abort | ✅ |

## Outcome — REJECT for production replacement

Cohere fails the Q-EN-corpus gate decisively (4.8 pp worse on English with CI excluding zero).  This is the gate that says "we don't lose English" — and Cohere demonstrably does.  No promotion of Cohere as a general-purpose ASR replacement.

### What the data actually says

The result is asymmetric across languages, and the asymmetry is large enough to matter:

- **On Japanese**, Cohere is *plausibly better* (point estimate +1.06 pp WER, CI marginally on the favorable side) and ~2× faster.
- **On English**, Cohere is *clearly worse* (−4.84 pp WER, CI excludes zero) despite still being faster.

A meeting-scribe deployment is multilingual by design — most live meetings are EN→JA, with both languages flowing through the live ASR path.  We can't take a 4.8 pp EN regression to gain a marginal JA improvement.

The original Track A "lean-promote" recommendation (based on TTS-synth audio) was wrong on direction: synthetic JA Qwen3-TTS produces phonetically-clean speech that masked Cohere's real-world weaknesses on natural recordings.

### Closure

- Production stays on `Qwen/Qwen3-ASR-1.7B`.
- Cohere container scaffolding (`containers/cohere-transcribe/`, the bench-profile compose entry) **stays on disk** as ready-to-go infrastructure for a *future* JA-only refinement-path evaluation if/when the cost-benefit ever favors splitting the ASR path by language.  Tear-down is deferred to a separate PR; nothing references it from production code.
- Phase C1 (refinement-side Cohere staging) of the 2026-Q3 plan is **skipped** — it was gated by B1=PROMOTE, and B1 came back REJECT.

### What this taught us about the methodology

Three lessons that propagate to future ASR challenger evaluations:

1. **Synthetic audio is not a valid proxy for real audio for ASR benches.**  The TTS-synth Track A run showed Cohere strongly favored on JA; the real-corpus run reversed that.  TTS produces speech that's phonetically cleaner than what either ASR backend was trained on — both backends ace it for different reasons, and the small differences flatter whichever backend happens to handle synthetic better.  Future ASR benches must use real recordings as the primary corpus from the start (Phase A1 of the 2026-Q3 plan now provides exactly that, on disk).
2. **Multilingual ASR evaluation must be per-language, not aggregated.**  The aggregate corpus WER on this 200-utterance run was −2.92 pp — looks bad until you split it.  JA was actually neutral-to-positive; EN was the dragger.  An aggregate that mixes language-specific WERs hides the asymmetry that drives the production decision.  The B1 harness initially hardcoded `language: "ja"` for every row; we caught it because the aggregate didn't match the per-sample p50.  The reducer + the harness both now do per-language bucketing as the default.
3. **The Q-EN-corpus gate ("we don't lose English") earned its place.**  Without it the per-language story would still flag a 4.8 pp EN regression but the plan would have no formal way to disqualify on that ground.  The gate caught exactly what it was supposed to.

## Decision Log entry (to add to `MODEL_SELECTION.md`)

```
| 2026-04-28 | ASR | **Track A closes — REJECT Cohere Transcribe for production replacement** (Phase B1 real-corpus re-run) | Per-language: JA char WER 4.89% vs Qwen3-ASR 5.95% (Δ +1.06 pp Cohere-better, CI [-0.87, +3.00] includes zero — DEFER); EN char WER 11.07% vs Qwen3-ASR 6.23% (Δ -4.84 pp Cohere-WORSE, CI [-8.42, -1.58] excludes zero — Q-EN-corpus FAIL). Cohere 1.5-2× faster on p95 but the EN regression is disqualifying for live replacement. Production stays on Qwen3-ASR-1.7B. Phase C1 (refinement-side Cohere staging) skipped — gated by B1=PROMOTE. Lessons: (1) TTS-synth audio invalid as ASR-bench proxy; (2) aggregate WER hides per-language asymmetry; (3) Q-EN-corpus gate earned its place. See `reports/2026-Q2-bench/asr_cohere/decision_gate.md`. |
```
