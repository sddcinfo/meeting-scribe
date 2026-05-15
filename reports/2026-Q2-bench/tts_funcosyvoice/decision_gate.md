# Track B — TTS decision gate (NOT RUN — challenger does not exist as open weights, 2026-04-28)

## Pre-bench finding (the bench did not run)

The 2026-04-28 market survey listed **Fun-CosyVoice 3.5 (Tongyi, 2026-03-02)** as the primary JA-capable TTS challenger to `Qwen3-TTS-12Hz-0.6B-Base`.  The headline claim — "35 % first-packet-latency drop vs CosyVoice 3" — came from news coverage on gaga.art and pandaily.com.

**Reality, confirmed during the bench window:**

1. **`fun-cosyvoice` is not on PyPI** (HTTP 404 on `https://pypi.org/pypi/fun-cosyvoice/json`).
2. **The upstream `FunAudioLLM/CosyVoice` GitHub repo only ships through CosyVoice 3.0** (December 2025 release); the latest open weights are `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`.
3. **GitHub issue #1840** in that repo — _"Congratulations on the Fun-CosyVoice 3.5 Release!"_ — is community members asking *when* 3.5 will be open-sourced.  As of 2026-04-28 the answer is unknown.
4. The 3.5 announcement on 2026-03-02 was about a **Tongyi platform-only product**, not an open release.  Same pattern as MiMo-V2.5-TTS (Xiaomi platform-only).

So Fun-CosyVoice 3.5 belongs in the **same hard-pass bucket** as MiMo-V2.5-TTS, not the active-bench bucket.  This was a mis-classification in the original 2026-04-28 market survey.

## Pins (S3, plan)

The plan's S3 pinning rule is the gate that catches this: there's no model SHA to pin because there is no public model artifact to download.

| Field | Value |
|---|---|
| `challenger_model` | **NONE — does not exist as open weights** |
| `baseline_model` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (production) |
| `bench_window` | 2026-04-28 evening; `MEETING_SCRIBE_BENCH_WINDOW=1` |

## Outcome

- **HARD PASS — challenger reclassified as Tongyi-platform-only, in the same bucket as MiMo-V2.5-TTS**

The container scaffolding under `containers/funcosyvoice/` and the bench-profile entry in `docker-compose.gb10.yml` were built before the upstream check turned up the gap; both are kept on the branch as ready-to-go infrastructure for whenever Fun-CosyVoice 3.5 actually open-sources, but **not bringing them up** in this bench window.

## What was learned about the broader TTS landscape

A second-pass review of the open-source TTS field that's actually deployable today:

| Candidate | Status | Track-B fit |
|---|---|---|
| **Fun-CosyVoice 3.0** (Dec 2025) | Real, open, voice-cloning, FastAPI server | Predates the project baseline (we already considered + skipped older Cosy in Apr 2026 model selection) |
| **Higgs-Audio V2.5** (Boson, late 2025) | Open, but still non-streaming — V3 streaming TTS planned Q2 2026 | Re-evaluate when V3 lands |
| **Voxtral TTS** (Mistral) | Open, no zero-shot cloning | Doesn't meet our cloning gate |
| **OmniVoice** (k2-fsa, Mar 2026) | Apache 2.0, but model card flags "academic research only" + no streaming surface | Non-starter for prod |
| **VibeVoice** (Microsoft) | Open, but Microsoft itself says "research only, do not use in production" | Non-starter for prod |
| **MiMo-V2.5-TTS** (Xiaomi, Apr 2026) | Closed source | Non-starter |
| **Fun-CosyVoice 3.5** (Mar 2026 announce) | Closed source (this track's original target) | Non-starter |

Net: **no credible open-weights TTS challenger to Qwen3-TTS-12Hz-0.6B-Base exists as of 2026-04-28** beyond the older models already evaluated in our April baseline.  Production stays on Qwen3-TTS.

## Follow-up plan

1. Add a quarterly upstream check on the candidates above; the new `scripts/bench/check_stale_pins.py` workflow surfaces version drift but doesn't surface "license / weights status changed".
2. Specifically watch for Higgs-Audio V3 (Q2 2026 — may have shipped already; check release pages).
3. Specifically watch the FunAudioLLM/CosyVoice GitHub releases page for a 3.5-tagged open release.
4. Update `MODEL_SELECTION.md` to reflect the corrected Fun-CosyVoice 3.5 status (Tongyi-platform-only).

## Decision Log entry (to add to `MODEL_SELECTION.md`)

```
| 2026-04-28 | TTS | Hard pass on Fun-CosyVoice 3.5 — does NOT exist as open weights | Original 2026-04-28 survey listed it as Apache 2.0 based on news coverage; upstream check during the bench window confirmed it is a Tongyi-platform-only product (PyPI 404, GitHub issue #1840 has community asking when it'll open-source). Reclassify alongside MiMo-V2.5-TTS as closed-source. Production stays on Qwen3-TTS-12Hz-0.6B-Base. |
```
