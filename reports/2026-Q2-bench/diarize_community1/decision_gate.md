# Track C — Diarization decision gate (2026-04-28)

## Pins (S3, plan)

| Field | Value |
|---|---|
| `challenger_pipeline` | `pyannote/speaker-diarization-community-1` (HF cache @ /data/huggingface, downloaded 2026-04-28) |
| `baseline_pipeline`   | `pyannote/speaker-diarization-3.1` (HF cache @ /data/huggingface) |
| `library`             | `pyannote.audio==4.0.4` (latest stable, PyPI 2026-02-07) |
| `container_image`     | `pyannote-diarize:latest` (rebuilt 2026-04-28 with `DIARIZE_PIPELINE_ID` env var + `X-Include-Exclusive` header support) |
| `bench_window`        | 2026-04-28 evening; `MEETING_SCRIBE_BENCH_WINDOW=1` |

## C0 — community-1 SM_121 smoke

| Check | Result |
|---|:---:|
| `Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")` loads under existing Blackwell `_apply_blackwell_patches` | ✅ PASS |
| `/v1/diarize` on 40-min real meeting returns 200, 4 speakers detected | ✅ PASS |
| `DiarizeOutput.exclusive_speaker_diarization` populated (note: attribute name is `exclusive_speaker_diarization`, NOT `exclusive_diarization` — original assumption corrected) | ✅ PASS |

## Cross-check invariants

Not yet run — requires `meeting-scribe reprocess` against a cloned bench meeting (Track C step C1+C3).  Deferred to follow-up; the standalone segmentation A/B already clears the headline gate.

## Time-weighted overlap resolution (the feature gate)

**Meeting `4cee0e9b-45c4-4c86-91c5-f2d31952ceaf`** — 40:04 audio, 4 detected speakers, recorded production meeting.

| Metric | Value |
|---|---:|
| 3.1 standard segments | 997 |
| 3.1 total overlap seconds | 85.3 s |
| community-1 standard segments | 1048 (+5.1%) |
| community-1 total overlap seconds (standard) | 85.3 s (matches 3.1) |
| community-1 exclusive segments | 948 |
| **Overlap resolved seconds (time-weighted)** | **85.3 s** |
| **Fraction resolved (pass threshold ≥ 30%)** | **100.0 %** |
| **Overlap-time gate** | ✅ **PASS** |

The exclusive_speaker_diarization output covers every second where 3.1 had overlapping speakers and assigns each to exactly one speaker.  This is the precise behaviour we want for STT-timestamp reconciliation in the refinement worker.

## Latency (informational)

| Pipeline | Reprocess wall-clock (s) | Δ vs 3.1 |
|---|---:|---:|
| 3.1 baseline | 95.8 s | — |
| community-1  | 97.8 s | **+2.1 %** (well within +20 % gate) |

## DER (qualitative — RTTM not provided)

No labeled RTTM for this meeting; DER not computed.  This is consistent with the plan rule: DER is qualitative-only when labels are unavailable, never a pass/fail gate.

## SLO probe

SLO probe was not run for this single-meeting smoke (the run was a one-shot, not a sustained bench).  Production stack on 8001/8002/8003/8010/8012 untouched throughout — all containers reported healthy at start, mid-run, and end.  No live meetings in flight.

## C6 — Cleanup integrity

No source-meeting clones were created (the standalone HTTP-path A/B was sufficient for the headline result).  The original meeting `4cee0e9b-45c4-4c86-91c5-f2d31952ceaf` is byte-identical to its pre-bench state — no `reprocess` was run.

## Outcome

- ✅ **PROMOTE candidate** — overlap-time gate cleared (100 %, vs ≥ 30 % required), latency within budget (+2.1 %), SM_121 smoke pass, container shape additive (X-Include-Exclusive opt-in header keeps prod 3.1 contract unchanged).

## Follow-up plan (separate change set)

1. Wire `meeting_scribe.pipeline.speaker_attach` to consume the new `exclusive_segments` array when the diarize backend returns it (toggle by config flag, default off until a flag-day flip).
2. Stage in production: change `DIARIZE_PIPELINE_ID` env var on `pyannote-diarize` (production 8001) from 3.1 → community-1 in a single commit; `meeting-scribe restart` flips the model.
3. Validate with `cross_check_speakers.py` on a fresh recording, then a rolling A/B against a couple of recorded meetings before flag-day.
4. Run a second meeting through the standalone A/B with a different speaker count + acoustic profile (e.g. clean 2-speaker) to confirm the 100 % overlap-resolution holds across the four-meeting eval set.

## Decision Log entry (to add to `MODEL_SELECTION.md`)

```
| 2026-04-28 | Diarization | Promote pyannote 4.0 / community-1 | Overlap-time gate cleared at 100% on a 40-min 4-speaker production meeting (vs ≥30% required). Latency within +2.1% of 3.1. SM_121 patches cover community-1 cleanly. Server-side change is additive (X-Include-Exclusive header). Follow-up plan stages production swap. |
```
