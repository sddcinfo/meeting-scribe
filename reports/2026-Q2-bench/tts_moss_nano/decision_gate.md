# Track B.7 — MOSS-TTS-Nano (speculative sidecar, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

CPU-only TTS challenger. Speculative; pass would mean a fall-back
TTS path that doesn't compete with the GPU translate primary.

## Pins

| Field | Value |
|---|---|
| `challenger_model` | `MOSS-TTS-Nano @ <40-char-sha>` |
| `baseline_model` | Qwen3-TTS via faster-qwen3-tts (production) |
| `staging_port` | 8025 (loopback-only) |
| `bench_window` | TBD |

## Files

* `meeting-scribe/containers/moss-tts-nano/Dockerfile` — CPU-only
  base.
* `meeting-scribe/containers/moss-tts-nano/server.py` — FastAPI
  `/v1/audio/speech`.
* `meeting-scribe/docker-compose.gb10.yml` — add `moss-tts-nano`
  under `profiles: ["bench"]` on `127.0.0.1:8025:8025`; no
  `--gpus all` (CPU-only).
* `meeting-scribe/benchmarks/_tts_backends.py:23` — add
  `"moss-tts-nano"` to the Backend literal.

## Gates

* JA MOS within −0.5 of Qwen3-TTS baseline
* TTFA p95 ≤ 200 ms
* voice-cloning quality on 3-second reference: pass/fail by reviewer

## Result

(populated post-run)
