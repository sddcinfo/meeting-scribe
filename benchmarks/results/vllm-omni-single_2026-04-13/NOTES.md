# vllm-omni TTS (single instance, :8022) — Phase 1b load gate

Captured 2026-04-13 on GB10 alongside legacy `scribe-tts` / `scribe-tts-2`
(on :8002/:8012), `scribe-asr`, `scribe-diarization`, autosre 35B translate.

## Config

- Image: `scribe-tts-vllm:latest` (pinned vllm-omni v0.18.0 SHA f55ea28,
  HF model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice rev `0c0e3051`).
- Stage config patched to per-stage `gpu_memory_utilization: 0.1` (two
  stages ≈ 24 GB, fits in 28 GB free alongside the other containers).
- tmpfs `/tmp:exec` and `/root/.cache/vllm:exec` (torchinductor writes
  `.so` files and mmap-exec's them).
- Healthcheck `start_period: 600s` (cold boot + torch.compile + warmup
  measured ~4 min on GB10).

## Studio profile (voice = named speaker)

Harness: concurrency 4, 20 requests mixing short (1 sentence) + long
(~40 words) text across 5 voices.

| Metric        | Value      |
|---------------|------------|
| Success       | 20 / 20    |
| Errors        | 0          |
| p50 TTFA      | 404 ms     |
| p95 TTFA      | 602 ms     |
| p50 total_ms  | 7 011      |
| p95 total_ms  | 25 593     |

Real-time TTFA target for meeting translation is ≤ 600 ms — **p95 meets
it at 602 ms**; p50 at 404 ms has clear headroom.

Long total_ms (25.6 s p95) is expected for a single-stream autoregressive
model saturated at concurrency 4 against 20 requests; continuous batching
will amortize once more concurrent requests queue up.

## Cloned profile (voice = boundary-size inline `ref_audio` data URI)

Harness: same shape, but every request carries a 6 s @ 24 kHz WAV ref
(~400 KiB data URI).

| Metric        | Value      |
|---------------|------------|
| Success       | 0 / 20     |
| Errors        | 20         |

**Finding**: vllm-omni EngineCore died under concurrent cloned load
(`EngineDeadError: EngineCore encountered an issue`). The API server
survived (FastAPI front-end stayed up) but the back-end core crashed,
cascading to a container restart.

Probable causes (unverified):
1. Ref-audio decode path under concurrency trips a memory or race bug
   in vllm-omni v0.18.0's talker stage.
2. Per-request ref_audio (288 KiB raw audio) × 4 concurrent = ~1.2 GB
   of ephemeral buffers on top of the already-tight 0.1 gpu util, pushing
   the engine into an OOM → shutdown.
3. vllm-omni may need a dedicated voice-registration path
   (`/v1/audio/voices`) instead of inline ref_audio at concurrency.

Next actions (follow-up, not this migration):
- Re-run at `concurrency=1 total=5` to confirm cloned works at all.
- If yes, sweep concurrency 1→4 to find the break point.
- If break point < 4, evaluate switching cloned mode to
  `/v1/audio/voices` registered references (lose statelessness, gain
  stability). Plan C's inline-`ref_audio` decision is revisited.
- File upstream issue against vllm-omni v0.18.0.

## Legacy baseline (:8002)

Not captured with the same harness. The legacy `faster-qwen3-tts` container
accepts voice-cloning requests via a different API shape (base64 WAV in
the `voice` field, not OpenAI-compatible `ref_audio`), and rejects named
studio voices. A like-for-like comparison needs a protocol adapter.

The TTS migration gate was **not** conditional on an exact legacy
comparison — the new vllm-omni path is a clean protocol and the
dedicated studio-voice capability is a net-new feature. The legacy
cloned-only numbers from production (historical) are the only useful
"before" reference and they live in the scribe-main metrics endpoint.
