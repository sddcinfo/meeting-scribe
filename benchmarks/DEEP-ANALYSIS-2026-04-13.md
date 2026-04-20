# Deep pipeline analysis — 2026-04-13

Complement to `BASELINE-2026-04-13.md`. That doc captured numbers; this
one walks every stage of the ingest → ASR → diarize → translate → TTS
→ broadcast pipeline and surfaces concrete improvement opportunities
with file:line pointers. Ranked by effort × impact at the end.

Live-meeting samples used throughout: meetings `afa9883c` and `99c4f3f8`
captured 2026-04-13 15:09–15:20, `/tmp/meeting-scribe.log`.

---

## Stage 1 — Audio ingress (browser → server WS)

### Code path
- Client sends 48 kHz s16le PCM frames over `/api/ws`.
- Handler: `websocket_audio` at `server.py:5559`.
- Every ~40 audio chunks, logged at WARNING: `Audio chunk #N: 48000Hz→16kHz, 12032→4011 samples, peak=... rms=...`.

### Observations
| | expected | observed |
|---|---|---|
| Chunk cadence | ~40 chunks/s × 4000 samples = 250 ms windows | consistent in the log |
| WS lifetime | entire meeting | **drops every 60–75 s**, auto-reconnects |

WS drop cadence:

```
15:12:44 connected → 15:13:49 disconnected = 65 s
15:13:49 reconnected immediately
```

### Root cause hypothesis
`uvicorn.Config` at `server.py:7209` and `:7233` does **not** set
`ws_ping_interval` or `ws_ping_timeout`. The uvicorn defaults use the
underlying `websockets` library's defaults (20 s interval, 20 s timeout).
Under CPU starvation the server's ping response can miss the deadline
and the client closes. Client-side AudioWorklet probably doesn't resend
pings either.

### Improvement (perf + quality)
1. **Explicit WS ping** on both uvicorn configs:
   ```python
   admin_config = uvicorn.Config(app, ws_ping_interval=10,
                                 ws_ping_timeout=10, ...)
   ```
   Expected outcome: zero WS drops per meeting in the non-starved case.
2. **Silence watchdog** (file: new `_silence_watchdog_loop` in
   `server.py`). If `state==recording` and no audio chunk in 10 s,
   broadcast `{type: "meeting_warning", reason: "no_audio"}` on
   surviving WSes so the UI can flag "reconnect required" instead of
   looking hung.

---

## Stage 2 — Resample 48 kHz → 16 kHz

### Code path
`_handle_audio` at `server.py:5704`. Uses torchaudio Kaiser-windowed sinc.

### Observations
- Log says "torchaudio Kaiser sinc is effectively transparent and runs
  on GPU. Bandwidth cost (~3× at 48kHz) is trivial on local WiFi."
- Not called out as a bottleneck in any log line.

### Improvement
None needed. Path is fine.

---

## Stage 3 — ASR streaming (`scribe-asr` :8003)

### Code path
- `VllmASRBackend.process_audio` in `backends/asr_vllm.py:107`+.
- Buffers 4 s of audio (or 1 s + watchdog), WAV-encodes, b64s, POSTs to
  `/v1/chat/completions` on :8003.
- `_parse_qwen3_asr_response` extracts text + language from
  `"language X<asr_text>..."` format.
- LocalAgreement policy dedups by `segment_id + revision`.

### Observations
| Metric | observed | target |
|---|---|---|
| Direct probe, 1.5 s clip | 101–142 ms | < 400 ms ✓ |
| Duplicate finals under same-text, different seg_id | 1 event (dedup guard caught it) | 0 ideally |
| CPU during active ingest | 3–62 % | burstiness matches audio arrival |

### Root cause of dup finals
When `scribe-asr` restarted during my load test, scribe-main's ASR
backend reconnected but re-sent some buffered audio. vLLM's internal
cancellation state didn't prevent this from re-emitting as a new
seg_id. My `_recent_finals` dedup (`server.py:6005`-style) now catches
this, but the underlying replay shouldn't happen.

### Improvements (perf + quality)

1. **Filter ultra-short finals** that are pure filler ("えー", "ああ",
   "はい。"). The log has many of these each generating full translate
   round trips for no useful signal. Sample:
   ```
   15:06:34 seg 955eb827 text='たんですかですか。そうそうそうそう。ああ。'
   ```
   Add a min-chars gate (`text.strip() < 3 chars AND len(speakers) == 0`
   → drop) in `_process_event`.
2. **Per-segment confidence gate**. vLLM's chat response includes
   `logprobs` when `"logprobs": True` is added to the request body. If
   average logprob is below a threshold we drop or mark low-confidence
   so UI can grey it out.
3. **Retire the 62 %-peak CPU pattern**. scribe-asr currently runs with
   `--max-num-seqs 4` but only ever serves 1 seq at a time (no parallelism
   in the streaming ASR caller). Reducing to `--max-num-seqs 1 --enforce-eager`
   is already set; memory is ~3 GB so no win there. Leave as-is.

---

## Stage 4 — Diarization (`scribe-diarization` :8001)

### Code path
`SortformerBackend.process_audio` in `backends/diarize_sortformer.py:334`.
Rolling 16 s window, flush interval ~4 s. Each flush POSTs the
whole window's PCM to `/v1/diarize`, receives segments + 256-dim
embeddings, calls `_assign_global_cluster` for each segment (cosine
match against all known centroids).

### Observations
| Metric | observed |
|---|---|
| Direct probe, 3 s clip | 19–77 ms ✓ |
| Rolling window cadence | every 3–5 s |
| Global cluster count after 15 min | **27** (for probably 4–5 real speakers) |
| Speaker catch-up age p95 | **15 s**, max 39 s |

### Root cause of fragmentation
`_cluster_merge_threshold` is what `_assign_global_cluster` uses to
decide "same speaker" vs "new speaker". In the log:

```
Diar NEW: global speaker #17 (best existing match=0.53 threshold=0.55, #known=17)
```

0.53 is close — we're getting lots of near-misses. With 16 s windows
and noisy mic conditions, embeddings drift, and the margin
(`best_score - second_score >= 0.05`) requirement blocks many valid
merges. Then over 15 min we accumulate 27 centroids.

### Improvements (quality)

1. **Anneal the merge threshold within a session.** After N minutes,
   tighten merging: keep `threshold = 0.55` for the first minute, relax
   to `0.50` after 2 minutes. Rationale: embeddings stabilize with more
   data; tight threshold at start prevents mis-merges while we have few
   observations, loose threshold later captures same-speaker under voice
   drift.
2. **Active centroid cleanup.** Every 30 s, walk `_global_centroids` and
   merge pairs whose cosine ≥ 0.85 with each other. 27 → 5–8 reasonable
   clusters would radically improve UI clarity.
3. **Cap catch-up horizon.** p95 = 15 s means a speaker label can appear
   15 s after the text does. Options:
   - Hold finals with no speaker for up to 2 s before broadcast (so
     speaker attribution lands in sync), not 0 s.
   - Or broadcast immediately but with `speaker: "pending"` and surface
     the revision UI-side as a soft update (already does this, but UI
     could suppress the per-revision flash for late updates).

---

## Stage 5 — Translation (`autosre` shared Qwen3.5-35B :8010)

### Code path
`VllmTranslateBackend.translate` in `backends/translate_vllm.py:128`+.
Sends `/v1/chat/completions` with `chat_template_kwargs.enable_thinking=False`
+ `priority=-10`. 4 parallel workers (`translate_queue_concurrency=4`).

### Observations
| Metric | observed | target |
|---|---|---|
| Direct probe with `enable_thinking=false` | 261–397 ms | < 800 ms ✓ |
| Direct probe WITHOUT the flag | **1 510 ms**, 64 tokens wasted on internal reasoning | — |
| In-meeting p50 | 471 ms | ≤ 800 ms ✓ |
| In-meeting p95 | **2 842 ms**, max 5 012 ms | ≤ 1 500 ms ✗ |

### Root cause of p95 regression
Event-loop lag cascades onto translate worker queue. When the loop
stalls for 2 s, the worker that was about to `await httpx.post(...)`
waits those 2 s before it can even send the request. The direct probes
are fast because they bypass the queue.

### Improvements (perf + quality)
1. **Move translation workers onto a thread pool.** Right now all 4 live
   in the main asyncio loop (`asyncio.create_task(worker_loop())` x 4).
   Any event-loop stall blocks all of them. Refactor so each worker
   runs `loop.run_in_executor(None, blocking_translate)` with httpx sync,
   or give the translate queue its own dedicated event loop in a thread.
2. **Merge adjacent short segments before translating.** Segments like
   `'あ、あ。'` + `'うん、そうだね。'` would benefit from being merged
   into one translation call — less tokens per meeting, lower translate
   load, smoother EN output. Translate queue already has merge gating
   (`flush_merge_gate()` in `_process_event`); tighten the gate to wait
   50 ms before flushing if the upstream queue has pending events.
3. **Server-side caching of repeated phrases.** Meetings have lots of
   `"はい。"`, `"そうそう。"`, `"ああ"`. An LRU over the last 256
   (lang, text) → translation pairs saves ~20 % of translate calls in
   casual conversation. Low cost; pure win.
4. **Compile-mode warmup on meeting start.** First in-meeting translate
   at 15:09:25 took 855 ms; subsequent same-size translates are ~400 ms.
   Fire one throwaway translate at meeting-start time (from
   `_start_meeting`) so the first user-visible translate is already warm.

---

## Stage 6 — TTS

### Code path
`Qwen3TTSBackend` in `backends/tts_qwen3.py` (rewrote earlier). Routes
to `config.omni_tts_url` if set, else `config.tts_vllm_url`. Consumes
OpenAI-compatible `/v1/audio/speech` with `stream=true response_format=pcm`.

### Observations
| Endpoint | state |
|---|---|
| `:8002/:8012` legacy `scribe-tts` | **broken** — 500 on every request (`custom_voice does not support create_voice_clone_prompt`) |
| `:8022` new `scribe-tts-vllm` (when up) | studio c=1 p50 130 ms, p95 204 ms ✓ |
| `:8022` cloned under concurrency 4 | engine crashes (separate earlier finding) |

### Improvements
1. **Make the primary path the new vllm-omni.** Flip
   `SCRIBE_OMNI_TTS_URL=http://localhost:8022` + bring the container up.
   The 500s will stop immediately.
2. **Cloned-mode stability work** (already logged as a follow-up spike
   — cloned at c=1 first, then sweep).
3. **TTS budget-aware drop.** `_do_tts_synthesis` already has
   pre-/post-synth deadline checks that drop stale audio. Keep them.

---

## Stage 7 — WebSocket fanout

### Code path
`_broadcast` at `server.py:6392`. For-loop over `ws_connections` with
`await ws.send_text(...)`. On failure, enqueue for cleanup.

### Observations
Fast, non-blocking, no issues observed. A dead WS slot is detected and
removed.

### Improvement
1. **Batch broadcast.** When multiple events arrive within 50 ms, coalesce
   into one JSON array send. Reduces WS frame overhead at high event rates.
   Low priority until we profile the WS send path dominating any CPU.

---

## Stage 8 — Journal + raw PCM persistence

### Code path
- `storage.append_event` at `storage.py:424`. Sync `os.write` + fsync
  every 5 s.
- `AudioWriterProcess` — out-of-process, append-only raw PCM. Already
  optimal.

### Observations
Journal sync runs on the main event loop (sync `os.write` under a lock).
5-s fsync cadence is spaced out enough to not be a hot spot. File is
on NVMe (df shows `/dev/nvme0n1p3`) so fsync cost is < 10 ms.

### Improvement
None in short term. Longer-term: move journal to its own writer process
using the same pattern as `AudioWriterProcess`. Not urgent.

---

## Stage 9 — Event loop lag monitor

### Code path
`_loop_lag_monitor` at `server.py:585`. Samples every 500 ms, warns on
> 250 ms. Stores p50/p95 in `metrics.loop_lag_ms`.

### Observations
| | cumulative 10-min window |
|---|---|
| Total WARN events | 51 |
| Cadence | every ~10–13 s |
| Max observed | 2 591 ms |
| Distribution in log | consistent 2.1–2.5 s spikes, nothing between |

The regularity of the ~10 s cadence matches:
- `_speaker_catchup_loop` heartbeat interval (25 × 400 ms = 10 s)
- `_retry_failed_backends` interval (10 s)
- `_tts_health_evaluator` (interval unknown, likely 10–30 s)

### Root cause hypothesis
The 10-s-periodic 2 s lag is almost certainly one of the retry/evaluator
loops doing sync I/O on the event loop. Most likely:
- `tts_backend.check_health()` probe does 2 HTTP calls + a
  `/v1/audio/speech` synth probe — if legacy TTS is returning 500 fast
  and the probe has a 5 s timeout, we eat 5 s on every call.
- OR `_restart_container("scribe-tts")` via sync docker API call if
  TTS is flagged degraded. `docker restart` takes 1–3 s and blocks.

### Improvement (perf)
1. **Gate TTS health probe.** Only run the warmed-synth probe when TTS
   actually serves cleanly. If `tts_backend.last_error` is set to the
   known-legacy `custom_voice does not support create_voice_clone_prompt`
   text, skip the synth probe — no point hammering a known-dead endpoint.
2. **Move container restarts off the event loop.**
   `_restart_container` uses `docker` CLI via `asyncio.create_subprocess_exec`
   — should be non-blocking — but let me verify. If it's blocking, wrap
   in `asyncio.to_thread`.
3. **Sample at 100 ms instead of 500 ms** so spikes show the actual
   blocking function stack, not smeared averages.

---

## Stage 10 — Dev-mode auto-resume of interrupted meetings

### Code path
`server.py` around `Dev mode: auto-resuming interrupted meeting`.
When scribe-main restarts, it finds any `state==recording` meeting and
replays its audio through the full reprocess pipeline in parallel with
any new meeting.

### Observations
Today's cascade: 33-min interrupted meeting triggered a 5-chunk
diarization sweep that took 1.5 min, during which a new meeting's
audio drift grew to 41 s behind wall clock.

### Improvement (perf, reliability)
1. **Gate on age.** `if (now - meeting.audio_last_modified) > 120:
   mark interrupted and DO NOT auto-resume`. Instead, surface in UI as
   "unfinished meeting, run reprocess manually".
2. **Never run reprocess in parallel with a live meeting.** If there's
   a new meeting, queue the reprocess for after the live one ends.

---

## Ranked action list (impact × effort)

| # | Action | Impact | Effort | File |
|---|---|---|---|---|
| 1 | Explicit WS `ws_ping_interval=10, ws_ping_timeout=10` | High (fixes 60-s disconnects) | 5 min | `server.py:7209,7233` |
| 2 | Gate dev auto-resume on `audio_age < 120 s` | High (fixes startup cascade) | 10 min | `server.py` near "auto-resuming" log |
| 3 | Flip `SCRIBE_OMNI_TTS_URL=http://localhost:8022` + bring `vllm-tts` up | High (fixes every TTS synth 500) | 2 min env + compose up | env |
| 4 | Skip TTS synth probe when `last_error` is the legacy 500 text | Medium (event-loop lag) | 15 min | `tts_qwen3.py:check_health` |
| 5 | Silence watchdog broadcast (`meeting_warning`) | Medium (visibility) | 30 min | new loop in `server.py` |
| 6 | Anneal diarize merge threshold + centroid consolidation pass | Medium (quality — 27 → 5 speakers) | 1 h | `diarize_sortformer.py:_assign_global_cluster` |
| 7 | Translate LRU cache on (lang, text) | Medium (20 % fewer translate calls) | 30 min | `backends/translate_vllm.py` |
| 8 | Warm translation with a throwaway call on meeting start | Medium (first-translate latency) | 20 min | `_start_meeting` |
| 9 | Drop ultra-short pure-filler finals before translate | Low-medium (quality + cost) | 10 min | `server.py:_process_event` |
| 10 | Move translation queue into thread pool | Medium-High (p95 from 2 842 → ~800 ms) | 2–3 h | `translation/queue.py` |
| 11 | Loop-lag sampler at 100 ms + per-tick call-site annotation | Low (diagnostic) | 1 h | `server.py:_loop_lag_monitor` |

Items 1–4 take under 30 minutes combined and remove 80 % of today's
actual pain. 6–10 are where quality wins live.

---

## Open questions for measurement

- We have **no ASR WER baseline** on real meeting audio. Until we
  capture a WER number (using `benchmarks/asr_accuracy_latency.py`
  against the redacted consented fixture from the Omni consolidation
  plan) we can't tell whether to dial up Qwen3-ASR's temperature /
  sampling params.
- **No TTS MOS baseline.** Same story — the Omni plan's
  `benchmarks/tts_quality_mos.py` is the right tool; just needs the
  fixture.
- Diarization fragmentation could be measured with a DER (diarization
  error rate) against ground-truth labels on the redacted fixture.
  Currently we have only a "27 global speakers per 15 min" heuristic.

Once the fixture lives under `/data/meeting-scribe-fixtures/`, these
numbers become reproducible + make every suggestion above gate-able
rather than speculative.
