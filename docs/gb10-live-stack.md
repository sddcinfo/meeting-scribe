# GB10 Live Stack

This document records the production architecture for live interpretation on
the GB10 and the benchmark evidence required before asking for manual testing.

## Runtime Topology

`meeting-scribe.service` owns the web app, capture pipeline, meeting state,
WebSocket fanout, and local room playback routing. Model services run beside it:

| Service | Port | Owner | Purpose |
|---|---:|---|---|
| `autosre-vllm-local` | 8010 | auto-sre | Qwen3.6-35B-A3B-FP8 translation, name extraction, optional refinement |
| `scribe-asr` | 8003 | `docker-compose.gb10.yml` | Qwen3-ASR-1.7B real-time transcription |
| `scribe-diarization` | 8001 | `docker-compose.gb10.yml` | pyannote community-1 speaker diarization |
| `scribe-tts` | 8002 | `docker-compose.gb10.yml` | Qwen3-TTS interpretation audio |

The live meeting path is:

1. Capture PCM from the selected source, normally the Poly USB input.
2. Segment and submit audio to ASR without blocking capture.
3. Translate finalized utterances through the shared vLLM endpoint.
4. Queue translated text for TTS only when the target language is TTS-native.
5. Fan out TTS audio to web listeners and optionally to the local room sink.
6. Persist raw PCM, transcript events, translations, and speaker timelines for
   replay and future regression tests.

## Production TTS Choice

Production TTS is `Qwen/Qwen3-TTS-12Hz-0.6B-Base` through
`faster-qwen3-tts==0.2.6` on the TTS image's pinned PyTorch 2.11 / CUDA 13
wheel stack. The older baseline backend and `faster-qwen3-tts==0.2.5` were not
stable enough on GB10:

- baseline Qwen TTS repeatedly wedged on short English phrases such as
  `Good evening.`
- the older faster runtime hit CUDA graph / CUDA state failures on GB10
- unguarded faster 0.2.6 could intermittently produce 163 s of audio for a
  short phrase

The current configuration keeps the faster backend, but bounds output by phrase
length and marks the worker fatal on runaway generation. Docker then restarts
the container and the app retries once after `/health` returns healthy.

Important environment variables:

| Variable | Production value | Why |
|---|---:|---|
| `TTS_BACKEND` | `faster` | Required for real streaming and acceptable latency on GB10. |
| `TTS_FASTER_XVEC_ONLY` | `1` | Keeps cloning on x-vector references instead of full prompt cloning. |
| `TTS_FASTER_NON_STREAMING_MODE` | `0` | Keeps upstream-style streaming generation. |
| `TTS_SERVER_SYNTH_TIMEOUT_S` | `45` | Long enough for cold graph capture; still fatal if synthesis wedges. |
| `TTS_MAX_AUDIO_S` | `12` | Absolute upper bound for any generated clip. |
| `TTS_MAX_AUDIO_BASE_S` | `1.5` | Dynamic output budget intercept. |
| `TTS_MAX_AUDIO_PER_CHAR_S` | `0.22` | Dynamic output budget slope. |
| `SCRIBE_TTS_CONTAINER_CONCURRENCY` | `1` | One resident TTS replica on GB10; the `tts-pool` profile is experimental. |
| `SCRIBE_TTS_QUEUE_MAXSIZE` | `32` | Live TTS backlog cap. Prevents one slow synthesis burst from creating a long delayed playback tail. |
| `SCRIBE_TTS_WORKER_COUNT` | `2` | Enough app workers to keep translation handoff moving while the single TTS backend serializes GPU work. |
| `SCRIBE_TTS_MAX_SPEECH_LAG_S` | `20` | Drops TTS that can no longer play close to the source speech instead of speaking stale content into the room. |

The dynamic TTS budget is:

```text
min(TTS_MAX_AUDIO_S, TTS_MAX_AUDIO_BASE_S + TTS_MAX_AUDIO_PER_CHAR_S * chars)
```

This mirrors the saved-meeting benchmark's pathological-output gate. If a
generation exceeds the budget, the server emits only the bounded audio, marks
itself fatal, returns unhealthy on `/health`, and exits so Docker restarts it.

## Health And Recovery

TTS health is deliberately stricter than "model object exists":

- `/health` returns 503 while synthesis exceeds the configured server timeout.
- `/health` returns 503 after any fatal synthesis exception or runaway output.
- a minimal CUDA operation is part of health so a poisoned CUDA context does
  not keep reporting healthy.
- app-side TTS retries wait for `/health` before retrying once.

This was added because a prior failure mode returned HTTP 500 from synthesis
while `/health` still returned 200, causing the live app and benchmark to keep
routing into a broken container.

## Baseline Gate

Use the saved-meeting live stack regression before manual testing:

```bash
CI_LOCAL_PY=.venv/bin/python .venv/bin/python scripts/ci_local.py --only quality
```

The quality lane reads private meeting IDs from the gitignored
`.local/live_stack_meetings.txt` file, or from `LIVE_STACK_MEETING_IDS`.
Hook output is written to `/tmp` by default so transcript-bearing reports do
not land in the repo.

Current accepted private GB10 baseline passed with:

- ASR error rate: 0.0
- ASR p95 latency: 487 ms
- ASR same-script p95 character distance: 0.385
- translation error rate: 0.0
- translation wrong-script count: 0
- TTS error rate: 0.0
- TTS max audio duration: 5.12 s
- TTS pathological count: 0

The benchmark intentionally includes ASR language mismatches in the report but
does not fail on them when the saved journal's language label disagrees with
the currently transcribed script. Same-script rows drive the ASR distance gate.

## Manual Test Readiness

Do not ask for live manual audio playback until all of these are true:

1. `curl http://localhost:8002/health` returns `status=healthy` and
   `backend=faster`.
2. `docker ps` shows `scribe-tts`, `scribe-asr`, and `scribe-diarization`
   healthy.
3. The saved-meeting live stack regression passes.
4. The benchmark output is saved under `benchmarks/results/` with a name that
   describes the runtime being tested.
