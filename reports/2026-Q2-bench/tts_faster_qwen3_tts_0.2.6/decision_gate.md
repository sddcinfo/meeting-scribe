# Track B.3 — faster-qwen3-tts 0.2.5 → 0.2.6 (sidecar bench, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

Patch upgrade. Bench-first: the new image runs as a `--profile bench`
sidecar before any production cutover.

## Pins

| Field | Value |
|---|---|
| `challenger_image` | `tts-bench-0.2.6` (FROM `containers/tts/Dockerfile`, only the `faster-qwen3-tts==0.2.6` pin changed) |
| `baseline_image` | `scribe-tts:0.2.5` (production digest captured at bench-time) |
| `staging_port` | 8022 (loopback-only) |
| `bench_window` | TBD |

## Files

* `meeting-scribe/containers/tts-bench-0.2.6/Dockerfile` — clone of
  `containers/tts/Dockerfile`, only the `faster-qwen3-tts` pin
  changed.
* `meeting-scribe/docker-compose.gb10.yml` — add
  `scribe-tts-bench-0.2.6` under `profiles: ["bench"]` on
  `127.0.0.1:8022:8022`.
* `meeting-scribe/scripts/bench/run_track_b3.sh` — runs
  `tts_quality_mos.py` + `tts_concurrent_load.py` against 8022;
  compares against production 8002.

## Gates (sidecar bench)

* TTFA p95 ≤ 110 ms
* concurrent-load p95 within ±10% of production 8002
* no CUDA-graph errors

## Production cutover (only after bench PASS)

Capture the deployed image digest:

```bash
docker inspect scribe-tts --format '{{index .RepoDigests 0}}'
```

Record it in this file under "Rollback identifier" — that immutable
digest is the rollback target. NOT a mutable `rollback-pre-*` tag.

Cutover steps: rebuild + restart `scribe-tts:8002` at the new image,
60s health check, auto-rollback re-pulls the recorded immutable
digest on TTFA p95 > 110 ms or error rate > 1% in 10-min post-cutover
monitoring.

## Result

(populated post-run)
