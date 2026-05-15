# Track B.5 — vLLM 0.20.1 ASR (sidecar bench, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

Bench-first: the ASR slice runs as a `--profile bench` sidecar
on a staging port. Production `containers/vllm-asr/Dockerfile` is
NOT edited until cutover.

Slice is intentionally narrow — only meeting-scribe ASR. The
autosre vLLM bump (port 8010 translation) is OUT OF SCOPE.

## Pins

| Field | Value |
|---|---|
| `challenger_image` | `vllm/vllm-openai:v0.20.1-aarch64-cu130-ubuntu2404@sha256:<digest>` |
| `baseline_image` | `containers/vllm-asr/Dockerfile` production digest |
| `staging_port` | 8033 (loopback-only) |
| `production_port` | 8003 |
| `bench_window` | TBD |

## Pre-flight

`docker manifest inspect vllm/vllm-openai:v0.20.1-aarch64-cu130-ubuntu2404`
must return — if the tag is not yet published, defer the track.

## Files

* `meeting-scribe/containers/vllm-asr-bench-0.20.1/Dockerfile` —
  `FROM <pinned-digest>` (NEVER `:latest`).
* `meeting-scribe/docker-compose.gb10.yml` — add
  `qwen3-asr-bench-0.20.1` under `profiles: ["bench"]` on
  `127.0.0.1:8033:8033`.
* `meeting-scribe/scripts/bench/run_track_b5.sh` — runs
  `asr_accuracy_latency.py` against 8033 with the existing Fleurs
  MANIFEST; compares against production 8003 captured separately.

## Gates (sidecar bench)

* ASR p50 (8033) within ±10% of production p50 (8003)
* `meeting-scribe demo-smoke` against the bench sidecar passes

## Production cutover (separate maintenance window)

Capture current ASR image digest:

```bash
docker inspect qwen3-asr --format '{{index .RepoDigests 0}}'
```

Record as immutable rollback identifier. Bump
`containers/vllm-asr/Dockerfile:21`; rebuild + restart; 60s health
check; auto-rollback re-pulls recorded immutable digest on
regression > 10% or error rate > 1%.

Runbook: `meeting-scribe/docs/runbooks/vllm_0.20.1_asr_cutover.md`
(create at cutover time).

## Result

(populated post-run)
