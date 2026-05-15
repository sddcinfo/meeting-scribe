# Track B.4 — transformers ==5.6.2 (TTS-only sidecar, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

Test variable is `transformers==5.6.2` exactly (NOT a range). Two
images are built from the SAME Dockerfile — only the `transformers`
pin line differs. This isolates the migration as a single-line delta
so the bench can attribute any regression to the library bump.

Decoupled from B.3: B.4 is its own maintenance window. If B.3
promoted, the B.4 baseline image incorporates the B.3-accepted
TTS pin; if B.3 rejected, the baseline carries the production TTS pin
unchanged.

## Pins

| Field | Value |
|---|---|
| `baseline_image` | `tts-bench-tx4-baseline:<digest>` — clone of B.3-accepted Dockerfile, current `transformers` pin |
| `challenger_image` | `tts-bench-tx5-challenger:<digest>` — same Dockerfile, ONLY `transformers==5.6.2` line changed |
| `baseline_port` | 8023 (loopback-only) |
| `challenger_port` | 8024 (loopback-only) |
| `bench_window` | TBD |

## Files

* `meeting-scribe/containers/tts-bench-tx4-baseline/Dockerfile`
  — explicit baseline image
* `meeting-scribe/containers/tts-bench-tx5/Dockerfile` — challenger,
  only differs in `transformers==5.6.2`
* `meeting-scribe/docker-compose.gb10.yml` — add
  `scribe-tts-bench-tx4-baseline` (8023) and
  `scribe-tts-bench-tx5-challenger` (8024) under
  `profiles: ["bench"]`; both bound `127.0.0.1:<port>:<port>`
* `meeting-scribe/scripts/bench/run_track_b4.sh`
  — bench challenger 8024 vs baseline 8023; production 8002
  untouched.

## Gates

* TTS-on-tx5 challenger delta within ±5% of tx4 baseline-sidecar
* smoke pass

## Production cutover (separate maintenance window from B.3)

Capture deployed image digest:

```bash
docker inspect scribe-tts --format '{{index .RepoDigests 0}}'
```

Record as the immutable rollback identifier (NO mutable
`rollback-pre-*` tag). Rebuild + restart at the new image; 60s
health check; auto-rollback re-pulls the recorded immutable digest.

## Result

(populated post-run)
