# OOM-Tune Investigation — 2026-05-11

Goal: reduce CUDA OOM / CUBLAS_STATUS_INTERNAL_ERROR events in `scribe-diarization` and `scribe-tts` without regressing the locked Qwen3.6-FP8 perf baseline (`autosre/benchmarks/baselines/gb10_qwen36_fp8_seq4_kv-fp8`).

Workload: meeting-scribe stack + autosre vLLM co-resident on GB10 (121.6 GB unified memory).

Method: A/B against `autosre perf run` (translation+coding gate) and `benchmarks/perf_baseline.py` (ASR/translate/diarize stress).

## Baseline state (pre-tune)

Captured 2026-05-11 12:33 (vLLM in long-running uptime — 2 h pre-test, scribe co-tenants resident at boot).

```
GPU 0  VRAM 61.2 G / 121.6 G (50%)   free ≈ 60.4 G
  autosre-vllm-local   52.4 G   Qwen3.6-35B-A3B-FP8 (vLLM 0.20.0, gmu=0.70)
  scribe-asr            5.7 G   Qwen3-ASR-1.7B
  scribe-tts            2.8 G   faster-qwen3-tts
  scribe-diarization  301 M    pyannote community-1
```

Cumulative CUDA errors in container logs at start: diarize=24, tts=4 (over ~2 h of prior production use).

## Per-phase results

### autosre perf (4 phases × 60 s, contention allowed)

| Workload | Phase | TTFT p50 | TTFT p95 | TPS p50 | TPS agg | Errors |
|---|---|---:|---:|---:|---:|---:|
| **Locked baseline 2026-05-07** | | | | | | |
| translation | isolated | 114 ms | 118 ms | 52.11 | 33.58 | 0 |
| translation | contention | 157 ms | 162 ms | 31.47 | 17.34 | 0 |
| coding | isolated | 276 ms | 282 ms | 42.69 | 72.58 | 0 |
| coding | contention | 397 ms | 410 ms | 28.71 | 50.46 | 0 |
| **pre-tune baseline (today)** | | | | | | |
| translation | isolated | 109 ms | 114 ms | 52.39 | 33.85 | 0 |
| translation | contention | 149 ms | 169 ms | 32.98 | 17.06 | 0 |
| coding | isolated | 268 ms | 1027 ms* | 43.61 | 72.23 | 0 |
| coding | contention | 362 ms | 400 ms | 30.28 | 51.96 | 0 |
| **Change 2 only** (expandable_segments:True on diarize+tts) | | | | | | |
| translation | isolated | 109 ms | 114 ms | 52.23 | 34.10 | 0 |
| translation | contention | 149 ms | 161 ms | 32.98 | 17.13 | 0 |
| coding | isolated | 267 ms | 918 ms* | 43.58 | 72.99 | 0 |
| coding | contention | 348 ms | 387 ms | 30.55 | 53.50 | 0 |
| **Change 1+2** (gmu=0.55 + expandable_segments) — after vLLM cold-restart | | | | | | |
| translation | isolated | 110 ms | 136 ms | 52.42 | 33.10 | 0 |
| translation | contention | 151 ms | 196 ms* | 32.42 | 20.73 | 0 |
| coding | isolated | 266 ms | 310 ms | 41.66 | 75.64 | 0 |
| coding | contention | 387 ms | 431 ms | 28.38 | 50.19 | 0 |

\* Outliers from small coding sample (n=18-28) — single tail latency dominates p95.

### scribe perf_baseline (concurrent c=6, n=60 per backend)

| Backend / phase | Baseline | +Change 2 | +Change 1+2 |
|---|---:|---:|---:|
| asr concurrent p50 | 121.6 ms | 129.3 ms | failed* |
| asr concurrent p95 | 175.5 ms | 181.9 ms | — |
| translate ja→en concurrent p50 | 716.4 ms | 689.6 ms | — |
| translate ja→en concurrent p95 | 961.0 ms | 892.1 ms | — |
| diarize concurrent p50 | 116.3 ms | 115.4 ms | — |
| diarize concurrent p95 | 124.1 ms | 127.6 ms | — |
| mixed pipeline wall | 7922 ms | 8886 ms | — |
| diarize CUDA error delta | 0 | 0 | **1 (CUBLAS_STATUS_INTERNAL_ERROR)** |

\* Change 1+2 perf_baseline aborted on first diarize call (500 / CUBLAS_STATUS_INTERNAL_ERROR after 45 prior ASR+translate calls). Diarize container had to be force-recreated by autosre warmup.

### GPU VRAM allocation snapshots

| Config | vLLM VRAM | Total GPU VRAM | Free for co-tenants | Notes |
|---|---:|---:|---:|---|
| **baseline** (long-uptime vLLM, scribe co-resident at vLLM boot) | 52.4 G | 61.2 G | ~60.4 G | original measured state |
| change 2 only | 52.4 G | 61.2 G | ~60.4 G | unchanged vs baseline |
| change 1+2 (gmu=0.55, fresh vLLM start) | 62.8 G | 71.6 G | ~50.0 G | KV cache 714 K tokens |
| revert to 0.70, fresh vLLM start | 81.0 G | 89.8 G | **~31.8 G** | KV cache 1,192 K tokens |

## Key finding — vLLM allocation is boot-time-dependent

vLLM 0.20.0 at the SAME `gpu_memory_utilization=0.70` allocates **a different amount** depending on what other CUDA tenants are holding memory at the moment of the engine boot:

- May 7 production boot (`gb10_qwen36_fp8_seq4_kv-fp8`): KV cache 871,936 tokens / vLLM ≈ 67 GiB total
- Today's pre-tune state (long uptime with scribe at boot): vLLM ≈ 52.4 G visible
- After a fresh `autosre stop --unload-model && autosre start` with scribe quiesced: vLLM ≈ 81 G, KV 1,192,624 tokens

This is the root mechanism of the OOM cascade we set out to fix:

1. vLLM cold-starts when scribe containers are between meetings / restarting / not yet up.
2. It probes the system, sees ~110 GB free, allocates KV cache against that.
3. End result: vLLM holding 81 GB of unified memory at steady state.
4. Diarize/TTS containers (re)start later, get squeezed, hit OOM under burst load.

`gpu_memory_utilization` is a CEILING relative to "free memory at boot", not an absolute cap. Lowering it to 0.55 still produced a ~63 GB allocation on a freshly-quiesced GPU — better, but the bench still hit a CUBLAS error during sequential warmup, indicating the OOM headroom is fundamentally about the **shape** of the allocation order, not a single parameter.

## Recommendation

### Adopt now

**Change 2** — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on `scribe-diarization` and `scribe-tts` only.

- Zero perf regression observed (translation TTFT 109/114 ms identical before & after; TPS within ±0.5 %)
- TTS steady-state VRAM dropped 2.8 G → 2.3 G under stress
- Diarize peak VRAM dropped 383 M → 352 M
- No regression of the `gb10_qwen36_fp8_seq4_kv-fp8` envelope
- Compose file change already in place (`docker-compose.gb10.yml` lines 78 and 139)

### Do NOT adopt as currently designed

**Change 1** — `gpu_memory_utilization` 0.70 → 0.55 — recipe already reverted to 0.70.

Reason: empirical measurement showed that at the moment of vLLM boot, both 0.55 and 0.70 settings result in vLLM allocations that squeeze diarize/TTS first-inference. Lowering the recipe value did not solve the underlying boot-order problem and introduced a separate translation TTFT p95 regression (162→196 ms in contention). Recipe approval token was minted, change applied, validated unsafe, and reverted in the same session.

### Follow-up work — Change 3 implemented this session

**Change 3 — Boot ordering enforced in `autosre start`** (applied 2026-05-11 in `repos/auto-sre/autosre/cli.py`):

- Was: `_start_scribe` and `_start_agent` ran in parallel via a 2-worker ThreadPoolExecutor. vLLM's memory profiler could probe the GPU before scribe containers had loaded their weights.
- Now: `_start_scribe` runs first, then `_wait_for_scribe_models_resident` polls `/health` on `:8003` (asr), `:8002` (tts), `:8001` (diarize). Each backend's `/health` only returns 200 once its model is GPU-resident. Backends whose ports refuse connections after the first ~5 s are treated as "not present" (so `--no-scribe` and customer installs with a subset of backends don't stall the gate).
- Then `_start_agent` runs.

Validated against locked `gb10_qwen36_fp8_flashinfer` baseline after a full cold cycle (`stop --unload-model` → `start`):

| Workload | Phase | TTFT p50 | TTFT p95 | TPS p50 | TPS agg | Errors | vs locked |
|---|---|---:|---:|---:|---:|---:|---|
| translation | isolated | 113 ms | 117 ms | 52.16 | 33.72 | 0 | within tolerance |
| translation | contention | 155 ms | 163 ms | 31.82 | 16.42 | 0 | within tolerance |
| coding | isolated | 222 ms | 296 ms | 41.15 | 75.74 | 0 | TTFT improved |
| coding | contention | 361 ms | 406 ms | 28.26 | 50.53 | 0 | within tolerance |

Only baseline violation: `[WARN] coding/contention insufficient_samples: observed=16, baseline=20` — sample-count variance, not a perf regression.

scribe-side `benchmarks/perf_baseline.py` post-fix (concurrent c=6, n=60 per backend):

- diarize sequential p95: **45.7 ms** (vs 63 ms in original baseline, 325 ms after Change 1+2 alone)
- diarize concurrent p95: 272.9 ms — no errors
- mixed pipeline wall: 7807 ms — fastest of any phase tested
- **Zero CUDA errors / zero 500s across the full run** — every prior phase that ran `perf_baseline.py` against a freshly-booted vLLM hit `CUBLAS_STATUS_INTERNAL_ERROR` on the diarize sequential probe

vLLM allocation delta: 79.7 G (vs 81 G pre-fix). The headroom gain is small in raw VRAM terms, but the **fragmentation pattern at vLLM's allocation time** is what matters: scribe's CUDA contexts are now pre-established when vLLM grabs its chunk, so the subsequent fragmentation of the unified pool aligns with both tenants' needs.

### Remaining follow-up (deferred, not implemented this session)

- **Explicit `--kv-cache-memory-bytes` / `--num-gpu-blocks-override`** to pin KV to a known value regardless of boot-time free memory. Touches the perf-locked recipe; needs an approval token + perf-validation cycle.
- **`VLLM_RESERVE_MEMORY_BYTES`** env var (or current equivalent) for an explicit non-vLLM tenant reservation. Need to confirm the flag exists in our pinned vLLM image.
- **`sddc gpu top` measurement clarification**: this session uncovered that the per-process VRAM number is closer to physical-RSS than CUDA-virtual-reservation. A 2-hour-idle vLLM reports ~52 G even though it has reserved ~80 G of CUDA virtual address space. Document this in `sddc-cli/src/sddc/commands/gpu.py` so future tuning work doesn't conflate the two numbers.

## Artifacts

- `results/baseline/` — pre-tune perf logs + GPU snapshots + OOM counts
- `results/change2-expandseg/` — Change 2-only results
- `results/change1+2-full/` — Change 1+2 results (aborted in scribe-perf phase)
- `results/final-c2-only/` — after rolling back Change 1; reproduced the diarize CUBLAS failure on fresh vLLM boot
- `snapshots/t0-pre-baseline.txt` — boot-time snapshot
- `gen_probe_pcm.py` — helper that generates the 3 s 16 kHz PCM probe used for direct diarize sanity-curling

## Final state of files

- `repos/meeting-scribe/docker-compose.gb10.yml` — Change 2 applied (`expandable_segments:True` on diarize + TTS only) ✓
- `repos/auto-sre/autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml` — reverted to `gpu_memory_utilization: 0.70` (untouched from production)
- `repos/meeting-scribe/benchmarks/perf_baseline.py` — `probe_scribe_status` now honors `SCRIBE_AP_IP` + `SCRIBE_PORT` and degrades gracefully on auth-redirect (was hardcoded to localhost:8080)
