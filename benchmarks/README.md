# Benchmarks — Meeting Scribe on GB10

Performance benchmarks for the Meeting Scribe model stack on NVIDIA DGX
Spark (GB10).  This document tracks the **current production
configuration**.  Historical comparison points (Qwen3.5-INT4 +
optimisation sweeps from 2026-04-08) are archived in
`BASELINE-2026-04-13.md` and `DEEP-ANALYSIS-2026-04-13.md`.

## Hardware

- **GPU**: NVIDIA GB10 (Blackwell, SM121)
- **Memory**: 128 GB LPDDR5X unified (273 GB/s)
- **CPU**: Grace ARM64 (20 cores)
- **Driver**: 580.126.09 (required — 590.x has CUDAGraph deadlock on unified memory)
- **CUDA**: 13.0 (via forward compatibility from container)
- **vLLM**: `0.19.1rc1.dev391+g80b18230e` on `vllm/vllm-openai:nightly`

## Translation model — production

- **Model**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **Architecture**: hybrid Mamba+MoE — 35 B total, ~3 B active per token
- **Quantization**: native FP8 (no calibration; vLLM auto-detects)
- **Context**: native 262 144 tokens (no YaRN)
- **vLLM image**: `vllm/vllm-openai:nightly`
- **Recipe**: `repos/auto-sre/autosre/backends/recipes/qwen3.6-fp8-nightly.yaml`
- **Memory**: weights ~35 GB; vLLM allocation under
  `gpu_memory_utilization=0.70` reserves up to ~89 GB; combined-stack
  steady-state runs ~117 GB / 128 GB (≈93%).

## Live measurements — Qwen3.6-FP8 (post-hardening, 2026-04-19)

Captured against the live production endpoint after the
`CUBLAS_WORKSPACE_CONFIG=:4096:8` + `VLLM_MARLIN_USE_ATOMIC_ADD=1`
hardening landed.  Full reproducible numbers in
`reports/reliability/post_hardening_validation/`.

### Translation throughput + latency (single-resident perf bench)

Source: `reports/phase5/decision_gate_2026-04-18.md` (post-tuning
flashinfer profile, perf-bench numbers).  This is the current
production envelope.

| Workload   | Phase       | Samples | TTFT p50 | TTFT p95 | TTFT p99 | TPS p50 | TPS agg | Errors |
|------------|-------------|--------:|---------:|---------:|---------:|--------:|--------:|-------:|
| translation | isolated   | 184     | 116 ms   | 120 ms   | 122 ms   | 51.56   | 33.22   | 0      |
| coding      | isolated   | 32      | 270 ms   | 275 ms   | 275 ms   | 42.53   | 72.47   | 0      |
| translation | contention | 118     | 158 ms   | 175 ms   | 195 ms   | 31.87   | 18.89   | 0      |
| coding      | contention | 14      | 398 ms   | 414 ms   | 422 ms   | 27.29   | 51.06   | 0      |

The contention column is what matters for live meeting + coding-agent
coexistence: translation TTFT p99 195 ms holds the ~400 ms live-path
SLO with comfortable margin.

### Live translation regression suite

Source: `reports/reliability/post_hardening_validation/translate_qwen36_hardened.jsonl`.
Re-running the exact 72-pair JA↔EN burst that triggered the
`CUBLAS_STATUS_INTERNAL_ERROR` crash on 2026-04-18 23:45.

| Metric                           | Value |
|----------------------------------|------:|
| Pairs sent                       | 72    |
| Successful responses             | 72    |
| Errors / 500s / engine restarts  | 0     |

The hardening fully covered the failure mode.

### Slide pipeline regression suite

Source: `reports/reliability/post_hardening_validation/slides_qwen36_hardened.jsonl`.
Phase 4a fixture deck (`tests/fixtures/test_slides.pptx`) translated
through the production slides path in four directions.

| Metric                              | Value      |
|-------------------------------------|-----------:|
| Slide translations                  | 12 (4 × 3) |
| Parse failures                      | 0          |
| ID coverage failures                | 0          |
| `runs_returned == runs_requested`   | 12 / 12    |
| Latency range                       | 217–1030 ms |

### Refinement context-window sweep

Source: `reports/context_window_sweep/2026-04-19/summary.md`.  Sweep
across `[0, 2, 4, 6, 8, 12]` prior segments threaded into the system
prompt, against the 107 finalised utterances of meeting `28e55f5f`.

| Window | n   | Latency p50 | Latency p95 | Mean prompt tok | Mean completion tok |
|-------:|----:|------------:|------------:|----------------:|--------------------:|
| 0      | 107 | 333 ms      | 543 ms      | 118.4           | 10.9                |
| 2      | 107 | 351 ms      | 521 ms      | 182.8           | 10.6                |
| 4      | 107 | 382 ms      | 535 ms      | 231.0           | 10.7                |
| 6      | 107 | 366 ms      | 545 ms      | 277.3           | 10.7                |
| 8      | 107 | 392 ms      | 560 ms      | 322.4           | 10.8                |
| 12     | 107 | 389 ms      | 565 ms      | 405.4           | 10.7                |

Production default: **`refinement_context_window_segments = 4`**.
Quality plateaus at ≥4; anything above 8 is wasted prompt budget.
Live path stays stateless (latency-SLO sensitive); refinement path
absorbs the cost.

## Translation quality — Qwen3.6-FP8 vs prior 3.5-INT4 baseline

Source: `reports/phase3/translation_ab_2026-04-18.md` (sacreBLEU on the
72-pair `en_ja_meeting_v2` corpus, EN→JA scored with the MeCab
tokenizer).

| Direction | Qwen3.5-INT4 (baseline) | Qwen3.6-FP8 | Δ      | Gate (≥ −1.0) |
|-----------|------------------------:|------------:|-------:|--------------:|
| EN → JA   | 55.57                   | **66.97**   | **+11.40** | PASS — major improvement |
| JA → EN   | 55.44                   | 54.63       | −0.81  | PASS          |

EN → JA is scribe's primary live direction (English speakers →
Japanese viewers).  The +11.4 BLEU uplift is the headline reason 3.6
is in production despite the +13 GB weight delta.

## Reliability findings + mitigations

| Date       | Issue | Mitigation |
|------------|-------|------------|
| 2026-04-18 23:45 | `CUBLAS_STATUS_INTERNAL_ERROR` in MoE router gate BF16 GEMM under concurrent prefill — not OOM, GPU kernel internal error | `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `VLLM_MARLIN_USE_ATOMIC_ADD=1` env vars in the production recipe.  Documented in `reports/reliability/cublas_crash_2026-04-18.md`. |
| 2026-04-18 (Phase 5.5 soak) | PSI full-avg10 stalls (36 s on vllm, 35 s on scribe-asr) under synthetic perf-harness coding burst | Acknowledged as workload-specific (no concurrent coding agent during real meetings).  Re-soak on hardened profile deferred until a dedicated window. |
| Pre-tuning 3.6-FP8 | Coding TTFT p99 751 ms (3.2× over gate) | `--attention-backend=flashinfer` brought it to 279 ms.  Recipe-guard now flags `--attention-backend` as perf-sensitive (`repos/auto-sre/autosre/hooks_backend/recipe_guard.py`). |

## Benchmark scripts

### `scripts/context_window_sweep/sweep.py`

Sweeps the refinement context-window size against any completed
meeting.  Used to pick the production default of 4.

```bash
python3 scripts/context_window_sweep/sweep.py \
    --meeting-id <meeting-uuid> \
    --url http://localhost:8010 \
    --windows 0 2 4 6 8 12 \
    --limit 0 \
    --out reports/context_window_sweep/<date>
```

### `repos/meeting-scribe/benchmarks/translation_benchmark.py`

The 72-pair JA↔EN regression corpus.  Used for both the live regression
test and the Phase 3 quality A/B that produced the +11.4 BLEU EN→JA
number above.

```bash
python3 benchmarks/translation_benchmark.py \
    --url http://localhost:8010 \
    --corpus benchmarks/corpus/en_ja_meeting_v2.jsonl \
    --no-score --output /tmp/translate_smoke.jsonl
```

### `scripts/phase4/slides_ab.py`

Drives the fixture deck through the slide-translation path.  Used as
the slide-pipeline regression suite (parse failures, ID coverage,
latency).

```bash
python3 scripts/phase4/slides_ab.py dump \
    --url http://localhost:8010 \
    --label qwen36_hardened \
    --out-dir /tmp/slides_check
```

### `repos/auto-sre/` perf harness

`autosre perf run --baseline gb10_qwen36_fp8_flashinfer` runs the full
translation + coding workload mix and compares against the committed
baseline.  Default baseline pointer was switched from
`gb10_qwen35_int4_tuned` → `gb10_qwen36_fp8_flashinfer` on 2026-04-19
together with the production model default flip.

## Known issues + watch list

- **Driver 590.x**: CUDAGraph capture deadlock on GB10 unified memory.  Stay on 580.x.
- **Driver 595**: Beta, not in apt for DGX Spark — wait for official release.
- **Triton FP8 fallback**: vLLM upstream has no SM121-tuned kernel configs for Qwen3.6-FP8; the Triton path is the working but suboptimal default.  Re-evaluate when CUTLASS Blackwell-native FP8 kernels stabilise.
- **MoE Marlin disabled on SM121**: `_apply_sm121_fp8_fix` enforces the Triton MoE path because the Marlin kernel crashes on this device.

## Future work

- **Phase 5.5 re-soak on the hardened profile** in a realistic-pacing
  configuration (no concurrent coding agent), to lock in the
  combined-stack memory + PSI envelope.
- **Live-path context window** experiment.  Requires a window-2 sweep
  with the SLO test in place; deferred until refinement-path quality
  uplift is exercised in production.
- **Re-baseline the perf harness on the hardened profile**.  Current
  `gb10_qwen36_fp8_flashinfer` baseline pre-dates the cuBLAS env vars;
  capture `gb10_qwen36_fp8_hardened` once we have a dedicated quiesce
  window.
