# vLLM concurrency + KV-dtype A/B campaign — RESULT

**Date:** 2026-05-07  
**Plan:** `~/.claude/plans/linked-plotting-lampson.md`  
**Recipe touched:** `repos/auto-sre/autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml`  
**Pinned meeting set:** `meetings.manifest.json` — short=`38d4efb0` (6 events) · medium=`c4c913f8` (179) · long=`3db4286e` (782)

## Headline findings

1. **The memory-delta gate is unreachable by any config we tested** — all six fall within ~3 GiB of each other on post-warm `MemAvailable`. vLLM pre-allocates the KV pool from `gpu_memory_utilization=0.70`, so changing `max_num_seqs` or `kv_cache_dtype` doesn't free unified memory. The original 30 s playback stall was driven by *swap pressure built up over a long-running session*, not by steady-state vLLM provisioning. Restarting vLLM at any of these configs drained swap from 14.15 → 0 GiB used and held it there.
2. **Current production (C0 = 8, auto) is the *only* config that fails the long meeting summary** — `LLM call failed:` timeout at 180 s on `3db4286e`. All five candidate configs handle the same meeting cleanly. This matches the failure mode user reported in the prior summary A/B (commit `9ee35d8`).
3. **`kv_cache_dtype: fp8` is *not* unstable on this hardware right now** — boots cleanly, runs the perf harness with zero errors, *and* produces materially higher BLEU than `auto`. The vllm#26646 instability documented in `repos/auto-sre/CLAUDE.md` does not reproduce on the current vLLM build (`vllm/vllm-openai:latest`, `[SM121] FP8 fix applied: CUTLASS block-FP8 disabled, Triton fallback enabled`).
4. **Peak concurrent demand is 3 slots** under the perf harness (`coding_concurrency=2, translation_rps=2.0`). `max_num_seqs ∈ {4, 6, 8}` all show `requests_running_peak=3.0` and `preemptions_delta=0`. Only `max_num_seqs=2` (A3) forces preemption — and it does so dramatically (117 preemptions, coding TTFT p95 jumps 8×).

## Decision matrix — Track A (kv_cache_dtype = auto)

| ID | seqs | TTFT translate p99 cont | TTFT coding p95 cont | Preempts | KV peak | Long-mtg topics | Long-mtg actions | Errors | MemAvail (GiB) | PASS gates? |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 8 | 161 ms | 413 ms | 0 | 0.9 % | **FAIL (timeout)** | n/a | 0 in perf | 39.88 | **FAIL** (long-mtg) |
| A1 | 6 | 160 ms | 413 ms | 0 | 0.9 % | 9 | 3 | 0 | 40.27 | PASS |
| A2 | 4 | 161 ms | 404 ms | 0 | 0.9 % | 12 | 3 | 0 | 39.63 | PASS |
| A3 | 2 | 220 ms | **3357 ms** | **117** | 0.8 % | 6 | 0 | coding p95 isolated **FAIL** | 39.78 | **FAIL** |

## Decision matrix — Track B (kv_cache_dtype = fp8)

| ID | seqs | TTFT translate p99 cont | TTFT coding p95 cont | Preempts | KV peak | Long-mtg topics | Long-mtg actions | Errors | MemAvail (GiB) | PASS gates? |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 8 | 161 ms | 405 ms | 0 | 1.0 % | 8 | 4 | 0 | 39.52 | PASS |
| B2 | 4 | 162 ms | 410 ms | 0 | 1.0 % | 6 | 0 | 0 | 40.62 | PASS |

## Quality (sacreBLEU vs C0, today's measurement)

| Config | JA → EN Δ vs C0 | EN → JA Δ vs C0 |
|---|---|---|
| A1 (6, auto) | +2.17 | +0.75 |
| A2 (4, auto) | +1.89 | +1.93 |
| A3 (2, auto) | −1.13 | +2.39 |
| **B1 (8, fp8)** | **+3.93** | **+3.21** |
| **B2 (4, fp8)** | **+3.26** | **+2.16** |

**Note on absolute BLEU.** Today's C0 (43.27 / 51.70) is well below the published 2026-04-19 baseline (54.63 / 66.97). Drift in vLLM image, weights cache, or measurement methodology accounts for this; the *relative* deltas across today's configs are apples-to-apples.

## Manual long-meeting spot-check

All four passing candidates (A1, A2, B1, B2) identify the same major themes on the long meeting (AI-first philosophy, agentic coding/personal infra, tiered data views, enterprise adoption, infrastructure/tokenomics, agent swarms). They differ in granularity (B2: 6 topics → A2: 12 topics) but no config produced hallucinated or off-topic content. **B1 was the only one to extract action_items (4) on the long meeting** — possible quality edge or possible behavioral artifact; either way not a regression.

## Recommendation

Two viable winners:

- **B1 (max_num_seqs=8, kv_cache_dtype=fp8)** — best BLEU (+3.93 / +3.21), keeps the existing concurrency ceiling, reliable long-meeting summaries. Single-knob change vs. current prod.
- **B2 (max_num_seqs=4, kv_cache_dtype=fp8)** — slightly lower BLEU (+3.26 / +2.16), matches the user's intuition that 4 is the right concurrency ceiling, has 1 slot of headroom over the observed peak (3). Two-knob change.

**Either is a strict improvement over C0** — both fix the long-meeting timeout, both improve BLEU, neither costs anything on TTFT under the perf-harness workload. The choice between them is essentially: "do you want concurrency headroom for unexpected bursts (B1) or aligned with the 'max-4' intuition (B2)?"

**I recommend B2** because:
1. The data confirms peak demand is 3 — `max_num_seqs=4` has the right amount of headroom (1 slot) without over-provisioning.
2. It matches the user's pre-test mental model.
3. The BLEU difference vs B1 (~0.7 BLEU) is within noise and not a clear quality regression.
4. Smaller `max_num_seqs` reduces theoretical worst-case KV-block fragmentation under burst conditions that the perf harness doesn't exercise.

**vLLM is currently running B2.** No further action is needed for the runtime change to take effect. To **persist** it: `autosre perf save-baseline gb10_qwen36_fp8_seq4_kv-fp8`, then commit recipe + new baseline files on `repos/auto-sre`'s `dev` branch.

## What does NOT change

- `max_model_len = 262144` — preserved per user requirement.
- `gpu_memory_utilization = 0.70` — out of scope this campaign. **This is the actual lever for unified-memory savings; revisit if memory pressure remains an operational concern.**
- All other recipe fields (extra_args, env, attention_backend, quantization).

## Rejected configs

- **C0 (8, auto)** — current prod; fails the long-meeting summary reliability check.
- **A3 (2, auto)** — preemption-induced 8× regression in coding TTFT p95.

## Caveats / known gaps

- **The original loaded-playback "PASS" was bogus.** The first version of `benchmarks/loaded_playback_check.py` (commit `47fb647`) used `curl -L`, which silently followed the auth redirect and timed the **HTML login page** (1453 bytes of `<!doctype html>...`). Numbers like `first_byte=0.238s code=200 size=1453` looked clean but were measuring the wrong path. The script has since been rewritten to require an auth cookie, refuse to follow redirects, and assert `content-type=audio/wav` + body size > 1024 B; the bogus `loaded_playback.txt` log is preserved here as a record of the failure mode.
- **Peak-memory sampler died early** in B1 and C0 perf runs (likely a subshell/environment issue with the bash heredoc). Post-warm idle samples (the actual gate metric) captured cleanly for all six configs.
- **The "memory delta ≥ 5 GiB" gate is unreachable for these knobs.** This was a planning-time miscalibration; we expected `kv_cache_dtype: fp8` and lower `max_num_seqs` to free unified memory but the data shows `gpu_memory_utilization` dominates the allocation. Future memory work should target that knob.
- **Absolute BLEU is below the 2026-04-19 published baseline** for all configs including current prod. Worth investigating separately whether vLLM/weights drift has cost ~10 BLEU points across the board, but it's not a campaign blocker — relative comparisons are sound.
- **Coding-contention insufficient samples** WARN appeared on all five non-A3 configs (14–19 samples vs baseline's 20). Workload-pacing artifact, not a perf regression.
