# Track B.2 — Qwen3.6-27B-FP8 (translation A/B, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

Translation-only A/B against the 35B-A3B production primary. The
goal is a translation-only fallback that fits a smaller VRAM
envelope; primary swap is OUT OF SCOPE.

## Pins

| Field | Value |
|---|---|
| `challenger_model` | `Qwen/Qwen3.6-27B-FP8 @ <40-char-sha>` |
| `baseline_model` | `Qwen/Qwen3.6-35B-A3B-FP8` (production) |
| `challenger_image` | autosre production vLLM (recipe selects model) |
| `staging_port` | 8016 (loopback-only) |
| `bench_window` | TBD |

## Recipe (`auto-sre/autosre/backends/recipes/qwen3.6-27b-fp8.yaml`)

Clone of `qwen3.6-35b-a3b-fp8.yaml` with `model_id` swapped and
adjusted `gpu_memory_utilization` for the smaller weights set.

**The candidate is NOT added to `auto-sre/autosre/bench.py`'s
MODELS array.** B.2's runner consumes the recipe via an explicit
`--recipe` CLI override so the experiment stays isolated.
Promotion to the global MODELS matrix requires an explicit
follow-up decision in a future cycle.

## Run script (`meeting-scribe/scripts/bench/run_phase_a.sh`)

Mirrors `run_phase_6c.sh`. Passes the recipe path explicitly via
CLI override, NOT via the global MODELS array.

## Gates

* EN↔JA sacreBLEU within ±1.0 of 35B-A3B
* slide BLEU within ±1.0
* COMET ≥ baseline − 0.02
* TTFT p99 ≤ 195 ms
* SLO probe zero ABORTs

## Decision frame

Pass → keep on watchlist as translation-only fallback if the larger
model becomes unavailable. Fail → REJECT. No production swap either way.
