# Track B.6 — Fun-ASR 1.5 (speculative sidecar, scaffold)

**Status: SCAFFOLD — awaiting hardware bench run on the GB10.**

Speculative — research-grade challenger. PROMOTE-candidate only if
the JA CER gate clears the production baseline by a meaningful
margin AND we don't lose English.

## Pins

| Field | Value |
|---|---|
| `challenger_model` | `csukuangfj/Fun-ASR-Nano-2512-hf @ <40-char-sha>` |
| `baseline_model` | `Qwen/Qwen3-ASR-1.7B` (production) |
| `staging_port` | 8014 (loopback-only) |
| `bench_window` | TBD |

## Files

* `meeting-scribe/containers/fun-asr-15/Dockerfile` — mirror of
  `containers/cohere-transcribe`; install
  `csukuangfj/Fun-ASR-Nano-2512-hf`.
* `meeting-scribe/containers/fun-asr-15/server.py` — FastAPI
  `/v1/audio/transcriptions`.
* `meeting-scribe/docker-compose.gb10.yml` — add `fun-asr-15`
  under `profiles: ["bench"]` on `127.0.0.1:8014:8014`.
* `meeting-scribe/benchmarks/asr_accuracy_latency.py` — accept
  `--url`; add `--backend funasr` if endpoint differs.

## Gates (CER/WER lower-is-better)

* JA CER (challenger) ≤ Qwen3-ASR-1.7B − 0.5 pp AND bootstrap
  95% CI excludes zero → PROMOTE-candidate
* EN WER (challenger) ≤ Qwen3-ASR-1.7B + 0.5 pp ("we don't lose
  English")
* No language-mismatch container wedge
* Real corpus only — Fleurs JA + EN_US (CC BY 4.0, already in
  `benchmarks/fixtures/`)

## Result

(populated post-run)
