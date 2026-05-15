# 2026-Q2 bench cycle B-tracks — scaffolding status

Tracks B.1 – B.7 from the composed-finding-giraffe plan. All are
scaffolds at present — the per-track decision_gate.md files capture
the recipe pins, file lay-out, and gate criteria so the operator can
walk into each bench run with the contract pre-locked.

The actual recipes/Dockerfiles/run-scripts are NOT created here —
they're hardware-gated and would bit-rot if landed without an imminent
run window. Each decision_gate.md spells out the file lay-out so
they can be created cleanly when the run is scheduled.

## Discipline (carried forward to every track)

* **Loopback-only binding** — every NEW bench-profile service in
  `docker-compose.gb10.yml` binds to `127.0.0.1:<port>:<port>`,
  never `0.0.0.0:` or unqualified `<port>:<port>`. Bench FastAPI
  endpoints are not authenticated; loopback keeps them off the
  host's reachable interfaces.
* **Pinned model revisions** — every `sddc hf download` for a new
  bench model uses `--revision <40-char-sha>`. The recipe pins the
  same SHA. The decision_gate.md header records it alongside
  container digests.
* **No `:latest` images** — pin via digest in every NEW Dockerfile.
* **Production untouched until cutover** — sidecar tracks (B.3,
  B.4, B.5) bench the challenger on a staging port; production
  promotion is a separate maintenance-window step with an
  immutable-digest rollback artifact captured FIRST.

## Tracks

| Track | Decision gate | Status |
|---|---|---|
| B.1 — NVFP4 + FlashInfer 0.6.10 | [translate_phase_6c_nvfp4_flashinfer_0.6.10/](translate_phase_6c_nvfp4_flashinfer_0.6.10/decision_gate.md) | SCAFFOLD |
| B.2 — Qwen3.6-27B-FP8 | [translate_qwen3.6-27b-fp8/](translate_qwen3.6-27b-fp8/decision_gate.md) | SCAFFOLD |
| B.3 — faster-qwen3-tts 0.2.6 | [tts_faster_qwen3_tts_0.2.6/](tts_faster_qwen3_tts_0.2.6/decision_gate.md) | SCAFFOLD |
| B.4 — transformers ==5.6.2 | [transformers_v5_migration_tts/](transformers_v5_migration_tts/decision_gate.md) | SCAFFOLD |
| B.5 — vLLM 0.20.1 ASR | [vllm_0.20.1_asr/](vllm_0.20.1_asr/decision_gate.md) | SCAFFOLD |
| B.6 — Fun-ASR 1.5 | [asr_funasr_15/](asr_funasr_15/decision_gate.md) | SCAFFOLD |
| B.7 — MOSS-TTS-Nano | [tts_moss_nano/](tts_moss_nano/decision_gate.md) | SCAFFOLD |
