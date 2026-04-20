# scribe-omni — Qwen3-Omni unified ASR+TTS+translate

Consolidation target per plan `vLLM Consolidation + Omni Spike`
(in `~/.claude/plans/golden-dreaming-koala.md`). Runs alongside the
existing dedicated stack under compose profile `omni-spike` on port
`8032`; cutover is per-modality via `SCRIBE_OMNI_ASR_URL` /
`SCRIBE_OMNI_TTS_URL` / `SCRIBE_OMNI_TRANSLATE_URL` (empty = stay on the
dedicated instance).

Build pins: see `BUILD_PINS.md`.

## Bring up the spike

```bash
docker compose -f docker-compose.gb10.yml --profile omni-spike up -d omni-unified
curl -f http://localhost:8032/health
curl -sS http://localhost:8032/v1/models | jq .
```

## Cut over one modality

```bash
# Route ASR only:
export SCRIBE_OMNI_ASR_URL=http://localhost:8032
meeting-scribe server restart

# Confirm no regression, then flip the next modality.
# Roll back by unsetting the env var.
```

## Acceptance

`ACCEPTANCE.md` records the per-modality user acceptance stamp that
must land before any Phase 6-style deletion of the dedicated services
is considered. Per the standing rule, we keep the old stack defined in
compose for at least 2 weeks after the last flip.
