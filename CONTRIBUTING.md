# Contributing to Meeting Scribe

## Development Setup

Meeting Scribe runs exclusively on the NVIDIA GB10 (aarch64 Linux, 128GB unified memory).

```bash
git clone https://github.com/sddcinfo/meeting-scribe.git
cd meeting-scribe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
git config core.hooksPath scripts/hooks/
```

## Dependencies

| Tier | Packages | Required? |
|------|----------|-----------|
| **Core** | fastapi, uvicorn, numpy, pydantic, click, httpx, soundfile | Always |
| **ML** | torch, torchaudio, av | Full pipeline |
| **Dev** | pytest, pytest-asyncio, pytest-cov, websockets, ruff | Development |

Model inference is handled by external vLLM and diarization containers managed via `docker-compose.gb10.yml`. The application connects to these over HTTP.

### Security upkeep

We run a three-layer defense against vulnerable dependencies:

1. **`security` CI lane** (`.github/workflows/tests.yml`) — runs on every push + PR. Uses `pip-audit` (PyPI advisories via OSV) and `npm audit` against the overlay. Fails the build on any **HIGH+** severity finding so vulnerabilities can't merge to `dev`. Moderate-and-below are visible in the log but do not block.
2. **Dependabot** (`.github/dependabot.yml`) — opens grouped weekly PRs against `dev` for security and patch-level updates across pip, npm, and github-actions ecosystems. ML deps with numerical-behavior risk (`torch`, `torchaudio`, `numpy`, `av`) are excluded; bumps for those are reviewed manually via the freshness issue below.
3. **Weekly freshness issue** (`.github/workflows/dependency-freshness.yml`) — opens a single tracking issue when any pin is ≥30 days behind latest stable, so manual-review deps don't quietly stagnate.

When a Dependabot PR lands: the security lane already gated it on the actual CVE database, so the merge bar is "tests pass + targeted manual smoke if it touches a hot path (translation_queue, slides/job, runtime/lifespan)." Routine version bumps (patch-level non-security) auto-group and can be batch-merged.

When the security lane fails on a PR: the failing tool prints which package + advisory. Bump the pin in `pyproject.toml` (Python) or `overlay/package.json` (npm) to the patched version listed in the advisory, push, and the lane should clear. If the patched version requires a major bump that breaks something, open a Bug-class commit (see below) so the regression is tracked.

## Code Style

- **Python**: ruff (config in `pyproject.toml`, line length 100)
- **JavaScript**: Vanilla ES modules, no build tools, no framework
- **CSS**: CSS custom properties, no preprocessor

Run checks manually:

```bash
ruff check src/ tests/
ruff format src/ tests/
python3 scripts/check_secrets.py
```

## Testing

```bash
# Start model backends
docker compose -f docker-compose.gb10.yml up -d

# Start the server
meeting-scribe start

# Run all tests
PYTHONPATH=src pytest tests/ -v --tb=short

# Stop the server
meeting-scribe stop
```

Non-integration tests (lint, syntax, name extraction) run without a server. Integration tests require the server and model backends to be running.

## Pre-commit Hooks

The hook at `scripts/hooks/` runs ruff lint, ruff format, secret scanning, and JS syntax checks.

Install with:

```bash
git config core.hooksPath scripts/hooks/
```

## Commit Conventions

```
feat: new feature or capability
fix:  bug fix
chore: tooling, deps, CI changes
```

Keep commit messages to 1-2 lines. Focus on the "why" not the "what".

## Known Limitations

- **pyannote diarization**: Requires `HF_TOKEN` environment variable for downloading pyannote.audio model weights from Hugging Face (gated model, accept license first).
- **Translation**: Uses vLLM (Qwen3.6-35B) for GPU-accelerated translation.
- **Room layout**: Session-scoped via `scribe_session` cookie. Multiple browsers can have independent layouts.
- **Speaker matching**: Runs in ThreadPoolExecutor. May lag with rapid multi-speaker input.

## Architecture

See [README.md](README.md) for the full architecture overview and model stack.

## Bug-class taxonomy & PR hygiene

Every fix(*) commit requires a `Bug-class:` git trailer naming one of:

| Slug                  | What it covers |
|-----------------------|----------------|
| `cross-window-sync`   | Admin ↔ popout ↔ guest state divergence |
| `ws-lifecycle`        | WebSocket connect/disconnect/reconnect/replay |
| `event-dedup`         | Translation merge, segment dedup, listener fan-out |
| `async-render`        | Empty-state timing, scroll behavior, overlay state machines |
| `platform-quirk`      | iOS/Android/WebKit/captive-portal OS-level behavior |
| `data-shape`          | Wire-format / event-type drift, contract violations |
| `backend-lifecycle`   | vLLM container lifecycle, model swap, API drift |

Add the trailer with:

```
git commit --amend --trailer "Bug-class: <slug>"
```

The pre-push hook (`scripts/hooks/classify_push.py`) uses local Opus 4.7
to classify your diff and aborts the push if a fix doesn't have a
regression test. To bypass with audit trail, write a one-line reason
to `.git-waivers/<sha>.txt` and `git push --no-verify`.

The dashboard:

```
scripts/bug_class_report.py            # last 30 days
```

Manual-test runbook (real iPhone, real GB10, etc.) lives at
`tests/manual/README.md`. Bump verification stamps with
`scripts/manual_test_status.py --bump <slug>`.
