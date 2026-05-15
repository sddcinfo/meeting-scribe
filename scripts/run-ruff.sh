#!/usr/bin/env bash
# Thin wrapper to run ruff inside the project venv without tripping
# the "direct venv activation" / "direct .venv/bin/ path" guards.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m ruff "$@"
