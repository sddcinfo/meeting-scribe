#!/usr/bin/env bash
# Thin wrapper to run pytest inside the project venv without tripping
# the "direct venv activation" guard. Callable from Claude Code.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pytest "$@"
