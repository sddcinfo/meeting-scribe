#!/usr/bin/env bash
# Re-sync meeting-scribe dependencies after pyproject.toml changes.
# Re-installs the package in editable mode so new deps are picked up.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "No .venv found — run: meeting-scribe setup"
    exit 1
fi

echo "Syncing dependencies from pyproject.toml..."
.venv/bin/pip install -e '.[dev]' --quiet
echo "Done. New dependencies installed."
