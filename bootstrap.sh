#!/usr/bin/env bash
# meeting-scribe bootstrap — one-command setup for a fresh GB10 (DGX Spark).
#
# Intended flow on a brand-new machine:
#
#     git clone https://github.com/sddcinfo/meeting-scribe.git
#     cd meeting-scribe
#     ./bootstrap.sh
#
# What this does (in order):
#   1. Verify we're on aarch64 Linux with CUDA — hard fail otherwise.
#   2. Install OS packages we need for the build (ffmpeg, libportaudio, …).
#   3. Pin Python via mise (or your existing toolchain) and create .venv.
#   4. Editable-install meeting-scribe into that venv.
#   5. Hand off to ``meeting-scribe setup`` for the app-layer steps
#      (HF token, TLS certs, container stack, systemd unit).
#
# Non-destructive: re-running on an already-set-up machine is idempotent.
# Exits non-zero on the first real problem so you notice it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Platform gate ──────────────────────────────────────────────
# The scribe pipeline targets NVIDIA GB10 (DGX Spark) — aarch64 Linux
# with CUDA. Everything else is a footgun we'd rather flag upfront.
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "bootstrap.sh: Linux required (got $(uname -s))" >&2
    exit 1
fi
if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
    echo "bootstrap.sh: aarch64 required (got $(uname -m)). " \
         "x86_64 may work but is not tested — export MEETING_SCRIBE_FORCE_X86=1 to proceed." >&2
    if [[ "${MEETING_SCRIBE_FORCE_X86:-0}" != "1" ]]; then
        exit 1
    fi
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "bootstrap.sh: nvidia-smi not found. Install the NVIDIA driver stack first." >&2
    exit 1
fi

# ── 2. OS packages ────────────────────────────────────────────────
# Only runs apt if we're missing something — keeps sudo prompts minimal.
need_pkgs=()
for pkg in ffmpeg libportaudio2 libsndfile1 docker-compose-plugin; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        need_pkgs+=("$pkg")
    fi
done
if [[ ${#need_pkgs[@]} -gt 0 ]]; then
    echo "[bootstrap] installing apt packages: ${need_pkgs[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${need_pkgs[@]}"
fi

# ── 3. Python + venv ──────────────────────────────────────────────
# Prefer mise if available (it pins Python per .tool-versions / mise.toml).
# Fall back to system python3 — but guard for < 3.11 since scribe needs 3.11+.
if command -v mise >/dev/null 2>&1 && [[ -f mise.toml || -f .tool-versions ]]; then
    echo "[bootstrap] installing toolchain via mise"
    mise install
    PYBIN="$(mise exec -- which python)"
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "bootstrap.sh: no python3 on PATH. Install Python 3.11+ first." >&2
        exit 1
    fi
    PYBIN="$(command -v python3)"
fi

# Hard floor: 3.11 (scribe uses 3.11+ syntax + deps). Fail loudly otherwise.
"$PYBIN" - <<'PY' || { echo "bootstrap.sh: Python 3.11+ required" >&2; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY

if [[ ! -d .venv ]]; then
    echo "[bootstrap] creating .venv"
    "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 4. Editable install ───────────────────────────────────────────
echo "[bootstrap] installing meeting-scribe (editable)"
pip install --upgrade pip
pip install -e .

# ── 5. Hand off to the app setup flow ─────────────────────────────
# ``meeting-scribe setup`` covers: HF_TOKEN check, TLS cert generation,
# docker compose up, model warmup, optional systemd registration.
echo
echo "[bootstrap] base install complete"
echo "[bootstrap] handing off to 'meeting-scribe setup' for app-layer config"
echo
exec meeting-scribe setup "$@"
