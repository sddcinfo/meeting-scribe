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
# Prefer mise (it pins Python per mise.toml). Auto-install mise if it's
# missing AND we'd otherwise fall back to a too-old system python — fresh
# customer GB10s do not ship with 3.14, so a silent fallback would just
# fail at the next step.
need_mise_install=0
if ! command -v mise >/dev/null 2>&1; then
    if [[ -f mise.toml || -f .tool-versions ]]; then
        # We have a toolchain spec but no mise binary. Best path: install mise.
        need_mise_install=1
    elif ! command -v python3 >/dev/null 2>&1; then
        echo "bootstrap.sh: no python3 and no mise. Install one or the other first." >&2
        exit 1
    fi
fi
if [[ "${need_mise_install}" == "1" ]]; then
    echo "[bootstrap] mise missing — installing via the upstream installer"
    curl -fsSL https://mise.jdx.dev/install.sh | sh
    # The installer drops mise at ~/.local/bin/mise; activate for this script
    # without depending on shell rc files (we don't touch user shell config).
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v mise >/dev/null 2>&1; then
        echo "bootstrap.sh: mise install reported success but the binary is not on PATH." >&2
        echo "  Looked for: ${HOME}/.local/bin/mise" >&2
        exit 1
    fi
fi

if command -v mise >/dev/null 2>&1 && [[ -f mise.toml || -f .tool-versions ]]; then
    echo "[bootstrap] installing toolchain via mise"
    mise install --yes
    PYBIN="$(mise exec -- which python)"
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "bootstrap.sh: no python3 on PATH. Install Python 3.14+ first." >&2
        exit 1
    fi
    PYBIN="$(command -v python3)"
fi

# Hard floor: 3.14 (matches pyproject.toml requires-python). Fail loudly otherwise.
"$PYBIN" - <<'PY' || { echo "bootstrap.sh: Python 3.14+ required" >&2; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 14) else 1)
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

# ── 5. Sister-clone auto-sre (translate vLLM helper) ──────────────
# meeting-scribe's translation backend is a vLLM at :8010, served by
# auto-sre. We clone it as a sibling directory so a customer install
# is one bootstrap end-to-end. Skip with MEETING_SCRIBE_SKIP_AUTOSRE=1
# if the user is bringing their own translate backend.
if [[ "${MEETING_SCRIBE_SKIP_AUTOSRE:-0}" != "1" ]]; then
    AUTOSRE_DIR="${SCRIPT_DIR}/../auto-sre"
    if [[ ! -d "${AUTOSRE_DIR}/.git" ]]; then
        echo "[bootstrap] cloning auto-sre into ${AUTOSRE_DIR}"
        git clone https://github.com/sddcinfo/auto-sre.git "${AUTOSRE_DIR}"
    else
        echo "[bootstrap] auto-sre already present at ${AUTOSRE_DIR}"
        git -C "${AUTOSRE_DIR}" fetch --quiet origin || true
    fi
    if [[ ! -d "${AUTOSRE_DIR}/.venv" ]]; then
        echo "[bootstrap] creating auto-sre .venv"
        "${PYBIN}" -m venv "${AUTOSRE_DIR}/.venv"
    fi
    echo "[bootstrap] installing auto-sre (editable)"
    "${AUTOSRE_DIR}/.venv/bin/pip" install --upgrade pip --quiet
    "${AUTOSRE_DIR}/.venv/bin/pip" install -e "${AUTOSRE_DIR}" --quiet
    echo "[bootstrap] auto-sre ready — run 'autosre start' in another shell to bring up translate"
else
    echo "[bootstrap] MEETING_SCRIBE_SKIP_AUTOSRE=1 — skipping auto-sre clone"
fi

# ── 6. App-layer setup (TLS cert, HF_TOKEN check, port-80 cap) ───
# ``meeting-scribe setup`` is a configuration validator — it does NOT
# pull images or start containers. We invoke it for the cert/cap/HF
# checks, then bring the stack up in step 7.
echo
echo "[bootstrap] base install complete"
echo "[bootstrap] running 'meeting-scribe setup' for app-layer config"
echo
meeting-scribe setup "$@"

# ── 7. Bring up the in-tree model backends ───────────────────────
# ``meeting-scribe gb10 up`` builds + starts the pyannote-diarize, scribe-asr
# and scribe-tts containers. First run pulls the ~24 GB vllm-openai base
# image, builds the local ASR layer, and pulls HF model weights. Plan
# 15–30 min on a cold customer device. Idempotent on rerun (no-ops if the
# containers are already healthy). Translate is NOT started here — it
# lives in auto-sre (sister-cloned in step 5); run ``autosre start`` after.
echo
echo "[bootstrap] starting model backends ('meeting-scribe gb10 up')"
echo "[bootstrap] first run is 15–30 min for the image pull + HF weights"
meeting-scribe gb10 up || \
    echo "[bootstrap] gb10 up reported issues — try 'meeting-scribe gb10 status' for details"

# ── 8. Smoke-test ──────────────────────────────────────────────────
# 5-second sweep across all four backends. Non-fatal — translate will
# still be down (it lives in auto-sre, not started by this script).
echo
echo "[bootstrap] running 'meeting-scribe validate --quick'"
meeting-scribe validate --quick || true

# ── 9. Operator next steps ────────────────────────────────────────
echo
cat <<'NEXT_STEPS'
─────────────────────────────────────────────────────────────────────
[bootstrap] meeting-scribe install complete.

Translate (vLLM @ :8010) is not running yet — start it via auto-sre:

    cd ../auto-sre
    .venv/bin/autosre setup         # one-time (selects vLLM backend on GB10)
    .venv/bin/autosre start         # cold-loads the 35 B FP8 model (3+ min)

Then start the scribe server:

    meeting-scribe start

Verify everything is green:

    meeting-scribe validate --quick
─────────────────────────────────────────────────────────────────────
NEXT_STEPS
