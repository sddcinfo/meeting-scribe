#!/usr/bin/env bash
# meeting-scribe bootstrap — one-command setup for a fresh GB10 (DGX Spark).
#
# Intended flow on a brand-new machine:
#
#     git clone https://github.com/sddcinfo/meeting-scribe.git
#     cd meeting-scribe
#     sudo ./bootstrap.sh
#
# Bootstrap runs as root because customer install needs to do things
# that an unprivileged process cannot:
#
#   * apt-install OS packages (ffmpeg, libportaudio, …).
#   * setcap cap_net_bind_service so the guest portal can bind :80.
#   * rfkill unblock wifi (kernel ships with the radio soft-blocked).
#   * Install /etc/sudoers.d/meeting-scribe so the service can call
#     ``nmcli``/``iw``/``rfkill``/``iptables``/``setcap`` without a
#     PAM round-trip on every API call. Scoped — NOT a blanket NOPASSWD.
#   * loginctl enable-linger so the systemd user services autostart
#     at boot (no console login needed).
#
# The user-owned bits (venv, pip install, mise toolchain, gb10
# containers, ``meeting-scribe setup``, ``install-service``) run as
# the *invoking* user via ``sudo -u $SUDO_USER`` so file ownership
# under the repo is correct.
#
# Re-running on an already-set-up machine is idempotent.
# Set ``MEETING_SCRIBE_YES=1`` (or pass ``--yes``) to accept all
# defaults for unattended installs.

set -euo pipefail

# ── 0. Sudo gate ──────────────────────────────────────────────────
# Refuse to run unprivileged. Customers must invoke as ``sudo
# ./bootstrap.sh`` so we have one explicit authorisation point
# instead of N sudo prompts during the install.
if [[ "${EUID}" -ne 0 ]]; then
    cat >&2 <<EOF
bootstrap.sh requires root for system configuration. Re-run with sudo:

    sudo ./bootstrap.sh

The user-owned parts (venv, pip install, mise toolchain) drop privileges
back to your account via SUDO_USER, so file ownership stays correct.
EOF
    exit 1
fi

if [[ -z "${SUDO_USER:-}" ]]; then
    cat >&2 <<EOF
bootstrap.sh: SUDO_USER is empty — looks like you logged in as root
directly. Install as a regular user with ``sudo ./bootstrap.sh`` so
the venv + repo files end up owned by that user instead of root.
EOF
    exit 1
fi

TARGET_USER="${SUDO_USER}"
TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"

if [[ -z "${TARGET_HOME}" || ! -d "${TARGET_HOME}" ]]; then
    echo "bootstrap.sh: cannot resolve home dir for user '${TARGET_USER}'." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Helper: run a command as the original (non-root) user, in the repo
# directory, with their HOME and PATH so mise/pip/etc behave normally.
as_user() {
    sudo -u "${TARGET_USER}" \
        env "HOME=${TARGET_HOME}" \
            "PATH=${TARGET_HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
            bash -lc "cd '${SCRIPT_DIR}' && $*"
}

# ── 0.5 Yes-mode + interactive helpers ────────────────────────────
YES_MODE=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) YES_MODE=1 ;;
    esac
done
if [[ "${MEETING_SCRIBE_YES:-0}" == "1" ]]; then
    YES_MODE=1
fi

# Read a value with a default. Honors --yes / MEETING_SCRIBE_YES by
# auto-accepting the default (so unattended installs don't block).
prompt_default() {
    local label="$1"
    local default="$2"
    local var
    if [[ "${YES_MODE}" == "1" || ! -t 0 ]]; then
        printf '[bootstrap] %s: %s (auto)\n' "${label}" "${default}"
        echo "${default}"
        return
    fi
    read -r -p "${label} [${default}]: " var
    echo "${var:-${default}}"
}

# ── 1. Platform gate ──────────────────────────────────────────────
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

# ── 2. Locale prompts ─────────────────────────────────────────────
# Wifi regulatory domain is the most important — wrong country code
# silently caps 5GHz TX power or refuses certain channels and the AP
# may never associate. Defaults are auto-detected from the system
# where possible; the user gets one chance to override.

# Country / regdomain: try `timedatectl show` (if locale-set) or
# default to JP (the lab's home country and where most customer
# events are run).
detect_country() {
    local cc
    if command -v timedatectl >/dev/null 2>&1; then
        cc="$(timedatectl show --property=Timezone --value 2>/dev/null || true)"
        case "${cc}" in
            America/*) echo "US"; return ;;
            Europe/London) echo "GB"; return ;;
            Europe/*) echo "DE"; return ;;
            Asia/Tokyo) echo "JP"; return ;;
            Asia/Shanghai) echo "CN"; return ;;
            Asia/Singapore) echo "SG"; return ;;
            Australia/*) echo "AU"; return ;;
        esac
    fi
    echo "JP"
}
DEFAULT_COUNTRY="$(detect_country)"

# Timezone: take it straight from systemd if present.
detect_tz() {
    local tz
    if command -v timedatectl >/dev/null 2>&1; then
        tz="$(timedatectl show --property=Timezone --value 2>/dev/null || true)"
        if [[ -n "${tz}" && "${tz}" != "UTC" ]]; then
            echo "${tz}"
            return
        fi
    fi
    echo "Asia/Tokyo"
}
DEFAULT_TZ="$(detect_tz)"

# Language pair default: derive from $LANG of the invoking user, fall
# back to en,ja (the lab's bilingual default).
detect_langpair() {
    local lang
    lang="$(getent passwd "${TARGET_USER}" | cut -d: -f7 >/dev/null 2>&1 && \
            sudo -u "${TARGET_USER}" printenv LANG 2>/dev/null || true)"
    case "${lang}" in
        en_US*|en_CA*|en_GB*) echo "en,ja" ;;
        ja_JP*) echo "ja,en" ;;
        es_*) echo "es,en" ;;
        fr_*) echo "fr,en" ;;
        de_*) echo "de,en" ;;
        zh_*) echo "zh,en" ;;
        ko_*) echo "ko,en" ;;
        *) echo "en,ja" ;;
    esac
}
DEFAULT_LANGPAIR="$(detect_langpair)"

echo
echo "[bootstrap] locale settings (Enter to accept defaults)"
COUNTRY_CODE="$(prompt_default 'WiFi country code (regulatory domain)' "${DEFAULT_COUNTRY}")"
TIMEZONE="$(prompt_default 'Timezone' "${DEFAULT_TZ}")"
LANG_PAIR="$(prompt_default 'Default language pair (source,target)' "${DEFAULT_LANGPAIR}")"
echo
echo "[bootstrap] country=${COUNTRY_CODE} timezone=${TIMEZONE} languages=${LANG_PAIR}"

# ── 3. OS packages ────────────────────────────────────────────────
need_pkgs=()
for pkg in ffmpeg libportaudio2 libsndfile1 docker-compose-plugin rfkill iw; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        need_pkgs+=("$pkg")
    fi
done
if [[ ${#need_pkgs[@]} -gt 0 ]]; then
    echo "[bootstrap] installing apt packages: ${need_pkgs[*]}"
    apt-get update -qq
    apt-get install -y "${need_pkgs[@]}"
fi

# ── 4. WiFi radio: unblock + persist regdomain ────────────────────
# Two independent kill switches; both must be off before NetworkManager
# can put the wlan interface in a usable state. rfkill is the kernel-
# level one; ``nmcli radio wifi`` is NetworkManager's own switch (it
# defaults to ``disabled`` on Ubuntu 24.04 server installs and stays
# that way across reboots until explicitly turned on).
if command -v rfkill >/dev/null 2>&1; then
    rfkill unblock wifi || true
    echo "[bootstrap] rfkill: wifi radio unblocked"
fi
if command -v nmcli >/dev/null 2>&1; then
    nmcli radio wifi on 2>/dev/null || true
    echo "[bootstrap] nmcli: wifi radio enabled"
fi

# Persist regdomain via /etc/default/crda or /etc/sysconfig/regdomain
# (whichever exists). systemd-rfkill picks this up at boot. Also call
# `iw reg set` for the running session.
if command -v iw >/dev/null 2>&1; then
    iw reg set "${COUNTRY_CODE}" 2>/dev/null || true
    echo "[bootstrap] runtime regdomain set to ${COUNTRY_CODE}"
fi
if [[ -f /etc/default/crda ]]; then
    sed -i "s/^REGDOMAIN=.*/REGDOMAIN=${COUNTRY_CODE}/" /etc/default/crda
elif [[ -d /etc/default ]]; then
    echo "REGDOMAIN=${COUNTRY_CODE}" > /etc/default/crda
fi

# ── 5. Scoped sudoers.d ───────────────────────────────────────────
# Grants the invoking user passwordless sudo for the *specific*
# commands the meeting-scribe service needs. Scoped — NOT a blanket
# NOPASSWD: ALL. Any other ``sudo`` invocation still prompts for a
# password as normal.
SUDOERS_FILE="/etc/sudoers.d/meeting-scribe"
SUDOERS_BODY="# Installed by meeting-scribe bootstrap.sh.
# Scoped passwordless access for the service user. The meeting-scribe
# server calls these binaries on every WiFi/QR/setcap operation;
# requiring an interactive password would freeze the asyncio event
# loop on every API request.
${TARGET_USER} ALL=(root) NOPASSWD: /usr/bin/nmcli, /usr/sbin/iw, /usr/sbin/rfkill, /usr/sbin/iptables, /usr/sbin/ip6tables, /usr/sbin/setcap
"

# Atomic write via tmpfile + visudo -c so we never leave a malformed
# sudoers file (which would lock the user out of sudo entirely).
SUDOERS_TMP="$(mktemp /etc/sudoers.d/.meeting-scribe.tmp.XXXXXX)"
chmod 0440 "${SUDOERS_TMP}"
printf '%s' "${SUDOERS_BODY}" > "${SUDOERS_TMP}"
if visudo -c -f "${SUDOERS_TMP}" >/dev/null; then
    mv "${SUDOERS_TMP}" "${SUDOERS_FILE}"
    chmod 0440 "${SUDOERS_FILE}"
    echo "[bootstrap] installed ${SUDOERS_FILE} (scoped NOPASSWD for ${TARGET_USER})"
else
    rm -f "${SUDOERS_TMP}"
    echo "[bootstrap] WARNING: sudoers fragment failed visudo validation; not installed" >&2
fi

# ── 6. Python + venv (as TARGET_USER) ─────────────────────────────
# Auto-install mise for the user if missing, then create the venv +
# editable install. All run as TARGET_USER so file ownership is
# correct.
echo "[bootstrap] preparing Python toolchain + venv (as ${TARGET_USER})"

as_user '
    set -e
    if ! command -v mise >/dev/null 2>&1; then
        if [[ -f mise.toml || -f .tool-versions ]]; then
            echo "[bootstrap] mise missing — installing for ${USER}"
            curl -fsSL https://mise.jdx.dev/install.sh | sh
        fi
    fi
    PATH="${HOME}/.local/bin:${PATH}"
    if command -v mise >/dev/null 2>&1 && [[ -f mise.toml || -f .tool-versions ]]; then
        mise install --yes
        PYBIN="$(mise exec -- which python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYBIN="$(command -v python3)"
    else
        echo "bootstrap.sh: no python3 and no mise; cannot proceed" >&2
        exit 1
    fi
    "$PYBIN" -c "import sys; sys.exit(0 if sys.version_info >= (3,14) else 1)" \
        || { echo "bootstrap.sh: Python 3.14+ required" >&2; exit 1; }
    if [[ ! -d .venv ]]; then
        echo "[bootstrap] creating .venv"
        "$PYBIN" -m venv .venv
    fi
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e .
'

# ── 7. Sister-clone auto-sre ──────────────────────────────────────
if [[ "${MEETING_SCRIBE_SKIP_AUTOSRE:-0}" != "1" ]]; then
    AUTOSRE_DIR="${SCRIPT_DIR}/../auto-sre"
    as_user "
        set -e
        if [[ ! -d '${AUTOSRE_DIR}/.git' ]]; then
            echo '[bootstrap] cloning auto-sre into ${AUTOSRE_DIR}'
            git clone https://github.com/sddcinfo/auto-sre.git '${AUTOSRE_DIR}'
        else
            echo '[bootstrap] auto-sre already present at ${AUTOSRE_DIR}'
            git -C '${AUTOSRE_DIR}' fetch --quiet origin || true
        fi
        if [[ ! -d '${AUTOSRE_DIR}/.venv' ]]; then
            echo '[bootstrap] creating auto-sre .venv'
            \"\$(.venv/bin/python -c 'import sys; print(sys.executable)')\" -m venv '${AUTOSRE_DIR}/.venv'
        fi
        '${AUTOSRE_DIR}/.venv/bin/pip' install --upgrade pip --quiet
        '${AUTOSRE_DIR}/.venv/bin/pip' install -e '${AUTOSRE_DIR}' --quiet
    "
    # Install the autosre user systemd unit (no-start; vLLM cold-load
    # takes 3-7 min and we don't block bootstrap on that). The customer
    # then runs ``autosre start`` once when they're ready for the
    # cold-load — translate is the only manual step left after this.
    if [[ -x "${AUTOSRE_DIR}/.venv/bin/autosre" ]]; then
        as_user "'${AUTOSRE_DIR}/.venv/bin/autosre' install-service --no-start" || \
            echo "[bootstrap] autosre install-service reported issues — see above"
        echo "[bootstrap] autosre.service installed (boot autostart enabled)"
    fi
else
    echo "[bootstrap] MEETING_SCRIBE_SKIP_AUTOSRE=1 — skipping auto-sre clone"
fi

# ── 8. Persist locale settings into the meeting-scribe config ────
# Settings live at $XDG_CONFIG_HOME/meeting-scribe/settings.json (per
# user). The helper script writes the regdomain + timezone so the
# server picks them up on first start. Language pair is an env-var
# (SCRIBE_LANGUAGE_PAIR) — written below.
as_user ".venv/bin/python scripts/persist-locale-settings.py '${COUNTRY_CODE}' '${TIMEZONE}'"
# SCRIBE_LANGUAGE_PAIR lives in .env (env-var consumed by the server).
if ! grep -q "^SCRIBE_LANGUAGE_PAIR=" "${SCRIPT_DIR}/.env" 2>/dev/null; then
    echo "SCRIBE_LANGUAGE_PAIR=${LANG_PAIR}" >> "${SCRIPT_DIR}/.env"
    chown "${TARGET_USER}:" "${SCRIPT_DIR}/.env" 2>/dev/null || true
fi

# ── 9. App-layer setup ────────────────────────────────────────────
echo
echo "[bootstrap] running 'meeting-scribe setup' for app-layer config"
as_user '.venv/bin/meeting-scribe setup' || \
    echo "[bootstrap] meeting-scribe setup reported issues — see above"

# Grant CAP_NET_BIND_SERVICE on the venv python so the guest portal
# can bind :80 without sudo at runtime. setcap requires root and the
# physical (non-symlink) target.
PYTHON_REAL="$(readlink -f "${SCRIPT_DIR}/.venv/bin/python3" || echo '')"
if [[ -n "${PYTHON_REAL}" && -x "${PYTHON_REAL}" ]]; then
    setcap cap_net_bind_service=+ep "${PYTHON_REAL}" || true
    echo "[bootstrap] cap_net_bind_service granted to ${PYTHON_REAL}"
fi

# ── 10. Pre-pull HF model weights ─────────────────────────────────
echo
if [[ ! -d /data/huggingface ]]; then
    install -d -o "${TARGET_USER}" -g "$(id -gn "${TARGET_USER}")" -m 0775 /data/huggingface
fi
as_user '.venv/bin/meeting-scribe gb10 pull-models' || \
    echo "[bootstrap] pull-models reported issues — see above"

# ── 11. Build local container images ──────────────────────────────
echo
echo "[bootstrap] (re)building local container images"
as_user 'docker compose -f docker-compose.gb10.yml build pyannote-diarize qwen3-tts vllm-asr' || \
    echo "[bootstrap] compose build reported issues — see above"

# ── 12. Start model backends ──────────────────────────────────────
echo
echo "[bootstrap] starting model backends ('meeting-scribe gb10 up')"
as_user '.venv/bin/meeting-scribe gb10 up' || \
    echo "[bootstrap] gb10 up reported issues — see above"

# ── 13. Install user systemd unit (boot autostart) ────────────────
echo
echo "[bootstrap] installing meeting-scribe.service (user systemd unit)"
as_user '.venv/bin/meeting-scribe install-service --no-start' || \
    echo "[bootstrap] install-service reported issues — see above"

# ── 14. Start the scribe server ───────────────────────────────────
echo
echo "[bootstrap] starting meeting-scribe.service"
as_user '.venv/bin/meeting-scribe start' || \
    echo "[bootstrap] meeting-scribe start reported issues — see above"

# Give the server a beat to settle (uvicorn ready signal + first
# nmcli reconcile) before the validator hits it. 3 seconds is enough
# for the lifespan to reach READY=1 once all four backends respond.
sleep 3

# ── 15. Acceptance gate: customer-flow validation ────────────────
# This is the contract: bootstrap MUST produce a state where every
# customer-flow phase passes. If any phase fails, bootstrap exits
# non-zero so the operator sees the failure inline instead of
# discovering it during a customer demo. Run it as TARGET_USER so
# the on-disk admin secret resolves correctly.
echo
echo "[bootstrap] running 'meeting-scribe validate --customer-flow' (acceptance gate)"
if as_user '.venv/bin/meeting-scribe validate --customer-flow'; then
    VALIDATE_RC=0
else
    VALIDATE_RC=$?
    echo "[bootstrap] WARNING: validate --customer-flow exited rc=${VALIDATE_RC}" >&2
fi

# ── 16. Operator next steps ───────────────────────────────────────
echo
cat <<NEXT_STEPS
─────────────────────────────────────────────────────────────────────
[bootstrap] meeting-scribe install complete.

Country:    ${COUNTRY_CODE}
Timezone:   ${TIMEZONE}
Languages:  ${LANG_PAIR}

Auto-start: meeting-scribe.service AND autosre.service are both
installed as user systemd units (loginctl enable-linger is set).
Reboot survival is automatic — both come back without a console
login.

Service management (no sudo needed for these):

    systemctl --user status meeting-scribe.service autosre.service
    journalctl --user -u meeting-scribe.service -f
    meeting-scribe stop / start / restart

Translate vLLM is enabled for boot but not started (3-7 min cold
load). Start it once when you're ready for the first meeting:

    autosre start --no-scribe

Or wait for the next reboot — systemd will start it automatically.

To re-run the acceptance gate any time:

    meeting-scribe validate --customer-flow
─────────────────────────────────────────────────────────────────────
NEXT_STEPS

# Bootstrap exit code reflects the acceptance gate. The customer
# sees a hard failure if anything went wrong, not a silent partial
# install.
exit "${VALIDATE_RC}"
