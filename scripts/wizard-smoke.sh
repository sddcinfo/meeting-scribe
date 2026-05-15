#!/usr/bin/env bash
#
# wizard-smoke.sh — exercise the v1.0 setup wizard without containers.
#
# The first-touch wizard (Phase F routes + Phase B/D2/G plumbing) is
# pure FastAPI + a state machine + on-disk artifacts. None of the
# audio backends (vLLM, pyannote, qwen3-tts) are involved, so we
# don't have to wait for compose builds or model downloads to test
# the operator-visible flow.
#
# Boots ``meeting_scribe.server:app`` under uvicorn with
# ``--lifespan off`` so backend init is skipped, on a high port so
# CAP_NET_BIND is not required. Use this before the full
# ``meeting-scribe start`` for fast wizard iteration.
#
# Usage:
#   scripts/wizard-smoke.sh start [--port 8443]   # background-launch
#   scripts/wizard-smoke.sh stop                  # kill via PID file
#   scripts/wizard-smoke.sh status                # ps + curl probe
#   scripts/wizard-smoke.sh log                   # tail the log
#
# State files live under /tmp/wizard-smoke-* so the user does not
# need write perms anywhere outside their HOME + /tmp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE=/tmp/wizard-smoke.pid
LOG_FILE=/tmp/wizard-smoke.log
CAPTIVE_PID_FILE=/tmp/wizard-smoke-captive.pid
CAPTIVE_LOG_FILE=/tmp/wizard-smoke-captive.log
PORT="${WIZARD_SMOKE_PORT:-8443}"
# ``WIZARD_SMOKE_HOST`` overrides the bind address. Default
# loopback for in-place dev; set to ``10.42.0.1`` (with the AP
# already up) to test the production canonical-origin flow over
# real Wi-Fi without needing CAP_NET_BIND_SERVICE for port 443.
HOST="${WIZARD_SMOKE_HOST:-127.0.0.1}"
# When ``WIZARD_SMOKE_CANONICAL=1`` we run two listeners: the main
# HTTPS app on ``${HOST}:443`` and the captive HTTP sub-app on
# ``${HOST}:80``. Both need CAP_NET_BIND_SERVICE; we sudo the
# uvicorn invocation rather than setcap-ing python persistently.
CANONICAL="${WIZARD_SMOKE_CANONICAL:-0}"

cmd_start() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "wizard-smoke already running (pid $(cat "${PID_FILE}"))"
        exit 1
    fi

    local cert_dir="${REPO_ROOT}/certs"
    if [[ ! -f "${cert_dir}/cert.pem" || ! -f "${cert_dir}/key.pem" ]]; then
        echo "wizard-smoke: cert pair missing in ${cert_dir}" >&2
        echo "run: ${REPO_ROOT}/.venv/bin/meeting-scribe provision fingerprint" >&2
        exit 2
    fi

    # The setup-bootstrap-secret + fingerprint precondition was
    # dropped in the v1.0 simplification. The wizard now claims
    # credentials directly on GET /setup.

    cd "${REPO_ROOT}"
    # ``--lifespan off`` skips backend init (no vLLM, no pyannote)
    # and the lifespan's AP bring-up. We exercise the routes only.
    # ``SCRIBE_WIZARD_DEV_ALLOW_LOOPBACK=1`` lets the AP-origin
    # gate accept 127.0.0.1 — needed when the AP isn't up.
    export SCRIBE_WIZARD_DEV_ALLOW_LOOPBACK=1

    if [[ "${CANONICAL}" == "1" ]]; then
        # Canonical mode — bind 443 (main app) + 80 (captive sub-app)
        # so a phone joined to the open AP can drive the wizard at
        # the production URL. Both ports require CAP_NET_BIND_SERVICE
        # so we sudo the uvicorn invocation.
        local main_port=443
        local captive_port=80
        # Pre-clean log files. Sudo'd root sometimes can't reopen
        # a stale delldemo-owned tmp log for write — observed on the
        # GB10 walkthrough 2026-05-05. rm + fresh-create avoids the
        # "Permission denied" from inside the sudo'd bash -c.
        sudo -n rm -f "${LOG_FILE}" "${CAPTIVE_LOG_FILE}" 2>/dev/null || true
        sudo -n bash -c "
            SCRIBE_WIZARD_DEV_ALLOW_LOOPBACK=1 \\
            ${REPO_ROOT}/.venv/bin/python -m uvicorn \\
                meeting_scribe.server:app \\
                --host '${HOST}' \\
                --port '${main_port}' \\
                --ssl-keyfile '${cert_dir}/key.pem' \\
                --ssl-certfile '${cert_dir}/cert.pem' \\
                --lifespan off --log-level info \\
                > '${LOG_FILE}' 2>&1 &
            echo \$!" > "${PID_FILE}"
        sudo -n bash -c "
            SCRIBE_CANONICAL_HOST='10.42.0.1' \\
            ${REPO_ROOT}/.venv/bin/python -c '
import uvicorn
from meeting_scribe.hotspot.captive_http_app import build_captive_http_app
uvicorn.run(build_captive_http_app(), host=\"${HOST}\", port=${captive_port}, log_level=\"info\")
            ' > '${CAPTIVE_LOG_FILE}' 2>&1 &
            echo \$!" > "${CAPTIVE_PID_FILE}"
        sleep 2
        # Capture the actual python child PIDs (sudo bash -c '... &' echoes
        # the bash PID; the child python is ours to track + signal). Look
        # them up by matching the listening sockets.
        local main_real captive_real
        main_real=$(sudo -n ss -tlnpe sport = :${main_port} 2>/dev/null \
            | awk '/python/ {print $0}' \
            | grep -oP 'pid=\K[0-9]+' | head -1 || true)
        captive_real=$(sudo -n ss -tlnpe sport = :${captive_port} 2>/dev/null \
            | awk '/python/ {print $0}' \
            | grep -oP 'pid=\K[0-9]+' | head -1 || true)
        if [[ -n "${main_real}" ]]; then echo "${main_real}" > "${PID_FILE}"; fi
        if [[ -n "${captive_real}" ]]; then echo "${captive_real}" > "${CAPTIVE_PID_FILE}"; fi
        echo "wizard-smoke (canonical) up:"
        echo "  main:    https://${HOST}/setup (pid ${main_real:-unknown})"
        echo "  captive: http://${HOST}/      (pid ${captive_real:-unknown})"
        echo "  logs:    ${LOG_FILE}, ${CAPTIVE_LOG_FILE}"
        return 0
    fi

    # Default: single non-priv listener on a high port.
    nohup "${REPO_ROOT}/.venv/bin/python" -m uvicorn \
        meeting_scribe.server:app \
        --host "${HOST}" \
        --port "${PORT}" \
        --ssl-keyfile "${cert_dir}/key.pem" \
        --ssl-certfile "${cert_dir}/cert.pem" \
        --lifespan off \
        --log-level info \
        > "${LOG_FILE}" 2>&1 &
    echo "$!" > "${PID_FILE}"
    sleep 2

    if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "wizard-smoke: server died on start. Tail:" >&2
        tail -20 "${LOG_FILE}" >&2
        rm -f "${PID_FILE}"
        exit 3
    fi
    echo "wizard-smoke up at https://${HOST}:${PORT}/setup (pid $(cat "${PID_FILE}"))"
    echo "log: ${LOG_FILE}"
}

cmd_stop() {
    local stopped=0
    for pf in "${PID_FILE}" "${CAPTIVE_PID_FILE}"; do
        [[ -f "${pf}" ]] || continue
        local pid
        pid=$(cat "${pf}")
        if [[ -n "${pid}" ]] && (kill -0 "${pid}" 2>/dev/null || sudo -n kill -0 "${pid}" 2>/dev/null); then
            kill -TERM "${pid}" 2>/dev/null || sudo -n kill -TERM "${pid}" 2>/dev/null || true
            sleep 1
            if kill -0 "${pid}" 2>/dev/null || sudo -n kill -0 "${pid}" 2>/dev/null; then
                kill -KILL "${pid}" 2>/dev/null || sudo -n kill -KILL "${pid}" 2>/dev/null || true
            fi
            stopped=1
        fi
        rm -f "${pf}"
    done
    if [[ ${stopped} -eq 1 ]]; then
        echo "wizard-smoke stopped"
    else
        echo "wizard-smoke not running"
    fi
}

cmd_status() {
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "running pid $(cat "${PID_FILE}")"
        curl -k -s -o /dev/null -w "GET /setup → HTTP %{http_code}\n" \
            "https://${HOST}:${PORT}/setup" || true
    else
        echo "not running"
    fi
}

cmd_log() {
    tail -n "${1:-40}" "${LOG_FILE}"
}

case "${1:-}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    log)    shift; cmd_log "${1:-40}" ;;
    *)
        echo "usage: $(basename "$0") {start|stop|status|log [N]}" >&2
        exit 1
        ;;
esac
