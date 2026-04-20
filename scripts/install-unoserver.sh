#!/usr/bin/env bash
# Install + enable a unoserver user-systemd unit.
#
# unoserver keeps a single LibreOffice process resident on UNO port 2002,
# so subsequent PPTX/DOCX → PDF conversions skip the ~5s cold start.
# meeting-scribe's express-render path probes 127.0.0.1:2003 for
# `unoconvert`; if reachable, single-slide renders drop from ~3-5s
# (subprocess libreoffice on a 1-slide minimal PPTX) to well under 1s.
# When the daemon is not running, the subprocess fallback is used —
# nothing breaks if you skip this.
#
# Usage:
#   ./scripts/install-unoserver.sh                 # install + enable + start
#   MEETING_SCRIBE_UNOSERVER_PORT=2103 ./scripts/install-unoserver.sh
#
# Override the port if you already have a unoserver instance bound.

set -euo pipefail

PORT="${MEETING_SCRIBE_UNOSERVER_PORT:-2003}"
UNO_PORT="${MEETING_SCRIBE_UNO_PORT:-2002}"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_NAME="unoserver-meeting-scribe.service"

echo "==> Detecting LibreOffice's bundled Python"
LO_PY=""
for cand in \
    /usr/lib/libreoffice/program/python \
    /usr/lib64/libreoffice/program/python \
    /opt/libreoffice*/program/python ; do
    if [[ -x "$cand" ]]; then
        LO_PY="$cand"
        break
    fi
done

if [[ -z "$LO_PY" ]]; then
    echo "WARN: could not locate libreoffice's bundled python — falling back to system python3."
    echo "      unoserver requires UNO bindings; if the install fails, install LibreOffice first:"
    echo "        sudo apt install libreoffice libreoffice-script-provider-python python3-uno"
    LO_PY="$(command -v python3)"
fi

echo "==> Using python: $LO_PY"

echo "==> Installing unoserver into $LO_PY"
"$LO_PY" -m pip install --user --upgrade unoserver

mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/$UNIT_NAME" <<EOF
[Unit]
Description=unoserver — warm LibreOffice daemon for meeting-scribe slide renders
After=network.target

[Service]
Type=simple
ExecStart=${HOME}/.local/bin/unoserver --interface 127.0.0.1 --port ${PORT} --uno-port ${UNO_PORT}
Restart=on-failure
RestartSec=3
# unoserver loads LO once; tighten the working set so it doesn't grow unbounded
MemoryMax=2G

[Install]
WantedBy=default.target
EOF

echo "==> Wrote $UNIT_DIR/$UNIT_NAME"
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME"

echo "==> Status"
systemctl --user status --no-pager "$UNIT_NAME" | head -12

cat <<MSG

Done. Verify with:
  ss -lntp | grep ${PORT}                          # should show LISTEN on 127.0.0.1:${PORT}
  echo | unoconvert --host 127.0.0.1 --port ${PORT} --convert-to pdf - -

To point meeting-scribe at a non-default port, set:
  MEETING_SCRIBE_UNOSERVER_PORT=${PORT}
in the meeting-scribe service Environment= and restart it.
MSG
