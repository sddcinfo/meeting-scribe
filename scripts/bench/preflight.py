"""Production-safety preflight for the 2026-Q2 model-challenger bench.

Hard-fails the bench session unless ALL of the following hold:

1. ``meeting-scribe`` server is reachable on its API port AND reports
   no in-flight meeting (state in the empty-OK set).
2. ``MEETING_SCRIBE_BENCH_WINDOW=1`` is set in the operator's shell.
3. ``/tmp/scribe-bench-window.txt`` exists with a non-empty reason.

The bench can only run on the production GB10; this is the gate that
keeps it from interrupting live interpretation.  Every track's first
action runs::

    python3 scripts/bench/preflight.py --port 8080

and refuses to proceed on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Meeting states that mean "do not bench".  Mirrors the lifecycle
# documented in cli/meetings.py (the `interrupted` / `recording` /
# `stopping` / `reprocessing` set).
IN_FLIGHT_STATES = {"recording", "stopping", "reprocessing", "interrupted"}

WINDOW_REASON_FILE = Path("/tmp/scribe-bench-window.txt")
WINDOW_ENV_VAR = "MEETING_SCRIBE_BENCH_WINDOW"


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"OK:   {msg}")


def check_window_env() -> None:
    if os.environ.get(WINDOW_ENV_VAR) != "1":
        _fail(
            f"{WINDOW_ENV_VAR} not set to 1.  This bench requires an explicit "
            f"window declaration on the operator's shell."
        )
    _ok(f"{WINDOW_ENV_VAR}=1")


def check_window_reason() -> None:
    if not WINDOW_REASON_FILE.exists():
        _fail(
            f"{WINDOW_REASON_FILE} missing.  Record the bench window reason "
            f"(e.g. 'Track A Cohere Transcribe A/B, 2026-04-30 evening')."
        )
    text = WINDOW_REASON_FILE.read_text().strip()
    if not text:
        _fail(f"{WINDOW_REASON_FILE} is empty.  Record a non-empty reason.")
    _ok(f"window reason: {text!r}")


def check_no_meeting_in_flight(base_url: str) -> None:
    """``base_url`` like ``https://192.168.1.168:8080`` (no trailing slash)."""
    url = f"{base_url.rstrip('/')}/api/status"
    ctx = ssl.create_default_context()
    if base_url.startswith("https://"):
        # The dev server uses a self-signed cert.  Skip verify for the
        # preflight only — we are talking to localhost / LAN-IP, not a
        # third party.
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        _fail(f"Could not reach {url}: {exc}.  Start meeting-scribe before the bench.")
    meeting = data.get("meeting") or {}
    state = meeting.get("state")
    if state and state in IN_FLIGHT_STATES:
        _fail(
            f"Meeting {meeting.get('id')} is in state {state!r}.  Bench requires "
            f"no in-flight meeting (states {sorted(IN_FLIGHT_STATES)} are blocking)."
        )
    _ok(f"no in-flight meeting (current state: {state!r})")
    gpu = data.get("gpu") or {}
    if gpu:
        _ok(
            f"GPU vram {gpu.get('vram_used_mb')} MB / {gpu.get('vram_total_mb')} MB "
            f"({gpu.get('vram_pct')}%) — challenger headroom = "
            f"{(gpu.get('vram_total_mb') or 0) - (gpu.get('vram_used_mb') or 0)} MB"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--api-url",
        default=os.environ.get("SCRIBE_API_URL", "https://localhost:8080"),
        help="meeting-scribe API base URL (default https://localhost:8080 or $SCRIBE_API_URL)",
    )
    args = p.parse_args(argv)

    check_window_env()
    check_window_reason()
    check_no_meeting_in_flight(args.api_url)
    print("Preflight: all gates pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
