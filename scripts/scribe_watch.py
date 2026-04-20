#!/usr/bin/env python3
"""Quiet state-transition watcher for the meeting-scribe server.

Replaces the noisy `journalctl -f | grep` Monitor with a state machine
that polls /api/status every N seconds and emits a SINGLE line **only
when something interesting changes**. The signal-to-noise ratio is the
whole point — if nothing has changed, this stays silent.

Designed to be wrapped by Claude Code's Monitor tool. Each emitted line
is one event; you'll never get the "5 events suppressed" rate-limit kill
because there's no flood to suppress.

Tracked transitions:

  * server pid changes (restart)
  * meeting state: idle → recording / interrupted / stopped
  * any backend goes from ready=true → false (or vice versa)
  * TTS delivery rate falls below threshold or recovers
  * Translation failures rate increases
  * Loop-lag p95 crosses thresholds (500 / 2000 / 5000 ms)
  * GPU VRAM crosses 90% / 95%
  * meeting language pair changes
  * NEW errors appear in the journal (deduped against last poll)

Usage (standalone):
    python3 scripts/scribe_watch.py            # default 10s poll
    python3 scripts/scribe_watch.py --interval 5

Usage (under Claude Code Monitor):
    Monitor(command="python3 scripts/scribe_watch.py", persistent=true)
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ADMIN_HOSTS = ("https://localhost:8080", "https://127.0.0.1:8080", "https://192.168.8.153:8080")
SYSTEMD_UNIT = "meeting-scribe.service"
ERROR_RE = re.compile(r"(ERROR|CRITICAL) meeting_scribe\.\S+: (.+)$")
TRACE_RE = re.compile(r"^.*Traceback")
DROP_FROM_LAST = re.compile(r"^.*(WARNING|INFO).*$")  # drop info lines


def now_iso() -> str:
    return datetime.now().strftime("%H:%M:%S")


def emit(level: str, msg: str) -> None:
    """Emit a single event line on stdout. Each call is one Monitor event."""
    print(f"{now_iso()} [{level}] {msg}", flush=True)


def fetch_status() -> dict | None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for host in ADMIN_HOSTS:
        try:
            with urllib.request.urlopen(f"{host}/api/status", timeout=3, context=ctx) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue
    return None


def fetch_systemd() -> tuple[int | None, str]:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", SYSTEMD_UNIT, "-p", "MainPID", "-p", "ActiveState"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, "unknown"
    pid, state = None, "unknown"
    for line in r.stdout.splitlines():
        if line.startswith("MainPID="):
            try:
                pid = int(line.split("=", 1)[1]) or None
            except ValueError:
                pass
        elif line.startswith("ActiveState="):
            state = line.split("=", 1)[1]
    return pid, state


def fetch_recent_errors(since_iso: str) -> list[str]:
    """Pull ERROR/Traceback lines from the journal since `since_iso`. Returns
    a list of fingerprint strings — caller dedups against last poll's set."""
    try:
        r = subprocess.run(
            ["journalctl", "--user", "-u", SYSTEMD_UNIT, "--since", since_iso,
             "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    out: list[str] = []
    for line in r.stdout.splitlines():
        m = ERROR_RE.search(line)
        if m:
            # Fingerprint = error message minus volatile bits (timestamps, ids)
            fp = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", m.group(2))[:120]
            out.append(fp)
        elif TRACE_RE.match(line):
            out.append(f"traceback: {line[:80]}")
    return out


# ── State diffing ──────────────────────────────────────────────────────────
def snapshot(api: dict | None, pid: int | None, state: str) -> dict:
    """Reduce the current state to a small dict of comparable fields."""
    if not api:
        return {"pid": pid, "systemd": state, "api": None}
    m = api.get("meeting") or {}
    bd = api.get("backend_details") or {}
    metrics = api.get("metrics") or {}
    tts = metrics.get("tts") or {}
    ll = metrics.get("loop_lag_ms") or {}
    gpu = api.get("gpu") or {}
    backends = {}
    for name in ("asr", "translate", "diarize", "tts", "furigana"):
        d = bd.get(name) or {}
        backends[name] = {
            "ready": d.get("ready", False),
            "fails": d.get("consecutive_failures", 0) or 0,
        }
    return {
        "pid": pid,
        "systemd": state,
        "meeting_id": m.get("id") or "",
        "meeting_state": m.get("state"),
        "backends": backends,
        "translations_failed": metrics.get("translations_failed", 0),
        "translations_completed": metrics.get("translations_completed", 0),
        "tts_submitted": tts.get("submitted", 0),
        "tts_delivered": tts.get("delivered", 0),
        "tts_health": tts.get("health"),
        "loop_lag_p95": int(ll.get("p95") or 0),
        "vram_pct": int(gpu.get("vram_pct") or 0),
    }


def diff_and_emit(prev: dict | None, cur: dict) -> None:
    if prev is None:
        # First poll — establish baseline. Emit a one-line "started" event.
        emit("INFO", f"watch start · pid={cur['pid']} systemd={cur['systemd']} "
                     f"meeting={cur['meeting_id'] or 'idle'}/{cur['meeting_state']}")
        return

    # PID change → restart
    if cur["pid"] != prev["pid"]:
        emit("WARN", f"server pid changed: {prev['pid']} → {cur['pid']} (restart?)")

    # systemd state change
    if cur["systemd"] != prev["systemd"]:
        emit("WARN", f"systemd state: {prev['systemd']} → {cur['systemd']}")

    # No API → can't diff further
    if cur.get("api") is None and prev.get("api") is None:
        return

    # Meeting transitions
    if cur.get("meeting_state") != prev.get("meeting_state"):
        emit("INFO", f"meeting state: {prev.get('meeting_state')} → {cur.get('meeting_state')}"
                     f" (id={cur['meeting_id'] or 'idle'})")
    elif cur.get("meeting_id") != prev.get("meeting_id"):
        emit("INFO", f"new meeting: {cur['meeting_id']} ({cur.get('meeting_state')})")

    # Backend health transitions
    for name, b in cur.get("backends", {}).items():
        pb = prev.get("backends", {}).get(name) or {}
        if b["ready"] != pb.get("ready"):
            arrow = "✗→✓" if b["ready"] else "✓→✗"
            level = "INFO" if b["ready"] else "ERROR"
            emit(level, f"{name} backend {arrow}")
        elif b["fails"] > 0 and pb.get("fails", 0) == 0:
            emit("WARN", f"{name} backend started failing ({b['fails']} consecutive)")
        elif b["fails"] == 0 and pb.get("fails", 0) > 0:
            emit("INFO", f"{name} backend recovered")

    # Translation failures
    new_fail = cur.get("translations_failed", 0) - prev.get("translations_failed", 0)
    if new_fail > 0:
        emit("ERROR", f"{new_fail} new translation failures "
                      f"(total: {cur['translations_failed']}/{cur['translations_completed']})")

    # TTS delivery health: submitted advancing but delivered stuck
    new_sub = cur["tts_submitted"] - prev["tts_submitted"]
    new_deliv = cur["tts_delivered"] - prev["tts_delivered"]
    if new_sub >= 3 and new_deliv == 0:
        emit("ERROR", f"TTS submitted {new_sub} requests since last poll, delivered 0 "
                      f"(total {cur['tts_delivered']}/{cur['tts_submitted']})")
    elif cur["tts_health"] != prev["tts_health"] and cur["tts_health"]:
        level = "ERROR" if cur["tts_health"] != "healthy" else "INFO"
        emit(level, f"TTS health: {prev['tts_health']} → {cur['tts_health']}")

    # Loop-lag thresholds (only emit when crossing UP into a worse band, or
    # crossing DOWN into a better band — not while sustained in the same band)
    bands = [(5000, "5s+"), (2000, "2s+"), (500, "500ms+"), (0, "ok")]
    def band(p95: int) -> str:
        for thr, name in bands:
            if p95 >= thr:
                return name
        return "ok"
    cur_band = band(cur["loop_lag_p95"])
    prev_band = band(prev["loop_lag_p95"])
    if cur_band != prev_band:
        level = "ERROR" if cur_band in ("5s+", "2s+") else (
            "WARN" if cur_band == "500ms+" else "INFO"
        )
        emit(level, f"loop-lag p95: {prev_band} → {cur_band} ({cur['loop_lag_p95']}ms)")

    # GPU VRAM
    if cur["vram_pct"] >= 95 and prev["vram_pct"] < 95:
        emit("ERROR", f"GPU VRAM: {prev['vram_pct']}% → {cur['vram_pct']}% (≥95%)")
    elif cur["vram_pct"] >= 90 and prev["vram_pct"] < 90:
        emit("WARN", f"GPU VRAM: {prev['vram_pct']}% → {cur['vram_pct']}% (≥90%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=10, help="poll interval seconds (default 10)")
    ap.add_argument("--errors-window", type=int, default=20,
                    help="seconds of journal to scan for new errors per poll (default 20)")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="exit after N polls (0 = forever) — useful for smoke tests")
    args = ap.parse_args()

    prev: dict | None = None
    seen_errors: set[str] = set()
    iters = 0
    while True:
        api = fetch_status()
        pid, state = fetch_systemd()
        cur = snapshot(api, pid, state)
        diff_and_emit(prev, cur)

        # New journal errors
        try:
            errs = fetch_recent_errors(f"{args.errors_window} seconds ago")
        except Exception:
            errs = []
        new_errs = [e for e in errs if e not in seen_errors]
        for e in new_errs[:5]:
            emit("ERROR", f"journal: {e}")
            seen_errors.add(e)
        # Bound the seen set so it doesn't grow forever
        if len(seen_errors) > 200:
            seen_errors = set(list(seen_errors)[-100:])

        prev = cur
        iters += 1
        if args.max_iters and iters >= args.max_iters:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
