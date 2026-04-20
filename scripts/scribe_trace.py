#!/usr/bin/env python3
"""On-demand troubleshooting snapshot for the running meeting-scribe server.

Replaces the "tail -f journalctl | grep | flood the chat" workflow. This
script is **pull-only** — you run it whenever you want to see what the
server has been doing, instead of having a stream interrupt you on every
log line.

Defaults to a 90-second snapshot of the live state with:

  * server PID + uptime + active meeting
  * backend health (ASR / Translate / TTS / Diarize / Furigana)
  * last N transcript finals with per-segment join across ASR / language
    remap / furigana / translation / speaker catch-up — so you can see at
    a glance which dimension is missing for any given segment
  * deduplicated error summary (counts + first/last timestamps)
  * latency stats: median / p95 translation, furigana cadence, language
    remap rate

Subcommands:
  (default)               → snapshot of last 90 s
  --seg <id>              → trace one segment_id end-to-end
  --meeting <id>          → activity for one meeting
  --tail <n>              → last n finals (any meeting)
  --errors                → only the deduped error summary
  --since "5 min ago"     → override the journal lookback window

No external dependencies — stdlib only. Designed to be safe to re-run
during an active meeting.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import ssl
import statistics
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime

# ── Configuration ──────────────────────────────────────────────────────────
ADMIN_HOST_CANDIDATES = (
    "https://localhost:8080",
    "https://127.0.0.1:8080",
    "https://192.168.8.153:8080",
)
SYSTEMD_UNIT = "meeting-scribe.service"


# ── ANSI ───────────────────────────────────────────────────────────────────
def _supports_color() -> bool:
    return sys.stdout.isatty()


_COLOR = _supports_color()
def _c(code: str, s: str) -> str:
    if not _COLOR:
        return s
    return f"\033[{code}m{s}\033[0m"


def green(s: str) -> str: return _c("32", s)
def yellow(s: str) -> str: return _c("33", s)
def red(s: str) -> str: return _c("31", s)
def cyan(s: str) -> str: return _c("36", s)
def dim(s: str) -> str: return _c("2", s)
def bold(s: str) -> str: return _c("1", s)


# ── Journal parsing ────────────────────────────────────────────────────────
TS_RE = re.compile(r"^([A-Z][a-z]{2} \d{1,2} \d\d:\d\d:\d\d) ")
SUBMIT_RE = re.compile(
    r"Submitting for translation: seg=(?P<sid>[0-9a-f]+)\s+lang=(?P<lang>\S+)\s+text='(?P<text>[^']*)'",
)
WORKER_DONE_RE = re.compile(
    r"Worker (?P<wid>\d+) translated segment_id=(?P<sid>[0-9a-f-]+) in (?P<ms>\d+)ms \((?P<srcc>\d+)→(?P<dstc>\d+) chars\)"
)
WORKER_SKIP_RE = re.compile(
    r"Worker (?P<wid>\d+) .*?skipp(?:ed|ing)?.*? segment_id=(?P<sid>[0-9a-f-]+)"
)
FURIGANA_OK_RE = re.compile(
    r"furigana: seg=(?P<sid>[0-9a-f]+) text=(?P<text>.*?) → (?P<chars>\d+) chars$"
)
FURIGANA_NONE_RE = re.compile(
    r"furigana: no annotation for seg=(?P<sid>[0-9a-f]+) text=(?P<text>.*?) \(no kanji"
)
LANG_REMAP_RE = re.compile(
    r"Language remap: ASR=(?P<asr>\S+) → script=(?P<script>\S+)(?: → (?P<final>\S+))? \(text='(?P<text>[^']*)'\)"
)
CATCHUP_RE = re.compile(
    r"Speaker catch-up: seg=(?P<sid>[0-9a-f]+) age=(?P<age>[\d.]+)s → cluster (?P<cid>\d+)"
)
LOOPLAG_RE = re.compile(r"Loop-lag (?P<ms>\d+)ms")
TTS_DEGRADE_RE = re.compile(r"TTS marked degraded after (?P<n>\d+) consecutive failures")
ERROR_PREFIX_RE = re.compile(r"(?P<level>WARNING|ERROR|EXCEPTION|CRITICAL) (?P<module>\S+):")
TRACE_RE = re.compile(r"^Traceback")
SEGMENT_ID_PREFIX = re.compile(r"^[0-9a-f]{8}")


@dataclass
class SegmentRecord:
    sid: str
    first_seen: float | None = None
    asr_lang: str | None = None
    final_lang: str | None = None  # after language remap
    text: str = ""
    remapped: bool = False
    remap_reason: str | None = None
    furigana_chars: int | None = None  # 0 = no kanji, None = not run, >0 = annotated
    translate_ms: int | None = None
    translate_skip: bool = False
    speaker_cluster: int | None = None
    speaker_age: float | None = None


def _parse_ts(line: str) -> float | None:
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        # systemd journal format: "Apr 14 21:56:44" — no year, current year assumed
        now = datetime.now(UTC)
        dt = datetime.strptime(f"{now.year} {m.group(1)}", "%Y %b %d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        return None


def fetch_journal(since: str = "90 seconds ago") -> list[str]:
    cmd = ["journalctl", "--user", "-u", SYSTEMD_UNIT, "--since", since, "--no-pager"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(red(f"journalctl unavailable: {e}"), file=sys.stderr)
        return []
    return result.stdout.splitlines()


def parse_journal(lines: list[str]) -> tuple[dict[str, SegmentRecord], dict, dict]:
    segments: dict[str, SegmentRecord] = {}
    error_counts: dict[tuple[str, str], dict] = {}  # (kind, fingerprint) → {count, first, last}
    stats = {
        "translate_latencies_ms": [],
        "furigana_calls": 0,
        "furigana_no_kanji": 0,
        "language_remaps": 0,
        "loop_lag_count": 0,
        "loop_lag_max_ms": 0,
        "tts_degrade_events": 0,
    }

    def _bump_error(kind: str, fingerprint: str, ts: float | None, sample: str):
        key = (kind, fingerprint)
        rec = error_counts.setdefault(
            key, {"count": 0, "first": ts, "last": ts, "sample": sample}
        )
        rec["count"] += 1
        if ts is not None:
            if rec["first"] is None or ts < rec["first"]:
                rec["first"] = ts
            if rec["last"] is None or ts > rec["last"]:
                rec["last"] = ts

    for line in lines:
        ts = _parse_ts(line)
        body = line[16:] if ts else line  # drop the "Mon DD HH:MM:SS " prefix

        # Submit (first sighting of a segment)
        m = SUBMIT_RE.search(body)
        if m:
            sid = m.group("sid")
            rec = segments.setdefault(sid, SegmentRecord(sid=sid))
            if rec.first_seen is None:
                rec.first_seen = ts
            rec.text = m.group("text")
            rec.final_lang = m.group("lang")
            if rec.asr_lang is None:
                rec.asr_lang = m.group("lang")
            continue

        # Language remap (always logged BEFORE submit, so attach to next submit)
        m = LANG_REMAP_RE.search(body)
        if m:
            stats["language_remaps"] += 1
            # Match against the most-recent unattributed submit. Easiest is to
            # remember the remap and attach it to the next Submit by scanning
            # forward — but Submit lines for the SAME segment come within the
            # next ms, so we just stash the most recent remap and attach
            # heuristically via text equality.
            text = m.group("text")
            for rec in reversed(list(segments.values())):
                if rec.text == text and not rec.remapped:
                    rec.remapped = True
                    rec.asr_lang = m.group("asr")
                    rec.final_lang = m.group("final") or m.group("script")
                    rec.remap_reason = f"{m.group('asr')}→{m.group('script')}"
                    break
            else:
                # No matching segment yet — store as a pending remap so the
                # next Submit can claim it. Implement as a ring of one.
                stats.setdefault("_pending_remap", []).append(
                    (text, m.group("asr"), m.group("script"), m.group("final"))
                )
            continue

        # Translation done
        m = WORKER_DONE_RE.search(body)
        if m:
            sid = m.group("sid")
            ms = int(m.group("ms"))
            rec = segments.setdefault(sid, SegmentRecord(sid=sid))
            rec.translate_ms = ms
            stats["translate_latencies_ms"].append(ms)
            continue

        # Translation skipped
        m = WORKER_SKIP_RE.search(body)
        if m:
            sid = m.group("sid")
            rec = segments.setdefault(sid, SegmentRecord(sid=sid))
            rec.translate_skip = True
            _bump_error("translate_skip", "skipped", ts, body[:80])
            continue

        # Furigana annotated
        m = FURIGANA_OK_RE.search(body)
        if m:
            sid = m.group("sid")
            rec = segments.setdefault(sid, SegmentRecord(sid=sid))
            rec.furigana_chars = int(m.group("chars"))
            stats["furigana_calls"] += 1
            continue

        # Furigana ran but no kanji
        m = FURIGANA_NONE_RE.search(body)
        if m:
            sid = m.group("sid")
            rec = segments.setdefault(sid, SegmentRecord(sid=sid))
            rec.furigana_chars = 0
            stats["furigana_calls"] += 1
            stats["furigana_no_kanji"] += 1
            continue

        # Speaker catch-up
        m = CATCHUP_RE.search(body)
        if m:
            sid = m.group("sid")
            rec = segments.setdefault(sid, SegmentRecord(sid=sid))
            rec.speaker_cluster = int(m.group("cid"))
            rec.speaker_age = float(m.group("age"))
            continue

        # Loop-lag
        m = LOOPLAG_RE.search(body)
        if m:
            ms = int(m.group("ms"))
            stats["loop_lag_count"] += 1
            stats["loop_lag_max_ms"] = max(stats["loop_lag_max_ms"], ms)
            _bump_error("loop_lag", f">{ms // 500 * 500}ms", ts, body[:80])
            continue

        # TTS degraded
        if TTS_DEGRADE_RE.search(body):
            stats["tts_degrade_events"] += 1
            _bump_error("tts_500", "/v1/audio/speech", ts, body[:120])
            continue

        # Generic error / traceback
        m = ERROR_PREFIX_RE.search(body)
        if m and m.group("level") in ("ERROR", "CRITICAL", "EXCEPTION"):
            mod = m.group("module").rsplit(".", 1)[-1]
            _bump_error(mod, body[:60], ts, body[:200])
            continue
        if TRACE_RE.search(body):
            _bump_error("traceback", body[:60], ts, body[:200])
            continue

    return segments, error_counts, stats


# ── /api/status fetch ──────────────────────────────────────────────────────
def fetch_api_status() -> dict | None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for host in ADMIN_HOST_CANDIDATES:
        try:
            with urllib.request.urlopen(f"{host}/api/status", timeout=2, context=ctx) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue
    return None


def fetch_systemd_main_pid() -> tuple[int | None, str]:
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", SYSTEMD_UNIT,
             "-p", "MainPID", "-p", "ActiveState"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, "unknown"
    pid = None
    state = "unknown"
    for line in r.stdout.splitlines():
        if line.startswith("MainPID="):
            try:
                pid = int(line.split("=", 1)[1]) or None
            except ValueError:
                pass
        elif line.startswith("ActiveState="):
            state = line.split("=", 1)[1]
    return pid, state


def fmt_pid_uptime(pid: int | None) -> str:
    if not pid:
        return "(no pid)"
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime,user", "--no-headers"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return f"pid {pid}"
    parts = r.stdout.split()
    return f"pid {pid} up {parts[0]}" if parts else f"pid {pid}"


# ── Render helpers ─────────────────────────────────────────────────────────
def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = round((len(s) - 1) * pct / 100)
    return s[k]


def fmt_time_hhmmss(ts: float | None) -> str:
    if ts is None:
        return "  --:--:--"
    return datetime.fromtimestamp(ts, tz=UTC).astimezone().strftime("%H:%M:%S")


def fmt_age(ts: float | None, now: float) -> str:
    if ts is None:
        return ""
    age = now - ts
    if age < 60:
        return f"{age:4.1f}s"
    if age < 3600:
        return f"{age/60:4.1f}m"
    return f"{age/3600:4.1f}h"


def fmt_segment_row(rec: SegmentRecord, t0: float | None) -> str:
    if rec.first_seen and t0:
        rel = f"+{rec.first_seen - t0:5.1f}s"
    else:
        rel = "       "
    lang = rec.final_lang or "??"
    lang_chip = lang
    if rec.remapped:
        lang_chip = yellow(f"{lang}*")  # asterisk = remapped
    if rec.translate_skip:
        tr = red("SKIP    ")
    elif rec.translate_ms is not None:
        tr = f"{rec.translate_ms:4d}ms  "
    else:
        tr = dim("  --    ")
    if rec.furigana_chars is None:
        fur = dim("   --   ")
    elif rec.furigana_chars == 0:
        fur = dim(" no kanji")
    else:
        fur = green(f" {rec.furigana_chars:3d}ch  ")
    spk = f"c{rec.speaker_cluster}" if rec.speaker_cluster is not None else dim("--")
    text = rec.text
    if len(text) > 50:
        text = text[:47] + "..."
    return f"  {rel}  {lang_chip:<4}  {tr}  {fur}  {spk:>4}  {text}"


def render_snapshot(args, segments, error_counts, stats, api_status, pid, state):
    now = datetime.now(UTC).timestamp()
    print()
    print(bold("═══ meeting-scribe trace ═══"), dim(datetime.now().strftime("%H:%M:%S %Z")))
    print()

    # Server line
    server = fmt_pid_uptime(pid)
    color_state = green if state == "active" else (yellow if state == "activating" else red)
    print(f"  server:   {server} · {color_state(state)}")

    # Meeting line
    if api_status:
        m = api_status.get("meeting") or {}
        mid = m.get("id") or "(none)"
        mstate = m.get("state") or "idle"
        ws = api_status.get("connections", "?")
        ao = api_status.get("audio_out_connections", 0)
        print(f"  meeting:  {mid} {mstate} · {ws} ws · {ao} audio-out")

    # Backend health
    if api_status:
        bd = api_status.get("backend_details") or {}
        chips = []
        for name in ("asr", "translate", "diarize", "tts", "furigana"):
            data = bd.get(name) or {}
            ready = data.get("ready", False)
            failures = data.get("consecutive_failures", 0) or 0
            if ready and failures == 0:
                chips.append(green(f"{name}✓"))
            elif ready and failures > 0:
                chips.append(yellow(f"{name}⚠({failures}fails)"))
            else:
                detail = (data.get("detail") or "down")[:25]
                chips.append(red(f"{name}✗({detail})"))
        print(f"  backends: {' '.join(chips)}")

    # Live metrics block from /api/status — far richer than journal parsing
    if api_status and api_status.get("metrics"):
        m = api_status["metrics"]
        elapsed = m.get("elapsed_s", 0)
        finals = m.get("asr_finals", 0)
        eps = m.get("asr_eps", 0)
        sub = m.get("translations_submitted", 0)
        comp = m.get("translations_completed", 0)
        fail = m.get("translations_failed", 0)
        avg_ms = m.get("avg_translation_ms", 0) or 0
        line = f"  asr:      {finals} finals · {eps:.1f}/s · meeting elapsed {elapsed:.0f}s"
        print(line)

        translate_chip = green if fail == 0 else red
        line = (f"  translate:{translate_chip(f' {comp}/{sub}')} done"
                + (red(f' · {fail} failed') if fail else "")
                + (f" · avg {avg_ms:.0f}ms" if avg_ms else ""))
        print(line)

        tts = m.get("tts") or {}
        tts_sub = tts.get("submitted", 0)
        tts_deliv = tts.get("delivered", 0)
        tts_health = tts.get("health", "unknown")
        tts_drops = sum((tts.get("drops") or {}).values())
        tts_q = tts.get("queue_depth", 0)
        tts_qmax = tts.get("queue_maxsize", 0)
        tts_busy = tts.get("workers_busy", 0)
        tts_total = tts.get("workers_total", 0)
        # Smoking gun: submitted but nothing delivered
        tts_chip = green if tts_health == "healthy" else (
            yellow if tts_health == "degraded" else red
        )
        delivery_warn = ""
        if tts_sub > 0 and tts_deliv == 0:
            delivery_warn = red(f"  ⚠ {tts_sub} submitted, 0 delivered")
        elif tts_sub > 5 and tts_deliv / max(tts_sub, 1) < 0.5:
            delivery_warn = yellow(f"  ⚠ low delivery rate {tts_deliv}/{tts_sub}")
        print(f"  tts:      {tts_chip(tts_health)} · {tts_deliv}/{tts_sub} delivered · "
              f"queue {tts_q}/{tts_qmax} · workers {tts_busy}/{tts_total}"
              + (f" · {tts_drops} drops" if tts_drops else "")
              + delivery_warn)

        ll = m.get("loop_lag_ms") or {}
        if ll.get("sample_count"):
            p95 = ll.get("p95") or 0
            chip = red if p95 > 2000 else (yellow if p95 > 500 else green)
            print(f"  loop-lag: p50 {ll.get('p50', 0):.0f}ms · "
                  f"{chip(f'p95 {p95:.0f}ms')} · p99 {ll.get('p99', 0):.0f}ms"
                  f" ({ll.get('sample_count')} samples)")

    if api_status and api_status.get("gpu"):
        g = api_status["gpu"]
        pct = g["vram_pct"]
        chip = red if pct > 92 else (yellow if pct > 80 else green)
        used = g["vram_used_mb"]
        total = g["vram_total_mb"]
        print(f"  gpu:      vram {used}/{total} MB · {chip(f'{pct:.0f}%')}")

    # Activity table
    print()
    print(bold(f"  recent finals ({len(segments)} segments in window):"))
    if not segments:
        print(dim("    (no transcript activity in window)"))
    else:
        ordered = sorted(
            segments.values(),
            key=lambda r: (r.first_seen or 0),
        )
        t0 = ordered[0].first_seen if ordered else None
        print(dim("    +time   lang  trans     furigana  spk   text"))
        for rec in ordered[-args.tail:] if args.tail else ordered:
            print(fmt_segment_row(rec, t0))

    # Latency stats
    print()
    lat = stats["translate_latencies_ms"]
    if lat:
        med = int(statistics.median(lat))
        p95 = int(percentile(lat, 95) or 0)
        skip = sum(1 for s in segments.values() if s.translate_skip)
        chip = green if med < 500 else (yellow if med < 1000 else red)
        print(f"  translate: {chip(f'median {med}ms')} · p95 {p95}ms · {len(lat)} done · {skip} skipped")
    else:
        print(dim("  translate: (no completions in window)"))

    fur_calls = stats["furigana_calls"]
    if fur_calls:
        no_k = stats["furigana_no_kanji"]
        annot = fur_calls - no_k
        print(f"  furigana:  {annot} annotated · {no_k} no-kanji ({fur_calls} total)")

    rmaps = stats["language_remaps"]
    if rmaps:
        print(f"  remaps:    {yellow(f'{rmaps} language overrides')}")

    if stats["loop_lag_count"]:
        max_ms = stats["loop_lag_max_ms"]
        chip = red if max_ms > 5000 else yellow
        ll_count = stats["loop_lag_count"]
        print(f"  loop-lag:  {chip(f'{ll_count} spikes')} · max {max_ms}ms")

    # Errors
    if error_counts:
        print()
        print(bold("  errors / warnings (deduped):"))
        rows = sorted(
            error_counts.items(),
            key=lambda kv: (-kv[1]["count"], kv[0]),
        )
        for (kind, fingerprint), rec in rows[:15]:
            count = rec["count"]
            first = fmt_age(rec["first"], now)
            last = fmt_age(rec["last"], now)
            count_chip = red(f"{count:>4}×") if count > 5 else yellow(f"{count:>4}×")
            sample = rec["sample"]
            if len(sample) > 80:
                sample = sample[:77] + "..."
            print(f"    {count_chip}  {kind:<14} {dim(f'first {first} ago, last {last} ago')}")
            print(f"         {dim(fingerprint[:90])}")
    print()


def render_segment_trace(sid: str, segments, lines):
    rec = segments.get(sid)
    print()
    print(bold(f"═══ segment {sid} ═══"))
    print()
    if not rec:
        print(red(f"  no segment matching {sid} in journal window"))
        return
    print(f"  text:        {rec.text}")
    print(f"  lang:        ASR={rec.asr_lang or '??'} → final={rec.final_lang or '??'}"
          + (yellow("  (REMAPPED)") if rec.remapped else ""))
    print("  translate:   "
          + (red("SKIPPED") if rec.translate_skip
             else (f"{rec.translate_ms}ms" if rec.translate_ms is not None else dim("(none)"))))
    print("  furigana:    "
          + (dim("(not run)") if rec.furigana_chars is None
             else (dim("no kanji") if rec.furigana_chars == 0
                   else green(f"{rec.furigana_chars} chars"))))
    print("  speaker:     "
          + (f"cluster {rec.speaker_cluster} (catch-up @ {rec.speaker_age:.1f}s)"
             if rec.speaker_cluster is not None else dim("(none)")))
    print()
    print(dim("  raw journal lines:"))
    for line in lines:
        if sid in line:
            print(f"    {line}")
    print()


# ── Entrypoint ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="On-demand meeting-scribe troubleshooting snapshot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--seg", help="Trace a single segment_id")
    ap.add_argument("--meeting", help="Filter to one meeting_id")
    ap.add_argument("--tail", type=int, default=0, help="Show only last N finals")
    ap.add_argument("--errors", action="store_true", help="Errors only (no transcript table)")
    ap.add_argument("--since", default="90 seconds ago", help="Journal lookback (default: 90 seconds ago)")
    ap.add_argument(
        "--listeners",
        action="store_true",
        help="Show live audio-listener telemetry pushed by browsers (admin / guest / reader)",
    )
    args = ap.parse_args()

    if shutil.which("journalctl") is None:
        print(red("journalctl not on PATH"), file=sys.stderr)
        return 1

    lines = fetch_journal(args.since)
    segments, errors, stats = parse_journal(lines)
    api_status = fetch_api_status()
    pid, state = fetch_systemd_main_pid()

    if args.seg:
        render_segment_trace(args.seg, segments, lines)
        return 0

    if args.listeners:
        render_listener_diag()
        return 0

    render_snapshot(args, segments, errors, stats, api_status, pid, state)
    return 0


def fetch_listener_diag() -> list[dict]:
    """Pull POST /api/diag/listeners → list of live client states."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for host in ADMIN_HOST_CANDIDATES:
        try:
            with urllib.request.urlopen(f"{host}/api/diag/listeners", timeout=2, context=ctx) as r:
                return (json.loads(r.read().decode()) or {}).get("listeners", []) or []
        except Exception:
            continue
    return []


def render_listener_diag() -> None:
    """Pretty-print every audio-listener client's live state.

    The data is pushed by `guest.html` (and any other client wired into
    `_diagPushToServer`) every 2 s. Lets us debug "I tapped Listen but no
    audio" without asking the user to read text off a phone screen.
    """
    listeners = fetch_listener_diag()
    print()
    print(bold("═══ audio-listener clients ═══"), dim(datetime.now().strftime("%H:%M:%S %Z")))
    print()
    if not listeners:
        print(dim("  (no clients have pushed telemetry yet — tap Listen on a page running"))
        print(dim("   the latest guest.html / scribe-app.js with the diag poster wired in)"))
        return
    for li in listeners:
        cid = li.get("client_id") or "?"
        page = li.get("page", "?")
        ua = li.get("ua_short", "?")
        peer = li.get("_peer", "?")
        age = li.get("age_s", 0)
        ws_state = li.get("ws_state", "NULL")
        ctx_state = li.get("ctx_state", "null")
        ctx_rate = li.get("ctx_rate", 0)
        primed = li.get("primed", False)
        queue = li.get("queue", 0)
        bytes_in = li.get("bytes_in", 0)
        blobs_in = li.get("blobs_in", 0)
        decoded = li.get("decoded", 0)
        decode_err = li.get("decode_err", 0)
        played = li.get("played", 0)
        last_err = li.get("last_err", "") or ""

        ws_chip = green(ws_state) if ws_state == "OPEN" else (yellow(ws_state) if ws_state in ("CONNECTING",) else red(ws_state))
        ctx_chip = green(ctx_state) if ctx_state == "running" else red(ctx_state)
        primed_chip = green("primed") if primed else red("not-primed")
        age_chip = dim(f"{age}s ago") if age < 5 else (yellow(f"{age}s ago") if age < 30 else red(f"{age}s STALE"))

        print(f"  client {cid}…  {bold(page)}  {ua}  {dim(peer)}  {age_chip}")
        print(f"    ws:        {ws_chip}")
        print(f"    audioCtx:  {ctx_chip} @ {ctx_rate}Hz  ·  {primed_chip}")
        print(f"    pipeline:  bytes={bytes_in}  blobs={blobs_in}  q={queue}")
        print(f"               decoded={decoded}  decode_err={decode_err}  played={played}")
        # Diagnose-by-pattern: tell the operator what each combo means.
        if blobs_in == 0:
            print(red("    DIAG:      no bytes have arrived — server is not delivering to this client"))
        elif decoded == 0 and blobs_in > 0:
            print(red("    DIAG:      bytes arrive but decode never succeeds — bad WAV header / format mismatch"))
        elif decoded > 0 and played == 0:
            print(red("    DIAG:      decoded buffers but onended never fires — audio is muted at the device"))
        elif played > 0 and ctx_state != "running":
            print(yellow("    DIAG:      played count is climbing but context is suspended — check device output"))
        elif played > 0 and played < decoded - 2:
            print(yellow(f"    DIAG:      decoded {decoded} but only played {played} — playback can't keep up"))
        else:
            print(green("    DIAG:      pipeline is healthy"))
        if last_err:
            print(red(f"    ERR:       {last_err}"))
    print()


if __name__ == "__main__":
    sys.exit(main())
