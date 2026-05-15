#!/usr/bin/env python3
"""Loaded-playback verification — reproduces the original 30s playback stall failure mode.

Fires sustained translation load + a long-meeting summary regen, then
times audio playback first-byte under that contention. Pass gate is
first_byte < 1.5 s under load.

**This script asserts the response is real audio.** A previous version
followed redirects and silently passed when the server returned the
HTML auth page (1453 bytes of `<!doctype html>...`). To prevent that
class of false-pass:
  * the script does NOT follow redirects (-L) — a 302 is a hard fail;
  * the response Content-Type must be ``audio/wav`` (anything else fails);
  * the response body must be > 1024 bytes (1 s of 16 kHz / 16-bit mono
    PCM is ~32 KB before the WAV header — anything smaller is degenerate).

Auth: pass an admin session cookie via ``--cookie`` (or the
``MEETING_SCRIBE_ADMIN_COOKIE`` env var, e.g. exported from a logged-in
browser's devtools). Without it, /api/meetings/.../audio 302s to /auth
and the script fails fast with "AUTH GATE" rather than pretending to
pass.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from summary_param_ab import _run_one

LONG_MEETING_ID = "3db4286e-7164-4550-8101-9e5db69cfecc"
SCRIBE_BASE = "https://127.0.0.1"
VLLM_URL = "http://localhost:8010"

REQUIRED_CONTENT_TYPE = "audio/wav"
MIN_BODY_BYTES = 1024
MAX_TTFB_SECONDS = 1.500


def measure_playback(meeting_id: str, cookie: str) -> tuple[float, str, int, str]:
    """Return (ttfb_seconds, http_code, body_size_bytes, content_type).

    Does NOT follow redirects: a 302 to /auth indicates an auth failure
    and must be reported as such, not silently followed to an HTML page.
    """
    url = f"{SCRIBE_BASE}/api/meetings/{meeting_id}/audio?start_ms=0&end_ms=1000"
    out = subprocess.run(
        [
            "curl",
            "-k",
            "-s",
            "-o",
            "/dev/null",
            "--cookie",
            f"ms_admin={cookie}",
            "-w",
            "%{time_starttransfer},%{http_code},%{size_download},%{content_type}\n",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ","
    parts = line.split(",", 3)
    while len(parts) < 4:
        parts.append("")
    return float(parts[0] or 0), parts[1], int(parts[2] or 0), parts[3]


def fire_load() -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "vllm_optimization.py"),
            "--url",
            VLLM_URL,
            "--label",
            "landing_load_check",
            "--concurrency",
            "8",
            "--requests",
            "240",
            "--output-dir",
            "/tmp",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


async def summary_regen(meeting_id: str) -> None:
    t0 = time.monotonic()
    r = await _run_one(
        VLLM_URL,
        REPO_ROOT / "meetings" / meeting_id,
        "loaded_check",
        enable_thinking=True,
        max_tokens=8192,
    )
    dt = time.monotonic() - t0
    print(f"  summary regen: elapsed={dt:.1f}s topics={r.num_topics} err={r.error}")


async def main(args: argparse.Namespace) -> int:
    cookie = args.cookie or os.environ.get("MEETING_SCRIBE_ADMIN_COOKIE", "")
    if not cookie:
        print(
            "FAIL: no auth cookie. Pass --cookie <ms_admin value> or set\n"
            "      MEETING_SCRIBE_ADMIN_COOKIE. The audio endpoint is auth-gated\n"
            "      and the script refuses to silently follow the /auth redirect.",
            file=sys.stderr,
        )
        return 2

    # Pre-flight: confirm auth works at all by hitting the meeting-list
    # endpoint. If this 302's, the cookie is bad — fail fast.
    print("[0] Pre-flight auth check ...")
    pre = subprocess.run(
        [
            "curl",
            "-k",
            "-s",
            "-o",
            "/dev/null",
            "--cookie",
            f"ms_admin={cookie}",
            "-w",
            "%{http_code}",
            f"{SCRIBE_BASE}/api/meetings",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if pre.stdout.strip() != "200":
        print(f"FAIL: auth pre-flight returned {pre.stdout.strip()} (expected 200)")
        return 2

    print("[1] Firing sustained translation load (concurrency=8, 240 reqs ~ 3min) ...")
    load_proc = fire_load()
    print(f"    load PID={load_proc.pid}")

    print(f"[2] Firing long-meeting summary regen ({args.meeting_id}) ...")
    summary_task = asyncio.create_task(summary_regen(args.meeting_id))

    print("[3] Letting load engage (5s) ...")
    await asyncio.sleep(5)

    print("[4] Playback first-byte under load (3 samples):")
    samples = []
    for i in range(1, 4):
        ttfb, code, size, ctype = measure_playback(args.meeting_id, cookie)
        samples.append((ttfb, code, size, ctype))
        print(f"  attempt={i} first_byte={ttfb:.3f}s code={code} ctype={ctype} size={size}")
        await asyncio.sleep(2)

    print("[5] Waiting for load + summary to finish ...")
    load_proc.wait()
    print("    load done")
    await summary_task
    print("    summary done")

    print("\n=== gate check ===")
    failures = []
    worst_ttfb = max(s[0] for s in samples)
    print(
        f"playback first-byte under load (max of 3): {worst_ttfb:.3f}s (gate < {MAX_TTFB_SECONDS}s)"
    )
    if worst_ttfb >= MAX_TTFB_SECONDS:
        failures.append(f"first-byte {worst_ttfb:.3f}s exceeds {MAX_TTFB_SECONDS}s gate")

    for i, (ttfb, code, size, ctype) in enumerate(samples, 1):
        if code != "200":
            failures.append(f"attempt {i} returned http {code}; 200 expected (302 = auth gate hit)")
        if not ctype.startswith(REQUIRED_CONTENT_TYPE):
            failures.append(
                f"attempt {i} content-type {ctype!r}; {REQUIRED_CONTENT_TYPE!r} expected — "
                f"a previous version of this script falsely passed when the server "
                f"redirected to an HTML auth page."
            )
        if size < MIN_BODY_BYTES:
            failures.append(
                f"attempt {i} body size {size}B; min {MIN_BODY_BYTES}B expected — "
                f"the WAV header alone is 44B so anything < 1KB is degenerate."
            )

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cookie",
        default="",
        help="Admin session cookie value (ms_admin=...). Falls back to "
        "MEETING_SCRIBE_ADMIN_COOKIE env var.",
    )
    p.add_argument(
        "--meeting-id",
        default=LONG_MEETING_ID,
        help=f"Meeting UUID to fetch audio for (default: {LONG_MEETING_ID}).",
    )
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_build_parser().parse_args())))
