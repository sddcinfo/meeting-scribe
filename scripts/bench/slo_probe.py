"""Continuous live-path SLO probe for the 2026-Q2 model-challenger bench.

Runs in a side process during a bench window and aborts the bench if
either of the production live paths regresses beyond a threshold:

* **Translation**: one short EN→JA request to ``localhost:8010``
  every ``--interval`` seconds; abort if TTFT exceeds
  ``1.5 × translation_baseline_ms`` (default 195 ms × 1.5 ≈ 290 ms,
  per ``reports/phase5/decision_gate_2026-04-18.md``).
* **ASR**: one synthetic 5-second silent PCM clip to
  ``localhost:8003`` every ``--interval`` seconds; abort if total_ms
  exceeds ``1.5 × asr_baseline_ms`` (default 1500 ms × 1.5 = 2250 ms;
  conservative — refine after first observation window).

On abort, the probe writes ``ABORT`` + the reason to its log file and
sends ``SIGTERM`` to its parent PID (the bench shell), so the bench
halts before the next harness call.

Usage::

    python3 scripts/bench/slo_probe.py \\
        --log /tmp/scribe-slo-probe.log \\
        --interval 10 &
    SLO_PID=$!
    trap 'kill $SLO_PID 2>/dev/null' EXIT
    # ... run bench ...
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TRANSLATION_URL = "http://localhost:8010/v1/chat/completions"
DEFAULT_ASR_URL = "http://localhost:8003/v1/chat/completions"
DEFAULT_LOG = Path("/tmp/scribe-slo-probe.log")


def _discover_model_id(base_url: str) -> str | None:
    """Hit ``/v1/models`` next to ``base_url`` and return the first id.

    vLLM rejects POST /v1/chat/completions with HTTP 404 unless the
    request body's ``model`` matches a registered id (it doesn't accept
    "auto" the way some other servers do).  We discover the id once at
    probe startup so the probe body matches what's loaded.
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(base_url)
    models_url = f"{parsed.scheme}://{parsed.netloc}/v1/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            data = json.loads(resp.read())
        for entry in data.get("data", []):
            mid = entry.get("id")
            if mid:
                return mid
    except urllib.error.URLError:
        pass
    return None

# Baselines from reports/phase5/decision_gate_2026-04-18.md.  Override
# at run time with the matching env vars when re-baselining.
TRANSLATION_BASELINE_MS = float(os.environ.get("SCRIBE_TRANSLATION_BASELINE_MS", "195"))
ASR_BASELINE_MS = float(os.environ.get("SCRIBE_ASR_BASELINE_MS", "1500"))
ABORT_MULTIPLIER = float(os.environ.get("SCRIBE_SLO_ABORT_MULTIPLIER", "1.5"))


def _translation_probe(url: str, model_id: str) -> tuple[float, str | None]:
    """Send one short EN→JA translation request, return (ttft_ms, error)."""
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You translate English to Japanese."},
            {"role": "user", "content": "The meeting starts in five minutes."},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return (time.monotonic() - t0) * 1000.0, None
    except urllib.error.URLError as exc:
        return -1.0, f"{type(exc).__name__}: {exc}"


def _asr_probe(url: str, model_id: str) -> tuple[float, str | None]:
    """Send one short silent ASR request, return (total_ms, error).

    Uses a tiny silent WAV (1 s / 16 kHz mono) base64-embedded so the
    probe doesn't depend on any fixture files.
    """
    import base64
    import struct
    import wave
    from io import BytesIO

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(struct.pack("<h", 0) * 16_000)
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Transcribe in the spoken language."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ],
            },
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return (time.monotonic() - t0) * 1000.0, None
    except urllib.error.URLError as exc:
        return -1.0, f"{type(exc).__name__}: {exc}"


def _abort(log: Path, reason: str, parent_pid: int) -> None:
    log.open("a").write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} ABORT {reason}\n")
    print(f"SLO ABORT: {reason}", file=sys.stderr)
    try:
        os.kill(parent_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise SystemExit(3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--translation-url", default=DEFAULT_TRANSLATION_URL)
    p.add_argument("--asr-url", default=DEFAULT_ASR_URL)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--interval", type=int, default=10)
    p.add_argument(
        "--parent-pid",
        type=int,
        default=os.getppid(),
        help="PID to SIGTERM on abort (default: current parent shell)",
    )
    args = p.parse_args(argv)

    translation_threshold = TRANSLATION_BASELINE_MS * ABORT_MULTIPLIER
    asr_threshold = ASR_BASELINE_MS * ABORT_MULTIPLIER

    # Discover model ids dynamically.  vLLM 404s POST /v1/chat/completions
    # unless the request body model field matches a registered id.
    translation_model = _discover_model_id(args.translation_url) or "auto"
    asr_model = _discover_model_id(args.asr_url) or "auto"

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.open("a").write(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} START "
        f"translation_threshold_ms={translation_threshold:.0f} "
        f"asr_threshold_ms={asr_threshold:.0f} "
        f"translation_model={translation_model} asr_model={asr_model}\n"
    )

    while True:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        t_ms, t_err = _translation_probe(args.translation_url, translation_model)
        a_ms, a_err = _asr_probe(args.asr_url, asr_model)
        line = (
            f"{ts} translation_ttft_ms={t_ms:.0f} err={t_err!r} "
            f"asr_total_ms={a_ms:.0f} err={a_err!r}"
        )
        args.log.open("a").write(line + "\n")
        if t_err is None and t_ms > translation_threshold:
            _abort(
                args.log,
                f"translation TTFT {t_ms:.0f}ms > {translation_threshold:.0f}ms",
                args.parent_pid,
            )
        if a_err is None and a_ms > asr_threshold:
            _abort(
                args.log,
                f"ASR total {a_ms:.0f}ms > {asr_threshold:.0f}ms",
                args.parent_pid,
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
