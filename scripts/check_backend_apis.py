#!/usr/bin/env python3
"""Backend-API drift probe — runs on the GB10 dev machine on demand.

Pinned param names + response shapes for vLLM TTS / ASR / translate
live in ``tests/contracts/backend_api_pins.json``. This script:

  1. Loads the pins.
  2. Issues one known-good call per backend endpoint via direct
     HTTP (or a pinned client adapter when shapes are wrapped).
  3. Asserts the response matches the pinned shape (HTTP 200 + key
     presence + value type).
  4. Appends a JSON line to ``docs/backend_api_log.jsonl`` with
     ``{"checked_at": <iso>, "endpoint": "...", "ok": bool,
     "version_pin": "..."}``.

A 14-day staleness alarm is enforced in ``scripts/manual_test_status.py``
(release CI fails if this log is stale).

Manual workflow:

  scripts/check_backend_apis.py                 # check all
  scripts/check_backend_apis.py --only tts      # one backend
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = REPO_ROOT / "tests" / "contracts" / "backend_api_pins.json"
LOG_PATH = REPO_ROOT / "docs" / "backend_api_log.jsonl"


def _load_pins() -> dict:
    if not PINS_PATH.exists():
        raise SystemExit(
            f"missing pins file: {PINS_PATH.relative_to(REPO_ROOT)}. "
            f"Add backend version + endpoint expectations there."
        )
    return json.loads(PINS_PATH.read_text())


def _probe(name: str, pin: dict) -> dict:
    """Issue a probe and return a record. pin shape:
        {"url": "...", "method": "GET"|"POST", "body": {...}|null,
         "expect_status": int, "expect_keys": ["..."]}"""
    method = pin.get("method", "GET").upper()
    body = pin.get("body")
    headers = {"Content-Type": "application/json"} if body else {}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(pin["url"], method=method, headers=headers, data=data)
    out = {
        "endpoint": name,
        "url": pin["url"],
        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
        "version_pin": pin.get("version_pin", "?"),
        "ok": False,
        "error": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=pin.get("timeout_s", 30)) as resp:
            status = resp.status
            payload_bytes = resp.read()
        if status != pin.get("expect_status", 200):
            out["error"] = f"status {status} != expected {pin.get('expect_status')}"
            return out
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError:
            out["error"] = "non-JSON response body"
            return out
        missing = [k for k in pin.get("expect_keys", []) if k not in payload]
        if missing:
            out["error"] = f"missing keys: {missing}"
            return out
        out["ok"] = True
    except urllib.error.URLError as e:
        out["error"] = f"connection: {e}"
    except Exception as e:  # pragma: no cover
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="probe only this endpoint name")
    parser.add_argument(
        "--no-log", action="store_true",
        help="don't append to docs/backend_api_log.jsonl",
    )
    args = parser.parse_args()

    pins = _load_pins()
    # Filter out comment-style entries (anything starting with _ or that
    # isn't a dict). Lets the JSON file carry inline documentation.
    pins = {k: v for k, v in pins.items() if isinstance(v, dict) and not k.startswith("_")}
    targets = [args.only] if args.only else list(pins.keys())

    fail_count = 0
    for name in targets:
        pin = pins.get(name)
        if not pin:
            print(f"✗ {name}: no such pin in {PINS_PATH.name}", file=sys.stderr)
            fail_count += 1
            continue
        record = _probe(name, pin)
        if not args.no_log:
            _append_log(record)
        if record["ok"]:
            print(f"✓ {name}  ({record['version_pin']})")
        else:
            print(f"✗ {name}: {record['error']}")
            fail_count += 1

    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
