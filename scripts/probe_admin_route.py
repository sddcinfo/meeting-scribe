#!/usr/bin/env python3
"""End-to-end probe: mint an admin cookie via the live signer and
hit a target /api/admin/* path. Returns the actual HTTP status +
first 200 bytes of the response body so route-registration claims
can be verified without guessing.

Usage:
    PYTHONPATH=src python3 scripts/probe_admin_route.py /api/admin/bt/status
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Admin path, e.g. /api/admin/bt/status")
    parser.add_argument("--base", default="https://127.0.0.1", help="Origin")
    parser.add_argument("--method", default="GET")
    parser.add_argument("--field", help="Comma-separated keys to extract from JSON")
    args = parser.parse_args()

    # Mint a cookie with the same secret the live server uses. The signer
    # state lives in meeting_scribe.runtime.state — we have to import the
    # server module so the signer is initialised.
    import meeting_scribe.server  # noqa: F401  initialises state._terminal_cookie_signer
    from meeting_scribe.runtime import state

    signer = getattr(state, "_terminal_cookie_signer", None)
    if signer is None:
        print("FAIL: signer not initialised — server module didn't run setup")
        return 2
    cookie = signer.issue()

    url = args.base.rstrip("/") + args.path
    req = urllib.request.Request(url, method=args.method)
    req.add_header("Cookie", f"scribe_admin={cookie}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            body_full = resp.read()
            print(f"status: {resp.status}")
            print(f"content-type: {resp.headers.get('content-type', '-')}")
            print(f"size: {len(body_full)}")
            try:
                import json as _json

                parsed = _json.loads(body_full)
                if args.field:
                    keys = args.field.split(",")
                    out = {k: parsed.get(k) for k in keys}
                    print(_json.dumps(out, indent=2))
                else:
                    print(_json.dumps(parsed, indent=2))
            except Exception:
                print(f"body[:500]: {body_full[:500]!r}")
    except urllib.error.HTTPError as e:
        body = e.read()[:300] if hasattr(e, "read") else b""
        print(f"status: {e.code}")
        print(f"body[:300]: {body!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
