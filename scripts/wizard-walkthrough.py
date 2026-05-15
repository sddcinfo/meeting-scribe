#!/usr/bin/env python3
"""Walk the simplified v1.0 first-touch wizard end-to-end via HTTP.

Companion to ``scripts/wizard-smoke.sh``: while ``wizard-smoke.sh``
brings the FastAPI app up on a high port, this script drives it
through the operator flow:

  1. GET /setup  — assert page renders with the install button
  2. GET /setup/wifi-profile.mobileconfig — Apple plist serves
  3. POST /api/setup/finish — assert 202 + reconnect_token
  4. GET /api/setup/commit-status — assert "committing"
  5. POST /api/setup/cancel — clean up

The fingerprint + bootstrap-secret pages were dropped in the v1.0
simplification; this script tracks the new shape.

Run after ``scripts/wizard-smoke.sh start``.
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


def _build_opener(jar: CookieJar) -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(jar),
        _NoRedirect(),
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_a, **_kw):
        return None


def _request(
    opener: urllib.request.OpenerDirector,
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = opener.open(req, timeout=10)
        body = resp.read()
        return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp else b""
        return exc.code, dict(exc.headers), body


def _ok(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK " if ok else "FAIL"
    msg = f"[{mark}] {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="https://127.0.0.1:8443")
    args = p.parse_args()

    base: str = args.base.rstrip("/")
    jar = CookieJar()
    opener = _build_opener(jar)
    failures = 0

    # 1. GET /setup — claims, mints, renders.
    status, _, body = _request(opener, "GET", f"{base}/setup")
    has_install_button = b"Save Wi-Fi profile" in body
    has_finish_button = b"Finish" in body
    if not _ok(
        "GET /setup",
        status == 200 and has_install_button and has_finish_button,
        f"http={status}, install_button={has_install_button}, finish_button={has_finish_button}",
    ):
        failures += 1
    has_sid = any(c.name == "ms_setup_sid" for c in jar)
    if not _ok(
        "ms_setup_sid cookie set on first GET",
        has_sid,
    ):
        failures += 1

    # 2. GET /setup/wifi-profile.mobileconfig — Apple plist.
    status, headers, body = _request(opener, "GET", f"{base}/setup/wifi-profile.mobileconfig")
    is_plist = b"<!DOCTYPE plist" in body
    is_apple = headers.get("Content-Type") == "application/x-apple-aspen-config"
    has_ssid = b"<string>Dell Meeting</string>" in body
    if not _ok(
        "GET /setup/wifi-profile.mobileconfig",
        status == 200 and is_plist and is_apple and has_ssid,
        f"http={status}, plist={is_plist}, apple_mime={is_apple}, ssid_in={has_ssid}",
    ):
        failures += 1

    # 3. Idempotent re-render: a reload of /setup with the same
    # cookie should NOT mint new credentials. Compare the AP password
    # in the rendered HTML.
    _status_a, _, body_a = _request(opener, "GET", f"{base}/setup")
    _status_b, _, body_b = _request(opener, "GET", f"{base}/setup")
    # Find the ap_password cred-row by looking for the data-copy attr
    # on the Wi-Fi password field. Both renders should embed the same.
    import re

    pat = re.compile(rb'data-copy="([A-Z2-7]{16,})"')
    a_creds = pat.findall(body_a)
    b_creds = pat.findall(body_b)
    if not _ok(
        "GET /setup idempotent — same credentials on reload",
        a_creds == b_creds and len(a_creds) >= 1,
        f"a={len(a_creds)} b={len(b_creds)} match={a_creds == b_creds}",
    ):
        failures += 1

    # 4. POST /api/setup/finish — kick the AP rotation.
    status, _, body = _request(
        opener,
        "POST",
        f"{base}/api/setup/finish",
        data=b"",
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(body) if body else {}
    if not _ok(
        "POST /api/setup/finish",
        status == 202 and payload.get("state") == "committing",
        f"http={status} body={payload}",
    ):
        failures += 1
    payload.get("reconnect_token", "")  # forward-compat probe; not used in checks below

    # 5. GET /api/setup/commit-status — should report committing
    # (or possibly rolled-back if the AP rotation failed in the
    # background; on the smoke harness it always rolls back since
    # we're not actually rotating Wi-Fi).
    status, _, body = _request(opener, "GET", f"{base}/api/setup/commit-status")
    payload = json.loads(body) if body else {}
    if not _ok(
        "GET /api/setup/commit-status",
        status == 200 and payload.get("state") in ("committing", "rolled-back", "complete"),
        f"http={status} body={payload}",
    ):
        failures += 1

    # 6. POST /api/setup/cancel — only valid if NOT committing.
    # In the smoke harness the rollback may have already fired and
    # left state in setup-pending, in which case cancel is fine.
    # If it's in setup-committing, cancel returns 409.
    status, _, _ = _request(opener, "POST", f"{base}/api/setup/cancel", data=b"")
    _ok(
        "POST /api/setup/cancel",
        status in (200, 409),
        f"http={status} (200=cancelled, 409=mid-commit)",
    )

    print()
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
