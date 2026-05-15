#!/usr/bin/env python3
"""Re-trigger the slide pipeline for one or more past meetings.

Re-uploads each meeting's saved ``source.pptx`` through the live admin
``/api/meetings/<id>/slides/upload`` endpoint, which kicks off the full
validate -> render -> extract -> translate -> reinsert pipeline as if a
new upload had arrived. Useful for catching past decks up on pipeline
changes that only fire at upload time — e.g. the post-express
``translated/original.pdf`` finalizer added 2026-05-12.

Authenticates via the deterministic local admin password, the same way
``cli/_common._api_request`` does. Server must be running on the AP IP.

Usage:
    python3 scripts/reupload_slides.py <meeting_id> [<meeting_id> ...]
    python3 scripts/reupload_slides.py --poll-timeout 600 <meeting_id>
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

AP_IP = "10.42.0.1"
DEFAULT_PORT = 443


def _build_opener() -> tuple[urllib.request.OpenerDirector, CookieJar]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    jar = CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx),
        urllib.request.HTTPCookieProcessor(jar),
    )
    return opener, jar


def _authorize(opener: urllib.request.OpenerDirector, port: int) -> bool:
    from meeting_scribe.setup_state import _mint_admin_password

    password = os.environ.get("SCRIBE_ADMIN_PASSWORD") or _mint_admin_password()
    body = urllib.parse.urlencode({"password": password}).encode()
    req = urllib.request.Request(
        f"https://{AP_IP}:{port}/api/admin/authorize",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with opener.open(req, timeout=10) as resp:
        return resp.status in {200, 303}


def _resolve_source_pptx(meeting_id: str) -> Path:
    """Locate the active deck's source.pptx for a meeting. Picks the
    most-recently-modified deck directory if there are multiple."""
    slides_root = REPO / "meetings" / meeting_id / "slides"
    if not slides_root.is_dir():
        raise FileNotFoundError(f"{slides_root} not found")
    candidates = [d / "source.pptx" for d in slides_root.iterdir() if d.is_dir()]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(f"no source.pptx under {slides_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _upload(
    opener: urllib.request.OpenerDirector,
    port: int,
    meeting_id: str,
    pptx_path: Path,
) -> dict:
    """POST source.pptx as multipart/form-data and return the JSON reply."""
    import json
    import uuid

    boundary = f"----meetingscribereupload{uuid.uuid4().hex}"
    pptx_bytes = pptx_path.read_bytes()
    filename = pptx_path.name

    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: application/vnd.openxmlformats-officedocument.presentationml.presentation\r\n\r\n"
        ).encode()
        + pptx_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        f"https://{AP_IP}:{port}/api/meetings/{meeting_id}/slides/upload",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with opener.open(req, timeout=60) as resp:
        return json.loads(resp.read())


def _poll(
    opener: urllib.request.OpenerDirector,
    port: int,
    meeting_id: str,
    timeout_s: float,
) -> dict:
    """Poll /slides until stage=complete or timeout."""
    import json

    deadline = time.time() + timeout_s
    last_stage = None
    while time.time() < deadline:
        req = urllib.request.Request(
            f"https://{AP_IP}:{port}/api/meetings/{meeting_id}/slides",
            method="GET",
        )
        try:
            with opener.open(req, timeout=10) as resp:
                meta = json.loads(resp.read())
        except Exception as exc:
            print(f"  poll error: {exc}")
            time.sleep(2)
            continue
        stage = meta.get("stage")
        if stage != last_stage:
            print(f"  stage={stage}")
            last_stage = stage
        if stage == "complete":
            return meta
        time.sleep(2)
    raise TimeoutError(f"pipeline did not reach 'complete' within {timeout_s:.0f}s")


def _invalidate_deck_cache(meeting_id: str) -> int:
    """Strip ``content_hash`` from every existing deck's meta.json under
    this meeting so ``start_job`` can't short-circuit via the cache-hit
    path. Returns the number of decks invalidated. Non-destructive — the
    old decks stay on disk and remain visible in the deck switcher."""
    import json

    slides_root = REPO / "meetings" / meeting_id / "slides"
    if not slides_root.is_dir():
        return 0
    touched = 0
    for deck_dir in slides_root.iterdir():
        meta_path = deck_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            continue
        if "content_hash" not in m:
            continue
        m.pop("content_hash", None)
        meta_path.write_text(json.dumps(m, indent=2))
        touched += 1
    return touched


def reupload_one(
    meeting_id: str,
    port: int,
    poll_timeout: float,
    force: bool,
) -> int:
    print(f"\n== {meeting_id} ==")
    try:
        pptx_path = _resolve_source_pptx(meeting_id)
    except FileNotFoundError as exc:
        print(f"  FAIL: {exc}")
        return 1
    print(f"  source: {pptx_path.relative_to(REPO)} ({pptx_path.stat().st_size:,} bytes)")

    if force:
        n = _invalidate_deck_cache(meeting_id)
        print(f"  cache: invalidated content_hash on {n} existing deck(s)")

    opener, _ = _build_opener()
    if not _authorize(opener, port):
        print("  FAIL: admin authorize failed")
        return 1

    try:
        reply = _upload(opener, port, meeting_id, pptx_path)
    except urllib.error.HTTPError as exc:
        print(f"  FAIL: upload HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}")
        return 1
    deck_id = reply.get("deck_id", "")
    print(f"  uploaded → deck_id={deck_id[:16]}…")

    try:
        meta = _poll(opener, port, meeting_id, poll_timeout)
    except TimeoutError as exc:
        print(f"  FAIL: {exc}")
        return 1

    # Walk to the deck this upload produced (the upload reply has the
    # authoritative deck_id) and look for the post-express finalizer's
    # output. The express path returns stage=complete before the
    # finalizer task wraps up, so give it a moment.
    deck_dir = REPO / "meetings" / meeting_id / "slides" / deck_id
    translated_pdf = deck_dir / "translated" / "original.pdf"
    if translated_pdf.exists():
        print(f"  OK: translated.pdf produced ({translated_pdf.stat().st_size:,} bytes)")
        return 0
    for _ in range(60):  # up to 120s for the finalizer
        time.sleep(2)
        if translated_pdf.exists():
            print(
                f"  OK: translated.pdf produced ({translated_pdf.stat().st_size:,} bytes) "
                "via post-express finalizer"
            )
            return 0
    print(f"  WARN: stage=complete but {translated_pdf.relative_to(REPO)} missing")
    print(f"        stages={meta.get('stages')}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting_ids", nargs="+", help="One or more meeting UUIDs")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Admin HTTPS port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for stage=complete (default: 900)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Strip content_hash from existing decks before upload so the "
            "cache-hit short-circuit can't reuse a stale deck. Required when "
            "the meeting already has a deck — otherwise the upload no-ops."
        ),
    )
    args = parser.parse_args()

    failures = 0
    for mid in args.meeting_ids:
        if reupload_one(mid, args.port, args.poll_timeout, args.force) != 0:
            failures += 1

    if failures:
        print(f"\n{failures} meeting(s) failed.")
        return 1
    print("\nAll OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
