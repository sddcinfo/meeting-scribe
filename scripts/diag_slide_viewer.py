#!/usr/bin/env python3
"""Diagnostic Playwright session for the slide viewer (Phase 1.2-followup).

Drives the admin UI on the live dev server, opens an existing meeting
that has a finished deck on disk, and captures:

* DOM geometry of every relevant ``.popout-slides`` element
  (clientWidth/Height, computed display, getBoundingClientRect).
* Computed styles of the two ``.sv-pane`` children.
* The ``body`` / ``.popout-slides`` class lists (catches a stray
  ``monolingual-slides`` class).
* The natural dimensions of ``#sv-orig-img`` and ``#sv-trans``.
* Screenshots of the page at 1920x1080 / 1024x768 / 390x844, plus
  an element-only screenshot of ``.popout-slides`` at each viewport.

Output goes to ``tests/fixtures/diag/slide-viewer/`` (gitignored). Run:

    python scripts/diag_slide_viewer.py [--meeting-id MID] [--password PW] [--base URL]

Admin password is read from ``$MEETING_SCRIBE_ADMIN_PASSWORD``
(no hard-coded default — the secret-scan hook rejects the on-disk
``DellMeetingAdmin<NNNN>`` pattern in source). Base defaults to
``https://192.168.1.168``; meeting is auto-discovered as the
first on-disk meeting with a complete deck if ``--meeting-id`` is
not passed.

This script is one-shot diagnosis, NOT a regression test — it doesn't
assert correctness, it dumps observations so the operator can decide
what's broken. The actual locking tests come once the bug is
understood.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "diag" / "slide-viewer"


VIEWPORTS = [
    ("desktop-1920x1080", 1920, 1080),
    ("tablet-1024x768", 1024, 768),
    ("mobile-390x844", 390, 844),
]


# DOM-inspection probe — runs inside the page via page.evaluate.
# Returns a JSON-friendly snapshot of every element that matters for
# the slide-viewer geometry. Single statement per the project's
# bash-guard convention; not run via ``python -c``.
INSPECT_JS = r"""
() => {
  const out = {};
  out.bodyClasses = document.body.className;
  out.url = location.href;

  // ── Admin slide bar (the in-app side-by-side thumbnail strip) ──
  const adminBar = document.getElementById('admin-slide-bar');
  out.adminBarPresent = !!adminBar;
  if (adminBar) {
    out.adminBarDisplay = getComputedStyle(adminBar).display;
    out.adminBarRect = (() => {
      const r = adminBar.getBoundingClientRect();
      return { x: r.x, y: r.y, w: r.width, h: r.height };
    })();
    const thumbsRow = adminBar.querySelector('.admin-slide-thumbs');
    if (thumbsRow) {
      out.adminThumbsRect = (() => {
        const r = thumbsRow.getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height };
      })();
      out.adminThumbsFlexDir = getComputedStyle(thumbsRow).flexDirection;
    }
    for (const [name, id] of [['orig', 'admin-slide-thumb'], ['trans', 'admin-slide-thumb-translated']]) {
      const img = document.getElementById(id);
      if (!img) { out[`admin_${name}_imgPresent`] = false; continue; }
      const wrap = img.closest('.admin-slide-thumb-wrap');
      out[`admin_${name}_imgPresent`] = true;
      out[`admin_${name}_src`] = img.getAttribute('src') || '';
      out[`admin_${name}_natural`] = `${img.naturalWidth}x${img.naturalHeight}`;
      out[`admin_${name}_imgDisplay`] = getComputedStyle(img).display;
      out[`admin_${name}_imgRect`] = (() => {
        const r = img.getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height };
      })();
      if (wrap) {
        out[`admin_${name}_wrapRect`] = (() => {
          const r = wrap.getBoundingClientRect();
          return { x: r.x, y: r.y, w: r.width, h: r.height };
        })();
      }
    }
    const lab = document.getElementById('admin-slide-label');
    out.adminLabelText = lab ? lab.textContent : null;
    const status = document.getElementById('admin-slide-thumb-trans-status');
    if (status) {
      out.adminTransStatusVisible = !status.classList.contains('is-hidden');
      out.adminTransStatusDisplay = getComputedStyle(status).display;
    }
  }

  // ── Popout-slides viewer (mounted only in popout layouts) ──
  const sv = document.querySelector('.popout-slides');
  out.popoutSlidesPresent = !!sv;
  if (!sv) return out;
  out.popoutSlidesClasses = sv.className;
  out.popoutSlidesDisplay = getComputedStyle(sv).display;
  out.popoutSlidesRect = sv.getBoundingClientRect().toJSON
    ? sv.getBoundingClientRect().toJSON()
    : (() => {
        const r = sv.getBoundingClientRect();
        return { x: r.x, y: r.y, w: r.width, h: r.height };
      })();

  const slides = sv.querySelector('.sv-slides');
  out.svSlidesPresent = !!slides;
  if (slides) {
    out.svSlidesClientW = slides.clientWidth;
    out.svSlidesClientH = slides.clientHeight;
    out.svSlidesInlineHeight = slides.style.height;
    out.svSlidesUserResized = slides.dataset.userResized || null;
    out.svSlidesComputedDisplay = getComputedStyle(slides).display;
    out.svSlidesFlexDirection = getComputedStyle(slides).flexDirection;
  }

  const origPane = sv.querySelector('#sv-orig-pane');
  const transPane = sv.querySelector('#sv-trans-pane');
  for (const [name, p] of [['orig', origPane], ['trans', transPane]]) {
    if (!p) { out[`${name}PanePresent`] = false; continue; }
    out[`${name}PanePresent`] = true;
    out[`${name}PaneRect`] = (() => {
      const r = p.getBoundingClientRect();
      return { x: r.x, y: r.y, w: r.width, h: r.height };
    })();
    out[`${name}PaneDisplay`] = getComputedStyle(p).display;
    out[`${name}PaneFlex`] = getComputedStyle(p).flex;
  }

  const orig = document.getElementById('sv-orig-img');
  const trans = document.getElementById('sv-trans');
  for (const [name, img] of [['orig', orig], ['trans', trans]]) {
    if (!img) { out[`${name}ImgPresent`] = false; continue; }
    out[`${name}ImgPresent`] = true;
    out[`${name}ImgSrc`] = img.src;
    out[`${name}ImgDisplay`] = getComputedStyle(img).display;
    out[`${name}ImgNatural`] = `${img.naturalWidth}x${img.naturalHeight}`;
    out[`${name}ImgRect`] = (() => {
      const r = img.getBoundingClientRect();
      return { x: r.x, y: r.y, w: r.width, h: r.height };
    })();
    out[`${name}ImgComplete`] = img.complete;
  }
  return out;
}
"""


def _find_default_meeting() -> str | None:
    """Pick the first meeting on disk with a complete deck."""
    meetings_dir = REPO_ROOT / "meetings"
    if not meetings_dir.is_dir():
        return None
    for d in sorted(meetings_dir.iterdir()):
        active_path = d / "slides" / "active_deck_id"
        if not active_path.is_file():
            continue
        deck_id = active_path.read_text().strip()
        if not deck_id:
            continue
        meta_path = d / "slides" / deck_id / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("stage") == "complete" and meta.get("total_slides", 0) > 0:
            return d.name
    return None


def _ensure_meeting_with_deck(base: str, password: str) -> tuple[str, str, dict]:
    """Make sure the live server has an active meeting with a finished
    deck. Returns ``(meeting_id, deck_id, cookies)``. If a meeting is
    already active, reuse it; if it has no deck, upload one. Polls
    until the deck stage is ``complete``.
    """
    import ssl as _ssl
    import time
    import urllib.request as _u
    from http.cookiejar import CookieJar
    from urllib.parse import urlencode  # noqa: F401  # kept for future use

    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    jar = CookieJar()
    opener = _u.build_opener(_u.HTTPSHandler(context=ctx), _u.HTTPCookieProcessor(jar))

    def _post_json(path, body):
        req = _u.Request(
            f"{base}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Origin": base, "Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read())

    def _get_json(path):
        req = _u.Request(f"{base}{path}", headers={"Origin": base})
        with opener.open(req, timeout=15) as r:
            return json.loads(r.read())

    print("[diag] signing in")
    rc, _ = _post_json("/api/admin/authorize", {"password": password})
    if rc != 200:
        raise RuntimeError(f"sign-in failed: {rc}")

    status = _get_json("/api/status")
    mid = status.get("meeting", {}).get("id")
    if not mid:
        print("[diag] no live meeting — starting one")
        rc, body = _post_json("/api/meeting/start", {"language_pair": ["en", "ja"]})
        if rc != 200:
            raise RuntimeError(f"start meeting failed: {rc} {body}")
        mid = body["meeting_id"]
    else:
        print(f"[diag] reusing live meeting {mid}")

    # Decks?
    decks_resp = _get_json(f"/api/meetings/{mid}/decks")
    decks = decks_resp.get("decks") or []
    if decks:
        deck_id = decks[0]["deck_id"]
        print(f"[diag] reusing existing deck {deck_id}")
    else:
        # Upload a fixture .pptx via multipart.
        print("[diag] no deck; uploading fixture .pptx")
        fixture = REPO_ROOT / "tests" / "fixtures" / "test_slides.pptx"
        if not fixture.is_file():
            raise RuntimeError(f"fixture .pptx missing: {fixture}")
        boundary = "diagdiagdiagdiagdiag"
        body = (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="diag.pptx"\r\n'
                "Content-Type: application/vnd.openxmlformats-officedocument.presentationml.presentation\r\n\r\n"
            ).encode()
            + fixture.read_bytes()
            + f"\r\n--{boundary}--\r\n".encode()
        )
        req = _u.Request(
            f"{base}/api/meetings/{mid}/slides/upload",
            data=body,
            headers={
                "Origin": base,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with opener.open(req, timeout=60) as r:
            up = json.loads(r.read())
        deck_id = up["deck_id"]
        print(f"[diag] upload accepted deck_id={deck_id} status={up.get('status')}")

        # Poll until stage=complete or timeout.
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            decks_resp = _get_json(f"/api/meetings/{mid}/decks")
            for d in decks_resp.get("decks") or []:
                if d["deck_id"] == deck_id:
                    stage = d.get("stage")
                    print(f"[diag]   deck stage={stage} progress={d.get('stages')}")
                    if stage == "complete":
                        deadline = -1  # break outer
                        break
            if deadline == -1:
                break
            time.sleep(2)
        if deadline != -1:
            raise RuntimeError("deck did not reach stage=complete within 240s")

    cookies = []
    for c in jar:
        cookies.append(
            {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
                "secure": bool(c.secure),
                "httpOnly": False,  # JS reads scribe_admin via fetch credentials
                "sameSite": "Strict",
            }
        )
    return mid, deck_id, cookies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meeting-id",
        default=None,
        help="If set, skip the live-meeting setup; use this id directly.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("MEETING_SCRIBE_ADMIN_PASSWORD"),
        help=(
            "Admin password. Defaults to $MEETING_SCRIBE_ADMIN_PASSWORD; "
            "no built-in default because the dev-box pattern trips the "
            "secret-scan hook."
        ),
    )
    parser.add_argument("--base", default="https://192.168.1.168")
    args = parser.parse_args()

    if not args.password:
        print(
            "error: admin password required — pass --password or set "
            "$MEETING_SCRIBE_ADMIN_PASSWORD",
            file=sys.stderr,
        )
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.meeting_id:
        meeting_id = args.meeting_id
        print(f"[diag] using --meeting-id={meeting_id} (no upload)")
        cookies = []
    else:
        meeting_id, _deck_id, cookies = _ensure_meeting_with_deck(args.base, args.password)

    print(f"[diag] meeting_id={meeting_id}")
    print(f"[diag] base={args.base}")
    print(f"[diag] output dir={OUT_DIR}")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for label, w, h in VIEWPORTS:
                ctx = browser.new_context(
                    viewport={"width": w, "height": h},
                    ignore_https_errors=True,
                    storage_state=None,
                )
                # Inject the admin cookie we obtained outside the
                # browser context so the page loads authenticated.
                if cookies:
                    norm_cookies = [
                        {**c, "url": args.base} if "domain" not in c else c for c in cookies
                    ]
                    ctx.add_cookies(norm_cookies)
                else:
                    # Sign in via the page's request context for the
                    # explicit-meeting-id branch.
                    resp = ctx.request.post(
                        f"{args.base}/api/admin/authorize",
                        headers={"Origin": args.base, "Content-Type": "application/json"},
                        data=json.dumps({"password": args.password}),
                    )
                    if not resp.ok:
                        print(
                            f"[diag] {label}: sign-in failed: {resp.status} {resp.text()}",
                            file=sys.stderr,
                        )
                        ctx.close()
                        continue

                page = ctx.new_page()
                page.set_default_timeout(20_000)

                # Capture browser console + page errors. Without this
                # the diag flies blind past every TypeError / 404 the
                # user actually sees in DevTools. Buffers per-viewport
                # so we can correlate errors with layout state.
                console_events: list[dict] = []
                page_errors: list[dict] = []

                def _on_console(msg) -> None:
                    try:
                        loc = msg.location or {}
                        console_events.append(
                            {
                                "type": msg.type,
                                "text": msg.text,
                                "url": loc.get("url"),
                                "line": loc.get("lineNumber"),
                                "col": loc.get("columnNumber"),
                            }
                        )
                    except Exception:
                        pass  # malformed console message — skip rather than crash the diag run

                def _on_page_error(err) -> None:
                    page_errors.append({"message": str(err)})

                def _on_request_failed(req) -> None:
                    # Only surface non-trivial failures; skipping
                    # routine aborted requests on navigation.
                    fail = req.failure
                    if not fail:
                        return
                    if fail in ("net::ERR_ABORTED",):
                        return
                    console_events.append(
                        {
                            "type": "request-failed",
                            "text": f"{req.method} {req.url}: {fail}",
                        }
                    )

                page.on("console", _on_console)
                page.on("pageerror", _on_page_error)
                page.on("requestfailed", _on_request_failed)

                page.goto(f"{args.base}/", wait_until="domcontentloaded")

                # Once the page has loaded, install a structured capture
                # of CSP violations: the browser's plain console message
                # elides the blocked URI, but the
                # ``securitypolicyviolation`` event carries the full
                # report. Stash each event onto ``window.__cspViolations``
                # so we can read them back via ``page.evaluate`` after
                # the wait below.
                page.evaluate(
                    """
                    () => {
                      window.__cspViolations = window.__cspViolations || [];
                      if (!window.__cspViolationsHooked) {
                        window.__cspViolationsHooked = true;
                        window.addEventListener('securitypolicyviolation', (ev) => {
                          window.__cspViolations.push({
                            blockedURI: ev.blockedURI,
                            violatedDirective: ev.violatedDirective,
                            effectiveDirective: ev.effectiveDirective,
                            sourceFile: ev.sourceFile,
                            lineNumber: ev.lineNumber,
                            columnNumber: ev.columnNumber,
                            sample: (ev.sample || '').slice(0, 500),
                          });
                        });
                      }
                    }
                    """
                )

                # Two slide UIs to wait for:
                #   * Admin bar (#admin-slide-thumb) — always present in
                #     the admin SPA when there's a deck (single-image
                #     thumbnail strip).
                #   * Popout viewer (#sv-orig-img) — only mounted inside
                #     popout-layout panels that include a slides panel.
                # Wait for whichever appears first; if neither, capture
                # the page state for diagnosis anyway.
                try:
                    page.wait_for_function(
                        "() => { const a = document.getElementById('admin-slide-thumb');"
                        " const o = document.getElementById('sv-orig-img');"
                        " return (a && a.complete && a.naturalWidth > 0)"
                        " || (o && o.complete && o.naturalWidth > 0); }",
                        timeout=25_000,
                    )
                except Exception as e:
                    print(f"[diag] {label}: no slide UI loaded an image: {e}", file=sys.stderr)
                # Give layout heuristics a beat to settle.
                page.wait_for_timeout(1500)

                snapshot = page.evaluate(INSPECT_JS)
                snap_path = OUT_DIR / f"{label}.json"
                snap_path.write_text(json.dumps(snapshot, indent=2) + "\n")
                print(f"[diag] {label}: wrote DOM snapshot to {snap_path}")

                page_png = OUT_DIR / f"{label}.full.png"
                page.screenshot(path=str(page_png), full_page=True)

                sv = page.query_selector(".popout-slides")
                if sv:
                    el_png = OUT_DIR / f"{label}.viewer.png"
                    sv.screenshot(path=str(el_png))

                # Also capture the admin slide bar so the side-by-side
                # change is visually verifiable without scrolling around
                # the full-page screenshot. Wrap in try/except — if
                # the bar is hidden or thrashing, skip rather than
                # blow up the whole sweep so we still get tablet +
                # mobile snapshots.
                admin_bar = page.query_selector("#admin-slide-bar")
                if admin_bar and snapshot.get("adminBarDisplay") not in (None, "none"):
                    try:
                        bar_png = OUT_DIR / f"{label}.admin-bar.png"
                        admin_bar.scroll_into_view_if_needed(timeout=2_000)
                        admin_bar.screenshot(path=str(bar_png), timeout=5_000)
                    except Exception as e:
                        print(
                            f"[diag] {label}: admin-bar screenshot skipped: "
                            f"{type(e).__name__}: {str(e)[:120]}"
                        )

                # Drain the structured CSP-violation capture installed
                # above. These reports include ``blockedURI`` which the
                # plain console message elides — vital for tracking
                # down "where is fonts.gstatic.com being loaded from"
                # without trial-and-error grepping.
                try:
                    csp_reports = page.evaluate("() => window.__cspViolations || []")
                except Exception:
                    csp_reports = []

                # Persist console + error + CSP captures alongside the
                # DOM snapshot so a failed run produces a self-contained
                # bundle for triage.
                logs_path = OUT_DIR / f"{label}.console.json"
                logs_path.write_text(
                    json.dumps(
                        {
                            "console": console_events,
                            "page_errors": page_errors,
                            "csp_violations": csp_reports,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                err_count = sum(1 for e in console_events if e.get("type") == "error")
                req_fail_count = sum(1 for e in console_events if e.get("type") == "request-failed")
                print(
                    f"[diag] {label}: console events={len(console_events)} "
                    f"errors={err_count} request-failures={req_fail_count} "
                    f"page-errors={len(page_errors)} → {logs_path}"
                )
                if page_errors:
                    for pe in page_errors[:5]:
                        print(f"[diag]   pageerror: {pe['message'][:200]}")

                ctx.close()
        finally:
            browser.close()

    print(f"[diag] done. Outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
