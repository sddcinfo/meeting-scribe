"""Headless popout smoke test against the live meeting-scribe.

Launches Chromium against ``https://127.0.0.1:8080/?popout=view`` (self-signed
cert accepted), waits for init to complete, and prints:

  · console messages (warnings/errors)
  · network requests that failed
  · the transcript slot's children count + empty-state
  · the transcript-grid childCount
  · the popout-layout preset + currently-visible panels
  · a snippet of each recent compact-block's text (proves rendering)

Use: mise run python scripts/popout_smoke.py   (or python -m …).
"""

from __future__ import annotations

import json
import sys
import sys as _sys

from playwright.sync_api import sync_playwright

# Default: plain live-mode popout. Pass a meeting id as argv[1] to
# exercise the "review-mode Live button" URL (?popout=view#meeting/<id>).
_BASE = "https://127.0.0.1:8080/?popout=view"
URL = f"{_BASE}#meeting/{_sys.argv[1]}" if len(_sys.argv) > 1 else _BASE


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(ignore_https_errors=True)
        page = ctx.new_page()

        console_logs: list[str] = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))

        failed_requests: list[str] = []
        page.on(
            "requestfailed",
            lambda req: failed_requests.append(f"{req.method} {req.url} — {req.failure}"),
        )
        bad_responses: list[str] = []

        def _on_response(resp):
            if resp.status >= 400:
                bad_responses.append(f"{resp.status} {resp.url}")

        page.on("response", _on_response)

        page.goto(URL, wait_until="domcontentloaded")
        # Give the init IIFE time to fetch status, ingest replay, wire the WS,
        # and let a couple of live events land.
        page.wait_for_timeout(5000)

        # Optional: switch preset before snapshotting so we can exercise
        # multi-panel trees. Pass --sidebyside to enable.
        if "--sidebyside" in _sys.argv:
            page.select_option("#popout-layout-picker", "sidebyside")
            page.wait_for_timeout(800)

        snapshot = page.evaluate(
            """() => ({
              bodyClasses: document.body.className,
              hasRenderer: !!window._gridRenderer,
              transcriptGridPresent: !!document.getElementById('transcript-grid'),
              transcriptGridChildren: document.getElementById('transcript-grid')?.children?.length ?? -1,
              transcriptParentTag: document.getElementById('transcript-grid')?.parentElement?.tagName,
              transcriptParentClass: document.getElementById('transcript-grid')?.parentElement?.className,
              layoutState: (typeof window._popoutLayoutState === 'function') ? window._popoutLayoutState() : null,
              slotPanels: [...document.querySelectorAll('.lyt-slot-leaf')].map(s => ({
                panel: s.dataset.panel,
                empty: s.dataset.empty,
                text: s.dataset.emptyText,
                children: s.children.length,
              })),
              firstBlock: (() => {
                const b = document.querySelector('#transcript-grid .compact-block');
                if (!b) return null;
                return {
                  speaker: b.querySelector('.compact-speaker')?.textContent,
                  colA: b.querySelector('.compact-col-a')?.textContent?.slice(0, 120),
                  colB: b.querySelector('.compact-col-b')?.textContent?.slice(0, 120),
                };
              })(),
            })"""
        )

        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        print("--- console ---")
        for line in console_logs[-40:]:
            print(line)
        if failed_requests:
            print("--- failed requests ---")
            for line in failed_requests:
                print(line)
        if bad_responses:
            print("--- 4xx/5xx responses ---")
            for line in bad_responses:
                print(line)

        browser.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
