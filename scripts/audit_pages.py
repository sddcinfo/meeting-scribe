#!/usr/bin/env python3
"""Audit the deployed GitHub Pages site for runtime breakage.

Loads the public pages in Chromium and reports:

  * console errors / page errors (uncaught JS, deprecation warnings)
  * failed network requests (broken assets, 404s, blocked-by-CSP)
  * pages that render blank because a script throws at parse time

Use this against the live ``https://sddcinfo.github.io/meeting-scribe/``
deployment when something feels off in a browser; the report is JSON-on-
stderr-friendly so it composes with grep / jq.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urljoin

DEFAULT_BASE = "https://sddcinfo.github.io/meeting-scribe/"
# Pages.yml ships ONLY the public marketing surface (how-it-works.html
# + a redirect index.html). The admin app, captive portal, guest, and
# reader pages reference `/static/...` absolute paths that only resolve
# when served from the meeting-scribe FastAPI server (mounted at
# `/static`). Auditing them against GitHub Pages produces guaranteed
# 404s. Keep this list in sync with `.github/workflows/pages.yml`.
DEFAULT_PAGES = ["", "how-it-works.html"]


def audit(base: str, paths: list[str]) -> int:
    from playwright.sync_api import sync_playwright

    findings: dict[str, dict] = {}
    rc = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for path in paths:
                url = urljoin(base, path)
                ctx = browser.new_context()
                page = ctx.new_page()
                console_errors: list[str] = []
                page_errors: list[str] = []
                network_fails: list[dict] = []

                page.on(
                    "console",
                    lambda m: (
                        console_errors.append(f"{m.type}: {m.text}")
                        if m.type in ("error", "warning")
                        else None
                    ),
                )
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))

                def _on_response(resp):
                    if resp.status >= 400:
                        network_fails.append({"url": resp.url, "status": resp.status})

                page.on("response", _on_response)

                try:
                    page.goto(url, wait_until="networkidle", timeout=15000)
                    title = page.title()
                except Exception as exc:
                    findings[path or "(root)"] = {
                        "url": url,
                        "load_error": str(exc),
                    }
                    rc = 1
                    continue
                finally:
                    pass

                slot = path or "(root)"
                findings[slot] = {
                    "url": url,
                    "title": title,
                    "console": console_errors,
                    "page_errors": page_errors,
                    "network_fails": network_fails,
                }
                if console_errors or page_errors or network_fails:
                    rc = 1
                ctx.close()
        finally:
            browser.close()

    json.dump(findings, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("paths", nargs="*", default=DEFAULT_PAGES)
    args = parser.parse_args()
    return audit(args.base, args.paths)


if __name__ == "__main__":
    sys.exit(main())
