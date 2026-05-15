#!/usr/bin/env python3
"""Render the live meeting-scribe pages in Playwright + screenshot them.

The CSS bytes match local but visual / layout issues only show up
once a real engine renders the document. Run this against any
meeting-scribe instance reachable on the network to capture
screenshots for design audit.

The admin password on a v1.0 demo appliance is deterministically
``DellMeetingAdmin<NNNN>`` where ``<NNNN>`` is the SSID's
trailing 4-digit appliance ID. **Do not hardcode a real instance
of that string in source control** — read the value at run-time
(env var or interactive prompt) so a checkout doesn't leak the
exact password of any real device.

    # Compute the SSID-derived admin credential at run time — never
    # hardcode it. Shell example::
    #
    #     PIN=$(meeting-scribe wifi status --ssid | grep -oE '[0-9]{4}$')
    #     CRED="DellMeetingAdmin$PIN"
    #
    # Then drive the script:
    #
    #     PYTHONPATH=src .venv/bin/python scripts/screenshot-auth-page.py \\
    #         --base https://10.42.0.1 \\
    #         --password-env MS_AUTH_CRED \\
    #         --out /tmp/auth-shots
    #
    # ``--password-env`` reads the value from the named env var so
    # the password literal never lands on a process command line
    # (``ps aux`` would otherwise expose it).

Outputs:
    <out>/auth-mobile.png       (390x844, DPR 3) — pre-auth /auth page
    <out>/auth-desktop.png      (1440x900, DPR 2) — pre-auth /auth page
    <out>/index-mobile.png      authenticated / (admin index)
    <out>/index-desktop.png     authenticated / (admin index)
    <out>/summary.json          computed styles for canonical elements
                                + console errors + visible-element list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

_VIEWPORTS = (
    ("mobile", {"width": 390, "height": 844, "device_scale_factor": 3}),
    ("desktop", {"width": 1440, "height": 900, "device_scale_factor": 2}),
)


def _shoot(page, base: str, page_kind: str, out: Path, label: str) -> dict[str, object]:
    """Navigate the page + screenshot it. Returns a summary describing
    what's covering the screen (overlay detection) + any console errors."""
    console: list[str] = []
    page.on("console", lambda msg: console.append(f"[{msg.type}] {msg.text}"))
    page_errors: list[str] = []
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))

    target = f"{base}/" if page_kind == "index" else f"{base}/{page_kind}"
    response = page.goto(target, wait_until="networkidle")
    page.wait_for_timeout(500)  # let any auto-running scripts settle
    shot = out / f"{page_kind}-{label}.png"
    page.screenshot(path=str(shot), full_page=True)

    # Visible-element audit: ask the engine which elements at the top
    # layer are covering the viewport. A modal/overlay covering the
    # whole pane will show up as a high-z-index full-viewport child.
    overlays = page.evaluate(
        """() => {
            const out = [];
            const w = window.innerWidth, h = window.innerHeight;
            const all = document.querySelectorAll('body *');
            for (const el of all) {
                const r = el.getBoundingClientRect();
                const cs = window.getComputedStyle(el);
                const covers_w = r.width >= w * 0.8;
                const covers_h = r.height >= h * 0.8;
                const positioned = ['fixed', 'absolute'].includes(cs.position);
                const visible = cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0';
                if (positioned && covers_w && covers_h && visible) {
                    out.push({
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        cls: el.className || null,
                        z: cs.zIndex,
                        display: cs.display,
                        position: cs.position,
                        inline_display: el.style.display || '',
                        bg: cs.backgroundColor,
                        rect: {x: r.x, y: r.y, w: r.width, h: r.height},
                    });
                }
            }
            return out;
        }"""
    )

    return {
        "url": response.url if response else target,
        "shot": str(shot),
        "viewport": page.viewport_size,
        "console": console,
        "page_errors": page_errors,
        "overlays": overlays,
        "doc_height": page.evaluate("document.documentElement.scrollHeight"),
        "title": page.title(),
    }


def _capture(base: str, password: str | None, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"base": base, "viewports": {}}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for label, viewport in _VIEWPORTS:
            ctx = browser.new_context(
                ignore_https_errors=True,
                viewport={"width": viewport["width"], "height": viewport["height"]},
                device_scale_factor=viewport["device_scale_factor"],
            )
            entries: dict[str, object] = {}

            # Pre-auth /auth screenshot.
            page = ctx.new_page()
            entries["auth"] = _shoot(page, base, "auth", out, label)
            page.close()

            # Authenticated / screenshot — sign in via the form so the
            # browser's cookie jar carries the resulting Set-Cookie
            # over to the next page navigation.
            if password:
                page2 = ctx.new_page()
                page2.goto(f"{base}/auth", wait_until="networkidle")
                page2.fill("input[name=password]", password)
                page2.click("button.secondary")  # "Sign in as admin"
                page2.wait_for_url(f"{base}/", timeout=5000)
                entries["index"] = _shoot(page2, base, "index", out, label)
                page2.close()

            summary["viewports"][label] = entries
            ctx.close()
        browser.close()

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    for label, vp in summary["viewports"].items():
        for kind, data in vp.items():
            overlays = data.get("overlays", [])
            errors = data.get("page_errors", [])
            print(f"  {label}/{kind}: {data['shot']}")
            print(f"    title={data['title']!r}  doc_height={data['doc_height']}px")
            if overlays:
                print("    OVERLAYS COVERING ≥80%:")
                for o in overlays[:5]:
                    print(f"      - {o['tag']}#{o['id']}.{o['cls']}  z={o['z']}  bg={o['bg']}")
            if errors:
                print(f"    PAGE ERRORS: {errors[:3]}")


def _capture_guest(base: str, pin: str, out: Path) -> None:
    """Sign in as a guest with ``pin`` and screenshot the resulting
    page so the operator can see exactly what a guest sees after
    typing their 4-digit code on the AP captive form."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for label, viewport in _VIEWPORTS:
            ctx = browser.new_context(
                ignore_https_errors=True,
                viewport={"width": viewport["width"], "height": viewport["height"]},
                device_scale_factor=viewport["device_scale_factor"],
            )
            page = ctx.new_page()
            page.goto(f"{base}/auth", wait_until="networkidle")
            page.fill("input[name=pin]", pin)
            page.click("button.primary")  # "Join as guest"
            page.wait_for_url(f"{base}/", timeout=5000)
            page.wait_for_timeout(800)
            shot = out / f"guest-{label}.png"
            page.screenshot(path=str(shot), full_page=True)
            print(f"  guest/{label}: {shot}  (title={page.title()!r})")
            ctx.close()
        browser.close()


def main() -> None:
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="https://10.42.0.1 or https://192.168.1.100")
    p.add_argument(
        "--password-env",
        default="MS_AUTH_CRED",
        help=(
            "Read the admin password from this env var (default: "
            "MS_AUTH_CRED). Reading from env keeps the literal off "
            "the command line so ``ps aux`` can't expose it."
        ),
    )
    p.add_argument(
        "--guest-pin-env",
        default="MS_GUEST_PIN",
        help="Read the 4-digit guest PIN from this env var.",
    )
    p.add_argument("--out", default="/tmp/auth-shots", type=Path)
    args = p.parse_args()
    password = os.environ.get(args.password_env) or None
    guest_pin = os.environ.get(args.guest_pin_env) or None
    _capture(args.base, password, args.out)
    if guest_pin:
        _capture_guest(args.base, guest_pin, args.out)


if __name__ == "__main__":
    main()
