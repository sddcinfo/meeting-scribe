"""HTTP captive sub-app — port-80 sign-in surface for the captive gateway.

Port 80 hosts:

* OS captive-portal probes (Apple `/hotspot-detect.html`, Android
  `/generate_204`, Windows `/connecttest.txt`, Firefox `/success.txt`,
  RFC 8910 `/api/captive`) re-registered from
  `hotspot/captive_portal.py` so the captive-ack state machine stays
  the single source of truth.
* `GET /` → inline sign-in page. Guest PIN form posts back to
  `POST /captive/guest-pin` on port 80 (works inside the iOS/Android
  CNA sandbox); admin sign-in is a vanilla `<a target="_blank">` link
  to the canonical HTTPS origin (the admin password is too sensitive
  to accept over HTTP — the user clicks through the cert warning on
  the canonical origin instead).
* `POST /captive/guest-pin` → validate the 4-digit PIN against the
  same HMAC the canonical HTTPS bridge uses, drop the caller IP into
  the `ms-allowed-guests` ipset on success, render an inline success
  page that triggers `/hotspot-detect.html` so the OS CNA dismisses.
* `GET/HEAD /{path:path}` → 308 to the canonical HTTPS origin so
  operator-typed paths still hand off correctly.
"""

from __future__ import annotations

import logging
import os

import fastapi
from fastapi import FastAPI
from fastapi.responses import Response

from meeting_scribe.hotspot.captive_portal import (
    captive_204,
    captive_apple,
    captive_firefox,
    captive_ncsi,
    captive_rfc8910,
    captive_windows,
)

logger = logging.getLogger(__name__)


def _canonical_https_origin() -> str:
    """Resolve the canonical HTTPS origin for 308 redirects.

    Defaults to ``https://10.42.0.1`` for the v1.0 appliance. Honors
    ``SCRIBE_CANONICAL_HOST`` so dev deployments and tests can point
    elsewhere without rewriting the binary.
    """
    host = os.environ.get("SCRIBE_CANONICAL_HOST", "").strip() or "10.42.0.1"
    return f"https://{host}"


def _mdns_https_origin() -> str:
    """Resolve the HTTPS origin used for in-app redirects from captive.

    Returns ``https://10.42.0.1`` by default. The IP is in the cert
    SAN list so the hostname matches; only the trust chain is missing
    (self-signed) which is an unavoidable one-time warning on iOS for
    any private appliance.

    Why not ``https://meeting-<pin>.local``? iOS treats ``.local``
    names as mDNS-only (RFC 6762 strictness) and skips unicast DNS
    for them. mDNS multicast over WiFi is unreliable on consumer
    hardware — when it fails Safari hangs on the redirect with no
    feedback, which trapped the user during the GB10 phone test
    2026-05-13. Using the IP origin always resolves and matches the
    cert SAN.
    """
    return _canonical_https_origin()


def build_captive_http_app() -> FastAPI:
    """Construct the strict HTTP sub-app — call once at server startup.

    Returns a fresh FastAPI app distinct from the main TLS app. Wire it
    onto the port-80 listener via uvicorn; the main app stays on the
    port-443 listener.
    """
    app = FastAPI(title="meeting-scribe captive (HTTP)")

    # Re-register the captive-portal probe handlers — they own the
    # captive-ack state machine and must stay the single source of
    # truth across both the HTTP sub-app and any future variant.
    app.add_api_route(
        "/hotspot-detect.html",
        captive_apple,
        methods=["GET", "HEAD"],
        response_model=None,
    )
    for path in ("/generate_204", "/gen_204", "/canonical.html", "/redirect"):
        app.add_api_route(path, captive_204, methods=["GET", "HEAD"], response_model=None)
    app.add_api_route(
        "/connecttest.txt",
        captive_windows,
        methods=["GET", "HEAD"],
        response_model=None,
    )
    app.add_api_route("/ncsi.txt", captive_ncsi, methods=["GET", "HEAD"], response_model=None)
    app.add_api_route(
        "/success.txt",
        captive_firefox,
        methods=["GET", "HEAD"],
        response_model=None,
    )
    app.add_api_route(
        "/api/captive",
        captive_rfc8910,
        methods=["GET", "HEAD"],
        response_model=None,
    )

    @app.get("/", response_model=None)
    @app.head("/", response_model=None)
    async def root_signin(request: fastapi.Request) -> Response:
        """Captive sign-in landing on port 80.

        Behavior by caller identity:

        * **Unauthorized** (IP not in either ipset): render the
          sign-in page with the guest PIN form + admin sign-in link.
          Calls ``_captive_ack(request)`` so iOS' next captive probe
          gets the Success body and the **Done button appears in the
          CNA** (per legacy learning, commit 91920dc 2026-05-05).
          The Done button does NOT auto-dismiss the CNA — the user
          can still interact with the PIN form. It just lets them
          finish the captive flow once they've signed in, instead of
          iOS trapping them in a re-probe loop.
        * **Authorized guest** (IP in ``ms-allowed-guests``): 302 to
          the meeting view at ``https://10.42.0.1/`` so any HTTP URL
          a guest types (``http://apple.com``, ``http://example.com``,
          …) funnels onto the meeting hub after the one-time cert
          click-through. HTTPS hostname typing still hits the FORWARD
          REJECT (Connection Refused fast) — that's by design.
        * **Authorized admin** (IP in ``ms-allowed-admins``): pass
          through to the sign-in surface; admins don't go through the
          captive REDIRECT in the first place (admin ipset excludes
          them) so they shouldn't reach here unless they typed our
          IP directly.
        """
        # Lazy import — avoids hard dependency for unit tests that
        # don't exercise the ipset path.
        try:
            from meeting_scribe.server_support import firewall_allowlist

            client = getattr(request, "client", None)
            ip = client.host if client is not None else ""
            if ip and firewall_allowlist.is_guest(ip):
                return Response(
                    status_code=302,
                    headers={
                        "Location": f"{_mdns_https_origin()}/",
                        "Cache-Control": "no-store, max-age=0",
                    },
                )
        except Exception:
            logger.exception("captive: authed-guest redirect lookup failed")

        # Unauthorized: ack so iOS enables the Done button (legacy
        # pattern from 91920dc — without this the CNA re-probe loop
        # never breaks). Doesn't auto-dismiss; user still sees the
        # sign-in form and interacts with it normally.
        from meeting_scribe.server_support.captive_ack import _captive_ack

        _captive_ack(request)
        canonical = _canonical_https_origin()
        body = _CAPTIVE_SIGNIN_HTML.replace("{{CANONICAL}}", canonical)
        return Response(
            content=body,
            status_code=200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Cache-Control": "no-store, max-age=0",
            },
        )

    @app.post("/captive/guest-pin", response_model=None)
    async def post_captive_guest_pin(request: fastapi.Request) -> Response:
        """Validate the 4-digit guest PIN inside the captive sub-app.

        Mirrors the canonical ``POST /api/auth/guest-pin`` verifier
        (same HMAC, same ``setup_state.verify_guest_pin``) but adapted
        for the port-80 captive sheet context:

        * No cookie is set — iOS/Android CNAs isolate cookies from the
          real browser, so a cookie set here cannot carry over. The
          ``ms-allowed-guests`` ipset entry is the durable artifact.
        * Failure re-renders the sign-in page with an inline error.
        * Success renders an inline "you're online" page that probes
          ``/hotspot-detect.html`` after a short delay so the CNA
          dismisses with the platform success indicator.
        """
        from meeting_scribe import setup_state
        from meeting_scribe.routes.guest_auth import _admin_secret_store
        from meeting_scribe.server_support import firewall_allowlist

        form = await request.form()
        candidate_raw = form.get("pin") or ""
        candidate = candidate_raw.strip() if isinstance(candidate_raw, str) else ""
        canonical = _canonical_https_origin()
        secret = _admin_secret_store().secret
        if not setup_state.verify_guest_pin(candidate, admin_secret=secret):
            body = _CAPTIVE_SIGNIN_HTML.replace("{{CANONICAL}}", canonical).replace(
                "<!--GUEST_ERR_SLOT-->",
                '<p class="error" role="alert">Incorrect PIN — try again.</p>',
            )
            return Response(
                content=body,
                status_code=401,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                    "Cache-Control": "no-store, max-age=0",
                },
            )

        client = getattr(request, "client", None)
        ip = client.host if client is not None else ""
        if ip:
            try:
                await firewall_allowlist.add_guest(ip)
            except Exception:
                logger.exception("captive: add_guest hook failed for %s", ip)

        # Send the guest straight into the live meeting view by
        # redirecting to the cert-matching mDNS origin. The HTTPS
        # ``hotspot_guard`` will see the IP in ``ms-allowed-guests`` and
        # mint a fresh guest cookie inline (Phase H captive-gateway,
        # 2026-05-13) instead of bouncing the user back to ``/auth``.
        mdns_origin = _mdns_https_origin()
        body = _CAPTIVE_GUEST_SUCCESS_HTML.replace("{{CANONICAL}}", canonical).replace(
            "{{MDNS_ORIGIN}}", mdns_origin
        )
        return Response(
            content=body,
            status_code=200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Cache-Control": "no-store, max-age=0",
            },
        )

    @app.get("/{path:path}", response_model=None)
    @app.head("/{path:path}", response_model=None)
    async def catch_all(path: str, request: fastapi.Request) -> Response:
        """Funnel arbitrary AP-client HTTP traffic into the meeting view.

        Two regimes:

        * **Authorized guest/admin** (IP in either ipset): 302 to the
          meeting root (``https://10.42.0.1/``). When a guest types
          ``http://apple.com/news`` the PREROUTING REDIRECT rewrites
          the destination to us with path ``/news`` and Host
          ``apple.com``; we don't carry that path forward (it would
          404 on the canonical app), we just deposit the user at the
          meeting hub.
        * **Unauthorized**: 308 to the canonical HTTPS origin with
          the original path. Operator-typed
          ``http://10.42.0.1/whatever`` still works for admins who
          know the canonical surface.
        """
        try:
            from meeting_scribe.server_support import firewall_allowlist

            client = getattr(request, "client", None)
            ip = client.host if client is not None else ""
            if ip and (firewall_allowlist.is_guest(ip) or firewall_allowlist.is_admin(ip)):
                return Response(
                    status_code=302,
                    headers={
                        "Location": f"{_mdns_https_origin()}/",
                        "Cache-Control": "no-store, max-age=0",
                    },
                )
        except Exception:
            logger.exception("captive: authed catch-all redirect lookup failed")

        canonical = _canonical_https_origin()
        target_path = "/" + path if not path.startswith("/") else path
        if path == "":
            target_path = "/"
        query = request.url.query
        suffix = f"?{query}" if query else ""
        return Response(
            status_code=308,
            headers={"Location": f"{canonical}{target_path}{suffix}"},
        )

    return app


_CAPTIVE_BASE_CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-text-size-adjust: 100%; }
body {
  min-height: 100dvh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
  color: #1a1a1f;
  background: #f5f1e8;
  background-image:
    radial-gradient(ellipse 90% 60% at 12% -10%,
      rgba(200,155,60,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 60% 80% at 110% 110%,
      rgba(27,58,91,0.10) 0%, transparent 55%);
  background-attachment: fixed;
  -webkit-font-smoothing: antialiased;
  display: flex; flex-direction: column; align-items: center;
  padding: 3rem 1rem;
  padding-top: calc(3rem + env(safe-area-inset-top, 0));
  padding-bottom: calc(3rem + env(safe-area-inset-bottom, 0));
}
main {
  width: 100%; max-width: 30rem;
  background: #fff;
  border: 1px solid #d8cfb6;
  border-radius: 18px;
  padding: 2rem 2rem 2.5rem;
  box-shadow: 0 1px 0 rgba(26,26,31,0.04),
              0 24px 60px -36px rgba(27,58,91,0.30);
  position: relative; overflow: hidden;
}
main::before {
  content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
  background: linear-gradient(90deg,
    #1b3a5b 0%, #1b3a5b 22%,
    #c89b3c 22%, #c89b3c 28%,
    #1b3a5b 28%, #1b3a5b 100%);
}
@media (max-width: 480px) {
  main { padding: 1.5rem 1.25rem 2rem; border-radius: 12px; }
}
.brand {
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.22em;
  text-transform: uppercase; color: #1b3a5b;
  margin-bottom: 0.5rem;
}
h1 {
  font-family: 'Iowan Old Style', Charter, Cambria, 'Hoefler Text',
               Georgia, 'Times New Roman', serif;
  font-weight: 600;
  font-size: clamp(1.5rem, 5vw, 1.9rem);
  line-height: 1.1; letter-spacing: -0.012em;
  color: #1a1a1f; margin-bottom: 1.5rem;
}
h2 {
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: #1b3a5b;
  margin-top: 1.5rem; margin-bottom: 0.5rem;
}
.section + .section {
  margin-top: 0.5rem; padding-top: 1.25rem;
  border-top: 1px solid #e8dfc7;
}
p {
  font-size: 0.95rem; line-height: 1.55; color: #4a4a52;
  margin-bottom: 0.85rem;
}
p strong { color: #1a1a1f; font-weight: 600; }
label {
  display: block; font-size: 0.85rem; font-weight: 600;
  color: #1a1a1f; margin-bottom: 0.5rem;
}
input[type="text"] {
  width: 100%; font-size: 1.6rem; letter-spacing: 0.5rem;
  text-align: center; font-family: ui-monospace, 'SF Mono', Menlo,
              Consolas, monospace;
  padding: 0.75rem 0.5rem; border: 1px solid #d8cfb6;
  border-radius: 10px; background: #fdfbf4;
  margin-bottom: 0.85rem;
}
input[type="text"]:focus {
  outline: none; border-color: #1b3a5b;
  box-shadow: 0 0 0 3px rgba(27,58,91,0.18);
}
button.primary, a.button {
  display: inline-flex; align-items: center; justify-content: center;
  gap: 0.4rem; width: 100%;
  background: #1b3a5b; color: #fff;
  font-weight: 600; font-size: 1rem; text-decoration: none;
  padding: 0.85rem 1.2rem; border: 0; border-radius: 10px;
  box-shadow: 0 14px 28px -20px rgba(27,58,91,0.55);
  cursor: pointer;
}
button.primary:active, a.button:active { background: #2b5d8f; }
.error {
  font-size: 0.9rem; color: #8a2c1a;
  background: #fbe9e3; border-left: 3px solid #c0392b;
  padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0;
  margin-bottom: 0.85rem;
}
.hint {
  font-size: 0.85rem; line-height: 1.55; color: #4a4a52;
  background: #e9eef4; border-left: 3px solid #1b3a5b;
  padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
  margin-top: 0.75rem;
}
.hint code {
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.85em; background: #ede7d8;
  padding: 1px 5px; border-radius: 4px;
  border: 1px solid #e8dfc7;
}
"""


_CAPTIVE_SIGNIN_HTML = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f5f1e8">
<title>Meeting Scribe — Sign in</title>
<style>
"""
    + _CAPTIVE_BASE_CSS
    + """\
</style>
</head>
<body>
<main>
  <p class="brand">Meeting Scribe</p>
  <h1>Sign in to continue</h1>

  <section class="section">
    <h2>Guest</h2>
    <!--GUEST_ERR_SLOT-->
    <form method="POST" action="/captive/guest-pin" autocomplete="off">
      <label for="pin">4-digit PIN</label>
      <input id="pin" name="pin" type="text"
             inputmode="numeric" pattern="[0-9]{4}"
             maxlength="4" minlength="4" required
             placeholder="0000" autocomplete="off">
      <button type="submit" class="primary">Join as guest</button>
    </form>
  </section>

  <section class="section">
    <h2>Admin</h2>
    <p>Open the admin sign-in page in your phone's real browser.
    You'll see a <strong>"Not Secure"</strong> warning the first time
    &mdash; tap <strong>Visit Anyway</strong>; that's expected on the
    appliance's self-signed certificate.</p>
    <a class="button" href="{{CANONICAL}}/auth"
       target="_blank" rel="noopener noreferrer">
      Open admin sign-in
    </a>
  </section>
</main>
</body>
</html>
"""
)


_CAPTIVE_GUEST_SUCCESS_HTML = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f5f1e8">
<title>Success</title>
<style>
"""
    + _CAPTIVE_BASE_CSS
    + """\
.check {
  width: 48px; height: 48px; margin: 0 auto 1rem;
  border-radius: 50%; background: #2d8a4f;
  display: flex; align-items: center; justify-content: center;
  color: #fff;
}
.check svg { width: 26px; height: 26px; stroke-width: 3; }
.next-steps { text-align: left; margin-top: 1.25rem; }
.next-steps li {
  list-style: none; padding: 0.6rem 0;
  border-top: 1px solid #e8dfc7;
  font-size: 0.92rem; line-height: 1.45; color: #4a4a52;
}
.next-steps li:first-child { border-top: 0; }
.next-steps strong { color: #1a1a1f; }
</style>
</head>
<body>
<main>
  <div class="check">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M5 12.5l4.5 4.5 9.5-10"/>
    </svg>
  </div>
  <p class="brand">Meeting Scribe</p>
  <h1>Signed in — Success</h1>
  <a class="button" href="{{MDNS_ORIGIN}}/"
     target="_blank" rel="noopener noreferrer">
    Open meeting in Safari
  </a>
  <ol class="next-steps">
    <li><strong>Tap Done</strong> at the top right to close this
      sign-in screen.</li>
    <li>Open <code>{{MDNS_ORIGIN}}/</code> in <strong>Safari</strong>
      (not this captive screen). You may see a one-time
      "Not Secure" warning — tap <strong>Visit Anyway</strong>.</li>
  </ol>
</main>
<!--
iOS Captive Network Assistant accepts ``<title>Success</title>`` as
a captive-complete signal, so the Done button (blue tick) appears in
the top-right as soon as this page renders. No JS-driven navigation
needed — keeping the user on this page lets them tap the meeting
link with target="_blank" to launch Safari, which can click through
the self-signed cert (the CNA WebView itself cannot).
-->
</body>
</html>
"""
)
