"""HTTP captive sub-app — strict, minimal port-80 surface for v1.0.

The unified-hotspot model binds a single TLS listener at ``10.42.0.1:443``
(see Plan §A.1). Port 80 still has to exist for two reasons:

1. **OS captive-portal probes** (Apple ``/hotspot-detect.html``, Android
   ``/generate_204``, Windows ``/connecttest.txt``, Firefox
   ``/success.txt``, RFC 8910 ``/api/captive``) only run over HTTP and
   only against the gateway IP.
2. **Misdirected human traffic** — operators who type a host name and
   reach the captive HTTP listener should be redirected to the canonical
   HTTPS URL with a clear method-guard 426 fallback for tooling that
   POSTs.

The sub-app's surface is therefore:

* The four OS captive probe routes (re-registered from
  ``hotspot/captive_portal.py`` so the captive-ack state machine remains
  the single source of truth).
* ``GET/HEAD /{path:path}`` → ``308 Permanent Redirect`` to
  ``https://10.42.0.1{path}{?query}`` with the query string preserved
  verbatim.
* **Any non-GET/HEAD method on any path** → ``426 Upgrade Required``
  with header ``Upgrade: TLS/1.2, HTTP/1.1`` and ``Connection: Upgrade``,
  empty body, returned BEFORE route matching so the body is never read.

Everything else — the main FastAPI app's routes (``/admin/*``,
``/api/admin/*``, ``/api/ws*``, ``/static/*``, ``/`` itself) — is **not**
registered on the HTTP sub-app at all. Confidentiality is provided by
code path (the official UI never POSTs to HTTP); we don't claim wire-
level secrecy of misdirected POSTs.
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


def build_captive_http_app() -> FastAPI:
    """Construct the strict HTTP sub-app — call once at server startup.

    Returns a fresh FastAPI app distinct from the main TLS app. Wire it
    onto the port-80 listener via uvicorn; the main app stays on the
    port-443 listener.
    """
    app = FastAPI(title="meeting-scribe captive (HTTP)")

    @app.middleware("http")
    async def method_guard(request: fastapi.Request, call_next):
        """Reject every non-GET/HEAD method with 426 before route matching.

        Plan §A.1 / §Security Boundary. We do NOT claim wire-level secrecy
        of the request body — many HTTP clients pipeline body with headers,
        so the body of an unsanctioned POST may already be on the wire
        before this middleware fires. The 426 is a fail-closed signal to
        misconfigured tooling, not a transport guarantee. The official UI
        never POSTs to HTTP; sign-in always happens over HTTPS.
        """
        method = request.method.upper()
        if method in ("GET", "HEAD"):
            return await call_next(request)
        return Response(
            status_code=426,
            headers={
                "Upgrade": "TLS/1.2, HTTP/1.1",
                "Connection": "Upgrade",
                "Content-Length": "0",
            },
        )

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

    @app.get("/{path:path}")
    @app.head("/{path:path}")
    async def catch_all(path: str, request: fastapi.Request) -> Response:
        """308 every non-captive GET/HEAD to canonical HTTPS, query verbatim.

        Captive probe paths above match first via FastAPI's route order;
        this catch-all picks up everything else (operator-typed hostnames,
        bookmarks to ``http://10.42.0.1/whatever``, accidental loopback
        clicks). The 308 preserves method + body, so future tools that
        chain through the redirect see the same request shape.
        """
        canonical = _canonical_https_origin()
        target_path = "/" + path if not path.startswith("/") else path
        # FastAPI's path-parameter strips the leading slash; restore it so
        # the redirect URL matches the original request path exactly.
        if path == "":
            target_path = "/"
        query = request.url.query
        suffix = f"?{query}" if query else ""
        return Response(
            status_code=308,
            headers={"Location": f"{canonical}{target_path}{suffix}"},
        )

    return app
