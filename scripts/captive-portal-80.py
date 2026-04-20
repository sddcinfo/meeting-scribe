#!/usr/bin/env python3
"""Port 80 handler: captive-portal probes + HTTP → HTTPS redirect.

Behavior:
  - Captive-portal probes (iOS/Android/Windows/Firefox/RFC 8910): return
    "success" so the device's blue tick flips on without popping a CNA.
  - Everything else: 301 to ``https://<host>:<https_port><path>`` where the
    host is taken from the request's Host header. This lets the same script
    serve both generic dev machines (redirect to the box the user typed) and
    hotspot mode (redirect to the AP IP because that's what clients put in
    the Host header).

Environment:
  SCRIBE_HTTPS_PORT  Target HTTPS port for redirects (default 8080)
  SCRIBE_AP_IP       Fallback redirect host if the client sent no Host
                     header (default 10.42.0.1, matches the hotspot AP).

This script is intentionally stdlib-only so it can be launched by a
``python3`` with CAP_NET_BIND_SERVICE — no venv, no imports, no surprises.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTTPS_PORT = int(os.environ.get("SCRIBE_HTTPS_PORT", "8080"))
AP_IP_FALLBACK = os.environ.get("SCRIBE_AP_IP", "10.42.0.1")

SUCCESS = "<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"


def _redirect_target(host_header: str, path: str) -> str:
    hostname = host_header.split(":", 1)[0] if host_header else ""
    if not hostname or hostname == "0.0.0.0":
        hostname = AP_IP_FALLBACK
    return f"https://{hostname}:{HTTPS_PORT}{path}"


class Handler(BaseHTTPRequestHandler):
    server_version = "MeetingScribePort80/1.0"

    def do_GET(self):
        path = self.path.split("?")[0]

        # iOS: return Success → blue tick instantly
        if path == "/hotspot-detect.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(SUCCESS.encode())
            return

        # Android: 204 → connected, no portal
        if path in ("/generate_204", "/gen_204", "/canonical.html"):
            self.send_response(204)
            self.end_headers()
            return

        # Windows NCSI
        if path == "/connecttest.txt":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Microsoft Connect Test")
            return

        # Firefox
        if path == "/success.txt":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"success\n")
            return

        # RFC 8910 API
        if path == "/api/captive":
            self.send_response(200)
            self.send_header("Content-Type", "application/captive+json")
            self.end_headers()
            self.wfile.write(json.dumps({"captive": False}).encode())
            return

        # Everything else: redirect to the HTTPS server on the same host.
        # 302 (not 301) so clients don't cache the redirect once they leave
        # the hotspot — matches the prior captive-portal behavior.
        target = _redirect_target(self.headers.get("Host", ""), self.path)
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        target = _redirect_target(self.headers.get("Host", ""), self.path)
        self.send_response(307)  # 307 preserves method, 308 is permanent
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 80), Handler).serve_forever()
