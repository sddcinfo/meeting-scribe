"""Admin-bootstrap flow: paste the shared secret → receive a signed cookie.

The ``scribe_admin`` cookie this issues is what gates every terminal
endpoint. A LAN attacker without the secret file cannot obtain a cookie.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.terminal.auth import (
    COOKIE_NAME,
    AdminSecretStore,
    CookieSigner,
    decode_verified_cookie,
    revoke_session,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


BOOTSTRAP_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Authorize this browser · Meeting Scribe</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --error: #ff7b72;
    --success: #7ee787;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: radial-gradient(circle at 20% 0%, #1f2937 0, var(--bg) 60%);
    color: var(--text);
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }
  .card {
    width: 100%;
    max-width: 440px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.75rem 1.75rem 1.5rem;
    box-shadow: 0 24px 48px -18px rgba(0,0,0,.6);
  }
  h1 { margin: 0 0 .25rem; font-size: 1.15rem; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: .85rem; margin: 0 0 1.25rem; line-height: 1.5; }
  code { background: #0d1117; border: 1px solid var(--border); padding: 1px 6px; border-radius: 4px; font-size: .78rem; }
  form { display: flex; flex-direction: column; gap: .75rem; }
  label { font-size: .78rem; color: var(--muted); font-weight: 600; letter-spacing: .03em; text-transform: uppercase; }
  input[type=password] {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: .9rem;
    background: #0d1117;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .7rem .8rem;
    outline: none;
    transition: border-color .12s ease;
  }
  input[type=password]:focus { border-color: var(--accent); }
  button {
    background: var(--accent);
    color: #0d1117;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: .7rem 1rem;
    font-size: .9rem;
    cursor: pointer;
    transition: filter .12s ease, transform .12s ease;
    margin-top: .25rem;
  }
  button:hover { filter: brightness(1.05); }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: .6; cursor: progress; }
  .status { min-height: 1.2em; font-size: .82rem; margin-top: .25rem; }
  .status.err  { color: var(--error); }
  .status.ok   { color: var(--success); }
  .hint { color: var(--muted); font-size: .75rem; margin-top: 1rem; line-height: 1.5; }
  .hint strong { color: var(--text); }
</style>
</head>
<body>
<main class="card">
  <h1>Authorize this browser</h1>
  <p class="sub">Meeting Scribe's embedded terminal is admin-only. Paste the shared admin secret once to mint a signed cookie for this device. The cookie lasts 7 days.</p>
  <form id="f">
    <label for="secret">Admin secret</label>
    <input type="password" id="secret" name="secret" autocomplete="off" autofocus required>
    <button id="btn" type="submit">Authorize</button>
  </form>
  <div id="status" class="status" role="status"></div>
  <p class="hint">
    On the server host: <code>cat ~/.config/meeting-scribe/admin-secret</code><br>
    After authorization you can close this tab and open the meeting popout.
  </p>
</main>
<script>
const form = document.getElementById('f');
const btn  = document.getElementById('btn');
const input = document.getElementById('secret');
const status = document.getElementById('status');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  status.textContent = '';
  status.className = 'status';
  btn.disabled = true;
  try {
    const r = await fetch('/api/admin/authorize', {
      method: 'POST',
      credentials: 'include',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({secret: input.value}),
    });
    if (r.ok) {
      status.textContent = 'Authorized. You can close this tab.';
      status.className = 'status ok';
      input.value = '';
      setTimeout(() => { window.location.href = '/?popout=view'; }, 600);
    } else {
      const data = await r.json().catch(() => ({}));
      status.textContent = data.error || 'Authorization failed.';
      status.className = 'status err';
      input.focus();
      input.select();
    }
  } catch (err) {
    status.textContent = 'Network error: ' + err.message;
    status.className = 'status err';
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""


GuestScopeFn = Callable[[Any], bool]


@dataclass
class BootstrapConfig:
    admin_secret: AdminSecretStore
    cookie_signer: CookieSigner
    is_guest_scope: GuestScopeFn


def _close_admin_ws_for_session(session_id: str) -> None:
    """Best-effort: close every admin WS bound to ``session_id``.

    Logout / re-auth call this so an outstanding privileged WS from the
    revoked session can't keep streaming. Schedules close coroutines
    via the running event loop; never raises (the cookie revocation
    must remain idempotent).
    """
    bucket = state._admin_ws_by_session.pop(session_id, None)
    if not bucket:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not in an async context — best-effort no-op; the WS handler's
        # finally block will clean up when its connection eventually
        # closes naturally.
        return
    for ws in bucket:
        close = getattr(ws, "close", None)
        if close is None:
            continue
        coro = close(code=1008, reason="revoked")
        if asyncio.iscoroutine(coro):
            loop.create_task(coro)


async def _logout(request: Request, cfg: BootstrapConfig) -> JSONResponse:
    """Revoke the cookie's session_id, close active admin WS, delete the
    cookie. Idempotent — works even when the cookie is missing or
    invalid (no error path leaks state to unauthenticated callers).
    """
    cookie_value = request.cookies.get(COOKIE_NAME)
    ok, session_id, issued_at = decode_verified_cookie(cfg.cookie_signer, cookie_value)
    if ok and session_id is not None and issued_at is not None:
        revoke_session(
            session_id,
            expiry_epoch=issued_at + cfg.cookie_signer.max_age_seconds,
            revoked_sessions=state._revoked_sessions,
        )
        _close_admin_ws_for_session(session_id)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=COOKIE_NAME, path="/")
    resp.headers["Cache-Control"] = "no-store, private"
    return resp


def register_bootstrap_routes(app: FastAPI, cfg: BootstrapConfig) -> None:
    @app.get("/admin/bootstrap")
    async def bootstrap_page() -> HTMLResponse:
        return HTMLResponse(BOOTSTRAP_HTML)

    @app.post("/api/admin/authorize")
    async def authorize(request: Request) -> JSONResponse:
        if cfg.is_guest_scope(request):
            return JSONResponse({"error": "admin scope required"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid body"}, status_code=400)
        secret = (body or {}).get("secret", "") if isinstance(body, dict) else ""
        if not isinstance(secret, str) or not cfg.admin_secret.verify(secret):
            # Deliberate jittered delay on failure — mitigates online brute force
            # without enabling lockout-based DoS of the legitimate admin user.
            await asyncio.sleep(0.2 + random.random() * 0.4)
            peer = getattr(getattr(request, "client", None), "host", "?")
            logger.info("admin authorize failed (peer=%s)", peer)
            return JSONResponse({"error": "invalid secret"}, status_code=401)

        # Re-auth on the same browser revokes the prior session_id +
        # closes any privileged WS still alive under it before minting
        # the new cookie. This is what makes "logout from another tab"
        # close every tab's admin WS.
        prior_cookie = request.cookies.get(COOKIE_NAME)
        if prior_cookie:
            ok, prior_sid, prior_issued = decode_verified_cookie(
                cfg.cookie_signer, prior_cookie
            )
            if ok and prior_sid is not None and prior_issued is not None:
                revoke_session(
                    prior_sid,
                    expiry_epoch=prior_issued + cfg.cookie_signer.max_age_seconds,
                    revoked_sessions=state._revoked_sessions,
                )
                _close_admin_ws_for_session(prior_sid)

        cookie_value = cfg.cookie_signer.issue()
        resp = JSONResponse({"ok": True})
        # Plan §A.4 / §A.6: SameSite=Strict (single-origin admin/guest),
        # Secure=True (TLS-only), HttpOnly=True. The cookie gets the new
        # 3-part format <issued>.<session_id>.<hmac> via the upgraded
        # CookieSigner.
        resp.set_cookie(
            key=COOKIE_NAME,
            value=cookie_value,
            max_age=cfg.cookie_signer.max_age_seconds,
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",
        )
        return resp

    @app.post("/api/admin/deauthorize")
    async def deauthorize(request: Request) -> JSONResponse:
        # Backwards-compatible name kept for older clients; new code calls
        # /api/admin/logout. Both perform the same revoke-session-and-
        # close-WS dance.
        return await _logout(request, cfg)

    @app.post("/api/admin/logout")
    async def logout(request: Request) -> JSONResponse:
        return await _logout(request, cfg)

    @app.get("/api/admin/terminal-access")
    async def terminal_access(request: Request) -> JSONResponse:
        # Scope-gated (admin LAN only), not cookie-gated: the whole point is
        # surfacing the secret so the user can authorize a fresh browser
        # from the Settings panel without jumping to the bootstrap page.
        # An attacker on the admin LAN already has equivalent access.
        if cfg.is_guest_scope(request):
            return JSONResponse({"error": "admin scope required"}, status_code=403)
        cookie_value = request.cookies.get(COOKIE_NAME)
        authorized = bool(cookie_value) and cfg.cookie_signer.verify(cookie_value)
        return JSONResponse(
            {
                "secret": cfg.admin_secret.secret.decode(),
                "secret_path": str(cfg.admin_secret.path),
                "cookie_set": authorized,
                "cookie_max_age_seconds": cfg.cookie_signer.max_age_seconds,
            }
        )
