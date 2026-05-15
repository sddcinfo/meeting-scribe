"""FastAPI server — GB10 real-time bilingual meeting transcription.

Single meeting at a time. No auth (localhost only). No TLS needed.
Wires together: storage, audio resample, ASR (vLLM Qwen3-ASR), translation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from meeting_scribe.config import ServerConfig
from meeting_scribe.runtime import state
from meeting_scribe.runtime.net import (
    _make_tcp_socket,
    _NoSignalServer,
    _serve_three_apps,
    _serve_two_apps,
)
from meeting_scribe.server_support.page_cache import cache_html
from meeting_scribe.speaker.enrollment import SpeakerEnrollmentStore
from meeting_scribe.terminal.auth import (
    _KIOSK_COOKIE_HMAC_INFO,
    KIOSK_COOKIE_NAME,
    AdminSecretStore,
    CookieSigner,
    TicketStore,
    derive_cookie_subkey,
)
from meeting_scribe.terminal.bootstrap import BootstrapConfig, register_bootstrap_routes
from meeting_scribe.terminal.registry import ActiveTerminals
from meeting_scribe.terminal.router import TerminalRouterConfig, register_terminal_routes
from meeting_scribe.terminal.tmux_helper import write_tmux_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Force INFO level for our modules AND attach a handler so messages
# actually reach stdout. Uvicorn configures its own uvicorn.* loggers but
# leaves the root logger on the default lastResort handler at WARNING —
# so setLevel(INFO) alone isn't enough: INFO messages pass the logger
# filter, propagate up, and then get dropped by lastResort. The missing
# handler is why "TTS fire:", "TTS synthesize:", "cache_voice" INFO logs
# that should have been visible during live meetings were silently
# swallowed, making TTS failures look like nothing was happening at all.
_ms_logger = logging.getLogger("meeting_scribe")
_ms_logger.setLevel(logging.INFO)
if not _ms_logger.handlers:
    _ms_handler = logging.StreamHandler()
    _ms_handler.setLevel(logging.INFO)
    _ms_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _ms_logger.addHandler(_ms_handler)
    _ms_logger.propagate = False

# Ring buffer + rotating server.log are wired in lifespan() once the
# config is loaded (the file path is derived from meetings_dir.parent so
# tests can override it). See meeting_scribe.diagnostics.

# All shared globals live on ``meeting_scribe.runtime.state``:
# ``config`` / ``storage`` / ``resampler`` / ``translation_queue`` /
# ``metrics``. Reached as ``state.X``.

# Backend handles live in ``meeting_scribe.runtime.state`` so route
# modules, WS handlers, and the hotspot subsystem can reach them
# without importing this module. ``state`` declares typed placeholders
# (``state.asr_backend = None`` etc.) — real backends are constructed
# in ``lifespan``/``_init_*`` and assigned through ``state``.

# Speaker enrollment + room layout draft (session-scoped) — assigned
# through ``meeting_scribe.runtime.state`` so cross-module callers can
# reach them. The empty-dict layout drafts are declared in state.py;
# the speaker enrollment store is constructed here at import time.
state.enrollment_store = SpeakerEnrollmentStore()
# Session helpers (``_get_session_id``, ``_get_draft_layout``,
# ``_set_draft_layout``, ``ClientSession``) and the
# ``_DEFAULT_SESSION`` constant live in
# ``meeting_scribe.server_support.sessions``.


# Active meeting + WS connection sets live in
# ``meeting_scribe.runtime.state`` so route modules and WS handlers can
# reach them without importing this module.
# Background-task references and the lazy speaker-embedding executor
# moved to ``meeting_scribe.runtime.state`` so the lifecycle handlers
# can reach them once routes are extracted.

# Embedded terminal panel — admin-only in-browser shell
# (source of truth: docs in meeting_scribe.terminal). Constructed at
# import time because ``register_bootstrap_routes(...)`` and
# ``register_terminal_routes(...)`` (called at module level below)
# capture these as constructor arguments. Assigned through
# ``meeting_scribe.runtime.state`` so route modules can reach them.
state._terminal_admin_secret = AdminSecretStore.load_or_create()
# **Persistent** session salt: a 32-byte secret stored next to the
# admin secret, generated once on first boot. Was previously
# ``secrets.token_bytes(32)`` regenerated per-boot, which meant every
# server restart kicked every connected operator out of the admin UI.
# In an internal admin tool with one or two users that's pure UX
# friction — every CSS edit, every CI deploy, every transient crash
# locked everyone out. Persisting the salt means cookies survive
# restarts; explicit revocation still works through three other
# levers:
#   * ``_revoked_sessions`` (in-process set) — logout / re-auth path
#   * ``auth_version`` bump (factory_reset) — cross-rotation
#   * Cookie expiry (``max_age_seconds``) — natural ageing
# A determined attacker holding a stolen cookie loses access on the
# next factory_reset or expiry instead of on the next service
# restart; the security model documents the tradeoff explicitly.
_SESSION_SALT_PATH = (
    Path(os.environ.get("MEETING_SCRIBE_SESSION_SALT_PATH") or "")
    if os.environ.get("MEETING_SCRIBE_SESSION_SALT_PATH")
    else Path.home() / ".config" / "meeting-scribe" / "session-salt"
)
if _SESSION_SALT_PATH.exists():
    _data = _SESSION_SALT_PATH.read_bytes()
    if len(_data) >= 16:
        state.boot_session_id = _data
    else:
        state.boot_session_id = secrets.token_bytes(32)
        _SESSION_SALT_PATH.write_bytes(state.boot_session_id)
        os.chmod(_SESSION_SALT_PATH, 0o600)
else:
    _SESSION_SALT_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    state.boot_session_id = secrets.token_bytes(32)
    # Atomic-ish write via tmp + rename, mode 0600.
    _tmp = _SESSION_SALT_PATH.with_suffix(_SESSION_SALT_PATH.suffix + f".tmp.{os.getpid()}")
    _fd = os.open(str(_tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(_fd, state.boot_session_id)
    finally:
        os.close(_fd)
    os.replace(str(_tmp), str(_SESSION_SALT_PATH))
from meeting_scribe import setup_state as _setup_state

_admin_cookie_subkey = derive_cookie_subkey(
    state._terminal_admin_secret.secret,
    state.boot_session_id,
    _setup_state.auth_version(),
)
state._terminal_cookie_signer = CookieSigner(
    secret=_admin_cookie_subkey,
    is_revoked=lambda sid: sid in state._revoked_sessions,
)
# Kiosk cookie signer: shares the boot-derived subkey pool with the
# admin signer but uses a distinct HMAC info (``scribe-kiosk-cookie-v1``)
# so kiosk cookies and admin cookies cannot be replayed across roles.
# TTL is 90 days (vs 7 for admin) because the kiosk profile is a
# system-owned headless chromium that doesn't get "logged out" via UI;
# revocation happens via factory_reset (rotates the secret) or by
# removing the chromium profile.
state._kiosk_cookie_signer = CookieSigner(
    secret=_admin_cookie_subkey,
    is_revoked=lambda sid: sid in state._revoked_sessions,
    cookie_name=KIOSK_COOKIE_NAME,
    max_age_seconds=90 * 24 * 3600,
    hmac_info=_KIOSK_COOKIE_HMAC_INFO,
)
state._terminal_ticket_store = TicketStore(state._terminal_admin_secret.secret)
state._terminal_registry = ActiveTerminals(
    max_concurrent=int(os.environ.get("SCRIBE_TERM_MAX_SESSIONS", "4"))
)
# The tmux config is written once at startup; tmux picks it up only on
# `tmux server start`, so rewriting it is a no-op for live sessions.
write_tmux_config()

# Slide translation — initialized in lifespan if the worker container
# image is available. Both placeholders live in
# ``meeting_scribe.runtime.state``.


# Backend failure tracking, container auto-restart, and deep health
# probes (``_record_backend_failure``, ``_record_backend_success``,
# ``_backend_is_degraded``, ``_restart_container``,
# ``_deep_check_*``, ``_deep_backend_health``) live in
# ``meeting_scribe.server_support.backend_health``.


# Catchup queue (``state._pending_speaker_events``) and the
# matching monotonic-timestamp dict
# (``state._pending_speaker_timestamps``) both live in
# ``meeting_scribe.runtime.state``.

# TTS pipeline: a single long-lived worker task drains a FIFO queue of
# translation events. A real queue (instead of the old drop-latest-wins
# policy) means that under transient overload — e.g. translation gets
# slow for a few seconds — listeners still hear every segment, in order,
# just delayed. The queue is bounded so sustained overload can't grow
# memory without limit; when the cap is hit we drop the OLDEST item
# (front of queue) to make room, so audio stays roughly in sync with
# live speech rather than playing ancient backlog.
# TTS pipeline runtime handles (queue, semaphores, in-flight tracking,
# worker task list) live in ``meeting_scribe.runtime.state`` so the
# Metrics class, health evaluator, and TTS workers can reach them
# without circling through the server module.

# Short-window text-hash dedup for ASR finals. Guards against scribe-asr
# emitting the same final under a fresh segment_id after a container restart
# or other flakiness (LocalAgreement revision dedup keys on seg_id and can't
# catch that case). See _process_event.
# ``_DEDUP_WINDOW_S`` + ``_recent_finals`` (the short-window text-hash
# dedup state) live in ``meeting_scribe.pipeline.transcript_event``
# alongside ``_process_event``, the only consumer.

# ``_get_meeting_lifecycle_lock`` lives in
# ``meeting_scribe.server_support.lifecycle_lock`` so route modules
# can serialize start/stop/cancel without circling back here.


# Silence watchdog timestamps live in ``runtime.state``
# (``last_audio_chunk_ts``, ``silence_warn_sent``) so ws.audio_input and
# the watchdog loop don't sit on either side of a server-module import.

# TTS pipeline config constants live in ``runtime.state``. GB10 reliability
# mode keeps one resident faster-qwen3-tts container and relies on explicit
# lag gates plus bounded playback queues instead of a second GPU-resident TTS
# model.
# (``state.TTS_QUEUE_MAXSIZE``, ``state.TTS_WORKER_COUNT``,
# ``state.TTS_CONTAINER_MAX_CONCURRENCY``,
# ``state.TTS_EXPECTED_SYNTH_DEFAULT_S``) so the metrics + health
# evaluator can read them without circling back through this module.

# In-flight synthesis tracking lives in ``runtime.state.tts_in_flight``
# (incremented inside ``_do_tts_synthesis`` right before the backend call,
# decremented in a finally) and ``runtime.state.tts_inflight_started``
# (seg_id → monotonic ts when synth started; used by the health evaluator
# to detect "stuck" requests).

# ``_loop_lag_monitor`` lives in ``runtime.health_monitors`` (alongside
# the silence watchdog). Lifespan starts both via that module.


# ``Metrics`` and the ``tts_health_evaluator`` background loop live in
# ``meeting_scribe.runtime.metrics``. Construction now happens inside
# ``runtime.state`` at import time so the metrics object is available to
# every consumer (including tests) without a server-module round-trip.


# Lifespan (backend init, background tasks, dev-mode auto-resume,
# slide worker, post-ready bookkeeping) lives in
# ``meeting_scribe.runtime.lifespan``. server.py keeps the import so
# the FastAPI app construction below can wire it.
from meeting_scribe.runtime.lifespan import lifespan

# --- App ---

app = FastAPI(title="Meeting Scribe", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# GZip compression — cuts 67K CSS and 134K JS down to ~15K and ~35K over WiFi
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=500)

# Path validation (``_safe_meeting_dir`` / ``_safe_segment_path``)
# lives in ``meeting_scribe.server_support.safe_paths``.


# Multi-target translation feature flag — kept here at module scope
# because several call sites import it as ``_MULTI_TARGET_ENABLED``.


# ``_TTS_DEFER_UNTIL_CATCH_UP`` lives in
# ``meeting_scribe.pipeline.transcript_event`` (its only consumer).


# ── HTML page cache ──────────────────────────────────────────
# Read all HTML pages into memory at import time. Avoids disk I/O on
# every request (~10-50ms per read_text on slow storage).
#
# ``_HTML`` cache + ``cache_html`` live in
# ``meeting_scribe.server_support.page_cache`` so ``routes/views.py``
# can read the pre-loaded HTML without circling back through the
# server module.
cache_html(STATIC_DIR)


# Middlewares (hotspot_guard, request_timing, static_cache_headers) live in
# ``meeting_scribe.middlewares``; registration order matters and is fixed
# inside that module.
from meeting_scribe.middlewares import register_middlewares

register_middlewares(app)


# Static files mounted AFTER middleware — middleware intercepts first
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Terminal panel: bootstrap + REST + WS. Admin-only; cookie-gated; ticketed.
# Registered at module scope so the routes exist before lifespan runs,
# avoiding a race where the browser could hit them pre-init.
register_bootstrap_routes(
    app,
    BootstrapConfig(
        admin_secret=state._terminal_admin_secret,
        cookie_signer=state._terminal_cookie_signer,
    ),
)


def _terminal_history_path() -> Path | None:
    """Resolve where the embedded terminal's output log should live.

    - If a meeting is recording or just ended, the log lives inside the
      meeting folder alongside slides / journal / audio (matches the
      "slides live in the meeting folder" shape the user asked for).
    - With no meeting, fall back to ``<meetings_dir>/_no_meeting/terminal.log``
      so history still survives reloads.

    Returns None to disable logging if storage isn't initialized yet
    (e.g. extremely early in startup). The router treats None as
    "history disabled" and moves on.
    """
    try:
        meetings_dir = state.storage._meetings_dir
    except Exception:
        return None
    if state.current_meeting is not None:
        return meetings_dir / state.current_meeting.meeting_id / "terminal.log"
    return meetings_dir / "_no_meeting" / "terminal.log"


register_terminal_routes(
    app,
    TerminalRouterConfig(
        registry=state._terminal_registry,
        cookie_signer=state._terminal_cookie_signer,
        ticket_store=state._terminal_ticket_store,
        history_path_fn=_terminal_history_path,
    ),
)


# ── Extracted route modules ────────────────────────────────
# Live in ``meeting_scribe.routes.*`` and ``meeting_scribe.hotspot.*``;
# included here so they attach to the same FastAPI ``app`` instance.
# The route-parity baseline test guards against drift in path /
# handler identity.
from meeting_scribe.hotspot.ap_control import (
    AP_IP,
    _reap_orphan_captive_portals,
)
from meeting_scribe.hotspot.captive_portal import router as _captive_portal_router
from meeting_scribe.routes.admin import router as _admin_router
from meeting_scribe.routes.admin_audio import router as _admin_audio_router
from meeting_scribe.routes.admin_bt import router as _admin_bt_router
from meeting_scribe.routes.admin_speakerphone import (
    router as _admin_speakerphone_router,
)
from meeting_scribe.routes.admin_wan import router as _admin_wan_router
from meeting_scribe.routes.diagnostics import router as _diagnostics_router
from meeting_scribe.routes.guest_auth import router as _guest_auth_router
from meeting_scribe.routes.kiosk import router as _kiosk_router
from meeting_scribe.routes.kiosk_auth import router as _kiosk_auth_router
from meeting_scribe.routes.meeting_crud import router as _meeting_crud_router
from meeting_scribe.routes.meeting_lifecycle import (
    router as _meeting_lifecycle_router,
)
from meeting_scribe.routes.room import router as _room_router
from meeting_scribe.routes.setup import router as _setup_router
from meeting_scribe.routes.slides import router as _slides_router
from meeting_scribe.routes.speaker import router as _speaker_router
from meeting_scribe.routes.status import router as _status_router
from meeting_scribe.routes.views import router as _views_router
from meeting_scribe.ws.admin import router as _ws_admin_router
from meeting_scribe.ws.audio_input import router as _ws_audio_input_router
from meeting_scribe.ws.audio_output import router as _ws_audio_output_router
from meeting_scribe.ws.view_broadcast import router as _ws_view_broadcast_router

app.include_router(_admin_router)
app.include_router(_admin_audio_router)
app.include_router(_admin_bt_router)
app.include_router(_admin_speakerphone_router)
app.include_router(_admin_wan_router)
app.include_router(_captive_portal_router)
app.include_router(_diagnostics_router)
app.include_router(_guest_auth_router)
app.include_router(_kiosk_auth_router)
app.include_router(_kiosk_router)
app.include_router(_meeting_crud_router)
app.include_router(_meeting_lifecycle_router)
app.include_router(_room_router)
app.include_router(_setup_router)
app.include_router(_slides_router)
app.include_router(_speaker_router)
app.include_router(_status_router)
app.include_router(_views_router)
app.include_router(_ws_admin_router)
app.include_router(_ws_audio_input_router)
app.include_router(_ws_audio_output_router)
app.include_router(_ws_view_broadcast_router)


def main() -> None:
    """Entry point — single AP-bound HTTPS listener + captive HTTP sub-app.

    Binding (v1.0 setup-mode pivot):
      * Canonical HTTPS on ``10.42.0.1:443`` (``IP_FREEBIND``). One
        listener, one binding, every URL uses the implicit ``:443``.
        Setup-mode + operating-mode share the binding; only the AP
        SSID + auth differ.
      * Captive HTTP sub-app on ``10.42.0.1:80`` — static handoff
        page that deep-links into ``https://meeting-scribe-${id4}.local``
        plus the OS captive-portal probes. No stateful logic.

    LAN admin access is OUT of scope in v1.0 (codex P0/P1#1/P1#6 fix).
    Operators on LAN must SSH to the box or join the AP. Loopback
    (127.0.0.1) is also intentionally absent — the listener is on
    one IP only so cookie origin / cert SAN / browser autofill all
    line up on a single canonical URL.

    Tests use ``ASGITransport(app=app)`` for in-process route checks
    or spin up uvicorn directly via ``tests/test_server.py``; neither
    path goes through ``main()``, so this binding change is
    production-only.
    """
    server_config = ServerConfig.from_env()

    project_root = Path(__file__).resolve().parent.parent.parent
    ssl_key = project_root / "certs" / "key.pem"
    ssl_cert = project_root / "certs" / "cert.pem"
    if not (ssl_key.exists() and ssl_cert.exists()):
        raise RuntimeError(
            f"admin TLS certs missing: expected {ssl_key} and {ssl_cert}. "
            "Run `meeting-scribe setup` to generate a self-signed pair."
        )

    # Read-only SAN sanity check. Aborts startup with a remediation
    # message if the leaf cert lacks the AP IP SAN OR the per-device
    # mDNS DNS SAN — both are required because the wizard URL uses
    # the DNS name and the captive sub-app's 308 target uses the IP.
    from meeting_scribe.runtime.cert_check import CertConfigError, assert_cert_sans

    try:
        from meeting_scribe.cli._common import _required_leaf_dns_sans

        assert_cert_sans(
            ssl_cert,
            required_ips={AP_IP},
            required_dns=set(_required_leaf_dns_sans()),
        )
    except CertConfigError as cert_err:
        raise RuntimeError(str(cert_err)) from cert_err

    # Reap any orphaned captive subprocess holding port 80 from a
    # prior instance that was SIGKILL'd before its shutdown ran.
    _reap_orphan_captive_portals()

    # HTTPS listener on 0.0.0.0:<port>. Port defaults to 443 (production
    # canonical bind); SCRIBE_PORT (resolved into ``server_config.port``)
    # overrides for dev/test or when a second instance needs to coexist
    # with the production server on the same host — for example, a dev
    # sidecar on 8080. FREEBIND so the bind succeeds before
    # NetworkManager has finished bringing up the AP IP on ``wlP9s9``.
    # Admin auth is cookie-only.
    main_port = int(server_config.port)
    main_sockets = [_make_tcp_socket("0.0.0.0", main_port, freebind=True)]
    main_config = uvicorn.Config(
        app,
        log_level="info",
        ssl_keyfile=str(ssl_key),
        ssl_certfile=str(ssl_cert),
        lifespan="on",
        ws_ping_interval=20,
        ws_ping_timeout=45,
    )
    main_server = _NoSignalServer(main_config)

    # Captive HTTP sub-app at 10.42.0.1:80 — strict GET/HEAD redirect
    # to canonical HTTPS plus the OS captive probes. Built fresh here
    # so the captive routes don't leak into the canonical app.
    #
    # Disabled when SCRIBE_DISABLE_CAPTIVE_HTTP=1: lets a second
    # meeting-scribe instance run on the same host without colliding
    # with the production captive listener on port 80. Production
    # never sets the env; dev sidecars do.
    captive_disabled = os.environ.get("SCRIBE_DISABLE_CAPTIVE_HTTP") == "1"
    captive_server: _NoSignalServer | None = None
    captive_sockets: list = []
    if not captive_disabled:
        from meeting_scribe.hotspot.captive_http_app import build_captive_http_app

        captive_app = build_captive_http_app()
        captive_sockets = [_make_tcp_socket(AP_IP, 80, freebind=True)]
        captive_config = uvicorn.Config(
            captive_app,
            log_level="info",
            lifespan="off",
        )
        captive_server = _NoSignalServer(captive_config)

    # Kiosk loopback listener on 127.0.0.1:<port> (plain HTTP).
    # Serves only /kiosk, /kiosk-bootstrap, /api/kiosk/* via the
    # ``require_kiosk_listener`` dependency on those routes. Other
    # routes returned by the canonical app are technically reachable
    # over this socket too but every state-changing one goes through
    # role + origin checks that fail for the kiosk cookie.
    # Disabled when ``SCRIBE_DISABLE_KIOSK_LISTENER=1`` (unit tests;
    # dev sidecars where port 8444 is taken).
    kiosk_disabled = os.environ.get("SCRIBE_DISABLE_KIOSK_LISTENER") == "1"
    kiosk_port = int(os.environ.get("SCRIBE_KIOSK_LISTENER_PORT", "8444"))
    kiosk_server: _NoSignalServer | None = None
    kiosk_sockets: list = []
    if not kiosk_disabled:
        kiosk_sockets = [_make_tcp_socket("127.0.0.1", kiosk_port, freebind=False)]
        kiosk_config = uvicorn.Config(
            app,
            log_level="info",
            lifespan="off",
            ws_ping_interval=20,
            ws_ping_timeout=45,
        )
        kiosk_server = _NoSignalServer(kiosk_config)

    listener_summary = [f"main=https://{AP_IP}:{main_port}"]
    if captive_disabled:
        listener_summary.append("captive=disabled")
    else:
        listener_summary.append(f"captive=http://{AP_IP}:80")
    if kiosk_disabled:
        listener_summary.append("kiosk=disabled")
    else:
        listener_summary.append(f"kiosk=http://127.0.0.1:{kiosk_port}")
    logger.info("starting listeners: %s", ", ".join(listener_summary))

    try:
        if captive_server is None and kiosk_server is None:
            asyncio.run(main_server.serve(sockets=main_sockets))
        elif kiosk_server is None:
            asyncio.run(
                _serve_two_apps(
                    main_server,
                    main_sockets,
                    captive_server,  # type: ignore[arg-type]
                    captive_sockets,
                )
            )
        else:
            asyncio.run(
                _serve_three_apps(
                    main_server,
                    main_sockets,
                    captive_server,
                    captive_sockets,
                    kiosk_server,
                    kiosk_sockets,
                )
            )
    finally:
        for s in main_sockets:
            with suppress(OSError):
                s.close()
        for s in captive_sockets:
            with suppress(OSError):
                s.close()
        for s in kiosk_sockets:
            with suppress(OSError):
                s.close()


if __name__ == "__main__":
    main()
