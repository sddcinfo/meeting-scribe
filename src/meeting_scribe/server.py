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
    _detect_management_ip,
    _make_tcp_socket,
    _NoSignalServer,
    _serve_dual,
)
from meeting_scribe.server_support.page_cache import cache_html
from meeting_scribe.server_support.request_scope import (
    _is_guest_scope,
)
from meeting_scribe.speaker.enrollment import SpeakerEnrollmentStore
from meeting_scribe.terminal.auth import (
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
# ``metrics`` / ``refinement_worker``. Reached as ``state.X``.

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
# Per-boot signing subkey: regenerated every server start so a restart
# invalidates every previously-issued admin cookie. The CookieSigner takes
# the derived subkey rather than the master secret so the master never
# touches the HMAC. _revoked_sessions tracks live in-boot logout/re-auth
# revocations; per-boot rotation handles the cross-restart case.
state.boot_session_id = secrets.token_bytes(32)
_cookie_subkey = derive_cookie_subkey(
    state._terminal_admin_secret.secret,
    state.boot_session_id,
)
state._terminal_cookie_signer = CookieSigner(
    secret=_cookie_subkey,
    is_revoked=lambda sid: sid in state._revoked_sessions,
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

# Bound on the in-flight TTS backlog. Keep this SMALL so TTS stays
# near-live: faster-qwen3-tts produces ~1-2s of synthesis per segment,
# live speech produces segments every ~1s, and with a single serial
# worker a 40-slot queue would accumulate ~60s of backlog within a
# minute. 3 slots + drop-oldest policy means the listener always hears
# the most recent translations and silently misses at most a handful
# when the worker pool falls behind — preferable to playing ancient
# audio that's drifted 30s off the live transcript.
# With two replicas + streaming, two segments synthesise in parallel and
# listeners hear first audio within ~300 ms of synth start. Allowing four
# in the backlog plus two in flight (~6 segments buffered) is still
# comfortably under the lag budget while letting brief bursts coast
# without triggering oldest-drop.
# TTS pipeline config constants live in ``runtime.state``
# (``state.TTS_QUEUE_MAXSIZE``, ``state.TTS_WORKER_COUNT``,
# ``state.TTS_CONTAINER_MAX_CONCURRENCY``,
# ``state.TTS_EXPECTED_SYNTH_DEFAULT_S``) so the metrics + health
# evaluator can read them without circling back through this module.

# In-flight synthesis tracking lives in ``runtime.state.tts_in_flight``
# (incremented inside ``_do_tts_synthesis`` right before the backend call,
# decremented in a finally) and ``runtime.state.tts_inflight_started``
# (seg_id → monotonic ts when synth started; used by the health evaluator
# to detect "stuck" requests).

# ``refinement_worker`` lives on ``meeting_scribe.runtime.state``;
# the per-meeting instance is constructed in ``_setup_refinement_worker``
# and torn down in ``meeting_stop`` / ``meeting_cancel``.


# Refinement drain registry (``_DrainEntry``, ``_refinement_drains``,
# ``_drain_refinement`` etc.) lives in
# ``meeting_scribe.server_support.refinement_drains``. The seq counter
# is allocated via ``_next_drain_id()`` to keep the global mutation
# inside the module that owns it.


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
        is_guest_scope=_is_guest_scope,
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
        is_guest_scope=_is_guest_scope,
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
from meeting_scribe.routes.diagnostics import router as _diagnostics_router
from meeting_scribe.routes.meeting_crud import router as _meeting_crud_router
from meeting_scribe.routes.meeting_lifecycle import (
    router as _meeting_lifecycle_router,
)
from meeting_scribe.routes.room import router as _room_router
from meeting_scribe.routes.slides import router as _slides_router
from meeting_scribe.routes.speaker import router as _speaker_router
from meeting_scribe.routes.status import router as _status_router
from meeting_scribe.routes.views import router as _views_router
from meeting_scribe.ws.audio_input import router as _ws_audio_input_router
from meeting_scribe.ws.audio_output import router as _ws_audio_output_router
from meeting_scribe.ws.view_broadcast import router as _ws_view_broadcast_router

app.include_router(_admin_router)
app.include_router(_captive_portal_router)
app.include_router(_diagnostics_router)
app.include_router(_meeting_crud_router)
app.include_router(_meeting_lifecycle_router)
app.include_router(_room_router)
app.include_router(_slides_router)
app.include_router(_speaker_router)
app.include_router(_status_router)
app.include_router(_views_router)
app.include_router(_ws_audio_input_router)
app.include_router(_ws_audio_output_router)
app.include_router(_ws_view_broadcast_router)


def main() -> None:
    """Entry point — run admin (HTTPS) + guest (HTTP) listeners in one process.

    Binding:
      - Admin HTTPS on TWO sockets (or ONE in degraded mode):
          * ``127.0.0.1:<port>``  — local tools (tests, autosre TUI,
            healthcheck script, ``meeting-scribe status``).
          * ``<management_ip>:<port>`` — LAN admin access. Omitted when
            no management IP is detected (degraded mode); a background
            task hot-adds it when the network appears.
        Never ``0.0.0.0``: the admin socket must not exist on the
        hotspot (wlan0) interface under any circumstances.
      - Guest HTTP on ``<AP_IP>:<guest_port>`` (``IP_FREEBIND``). Pre-
        binds the hotspot IP before the AP is up so a single bind
        survives meeting rotations. Unaffected by degraded mode.

    Both servers share the same ``app`` object, so in-process globals
    (``state.current_meeting``, ``state._audio_out_clients``, ``state.tts_queue``,
    ``state.ws_connections``) are shared. The FastAPI ``hotspot_guard``
    middleware + per-WebSocket ``_is_guest_scope`` checks keep guest
    traffic restricted to the live-translation / TTS paths regardless of
    which listener accepted the connection.
    """
    server_config = ServerConfig.from_env()

    management_ip = _detect_management_ip()
    is_degraded = (
        management_ip == "127.0.0.1" and not os.environ.get("SCRIBE_MANAGEMENT_IP", "").strip()
    )
    if is_degraded:
        logger.warning(
            "no LAN IP detected — admin listener is localhost-only; "
            "will recover when network appears"
        )
    else:
        logger.info("management IP detected: %s", management_ip)

    project_root = Path(__file__).resolve().parent.parent.parent
    ssl_key = project_root / "certs" / "key.pem"
    ssl_cert = project_root / "certs" / "cert.pem"
    if not (ssl_key.exists() and ssl_cert.exists()):
        raise RuntimeError(
            f"admin TLS certs missing: expected {ssl_key} and {ssl_cert}. "
            "Run `meeting-scribe setup` to generate a self-signed pair."
        )

    # Read-only SAN sanity check. Aborts startup with a remediation message
    # if the leaf cert lacks the AP IP SAN — the v1.0 trust anchor (leaf-only,
    # no CA) requires every reachable address be in the SAN list. The check
    # never mutates the cert path; cert (re)generation is provisioning-time
    # only via `meeting-scribe setup`.
    from meeting_scribe.runtime.cert_check import CertConfigError, assert_cert_sans

    try:
        assert_cert_sans(ssl_cert, required_ips={AP_IP})
    except CertConfigError as cert_err:
        raise RuntimeError(str(cert_err)) from cert_err

    # Admin: pre-bind two sockets (loopback + management IP). Both serve
    # HTTPS via one uvicorn.Server so there's only one lifespan and one
    # set of in-process globals.
    admin_hosts: list[str] = ["127.0.0.1"]
    if management_ip != "127.0.0.1":
        admin_hosts.append(management_ip)
    admin_sockets = [_make_tcp_socket(h, server_config.port) for h in admin_hosts]

    admin_config = uvicorn.Config(
        app,
        log_level="info",
        ssl_keyfile=str(ssl_key),
        ssl_certfile=str(ssl_cert),
        lifespan="on",
        # Explicit WS keepalive. Bumped 2026-04-15 from 10/10 → 20/45 after
        # observing the audio-out WS getting closed mid-meeting by the
        # periodic deep-health probes (every 10–12 s) blocking the event
        # loop for ~2.5 s, which accumulated across cycles and pushed the
        # pong round-trip past the 10 s timeout. 20 s send interval × 45 s
        # receive timeout gives ~2× headroom for any single stall and still
        # detects a truly dead peer within ~65 s. Real fix for the stalls
        # is upstream in _DEEP_HEALTH_TTL and the shared httpx client, but
        # this is the direct "don't kill the WS" lever.
        ws_ping_interval=20,
        ws_ping_timeout=45,
    )
    admin_server = _NoSignalServer(admin_config)

    # Guest: plain HTTP on two sockets.
    #   * ``<AP_IP>:<guest_port>`` via IP_FREEBIND — the real hotspot
    #     entry point. FREEBIND lets us pre-bind before nmcli assigns
    #     10.42.0.1 to wlan0, so a single bind survives meeting rotations.
    #   * ``127.0.0.1:<guest_port>`` — keeps local tooling working
    #     (autosre TUI's portal probe, `curl http://127.0.0.1/`, tests).
    #     Middleware + per-WS ``_is_guest_scope`` still treat every HTTP
    #     request as guest-scope, so loopback cannot reach admin paths
    #     via this socket.
    # Port <1024 requires CAP_NET_BIND_SERVICE on the interpreter; the
    # meeting-scribe CLI grants that via `setcap` before spawn.
    #
    # Reap any orphaned captive-portal subprocesses before the bind.  If
    # a prior server instance was SIGKILL'd (OOM, autosre unload during
    # a benchmark window, etc.), its shutdown never ran and the
    # captive-portal-80.py subprocess it spawned can still be holding
    # port 80.  Doing this at startup (rather than only at shutdown) is
    # what makes the cleanup idempotent across crashes.
    _reap_orphan_captive_portals()
    guest_sockets = [
        _make_tcp_socket("127.0.0.1", server_config.guest_port),
        _make_tcp_socket(AP_IP, server_config.guest_port, freebind=True),
    ]
    guest_config = uvicorn.Config(
        app,
        log_level="info",
        lifespan="off",
        # See admin_config above for the 20/45 rationale — same stall
        # budget applies, and the audio-out WS that hotspot guests open
        # is the one actually exposed to this risk (admin WS is just
        # local operator / dev).
        ws_ping_interval=20,
        ws_ping_timeout=45,
    )
    guest_server = _NoSignalServer(guest_config)

    logger.info(
        "starting listeners: admin=https://{127.0.0.1,%s}:%d, guest=http://{127.0.0.1,%s}:%d",
        management_ip,
        server_config.port,
        AP_IP,
        server_config.guest_port,
    )

    deferred = (server_config.port, 15) if is_degraded else None

    try:
        asyncio.run(
            _serve_dual(
                admin_server,
                admin_sockets,
                guest_server,
                guest_sockets,
                deferred_admin_bind=deferred,
            )
        )
    finally:
        for s in admin_sockets:
            with suppress(OSError):
                s.close()
        for s in guest_sockets:
            with suppress(OSError):
                s.close()


if __name__ == "__main__":
    main()
