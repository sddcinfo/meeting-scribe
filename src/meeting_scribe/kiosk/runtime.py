"""Kiosk runtime: the long-running Python process inside the cage
session that owns the HDMI mirror.

Lifecycle (driven by ``meeting-scribe-kiosk.service``):

  1. Block until ``/run/meeting-scribe/helper.sock`` and the
     meeting-scribe server's loopback listener are reachable.
  2. Mint two single-use kiosk-bootstrap nonces via the admin REST
     endpoint (using the deterministic local admin password helper
     that the rest of the CLI uses).
  3. Use one nonce to fetch a ``scribe_kiosk`` cookie for this
     process's httpx client; use the second to launch chromium
     pointed at ``/kiosk-bootstrap?nonce=<B>`` so chromium gets its
     own independent cookie via Set-Cookie.
  4. Run ``wlr-randr --json`` to enumerate connector modes; write
     the parsed blob to ``/run/meeting-scribe/hdmi-status.json``
     (which the admin GET /api/admin/settings reads).
  5. Apply the persisted ``hdmi_mode``/``hdmi_rotation`` via
     wlr-randr.
  6. Watch the chromium process; restart on crash.
  7. Subscribe to ``ws://127.0.0.1:8444/api/ws/view`` with the kiosk
     cookie; reset an in-process "last activity" timer on every
     event. When the timer exceeds ``hdmi_idle_sleep_minutes`` minutes,
     ``wlr-randr --output HDMI-A-1 --off``; on the next event, ``--on``.
  8. inotify the settings file so resolution / rotation / sleep
     changes from the admin UI apply within ~1 s.

Hardening notes:

  * Single process, single owner: cage hands the Wayland socket to
    this process; wlr-randr commands run inside the same session so
    no XDG_RUNTIME_DIR juggling.
  * Never raises on a transient HTTP failure; every loop has a
    bounded retry with exponential back-off so a meeting-scribe
    restart doesn't kill the kiosk.
  * No external state besides the on-disk ``hdmi-status.json``
    snapshot and the chromium snap profile under
    ``~/snap/chromium/common/kiosk-profile``.

This module is invoked via the ``meeting-scribe kiosk run-runtime``
CLI verb, which the systemd unit's ExecStart resolves through the
``/usr/local/bin/meeting-scribe-kiosk-runtime`` launcher.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where the runtime publishes its snapshot for the admin REST handler.
# We'd prefer /run/meeting-scribe but the daemon runs unprivileged so we
# can't mkdir under /run without a systemd-tmpfiles config. The /tmp
# fallback path is owned by the kiosk user and survives until reboot,
# which is the same lifecycle as /run for our purposes. The admin REST
# reader (kiosk/hdmi_status.py) consults the same env var if set.
_HDMI_STATUS_PATH = Path(
    os.environ.get("MEETING_SCRIBE_HDMI_STATUS_PATH")
    or f"/tmp/meeting-scribe-{os.getuid()}/hdmi-status.json"
)

# Settings file the runtime watches with inotify; same shape as the
# admin UI persists into.
_SETTINGS_PATH = Path.home() / ".config" / "meeting-scribe" / "settings.json"

# Chromium snap-confined profile dir. Must live inside ~/snap/<app>/
# or chromium silently throws away the cookie store at exit.
_CHROMIUM_PROFILE = Path.home() / "snap" / "chromium" / "common" / "kiosk-profile"

# Loopback HTTP listener (matches server.main()).
_LOOPBACK_HOST = "127.0.0.1"
_LOOPBACK_PORT = int(os.environ.get("SCRIBE_KIOSK_LISTENER_PORT", "8444"))


def _load_settings() -> dict[str, Any]:
    """Best-effort read of the operator-managed settings overrides."""
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except Exception:
        return {}


def _publish_hdmi_status(payload: dict[str, Any]) -> None:
    """Atomically write the HDMI status snapshot.

    The admin REST handler reads this on every settings GET. Atomic
    write so a concurrent reader never sees a truncated JSON blob.
    Never raises: status publishing must not crash the runtime on a
    file-system permission glitch.
    """
    payload = {**payload, "updated_at": time.time()}
    try:
        _HDMI_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = _HDMI_STATUS_PATH.with_suffix(_HDMI_STATUS_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(str(tmp), str(_HDMI_STATUS_PATH))
    except OSError as exc:
        logger.warning("kiosk-runtime: status publish failed: %s", exc)


def _run_wlr_randr_json() -> dict[str, Any]:
    """Run ``wlr-randr --json`` inside the cage session and return the
    parsed dict. Returns the sentinel "no display" shape on failure.

    Ubuntu 24.04 ships wlr-randr 0.3.0 which does NOT support --json;
    that option landed in 0.4.0. On the old release we fall through to
    a plain ``wlr-randr`` text invocation and return an empty result -
    the admin UI still sees ``hdmi_status.source=sentinel`` which the
    panel renders as "no kiosk runtime data" until wlr-randr is
    upgraded. Mode + rotation applies still work because they don't
    depend on the JSON output."""
    for cmd in (["wlr-randr", "--json"], ["wlr-randr"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            logger.warning("wlr-randr not on PATH; HDMI status unavailable")
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("wlr-randr timed out")
            return {}
        if result.returncode != 0:
            # 0.3.0 prints "unrecognized option '--json'" + rc 1 on the
            # --json invocation; fall through to the text invocation.
            if "unrecognized option" in (result.stderr or "") and "--json" in cmd:
                continue
            logger.warning("wlr-randr exit %d stderr=%s", result.returncode, result.stderr)
            return {}
        # 0.4+ returns JSON; 0.3.0 returns text. We only parse JSON.
        if "--json" in cmd:
            try:
                return {"outputs": json.loads(result.stdout)}
            except json.JSONDecodeError as exc:
                logger.warning("wlr-randr JSON parse failed: %s", exc)
                return {}
        # Plain-text path: surface the raw blob so the admin UI can
        # still render something diagnostic, plus a degraded flag so
        # the operator knows JSON enumeration is missing.
        return {"outputs": [], "wlr_randr_text": result.stdout, "wlr_randr_degraded": True}
    return {}


def _enumerate_hdmi_status() -> dict[str, Any]:
    """Reshape the wlr-randr blob into the admin-UI-friendly schema.

    Looks for the HDMI-A-1 output by name; falls back to the first
    connected output (some monitors enumerate as DP-1 over a USB-C
    DP-alt adapter even though the GB10's physical port is HDMI).
    """
    raw = _run_wlr_randr_json()
    outputs = raw.get("outputs", []) if isinstance(raw, dict) else []
    if not outputs:
        return {
            "connected": False,
            "current_mode": None,
            "available_modes": [],
            "rotation": 0,
            "enabled": False,
            "edid_name": None,
        }
    # Preferred connector + first connected fallback.
    chosen = next((o for o in outputs if o.get("name") == "HDMI-A-1"), None)
    if chosen is None:
        chosen = next((o for o in outputs if o.get("enabled")), outputs[0])
    modes = []
    current_mode = None
    for mode in chosen.get("modes", []):
        identifier = f"{mode.get('width', 0)}x{mode.get('height', 0)}"
        refresh = mode.get("refresh")
        if refresh:
            identifier = f"{identifier}@{refresh:.3f}Hz"
        modes.append(identifier)
        if mode.get("current"):
            current_mode = identifier
    return {
        "connected": bool(chosen.get("enabled")),
        "current_mode": current_mode,
        "available_modes": modes,
        "rotation": int(chosen.get("transform", "normal").split("-")[-1])
        if chosen.get("transform", "normal").startswith("rotate-")
        else 0,
        "enabled": bool(chosen.get("enabled")),
        "edid_name": chosen.get("name"),
    }


def _apply_settings(settings: dict[str, Any]) -> None:
    """Push the operator's persisted HDMI settings into wlr-randr."""
    mode = settings.get("hdmi_mode", "auto")
    rotation = int(settings.get("hdmi_rotation", 0))
    args = ["wlr-randr", "--output", "HDMI-A-1"]
    if mode and mode != "auto":
        args += ["--mode", mode]
    if rotation in (0, 90, 180, 270):
        args += ["--transform", str(rotation)]
    try:
        subprocess.run(args, check=False, timeout=10)
    except Exception as exc:
        logger.warning("wlr-randr apply failed: %s", exc)


def _mint_kiosk_nonce_blocking(timeout_s: float = 120.0) -> str | None:
    """Loop ``_mint_kiosk_nonce`` until it returns a nonce or times out.

    The kiosk system service starts before the meeting-scribe USER
    systemd service is ready (the user-scope unit comes up after
    multi-user.target, while the kiosk hotplug oneshot fires AT
    multi-user.target). Without this loop the runtime mints once,
    fails, and launches chromium with an empty nonce - chromium then
    paints a permanent "invalid_nonce" / "site unreachable" error
    page until the cable is unplugged.

    Back-off is 1 s → 5 s with linear ramp; the 120 s cap covers
    even a worst-case scribe boot (asr cold-start, vLLM warm-up).
    Returns the nonce on success, ``None`` only after ``timeout_s``.
    """
    import time as _time

    deadline = _time.monotonic() + timeout_s
    delay = 1.0
    attempts = 0
    while _time.monotonic() < deadline:
        attempts += 1
        nonce = _mint_kiosk_nonce()
        if nonce:
            if attempts > 1:
                logger.info("kiosk-runtime: nonce ready after %d attempts", attempts)
            return nonce
        _time.sleep(delay)
        delay = min(delay + 0.5, 5.0)
    logger.warning("kiosk-runtime: nonce mint never succeeded after %.0fs", timeout_s)
    return None


def _mint_kiosk_nonce() -> str | None:
    """Ask meeting-scribe for a single-use bootstrap nonce.

    Uses the in-process CLI helper that handles admin auth so we
    don't need to reimplement the password flow here. The helper
    returns the parsed JSON dict (or ``None`` on failure).
    """
    try:
        from meeting_scribe.cli._common import _api_request
    except Exception as exc:
        logger.warning("kiosk-runtime: _api_request import failed: %s", exc)
        return None
    try:
        parsed = _api_request("/api/admin/kiosk/mint-nonce", method="POST")
    except Exception as exc:
        logger.warning("kiosk-runtime: nonce mint HTTP failed: %s", exc)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed.get("nonce")


def _launch_chromium(nonce: str | None) -> subprocess.Popen | None:
    """Launch chromium in kiosk mode pointed at the bootstrap URL.

    Returns the Popen handle for the watchdog loop. ``None`` is
    returned when chromium isn't installed (the systemd unit will
    log a warning and respawn us; the bootstrap step should have
    installed chromium).
    """
    if not nonce:
        logger.warning("kiosk-runtime: launching chromium without nonce - splash only")
    _CHROMIUM_PROFILE.mkdir(parents=True, exist_ok=True)
    target_url = f"http://{_LOOPBACK_HOST}:{_LOOPBACK_PORT}/kiosk-bootstrap?nonce={nonce or ''}"
    cmd = [
        "chromium",
        "--kiosk",
        "--ozone-platform=wayland",
        "--enable-features=UseOzonePlatform",
        f"--user-data-dir={_CHROMIUM_PROFILE}",
        "--noerrdialogs",
        "--disable-session-crashed-bubble",
        "--disable-features=TranslateUI",
        "--disable-infobars",
        "--no-default-browser-check",
        f"--app={target_url}",
    ]
    try:
        return subprocess.Popen(cmd)
    except FileNotFoundError:
        logger.error("kiosk-runtime: chromium not installed (snap install chromium)")
        return None


def _shutdown(*_args: Any) -> None:
    """Signal handler: clean exit on SIGTERM (systemd) / SIGINT."""
    logger.info("kiosk-runtime: received shutdown signal")
    sys.exit(0)


# Shared activity timestamp - written by the WS subscriber thread,
# read by the DPMS idle-timer poll loop. monotonic seconds; protected
# only by GIL since the access is a single Python-level assignment.
_last_activity_ts: float = 0.0


def _ws_subscriber_thread(get_cookie: Any) -> None:
    """Background thread: subscribe to /api/ws/view and reset
    ``_last_activity_ts`` on every kiosk-allowlisted event.

    Resilient by design: any disconnect (server restart, transient
    network blip) reconnects with exponential back-off. Never raises;
    failures log + sleep + retry forever.

    The cookie is fetched lazily via ``get_cookie()`` so a meeting-scribe
    restart that rotates the boot-derived secret can re-mint without
    restarting the runtime.
    """
    import asyncio as _asyncio
    import json as _json

    try:
        import websockets  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("kiosk-runtime: 'websockets' not installed; DPMS idle timer disabled")
        return

    backoff_s = 1.0
    backoff_max_s = 30.0
    url = f"ws://{_LOOPBACK_HOST}:{_LOOPBACK_PORT}/api/ws/view"

    async def _run() -> None:
        nonlocal backoff_s
        while True:
            try:
                cookie = get_cookie()
                headers = [("Cookie", f"scribe_kiosk={cookie}")] if cookie else []
                async with websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                ) as ws:
                    backoff_s = 1.0  # connected; reset back-off
                    logger.info("kiosk-runtime: WS subscriber connected to %s", url)
                    global _last_activity_ts
                    _last_activity_ts = time.monotonic()
                    async for raw in ws:
                        try:
                            msg = _json.loads(raw)
                        except Exception:
                            continue
                        # Any event resets activity. Server-side role
                        # filter already drops operator-only events
                        # from the kiosk fan-out so anything we see is
                        # legitimate activity.
                        if isinstance(msg, dict):
                            _last_activity_ts = time.monotonic()
            except Exception as exc:
                logger.warning("kiosk-runtime: WS subscriber error: %s", exc)
            await _asyncio.sleep(backoff_s)
            backoff_s = min(backoff_s * 2.0, backoff_max_s)

    _asyncio.run(_run())


def _start_ws_subscriber(get_cookie: Any) -> None:
    """Spawn the WS subscriber thread (daemon=True so process exit
    doesn't block on its join)."""
    import threading

    t = threading.Thread(target=_ws_subscriber_thread, args=(get_cookie,), daemon=True)
    t.start()


def main() -> None:
    """The cage ExecStart payload.

    Single big loop: refresh HDMI status, apply persisted settings,
    keep chromium alive, react to settings file changes. On
    unrecoverable error we exit non-zero and systemd respawns us.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s kiosk: %(message)s")
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Initial enumerate + publish so the admin UI sees something even
    # before chromium has rendered.
    _publish_hdmi_status(_enumerate_hdmi_status())
    settings = _load_settings()
    _apply_settings(settings)

    # Block on the nonce mint so chromium never launches against an
    # unreachable server - if it did, the page would render a permanent
    # error and the splash would never appear. The blocking helper
    # retries with back-off until either the server is up or the
    # 120 s cap elapses (after which we launch anyway and let the
    # WS subscriber + watchdog patch up the visible state).
    nonce = _mint_kiosk_nonce_blocking(timeout_s=120.0)
    chromium = _launch_chromium(nonce)

    # WS subscriber thread keeps ``_last_activity_ts`` fresh whenever
    # the server fans out a kiosk-allowlisted event. The DPMS poll
    # loop below reads it; no shared lock needed (single 64-bit write,
    # GIL-atomic on CPython).
    def _mint_ws_cookie_noredirect() -> str | None:
        """Mint a kiosk cookie for the WS subscriber.

        Refuses to follow the 302 from /kiosk-bootstrap so we can read
        the Set-Cookie header directly. Returns the cookie value (the
        ``<issued>.<sid>.<hmac>`` string) on success, ``None`` on any
        failure - the WS subscriber retries with back-off.
        """
        n = _mint_kiosk_nonce()
        if not n:
            return None
        try:
            import urllib.request

            class _NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *_a, **_kw):
                    return None

            opener = urllib.request.build_opener(_NoRedirect())
            req = urllib.request.Request(
                f"http://{_LOOPBACK_HOST}:{_LOOPBACK_PORT}/kiosk-bootstrap?nonce={n}",
                method="GET",
            )
            try:
                resp = opener.open(req, timeout=5)
            except urllib.error.HTTPError as e:
                resp = e
            cookie_header = resp.headers.get("Set-Cookie", "") if hasattr(resp, "headers") else ""
            for piece in cookie_header.split(";"):
                if piece.strip().startswith("scribe_kiosk="):
                    return piece.strip().split("=", 1)[1]
        except Exception as exc:
            logger.warning("kiosk-runtime: cookie mint (noredirect) failed: %s", exc)
        return None

    _start_ws_subscriber(_mint_ws_cookie_noredirect)

    last_settings_mtime = _SETTINGS_PATH.stat().st_mtime if _SETTINGS_PATH.exists() else 0.0
    last_status_publish = 0.0
    poll_interval_s = 5.0

    while True:
        time.sleep(poll_interval_s)

        # Watchdog the chromium process.
        if chromium is not None and chromium.poll() is not None:
            logger.warning(
                "kiosk-runtime: chromium exited (rc=%s); relaunching", chromium.returncode
            )
            # Use the blocking mint so a chromium crash during a
            # meeting-scribe restart doesn't leave the next session
            # on an "invalid_nonce" error page. Shorter timeout here
            # because we want the watchdog to be responsive.
            chromium = _launch_chromium(_mint_kiosk_nonce_blocking(timeout_s=30.0))

        # Re-apply settings on file change (admin saved new mode etc.).
        try:
            mtime = _SETTINGS_PATH.stat().st_mtime if _SETTINGS_PATH.exists() else 0.0
        except OSError:
            mtime = 0.0
        if mtime != last_settings_mtime:
            last_settings_mtime = mtime
            settings = _load_settings()
            _apply_settings(settings)
            logger.info("kiosk-runtime: re-applied settings (mtime=%.3f)", mtime)

        # Republish the status snapshot every ~30 s so the admin UI's
        # status block stays fresh without a per-call wlr-randr probe.
        now = time.monotonic()
        if now - last_status_publish > 30:
            last_status_publish = now
            _publish_hdmi_status(_enumerate_hdmi_status())

        # DPMS sleep gating. ``hdmi_idle_sleep_minutes`` of 0 disables
        # sleep entirely. ``_last_activity_ts`` is maintained by the
        # WS subscriber thread; 0 means "never seen any event yet" so
        # we treat it as fresh activity to avoid sleeping on startup.
        sleep_minutes = int(settings.get("hdmi_idle_sleep_minutes", 0))
        if sleep_minutes > 0 and _last_activity_ts > 0:
            idle = now - _last_activity_ts
            if idle > sleep_minutes * 60:
                subprocess.run(
                    ["wlr-randr", "--output", "HDMI-A-1", "--off"], check=False, timeout=5
                )


if __name__ == "__main__":
    main()
