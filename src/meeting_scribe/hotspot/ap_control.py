"""WiFi AP rotation orchestrator + captive-portal subprocess management.

The hotspot is the SSID-rotating AP that scribe brings up so guests
can join with their phones. This module owns:

* **Rotation guards** (``_AP_ROTATION_LOCK``,
  ``_LAST_ROTATED_MEETING_ID``, ``_reset_rotation_state_for_tests``).
* **Meeting-scoped lifecycle** (``_start_wifi_ap``, ``_stop_wifi_ap``).
* **Captive-portal subprocess** (``_start_captive_portal``,
  ``_stop_captive_portal``, ``_reap_orphan_captive_portals``,
  ``_write_hotspot_state``).

Primitives — nmcli helpers (``_wifi._run_nmcli_sync``, ``_parse_nmcli_fields``,
``_wifi._nmcli_read_live_ap_credentials``, ``_wifi._nmcli_ap_is_active``), the state-
file sync (``_wifi._write_hotspot_state_sync``), the firewall
(``_wifi._apply_meeting_firewall`` / ``_wifi._remove_firewall``) and AP-name /
hotspot-subnet constants — live in ``meeting_scribe.wifi`` and are
imported here. wifi.py is the canonical home per the wifi-port memory.

Pulled out of ``server.py`` so the lifecycle module can import
``_start_wifi_ap`` / ``_stop_wifi_ap`` at top level instead of via
lazy ``meeting_scribe.server`` imports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

from meeting_scribe import wifi as _wifi
from meeting_scribe.runtime import state
from meeting_scribe.server_support.regdomain import _ensure_regdomain
from meeting_scribe.server_support.settings_store import (
    _effective_regdomain,
    _is_dev_mode,
)

# Re-export wifi.py constants used in this module's hot path. Functions
# stay accessed via ``_wifi.X`` so monkeypatch on ``meeting_scribe.wifi.X``
# affects both ap_control's calls and wifi's own internal calls.
AP_CON_NAME = _wifi.AP_CON_NAME
AP_IP = _wifi.AP_IP
HOTSPOT_STATE_FILE = _wifi.HOTSPOT_STATE_FILE
_AP_ACTIVATION_POLL_INTERVAL = _wifi._AP_ACTIVATION_POLL_INTERVAL
_AP_ACTIVATION_WAIT_SECONDS = _wifi._AP_ACTIVATION_WAIT_SECONDS
_NMCLI_CON_DOWN_TIMEOUT = _wifi._NMCLI_CON_DOWN_TIMEOUT
_NMCLI_CON_MODIFY_TIMEOUT = _wifi._NMCLI_CON_MODIFY_TIMEOUT
_NMCLI_CON_UP_TIMEOUT = _wifi._NMCLI_CON_UP_TIMEOUT

logger = logging.getLogger(__name__)

# ── Rotation deduplication ──────────────────────────────────────────
#
# `_start_wifi_ap` is scheduled as a fire-and-forget task from BOTH
# `start_meeting` and `resume_meeting`. Without a dedup, each call
# rotates credentials afresh, which caused the production bug where the
# admin panel and the pop-out view displayed different SSIDs (each fetched
# `/api/meeting/wifi` between two rotations and cached a different answer).
#
# Guarantees:
#   - At most one rotation is in flight at any time (`_AP_ROTATION_LOCK`).
#   - At most one rotation per meeting_id (`_LAST_ROTATED_MEETING_ID`).
#   - A no-op call still reconciles the state file and starts the captive
#     portal + firewall so transient drift (e.g. state file clobbered by
#     an external process) still gets healed.
_AP_ROTATION_LOCK = asyncio.Lock()
_LAST_ROTATED_MEETING_ID: str | None = None


def _reset_rotation_state_for_tests() -> None:
    """Clear the rotation dedup state. Unit-test only."""
    global _LAST_ROTATED_MEETING_ID
    _LAST_ROTATED_MEETING_ID = None


async def _start_wifi_ap(meeting_id: str | None = None) -> None:
    """Rotate AP credentials, wait for NM to activate them, and sync state.

    Idempotent per ``meeting_id``: if called twice for the same meeting,
    the second call does NOT rotate credentials — it only re-syncs the
    state file and ensures the captive portal + firewall are applied. This
    prevents the consistency bug where admin view and pop-out view fetch
    ``/api/meeting/wifi`` between two rotations and display different SSIDs.

    The implementation is structured so that the state file in
    ``/tmp/meeting-hotspot.json`` can *never* drift from the live AP:

    1. Acquire rotation lock (serializes concurrent calls).
    2. If already rotated for this ``meeting_id`` AND the AP is still active,
       skip rotation but still reconcile state file + portal (no-op fast path).
    3. Otherwise: generate fresh credentials and push them into the NM profile.
    4. Bounce the connection (down → up). NOTE: ``nmcli con up`` can return
       non-zero (supplicant-timeout) even when NM subsequently auto-retries
       and succeeds — we do NOT treat the exit code as authoritative.
    5. **Poll** ``nmcli con show --active`` for up to 45s, catching the
       auto-retry window.
    6. Once the AP is active, read the live SSID/psk back from nmcli
       ``--show-secrets`` and write them to the state file. The state file
       is ALWAYS derived from the radio, never from in-memory values.
    7. Start the captive portal and apply the hotspot firewall. Idempotent.
    8. Record the ``meeting_id`` so the next call for the same meeting is
       a no-op.

    Fire-and-forget: this is scheduled via ``asyncio.create_task`` from
    ``start_meeting`` / ``resume_meeting``. All errors are logged; no
    exception escapes to the meeting-start flow.
    """
    import secrets

    global _LAST_ROTATED_MEETING_ID

    loop = asyncio.get_event_loop()

    async with _AP_ROTATION_LOCK:
        # Dev mode: skip SSID rotation entirely — keep the current AP
        # credentials so the hotspot network stays consistent across
        # meeting starts. Still reconcile state + portal + firewall.
        # If the AP isn't active yet, bring it up with existing credentials.
        if _is_dev_mode():
            logger.info(
                "Dev mode: skipping SSID rotation for meeting %s",
                meeting_id,
            )
            if not await loop.run_in_executor(None, _wifi._nmcli_ap_is_active):
                logger.info("Dev mode: AP not active, bringing up with existing credentials")
                await loop.run_in_executor(None, _ensure_regdomain)
                await loop.run_in_executor(
                    None,
                    lambda: _wifi._run_nmcli_sync(["con", "up", AP_CON_NAME], timeout=45),
                )
                deadline = asyncio.get_event_loop().time() + _AP_ACTIVATION_WAIT_SECONDS
                while asyncio.get_event_loop().time() < deadline:
                    if await loop.run_in_executor(None, _wifi._nmcli_ap_is_active):
                        break
                    await asyncio.sleep(_AP_ACTIVATION_POLL_INTERVAL)
            await loop.run_in_executor(None, _wifi._write_hotspot_state_sync)
            await _start_captive_portal()
            await _apply_hotspot_firewall()
            if meeting_id is not None:
                _LAST_ROTATED_MEETING_ID = meeting_id
            return

        # Step 0: make sure the WiFi regulatory domain is JP before we touch
        # the AP. If it drifted back to the default "world" domain (country 00),
        # the kernel would cap our 5 GHz TX power to a level where phones
        # can't associate. Done on EVERY rotation, not just once at boot,
        # because scans can silently reset the regdomain.
        if not await loop.run_in_executor(None, _ensure_regdomain):
            logger.error(
                "refusing to rotate AP: regulatory domain is not %s — "
                "a phone attempting to connect would fail at association",
                _effective_regdomain(),
            )
            return

        # Dedup fast path: same meeting already rotated and AP still active.
        if (
            meeting_id is not None
            and meeting_id == _LAST_ROTATED_MEETING_ID
            and await loop.run_in_executor(None, _wifi._nmcli_ap_is_active)
        ):
            logger.info(
                "WiFi AP rotation skipped (already rotated for meeting %s)",
                meeting_id,
            )
            # Still reconcile state + portal in case either drifted.
            await loop.run_in_executor(None, _wifi._write_hotspot_state_sync)
            await _start_captive_portal()
            await _apply_hotspot_firewall()
            return

        session_id = secrets.token_hex(2).upper()
        new_ssid = f"Dell Demo {session_id}"
        new_password = secrets.token_hex(4).upper()

        # First-run bootstrap: on a fresh customer GB10 the AP_CON_NAME
        # profile doesn't exist yet, so the rotation path's blind
        # ``nmcli con modify DellDemo-AP`` returns "unknown connection"
        # and the AP never comes up. Detect that case here and create
        # the profile via the full ``_bring_up_ap`` path with the
        # freshly-generated meeting credentials. Subsequent rotations
        # take the modify-and-bounce fast path below.
        if not await loop.run_in_executor(None, _wifi._nmcli_connection_exists):
            logger.info(
                "WiFi AP profile %r missing on this host — creating it via "
                "first-run bootstrap with the meeting's fresh credentials",
                AP_CON_NAME,
            )
            try:
                await _wifi._bring_up_ap(
                    _wifi.WifiConfig(
                        mode="meeting",
                        ssid=new_ssid,
                        password=new_password,
                        band=_wifi.DEFAULT_BAND,
                        channel=_wifi.DEFAULT_CHANNEL,
                        regdomain=_wifi._effective_regdomain(),
                        ap_ip=_wifi.AP_IP,
                    )
                )
            except Exception as exc:
                logger.error(
                    "WiFi AP first-run bootstrap failed: %s — guests won't see "
                    "the meeting's QR code until this is resolved",
                    exc,
                )
                return
            await loop.run_in_executor(None, _wifi._write_hotspot_state_sync)
            await _start_captive_portal()
            await _apply_hotspot_firewall()
            if meeting_id is not None:
                _LAST_ROTATED_MEETING_ID = meeting_id
            return

    def _rotate_profile_and_bounce() -> subprocess.CompletedProcess[str] | None:
        """Update the NM profile and bounce it. All sync to avoid race."""
        import time as _time

        modify = _wifi._run_nmcli_sync(
            [
                "con",
                "modify",
                AP_CON_NAME,
                "802-11-wireless.ssid",
                new_ssid,
                "802-11-wireless-security.psk",
                new_password,
            ],
            timeout=_NMCLI_CON_MODIFY_TIMEOUT,
        )
        if modify.returncode != 0:
            return modify

        _wifi._run_nmcli_sync(["con", "down", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT)
        _time.sleep(2)  # wifi driver needs a beat between down and up
        return _wifi._run_nmcli_sync(["con", "up", AP_CON_NAME], timeout=_NMCLI_CON_UP_TIMEOUT)

    # Step 1: rotate. Don't fail the whole flow if this raises — we still
    # want to reconcile in case the AP is running with stale credentials.
    try:
        rotate_result = await loop.run_in_executor(None, _rotate_profile_and_bounce)
        if rotate_result is None or rotate_result.returncode != 0:
            stderr = (rotate_result.stderr or "").strip()[:200] if rotate_result else ""
            logger.warning(
                "WiFi AP rotation nmcli failed (will still reconcile): ssid=%s err=%s",
                new_ssid,
                stderr or "<no stderr>",
            )
        else:
            logger.info("WiFi AP rotation submitted: ssid=%s", new_ssid)
    except Exception as exc:
        logger.warning("WiFi AP rotation raised (will still reconcile): %s", exc)

    # Step 2: poll for the AP to become active. NM may have auto-retried
    # even if our explicit `con up` timed out.
    deadline = asyncio.get_event_loop().time() + _AP_ACTIVATION_WAIT_SECONDS
    active = False
    while asyncio.get_event_loop().time() < deadline:
        if await loop.run_in_executor(None, _wifi._nmcli_ap_is_active):
            active = True
            break
        await asyncio.sleep(_AP_ACTIVATION_POLL_INTERVAL)

    if not active:
        logger.error(
            "WiFi AP did not become active within %ds — hotspot state file left unchanged",
            _AP_ACTIVATION_WAIT_SECONDS,
        )
        return

    # Step 3: reconcile state file from live nmcli. This is the authoritative
    # write — what's in the file now matches what the radio is broadcasting.
    if await loop.run_in_executor(None, _wifi._write_hotspot_state_sync):
        creds = await loop.run_in_executor(None, _wifi._nmcli_read_live_ap_credentials)
        live_ssid = creds[0] if creds else "<unknown>"
        logger.info("Hotspot state written from live nmcli: ssid=%s", live_ssid)
        if creds and creds[0] != new_ssid:
            logger.warning(
                "Live AP ssid %r does not match rotation target %r — "
                "rotation likely failed and NM served the previous profile",
                creds[0],
                new_ssid,
            )
    else:
        logger.error("Failed to sync hotspot state file after AP activation")

    # Step 4: captive portal + firewall. These are idempotent; calling them
    # after a successful reconciliation ensures clients can join whatever is
    # actually broadcasting.
    await _start_captive_portal()
    await _apply_hotspot_firewall()

    # Step 5: record the meeting_id so subsequent calls for the same meeting
    # are no-ops. Only set AFTER successful reconciliation — if rotation or
    # reconciliation failed above, we want a retry path to still attempt
    # rotation on the next call.
    if meeting_id is not None:
        _LAST_ROTATED_MEETING_ID = meeting_id


async def _write_hotspot_state() -> None:
    """Sync the hotspot state file from live nmcli.

    This is kept as an async wrapper around ``_wifi._write_hotspot_state_sync``
    for existing callers. New code should prefer the sync helper directly
    or call this from an async context.
    """
    loop = asyncio.get_event_loop()
    if await loop.run_in_executor(None, _wifi._write_hotspot_state_sync):
        logger.info("Hotspot state synced from live nmcli")
    else:
        logger.debug("Hotspot state sync: no change or failure")


SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
CAPTIVE_PID_80 = Path("/tmp/meeting-captive-80.pid")
# Stale PID file from the deleted port-443 TLS redirector. Kept as a
# constant only so _stop_captive_portal() can reap it on upgrade; nothing
# writes to it anymore. Safe to delete once all live hosts have cycled.
CAPTIVE_PID_443 = Path("/tmp/meeting-captive-443.pid")
# sddc-cli's captive portal PID file (port 80 only). We clean it up on
# stop so sddc-cli's redirector can't shadow ours when both writers are
# invoked in the same session.
SDDC_CLI_PORTAL_PID = Path("/tmp/meeting-captive-portal.pid")


async def _start_captive_portal() -> None:
    """Captive-portal lifecycle hook.

    The guest portal is **HTTP-only on port 80**. The in-process guest
    uvicorn binds ``{127.0.0.1,10.42.0.1}:80`` and serves every captive-
    portal probe route (``/hotspot-detect.html``, ``/generate_204``,
    ``/api/captive``, etc.) directly via the FastAPI routes registered
    above.

    There is **no TLS handler on port 443** and nothing is spawned
    here. HTTPS captive-portal MITM is dead since HSTS preload:
    apple.com, google.com, github.com, etc. cannot be intercepted with
    a self-signed cert because browsers refuse the click-through for
    HSTS-preloaded domains. Modern OS captive-portal detection uses
    HTTP probes (captive.apple.com/hotspot-detect.html,
    connectivitycheck.gstatic.com/generate_204, etc.) which hit port 80
    via the NAT REDIRECT rule in ``_apply_hotspot_firewall`` and work
    regardless. The firewall sends TCP RST on 443 so HTTPS attempts
    fail instantly and the OS falls back to its HTTP captive-portal
    detection.

    This function is kept as an explicit lifecycle seam (called by the
    hotspot bring-up flow) so future work can slot in — but today it
    just guarantees no stale 443 subprocess / PID files survive.
    """
    await _stop_captive_portal()


def _reap_orphan_captive_portals() -> None:
    """Synchronous best-effort reaping of any captive-portal subprocesses
    left over from a prior server instance that died hard.

    Called from ``main()`` BEFORE the guest uvicorn tries to bind
    127.0.0.1:80 / 10.42.0.1:80.  If the previous server was SIGKILL'd
    (OOM, docker rm -f, systemd force stop after cleanup timeout,
    autosre unload during a benchmark window), its shutdown path — which
    includes the async ``_stop_captive_portal`` — never runs and the
    ``captive-portal-80.py`` subprocess can survive holding port 80.
    The next startup then fails with ``OSError: [Errno 98] Address
    already in use`` at ``_make_tcp_socket("127.0.0.1", 80)``.

    This reaper mirrors ``_stop_captive_portal()`` but stays synchronous
    so it runs cleanly before the asyncio event loop starts.
    """
    import signal as _signal

    for pid_file in (CAPTIVE_PID_80, CAPTIVE_PID_443, SDDC_CLI_PORTAL_PID):
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                try:
                    os.kill(pid, _signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except Exception:
                pass
            pid_file.unlink(missing_ok=True)

    # Any un-PID-filed orphans.  Matches captive-portal-{80,443} and
    # sddc-cli's meeting-captive-portal.py.  `check=False` so an empty
    # match (no matching processes → pkill exit 1) is not an error.
    for pattern in ("captive-portal-[48]", "meeting-captive-portal.py"):
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError, subprocess.TimeoutExpired:
            pass


async def _stop_captive_portal() -> None:
    """Stop captive portal handlers.

    Cleans up BOTH meeting-scribe's own PID files (80 + 443) AND
    sddc-cli's PID file, because both writers bind port 80 and leaving
    the other writer's listener around would prevent us from rebinding.
    """
    import signal as _signal

    for pid_file in (CAPTIVE_PID_80, CAPTIVE_PID_443, SDDC_CLI_PORTAL_PID):
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                try:
                    os.kill(pid, _signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except Exception:
                pass
            pid_file.unlink(missing_ok=True)

    # Kill any orphaned captive portal processes (match script names)
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["pkill", "-f", "captive-portal-[48]"],
            capture_output=True,
            timeout=5,
            check=False,
        ),
    )
    # Also catch sddc-cli's script name (/tmp/meeting-captive-portal.py)
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["pkill", "-f", "meeting-captive-portal.py"],
            capture_output=True,
            timeout=5,
            check=False,
        ),
    )


async def _apply_hotspot_firewall() -> None:
    """Async wrapper around ``wifi._apply_meeting_firewall``.

    Same iptables rule set as ``meeting-scribe wifi up``, but reads the
    admin port from runtime ``state.config.port`` so the rule that REJECTs
    the admin HTTPS port from the hotspot subnet matches whatever port the
    server is actually bound to.
    """
    admin_port = state.config.port
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: _wifi._apply_meeting_firewall(admin_port)
    )


async def _remove_hotspot_firewall() -> None:
    """Async wrapper around ``wifi._remove_firewall``."""
    await asyncio.get_event_loop().run_in_executor(None, _wifi._remove_firewall)


async def _stop_wifi_ap() -> None:
    """Bring down the WiFi AP when a meeting ends.

    Stops the AP and captive portal. Firewall rules are intentionally
    left in place — they are harmless when no hotspot traffic exists
    and will be reused by the next meeting.
    """
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _wifi._run_nmcli_sync(
                ["con", "down", AP_CON_NAME], timeout=_NMCLI_CON_DOWN_TIMEOUT
            ),
        )
        HOTSPOT_STATE_FILE.unlink(missing_ok=True)
        await _stop_captive_portal()
        logger.info("WiFi AP + captive portal stopped (firewall rules retained)")
    except Exception as e:
        logger.debug("WiFi AP stop skipped: %s", e)
