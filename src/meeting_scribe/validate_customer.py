"""``meeting-scribe validate --customer-flow`` — post-install device check.

Runs *on the customer GB10* against a live install and exercises the
exact paths that broke in production today (PR #14):

* PPTX upload that hit ``AssertionError: python-multipart`` because
  the dep wasn't declared.
* WiFi/QR returning 503 because the AP's nmcli profile didn't exist.
* AP never associating because two independent kill switches
  (``rfkill`` + ``nmcli radio wifi``) needed to be off.
* /api/status latency spiking from per-call ``sudo nmcli`` PAM
  round-trips before sudoers.d was scoped.

Each phase is a pass/fail probe with a short detail string. The
report shape matches :mod:`meeting_scribe.validate` so the existing
JSON consumers (CI scripts, the diagnostics dashboard) don't need
any changes — they just see a new ``customer_flow`` mode.

The validator is **read-mostly**: the only state it changes is
starting a temporary meeting (immediately cancelled) so the AP
brings up. Anything else is observation only.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from meeting_scribe.config import ServerConfig
from meeting_scribe.validate import PhaseResult, ValidateReport, _print_phase

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_PPTX_FIXTURE = _FIXTURE_DIR / "test_slides.pptx"

# Per-call HTTP timeouts. Generous on /api/status because we want to
# *measure* its latency (not bail early), tight on most others.
_HTTP_FAST = httpx.Timeout(connect=2.0, read=3.0, write=2.0, pool=2.0)
_HTTP_SLOW = httpx.Timeout(connect=2.0, read=10.0, write=2.0, pool=2.0)
_HTTP_UPLOAD = httpx.Timeout(connect=2.0, read=30.0, write=30.0, pool=5.0)

# Latency budget for /api/status. The pre-fix customer device showed
# 3000+ms because every call did a sudo+nmcli PAM round-trip; with
# the cache + sudoers.d in place a healthy device sits well under
# 500ms. Tunable via env override for slower hardware.
_STATUS_BUDGET_MS = 500.0

# How long to wait for the AP to become broadcast-ready after meeting
# start. Real cold-boot can take 10-15s for the radio to associate
# and the captive portal to come up.
_AP_WAIT_S = 60.0

# meeting-scribe.service unit name.
_SCRIBE_SERVICE = "meeting-scribe.service"

# WiFi interface name. Best-effort lookup; if it fails the wifi
# phases skip rather than fail (a degraded install without wifi is a
# legitimate state on dev boxes that aren't running real meetings).
_WIFI_IFACE_DEFAULT = "wlP9s9"


def _server_url(config: ServerConfig) -> str:
    """Local URL we hit for HTTP probes. Assumes the validator runs
    on the same host as the server (the canonical deployment).

    The customer install runs an HTTPS listener on the management IP
    when ``certs/cert.pem`` exists (the standard ``meeting-scribe
    setup`` layout), so check that first; fall back to HTTP for dev
    boxes without TLS certs.
    """
    project_root = Path(__file__).resolve().parents[2]
    has_tls = (project_root / "certs" / "cert.pem").exists()
    scheme = "https" if has_tls else "http"
    return f"{scheme}://127.0.0.1:{config.port}"


def _signed_admin_cookie() -> tuple[str, str] | None:
    """Build a signed admin cookie from the on-disk secret.

    Returns ``(cookie_name, cookie_value)`` or ``None`` if the secret
    file is missing (degraded install — phases that need it will skip
    rather than fail). The validator runs on-device so we have
    filesystem access; building the cookie here is simpler than
    walking the bootstrap-OAuth flow.
    """
    try:
        from meeting_scribe.terminal.auth import AdminSecretStore, CookieSigner

        store = AdminSecretStore.load_or_create()
        signer = CookieSigner(store.secret)
        return signer.cookie_name, signer.issue()
    except Exception:
        return None


# ── Phase 1: meeting-scribe systemd unit is active ─────────────────


def _phase_systemd_unit() -> PhaseResult:
    t0 = time.monotonic()
    if shutil.which("systemctl") is None:
        return PhaseResult(
            name="systemd_unit",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail="systemctl not found",
        )

    proc = subprocess.run(
        ["systemctl", "--user", "is-active", _SCRIBE_SERVICE],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    state = proc.stdout.strip() or "unknown"

    if state == "active":
        return PhaseResult(
            name="systemd_unit",
            status="pass",
            elapsed_ms=elapsed_ms,
            detail=f"{_SCRIBE_SERVICE} active",
            metrics={"state": state},
        )

    return PhaseResult(
        name="systemd_unit",
        status="fail",
        elapsed_ms=elapsed_ms,
        detail=(
            f"{_SCRIBE_SERVICE} state={state} — install with "
            f"`meeting-scribe install-service` and check journalctl --user -u {_SCRIBE_SERVICE}"
        ),
        metrics={"state": state},
    )


# ── Phase 2: WiFi radio kill switches ──────────────────────────────


def _phase_wifi_radio() -> PhaseResult:
    """Both rfkill (kernel) and ``nmcli radio wifi`` (NetworkManager)
    must be enabled before NM can put the wlan interface in a usable
    state. Customer GB10 ships with both off; bootstrap clears them
    but a manual ``rfkill block`` or NM restart can re-arm them.
    """
    t0 = time.monotonic()
    rfkill_blocked: bool | None = None
    nmcli_state: str | None = None
    failures: list[str] = []

    if shutil.which("rfkill"):
        proc = subprocess.run(
            ["rfkill", "list", "wifi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0:
            rfkill_blocked = "Soft blocked: yes" in proc.stdout
            if rfkill_blocked:
                failures.append("rfkill: wifi soft-blocked (run `sudo rfkill unblock wifi`)")

    if shutil.which("nmcli"):
        proc = subprocess.run(
            ["nmcli", "radio", "wifi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode == 0:
            nmcli_state = proc.stdout.strip()
            if nmcli_state != "enabled":
                failures.append(f"nmcli radio wifi={nmcli_state} (run `sudo nmcli radio wifi on`)")

    elapsed_ms = (time.monotonic() - t0) * 1000
    metrics: dict[str, Any] = {"rfkill_blocked": rfkill_blocked, "nmcli_state": nmcli_state}

    if failures:
        return PhaseResult(
            name="wifi_radio",
            status="fail",
            elapsed_ms=elapsed_ms,
            detail="; ".join(failures),
            metrics=metrics,
        )
    return PhaseResult(
        name="wifi_radio",
        status="pass",
        elapsed_ms=elapsed_ms,
        detail=f"rfkill clear, nmcli radio={nmcli_state or 'n/a'}",
        metrics=metrics,
    )


# ── Phase 3: scoped sudoers.d is installed ─────────────────────────


def _phase_sudoers_d() -> PhaseResult:
    """``/etc/sudoers.d/meeting-scribe`` is what makes the per-call
    ``sudo nmcli`` cheap. Without it the running service falls back
    to PAM-on-every-call and ``/api/status`` blows past 3s.
    """
    t0 = time.monotonic()
    path = Path("/etc/sudoers.d/meeting-scribe")

    if not path.exists():
        return PhaseResult(
            name="sudoers_d",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=(
                f"{path} missing — re-run `sudo ./bootstrap.sh` so the scoped "
                "passwordless-sudo fragment lands"
            ),
        )

    # Don't try to read it (mode 0440, unreadable to non-root). Just
    # confirm the file is registered and has the right perms shape;
    # if the body were corrupt visudo would have refused and bootstrap
    # wouldn't have moved it into place.
    try:
        st = path.stat()
    except PermissionError:
        return PhaseResult(
            name="sudoers_d",
            status="pass",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=f"{path} present (mode unreadable as non-root, expected)",
        )

    elapsed_ms = (time.monotonic() - t0) * 1000
    return PhaseResult(
        name="sudoers_d",
        status="pass",
        elapsed_ms=elapsed_ms,
        detail=f"{path} present (mode {oct(st.st_mode & 0o777)})",
    )


# ── Phase 4: python-multipart is importable ────────────────────────


def _phase_multipart_dep() -> PhaseResult:
    """Catches the exact regression from PR #14: PPTX upload returned
    500 with ``AssertionError: python-multipart must be installed``
    because the dep wasn't pinned in pyproject.toml.
    """
    t0 = time.monotonic()
    try:
        import python_multipart  # noqa: F401

        return PhaseResult(
            name="multipart_dep",
            status="pass",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail="python-multipart importable",
        )
    except Exception as e:
        return PhaseResult(
            name="multipart_dep",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=(
                f"python-multipart import failed ({type(e).__name__}: {e}) — "
                "PPTX upload will 500 with form-parsing assertion. "
                "Re-run `pip install -e .` in the venv."
            ),
        )


# ── Phase 5: /api/status latency under budget ──────────────────────


async def _phase_status_latency(config: ServerConfig) -> PhaseResult:
    """Three back-to-back GET /api/status calls; assert p95 < budget.
    First call may pay the nmcli cache miss; the next two should hit
    the 5-second TTL cache and be fast.
    """
    t0 = time.monotonic()
    url = f"{_server_url(config)}/api/status"
    timings: list[float] = []

    try:
        async with httpx.AsyncClient(timeout=_HTTP_SLOW, verify=False) as c:
            for _ in range(3):
                tt = time.monotonic()
                r = await c.get(url)
                if r.status_code != 200:
                    return PhaseResult(
                        name="status_latency",
                        status="fail",
                        elapsed_ms=(time.monotonic() - t0) * 1000,
                        detail=f"GET /api/status returned HTTP {r.status_code}",
                    )
                timings.append((time.monotonic() - tt) * 1000)
    except httpx.HTTPError as e:
        return PhaseResult(
            name="status_latency",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=f"GET /api/status raised {type(e).__name__}: {e}",
        )

    elapsed_ms = (time.monotonic() - t0) * 1000
    p95 = sorted(timings)[-1]  # 3 samples → p95 == max
    metrics = {"samples_ms": [round(x, 1) for x in timings], "p95_ms": round(p95, 1)}

    if p95 > _STATUS_BUDGET_MS:
        return PhaseResult(
            name="status_latency",
            status="fail",
            elapsed_ms=elapsed_ms,
            detail=(
                f"p95={p95:.0f}ms > budget {_STATUS_BUDGET_MS:.0f}ms — "
                "check sudoers.d (each call is paying a PAM round-trip if missing) "
                "and the nmcli AP-state cache"
            ),
            metrics=metrics,
        )
    return PhaseResult(
        name="status_latency",
        status="pass",
        elapsed_ms=elapsed_ms,
        detail=f"p95={p95:.0f}ms (budget {_STATUS_BUDGET_MS:.0f}ms)",
        metrics=metrics,
    )


# ── Phase 6: PPTX upload through the live HTTP path ────────────────


async def _phase_slides_upload(config: ServerConfig) -> PhaseResult:
    """Real multipart POST to /api/meetings/{id}/slides/upload.

    A valid ``meeting_id`` is required — we start a temp meeting,
    upload the fixture, then cancel. Any non-500 response (200 with a
    deck_id, 503 with "slide processing unavailable", 400, 404, 401,
    etc.) is acceptable here — we only fail if the route returns 500
    *with the python-multipart assertion*, which is what we're
    actually guarding against.
    """
    t0 = time.monotonic()
    if not _PPTX_FIXTURE.exists():
        return PhaseResult(
            name="slides_upload",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=f"fixture {_PPTX_FIXTURE} missing — phase needs the test PPTX",
        )

    cookie = _signed_admin_cookie()
    if cookie is None:
        return PhaseResult(
            name="slides_upload",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail="admin secret unreadable; skipping protected endpoint check",
        )

    base = _server_url(config)
    cookies = {cookie[0]: cookie[1]}

    try:
        async with httpx.AsyncClient(timeout=_HTTP_UPLOAD, verify=False, cookies=cookies) as c:
            # 1. Start a temp meeting to get a meeting_id.
            start = await c.post(f"{base}/api/meeting/start", json={})
            if start.status_code != 200:
                return PhaseResult(
                    name="slides_upload",
                    status="fail",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    detail=(
                        f"could not start temp meeting (HTTP {start.status_code}): "
                        f"{start.text[:200]}"
                    ),
                )
            meeting_id = start.json().get("meeting_id")
            if not meeting_id:
                return PhaseResult(
                    name="slides_upload",
                    status="fail",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    detail="meeting/start returned 200 but no meeting_id",
                )

            try:
                # 2. POST the fixture.
                with _PPTX_FIXTURE.open("rb") as f:
                    upload = await c.post(
                        f"{base}/api/meetings/{meeting_id}/slides/upload",
                        files={
                            "file": (
                                "test_slides.pptx",
                                f.read(),
                                (
                                    "application/vnd.openxmlformats-officedocument."
                                    "presentationml.presentation"
                                ),
                            )
                        },
                    )

                # 3. Inspect the response. The specific failure we
                # guard against is the python-multipart assertion
                # surfaced as a 500 with that exact text.
                if upload.status_code == 500 and "python-multipart" in upload.text:
                    return PhaseResult(
                        name="slides_upload",
                        status="fail",
                        elapsed_ms=(time.monotonic() - t0) * 1000,
                        detail=(
                            "PPTX upload hit the python-multipart assertion — the "
                            "runtime dep is missing. Re-install with "
                            "`pip install -e .` inside the venv."
                        ),
                    )

                return PhaseResult(
                    name="slides_upload",
                    status="pass",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    detail=f"upload returned HTTP {upload.status_code} (no multipart error)",
                    metrics={"upload_status": upload.status_code},
                )
            finally:
                # 4. Cancel the temp meeting either way. No-op if the
                # canceller flag is wrong; we don't want to leak a
                # half-recorded session.
                try:
                    await c.post(f"{base}/api/meeting/cancel")
                except httpx.HTTPError:
                    pass
    except httpx.HTTPError as e:
        return PhaseResult(
            name="slides_upload",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=f"upload phase raised {type(e).__name__}: {e}",
        )


# ── Phase 7: Meeting → AP → QR end-to-end ──────────────────────────


async def _phase_meeting_qr(config: ServerConfig) -> PhaseResult:
    """Start a meeting, wait for the hotspot AP to come up, assert
    /api/meeting/wifi returns the QR payload, then cancel.

    Failure cases this catches:
    * Missing nmcli ``DellDemo-AP`` profile (rotation path's blind
      ``con modify`` pre-PR-#14).
    * ``wlP9s9:wifi:unavailable`` → ``con up`` returning ``No
      suitable device`` (rfkill / nmcli radio off).
    * /api/meeting/wifi returning 503 ``Hotspot not active`` after
      the wait deadline.
    """
    t0 = time.monotonic()
    cookie = _signed_admin_cookie()
    if cookie is None:
        return PhaseResult(
            name="meeting_qr",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail="admin secret unreadable; skipping protected endpoint check",
        )

    base = _server_url(config)
    cookies = {cookie[0]: cookie[1]}

    try:
        async with httpx.AsyncClient(timeout=_HTTP_FAST, verify=False, cookies=cookies) as c:
            # Cancel any leftover meeting from a previous run so we
            # start from a known state.
            try:
                await c.post(f"{base}/api/meeting/cancel")
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)

            start = await c.post(f"{base}/api/meeting/start", json={})
            if start.status_code != 200:
                return PhaseResult(
                    name="meeting_qr",
                    status="fail",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    detail=f"meeting/start HTTP {start.status_code}: {start.text[:200]}",
                )

            try:
                # Poll /api/meeting/wifi until it returns the QR or
                # we hit the deadline. The 503 is the canonical
                # "hotspot not yet active" response.
                deadline = time.monotonic() + _AP_WAIT_S
                last_status = 0
                last_body = ""
                while time.monotonic() < deadline:
                    r = await c.get(f"{base}/api/meeting/wifi")
                    last_status = r.status_code
                    last_body = r.text
                    if r.status_code == 200:
                        body = r.json()
                        ssid = body.get("ssid", "")
                        password = body.get("password", "")
                        qr_svg = body.get("wifi_qr_svg", "")
                        if ssid and password and "<svg" in qr_svg:
                            return PhaseResult(
                                name="meeting_qr",
                                status="pass",
                                elapsed_ms=(time.monotonic() - t0) * 1000,
                                detail=f"hotspot up: ssid={ssid!r}, QR SVG returned",
                                metrics={
                                    "ssid": ssid,
                                    "wait_s": round(time.monotonic() - t0, 1),
                                },
                            )
                    await asyncio.sleep(2.0)

                return PhaseResult(
                    name="meeting_qr",
                    status="fail",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    detail=(
                        f"hotspot did not come up within {_AP_WAIT_S:.0f}s; last "
                        f"/api/meeting/wifi: HTTP {last_status} {last_body[:150]}"
                    ),
                    metrics={"last_status": last_status},
                )
            finally:
                try:
                    await c.post(f"{base}/api/meeting/cancel")
                except httpx.HTTPError:
                    pass
    except httpx.HTTPError as e:
        return PhaseResult(
            name="meeting_qr",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000,
            detail=f"meeting_qr raised {type(e).__name__}: {e}",
        )


# ── Top-level runner ───────────────────────────────────────────────


async def run_customer_flow(*, json_only: bool = False) -> ValidateReport:
    """Run all customer-flow phases and return the report."""
    started_at = time.time()
    config = ServerConfig.from_env()
    phases: list[PhaseResult] = []

    # Synchronous, host-state phases first — fail fast before we
    # bother making HTTP calls.
    for sync_phase in (
        _phase_systemd_unit,
        _phase_wifi_radio,
        _phase_sudoers_d,
        _phase_multipart_dep,
    ):
        phases.append(sync_phase())
        if not json_only:
            _print_phase(phases[-1])

    # HTTP phases. Each one is self-contained (starts/cancels its
    # own meeting if needed), so reordering is safe.
    for async_phase in (
        _phase_status_latency,
        _phase_slides_upload,
        _phase_meeting_qr,
    ):
        phases.append(await async_phase(config))
        if not json_only:
            _print_phase(phases[-1])

    finished_at = time.time()
    report = ValidateReport(
        started_at=started_at,
        finished_at=finished_at,
        mode="customer-flow",
        hardware_class="gb10",
        phases=phases,
    )

    # Same diagnostics-dir persistence as the existing modes.
    try:
        diag_dir = config.meetings_dir.parent / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        out = diag_dir / f"validate-customer-flow-{ts}.json"
        out.write_text(json.dumps({**report.to_json(), "phases": [asdict(p) for p in phases]}))
    except Exception:
        pass

    return report
