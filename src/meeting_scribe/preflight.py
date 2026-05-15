"""Boot-time and shutdown-time preflight checks for meeting-scribe.

This module exposes a three-phase check runner that proves the system
can run cleanly *before* we try to start (or reboot into) it.

The checks are split by blast radius and retry semantics:

* **Phase 0 — hard gates** (filesystem-only, cheap, non-retriable).
  Things that can't be fixed by waiting: the compose file won't parse,
  the unit file is broken, required CLI binaries are missing, the HF
  model cache has a dangling symlink. If any of these fail, the next
  boot would fail for the same reason no matter how many times systemd
  restarted us, so we exit with a reserved non-retriable code.

* **Phase 1 — remediation** (Docker-dependent, retriable). Requires the
  Docker daemon and can run mutating commands like
  ``docker compose up -d``. Also runs a *semantic* compose validation
  (``docker compose config -q`` against the real env file) which catches
  interpolation / profile / merged-config errors that a Phase 0 YAML
  parse cannot — this is the load-bearing "can we actually boot the
  stack cold?" check.

* **Phase 2 — live readiness** (I/O, concurrent, retriable). All four
  backend health endpoints polled concurrently against a single shared
  wall-clock deadline so a 420s budget is a total budget, not per-check.

The runner is invoked from three entry points:

* ``meeting-scribe preflight --precondition`` → Phase 0 only.
  Used by systemd ``ExecCondition``. Exit 64 → non-retriable fail,
  65 → ``BOOT_BLOCKED`` present. Both go into
  ``RestartPreventExitStatus`` so we don't thrash.
* ``meeting-scribe preflight --boot --wait N`` → Phase 1 + Phase 2.
  Used by systemd ``ExecStartPre``. Exit 1 → soft retriable failure.
* ``meeting-scribe preflight --manual`` → all three phases, prints a
  human-readable report. Operator health check; also the documented
  recovery path (clears ``BOOT_BLOCKED`` on all-green).
* ``meeting-scribe preflight --shutdown`` → static-only subset, called
  from ``ExecStopPost`` for an audit trail during real shutdowns.

See ``docs/preflight.md`` (WIP) for the full design; the short version:
the Phase 0 ``compose_file_parses`` check is a cheap prefilter, the
Phase 1 ``compose_cold_start_valid`` check is the one that actually
proves the compose file will render at next boot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reserved exit codes
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_SOFT_FAIL = 1  # Phase 1/2 retriable failure (e.g. Docker late).
EXIT_HARD_FAIL = 64  # Phase 0 failure other than blocker marker.
EXIT_BLOCKED = 65  # BOOT_BLOCKED file present. Non-retriable.

# ---------------------------------------------------------------------------
# State locations
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    """Return the persistent preflight state directory.

    Honours ``$XDG_STATE_HOME``; defaults to ``~/.local/state/meeting-scribe``.
    Created lazily on first write.
    """
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "meeting-scribe"


def boot_blocked_path() -> Path:
    return _state_dir() / "BOOT_BLOCKED"


def last_good_boot_path() -> Path:
    return _state_dir() / "last-good-boot.json"


def preflight_audit_path() -> Path:
    return _state_dir() / "preflight-shutdown.json"


# ---------------------------------------------------------------------------
# Check dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    phase: int
    duration_ms: int
    warn_only: bool = False
    # Phase-6 demo gate — true when the check requires real hardware
    # (SP325 USB speakerphone, the GB10's wifi radio, the configured
    # mic actively capturing). ``--skip-hardware`` short-circuits these
    # to passed=True with a "skipped" detail so the same gate can run
    # in CI without a connected device.
    hardware_required: bool = False


@dataclass
class CheckContext:
    """Shared state passed to every check during a single preflight run."""

    repo_root: Path
    compose_file: Path
    env_file: Path
    deadline: float  # monotonic time at which the whole run must finish
    compose_services: list[str] = field(default_factory=list)
    compose_images: list[str] = field(default_factory=list)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


# ---------------------------------------------------------------------------
# Phase 0 — hard gates (filesystem-only, cheap, non-retriable)
# ---------------------------------------------------------------------------


def _check_no_boot_blocker(ctx: CheckContext) -> CheckResult:
    t0 = time.monotonic()
    path = boot_blocked_path()
    if path.exists():
        reasons = path.read_text(errors="replace").strip()
        return CheckResult(
            name="no_boot_blocker",
            passed=False,
            detail=f"{path} exists: {reasons or '(no reason recorded)'}",
            phase=0,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
    return CheckResult(
        name="no_boot_blocker",
        passed=True,
        detail="no blocker marker",
        phase=0,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _check_required_binaries(ctx: CheckContext) -> CheckResult:
    t0 = time.monotonic()
    required = ("docker", "ip", "curl", "systemctl")
    missing = [b for b in required if shutil.which(b) is None]
    if missing:
        return CheckResult(
            name="required_binaries_on_path",
            passed=False,
            detail=f"missing on $PATH: {', '.join(missing)}",
            phase=0,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
    return CheckResult(
        name="required_binaries_on_path",
        passed=True,
        detail=f"found: {', '.join(required)}",
        phase=0,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _check_compose_file_parses(ctx: CheckContext) -> CheckResult:
    """Cheap prefilter: YAML-parse the compose file to catch catastrophic
    edit damage before we try to run Docker. **Not load-bearing.** Real
    semantic validation lives in Phase 1's ``compose_cold_start_valid``.
    """
    t0 = time.monotonic()
    try:
        import yaml  # type: ignore[import-untyped]  # PyYAML, already transitive
    except ImportError:
        return CheckResult(
            name="compose_file_parses",
            passed=False,
            detail="PyYAML not installed",
            phase=0,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    if not ctx.compose_file.exists():
        return CheckResult(
            name="compose_file_parses",
            passed=False,
            detail=f"compose file not found: {ctx.compose_file}",
            phase=0,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    try:
        data = yaml.safe_load(ctx.compose_file.read_text())
    except yaml.YAMLError as e:
        return CheckResult(
            name="compose_file_parses",
            passed=False,
            detail=f"{ctx.compose_file}: {e}",
            phase=0,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    if not isinstance(data, dict) or "services" not in data:
        return CheckResult(
            name="compose_file_parses",
            passed=False,
            detail=f"{ctx.compose_file}: missing top-level 'services' key",
            phase=0,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    return CheckResult(
        name="compose_file_parses",
        passed=True,
        detail=f"{len(data['services'])} services declared",
        phase=0,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Phase 1 — remediation (Docker-dependent, retriable)
# ---------------------------------------------------------------------------


def _check_docker_daemon(ctx: CheckContext) -> CheckResult:
    """Probe the Docker daemon. Retries for up to 30s so a slow-to-start
    daemon on early boot doesn't fail the gate.
    """
    t0 = time.monotonic()
    deadline = min(t0 + 30.0, ctx.deadline)
    last_err: str = ""
    while time.monotonic() < deadline:
        try:
            r = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if r.returncode == 0:
                return CheckResult(
                    name="docker_daemon",
                    passed=True,
                    detail="docker info OK",
                    phase=1,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
            last_err = r.stderr.strip() or r.stdout.strip() or "(no stderr)"
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            last_err = repr(e)
        time.sleep(1)
    return CheckResult(
        name="docker_daemon",
        passed=False,
        detail=f"docker info failed for >{int(deadline - t0)}s: {last_err}",
        phase=1,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _check_compose_cold_start_valid(ctx: CheckContext) -> CheckResult:
    """Semantic validation of the compose file against real env inputs.

    Runs ``docker compose -f <compose> --env-file <env> config -q`` — the
    same render Docker does when you run ``docker compose up -d``. Catches
    env interpolation errors, profile misuse, merged-config errors, and
    unresolvable service references even when the current warm containers
    would otherwise mask the problem. This is what makes "the shutdown
    gate proves the next boot will come up" an actual guarantee and not
    just wishful thinking.

    Also caches the parsed service / image list on ``ctx`` for later
    phase 1 checks, so ``docker_images_present`` and ``containers_up``
    work against the live Docker view, not the raw YAML.
    """
    t0 = time.monotonic()

    cmd = [
        "docker",
        "compose",
        "-f",
        str(ctx.compose_file),
    ]
    if ctx.env_file.exists():
        cmd += ["--env-file", str(ctx.env_file)]
    cmd += ["config", "--format", "json"]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            cwd=ctx.repo_root,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return CheckResult(
            name="compose_cold_start_valid",
            passed=False,
            detail=f"compose config invocation failed: {e!r}",
            phase=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    if r.returncode != 0:
        return CheckResult(
            name="compose_cold_start_valid",
            passed=False,
            detail=(
                "docker compose config returned "
                f"{r.returncode}: {(r.stderr or r.stdout).strip()[:400]}"
            ),
            phase=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    try:
        rendered = json.loads(r.stdout or "{}")
        services = rendered.get("services") or {}
        ctx.compose_services = sorted(services.keys())
        ctx.compose_images = sorted({s.get("image") for s in services.values() if s.get("image")})
    except json.JSONDecodeError:
        # Older compose prints YAML by default; that's fine — we can still
        # accept the file as valid, we just don't get the parsed service list.
        pass

    return CheckResult(
        name="compose_cold_start_valid",
        passed=True,
        detail=f"docker compose config -q OK ({len(ctx.compose_services)} services)",
        phase=1,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _check_docker_images_present(ctx: CheckContext) -> CheckResult:
    """Every image referenced by the compose file must exist locally.

    Uses the images list parsed by ``compose_cold_start_valid`` so we
    iterate over what Docker would actually try to pull on a cold start,
    not just hard-coded names.
    """
    t0 = time.monotonic()
    if not ctx.compose_images:
        return CheckResult(
            name="docker_images_present",
            passed=True,
            detail="no images to verify (compose produced empty list)",
            phase=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
            warn_only=True,
        )
    missing = []
    for image in ctx.compose_images:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if r.returncode != 0:
            missing.append(image)
    if missing:
        return CheckResult(
            name="docker_images_present",
            passed=False,
            detail=f"images not present locally: {', '.join(missing)}",
            phase=1,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
    return CheckResult(
        name="docker_images_present",
        passed=True,
        detail=f"{len(ctx.compose_images)} images present",
        phase=1,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Phase 2 — live readiness
# ---------------------------------------------------------------------------


async def _probe_one(url: str, timeout: float) -> tuple[bool, str]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def _check_backends_healthy(ctx: CheckContext) -> CheckResult:
    """Run all four backend ``/health`` probes concurrently against the
    shared deadline. A single slow backend no longer eats the whole
    budget — per-check timeout is clamped to ``min(per_check, remaining)``.
    """
    t0 = time.monotonic()
    endpoints = {
        "translation": "http://localhost:8010/health",
        "diarization": "http://localhost:8001/health",
        "tts": "http://localhost:8002/health",
        "asr": "http://localhost:8003/health",
    }
    per_check = min(5.0, ctx.remaining())

    results = await asyncio.gather(
        *(_probe_one(url, per_check) for url in endpoints.values()),
        return_exceptions=False,
    )

    details = []
    failed = []
    for (name, url), (ok, info) in zip(endpoints.items(), results):
        if ok:
            details.append(f"{name}=ok")
        else:
            failed.append(f"{name}({info})")

    if failed:
        return CheckResult(
            name="backends_healthy",
            passed=False,
            detail=f"unhealthy: {', '.join(failed)}; healthy: {', '.join(details) or 'none'}",
            phase=2,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )
    return CheckResult(
        name="backends_healthy",
        passed=True,
        detail=", ".join(details),
        phase=2,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PHASE_0_CHECKS: list = [
    _check_no_boot_blocker,
    _check_required_binaries,
    _check_compose_file_parses,
]

PHASE_1_CHECKS: list = [
    _check_docker_daemon,
    _check_compose_cold_start_valid,
    _check_docker_images_present,
]

PHASE_2_CHECKS: list = [
    # Phase 2 entries are async — called separately in run_live_phases.
    _check_backends_healthy,
]


def _default_repo_root() -> Path:
    # preflight.py lives at src/meeting_scribe/preflight.py — three parents up
    return Path(__file__).resolve().parents[2]


def make_context(wait_seconds: float) -> CheckContext:
    repo = _default_repo_root()
    return CheckContext(
        repo_root=repo,
        compose_file=repo / "docker-compose.gb10.yml",
        env_file=repo / ".env.gb10",
        deadline=time.monotonic() + wait_seconds,
    )


def run_phase_0(ctx: CheckContext) -> list[CheckResult]:
    return [check(ctx) for check in PHASE_0_CHECKS]


def run_phase_1(ctx: CheckContext) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in PHASE_1_CHECKS:
        res = check(ctx)
        results.append(res)
        # If docker_daemon fails, the rest of Phase 1 is meaningless — skip.
        if not res.passed and res.name == "docker_daemon":
            break
    return results


async def run_phase_2(ctx: CheckContext) -> list[CheckResult]:
    """Poll every Phase 2 check against the shared wall-clock deadline.

    Each check is retried every ``_PHASE2_POLL_INTERVAL`` seconds until it
    passes or ``ctx.remaining()`` hits zero. The final (worst) result per
    check is what we return — so ``--wait 420`` becomes a real 7-minute
    budget for the backend stack to come up, not a per-probe timeout.
    """
    results: list[CheckResult] = []
    for check in PHASE_2_CHECKS:
        res = await check(ctx)
        while not res.passed and ctx.remaining() > 0:
            await asyncio.sleep(min(_PHASE2_POLL_INTERVAL, ctx.remaining()))
            if ctx.remaining() <= 0:
                break
            res = await check(ctx)
        results.append(res)
    return results


_PHASE2_POLL_INTERVAL = 5.0


def classify_exit(results: list[CheckResult]) -> int:
    """Map a list of check results to a reserved exit code.

    ``BOOT_BLOCKED`` present → 65. Any Phase 0 hard failure → 64.
    Phase 1/2 failures → 1 (soft, retriable). Otherwise 0.
    """
    if any((not r.passed) and r.name == "no_boot_blocker" for r in results):
        return EXIT_BLOCKED
    if any((not r.passed) and r.phase == 0 and not r.warn_only for r in results):
        return EXIT_HARD_FAIL
    if any((not r.passed) and not r.warn_only for r in results):
        return EXIT_SOFT_FAIL
    return EXIT_OK


def format_report(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        status = "PASS" if r.passed else ("WARN" if r.warn_only else "FAIL")
        lines.append(f"  [{status}] phase{r.phase} {r.name} ({r.duration_ms}ms): {r.detail}")
    return "\n".join(lines)


async def run_all(ctx: CheckContext, *, skip_blocker: bool = False) -> list[CheckResult]:
    """Run every phase in order, short-circuiting on a hard Phase 0 failure.

    ``skip_blocker=True`` omits ``no_boot_blocker`` from Phase 0 — used by
    ``--manual`` / ``--seed`` / ``unblock`` which are the documented
    recovery paths *from* a blocker state and must not themselves be
    blocked by the marker they are meant to clear.
    """
    checks = PHASE_0_CHECKS
    if skip_blocker:
        checks = [c for c in PHASE_0_CHECKS if c is not _check_no_boot_blocker]
    results = [c(ctx) for c in checks]
    if any(not r.passed and r.phase == 0 and not r.warn_only for r in results):
        return results
    results += run_phase_1(ctx)
    # If Phase 1 hard-failed (e.g. docker_daemon), skip the live probes.
    if any(not r.passed and not r.warn_only for r in results if r.phase == 1):
        return results
    results += await run_phase_2(ctx)
    return results


def write_post_ready() -> None:
    """Called from the server lifespan immediately before sd_notify('READY=1').

    Writes a small ``last-good-boot`` audit record and clears any stale
    ``BOOT_BLOCKED`` marker. Safe to call multiple times.
    """
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "pid": os.getpid(),
    }
    tmp = d / "last-good-boot.json.tmp"
    final = d / "last-good-boot.json"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, final)
    blocker = boot_blocked_path()
    if blocker.exists():
        try:
            blocker.unlink()
            logger.info("Cleared stale BOOT_BLOCKED marker after successful boot")
        except OSError as e:
            logger.warning("Failed to clear BOOT_BLOCKED marker: %s", e)


def write_audit(results: list[CheckResult], *, mode: str) -> None:
    """Write a preflight-run snapshot to the state dir for later inspection."""
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "mode": mode,
        "results": [
            {
                "name": r.name,
                "phase": r.phase,
                "passed": r.passed,
                "warn_only": r.warn_only,
                "detail": r.detail,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ],
    }
    tmp = preflight_audit_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, preflight_audit_path())


def write_boot_blocked(reason: str) -> None:
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    boot_blocked_path().write_text(f"{time.ctime()}: {reason}\n")


# ---------------------------------------------------------------------------
# CLI entry points (invoked from cli.py)
# ---------------------------------------------------------------------------


def cmd_manual(wait_seconds: float) -> int:
    """Run every phase live. Human-readable report. Operator tool.

    Skips the ``no_boot_blocker`` check so an existing ``BOOT_BLOCKED``
    marker doesn't fail the check that's supposed to *clear* it. On
    all-green, the marker is removed as the documented recovery path.

    Exit 0 on all-green, non-zero on any hard failure. Intentionally
    tolerant of warnings.
    """
    ctx = make_context(wait_seconds)
    results = asyncio.run(run_all(ctx, skip_blocker=True))
    print("meeting-scribe preflight (manual)")
    print(format_report(results))
    code = classify_exit(results)
    write_audit(results, mode="manual")
    if code == EXIT_OK:
        print("Result: GREEN")
        blocker = boot_blocked_path()
        if blocker.exists():
            blocker.unlink(missing_ok=True)
            print(f"Cleared {blocker}")
    else:
        print(f"Result: FAIL (exit {code})")
    return code


def cmd_precondition() -> int:
    """Phase 0 only. Used by systemd ExecCondition. Reserved exit codes."""
    ctx = make_context(wait_seconds=10.0)
    results = run_phase_0(ctx)
    code = classify_exit(results)
    # Short, single-line output per check so journalctl shows the state
    # of the gate without spamming the journal.
    for r in results:
        tag = "PASS" if r.passed else ("WARN" if r.warn_only else "FAIL")
        print(f"[{tag}] {r.name}: {r.detail}", file=sys.stderr)
    return code


def cmd_boot(wait_seconds: float) -> int:
    """Phase 1 + Phase 2. Used by systemd ExecStartPre."""
    ctx = make_context(wait_seconds)
    results = run_phase_1(ctx)
    if not any(not r.passed and not r.warn_only for r in results):
        results += asyncio.run(run_phase_2(ctx))
    for r in results:
        tag = "PASS" if r.passed else ("WARN" if r.warn_only else "FAIL")
        print(f"[{tag}] phase{r.phase} {r.name}: {r.detail}", file=sys.stderr)
    return EXIT_OK if all(r.passed or r.warn_only for r in results) else EXIT_SOFT_FAIL


# ---------------------------------------------------------------------------
# Phase 6 — `preflight --mode=demo` checks
#
# These run AFTER Phase 0 + Phase 1 + Phase 2 (so the inference stack is
# known-healthy). They cover the hardware-attached pieces the operator
# cares about for a customer demo: mic actually capturing, SP325 wideband
# compliance, AP profile rotatable, hotspot live.
#
# Each carries ``hardware_required=True`` so ``--skip-hardware`` can
# short-circuit them for CI runs that don't have the GB10 connected.
# ---------------------------------------------------------------------------


async def _check_mic_bound_and_live(_ctx: CheckContext) -> CheckResult:
    """Mic liveness check for the demo gate.

    The CLI's preflight subcommand runs in a separate Python process
    from the meeting-scribe server, so the in-process
    ``state.server_mic_active`` / ``state.last_nonzero_audio_ts``
    primitives reflect an empty CLI state, not the running daemon.
    Query ``/api/status`` instead so the demo gate sees the actual
    audio pipeline state. Phase 5's in-server preflight uses the
    direct ``probe_mic_liveness`` helper; the contract is the same
    (probe-local epoch, non-zero sample required) — just observed
    through different surfaces.
    """
    t0 = time.monotonic()
    try:
        from meeting_scribe.cli._common import (
            UnauthenticatedRedirect,
            _api_request,
        )

        try:
            data = _api_request("/api/status")
        except UnauthenticatedRedirect as exc:
            return CheckResult(
                name="mic_bound_and_live",
                passed=False,
                detail=(
                    f"/api/status redirected to {exc.suffix} — admin auth "
                    "required for demo-gate mic check"
                ),
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )
        if not isinstance(data, dict):
            return CheckResult(
                name="mic_bound_and_live",
                passed=False,
                detail="could not reach /api/status — is the server running?",
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )

        route_status = data.get("audio_route_status") or "ok"
        if route_status != "ok":
            return CheckResult(
                name="mic_bound_and_live",
                passed=False,
                detail=(
                    f"audio_route_status='{route_status}' — resolve the "
                    "admin-notifications banner before the demo"
                ),
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )

        server_mic_live = bool(data.get("server_mic_active_live"))
        if not server_mic_live:
            return CheckResult(
                name="mic_bound_and_live",
                passed=False,
                detail=(
                    "server-side mic capture is not running — enable "
                    "mic_active in the audio routing admin panel"
                ),
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )

        # Capture is bound + flagged live. Check that the audio_chunks
        # metric is moving — two reads, short interval. A stuck capture
        # process (subprocess hung, ALSA buffer wedged) shows up here
        # as a frozen counter even though server_mic_active_live=True.
        chunks0 = data.get("metrics", {}).get("audio_chunks")
        await asyncio.sleep(0.6)
        data2 = _api_request("/api/status") or {}
        chunks1 = data2.get("metrics", {}).get("audio_chunks")
        advanced = isinstance(chunks0, int) and isinstance(chunks1, int) and chunks1 > chunks0
        if not advanced:
            return CheckResult(
                name="mic_bound_and_live",
                passed=False,
                detail=(
                    f"audio capture stuck — audio_chunks frozen at {chunks0!r} "
                    "across consecutive reads (capture subprocess may be hung)"
                ),
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )

        return CheckResult(
            name="mic_bound_and_live",
            passed=True,
            detail=(
                f"server_mic_active_live=true, audio_chunks advanced "
                f"{chunks1 - chunks0} between reads"
            ),
            phase=2,
            duration_ms=int((time.monotonic() - t0) * 1000),
            hardware_required=True,
        )
    except Exception as exc:
        return CheckResult(
            name="mic_bound_and_live",
            passed=False,
            detail=f"probe raised: {exc!s}",
            phase=2,
            duration_ms=int((time.monotonic() - t0) * 1000),
            hardware_required=True,
        )


async def _check_sp325_compliance(_ctx: CheckContext) -> CheckResult:
    """SP325 wideband-mode compliance check.

    PASS on ``pass``, pass-with-warn on ``warn`` (recordable but thin
    audio — NR toggles on via Dell Peripheral Manager), FAIL on
    ``fail``. Wraps ``speakerphone.compliance.probe_device`` so the
    gate uses the same code path as the on-box CLI
    (``meeting-scribe speakerphone compliance``).
    """
    t0 = time.monotonic()
    try:
        from meeting_scribe.audio.audio_routing import (
            SETTINGS_AUDIO_MEETING_MIC_NODE,
        )
        from meeting_scribe.server_support.settings_store import _load_settings_override
        from meeting_scribe.speakerphone import compliance as _comp

        mic_node = str(_load_settings_override().get(SETTINGS_AUDIO_MEETING_MIC_NODE) or "").strip()
        if not mic_node:
            return CheckResult(
                name="sp325_compliance_ok",
                passed=False,
                detail="no mic configured — can't probe SP325 wideband state",
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )
        report = _comp.probe_device(mic_node)
    except Exception as exc:
        return CheckResult(
            name="sp325_compliance_ok",
            passed=False,
            detail=f"compliance probe raised: {exc!s}",
            phase=2,
            duration_ms=int((time.monotonic() - t0) * 1000),
            hardware_required=True,
        )
    # ``probe_device`` returns a ``ComplianceResult`` dataclass; status is
    # one of "pass" / "warn" / "fail".
    status = getattr(report, "status", "fail")
    passed = status in {"pass", "warn"}
    return CheckResult(
        name="sp325_compliance_ok",
        passed=passed,
        detail=f"compliance status={status}: {getattr(report, 'reason', '') or 'see report'}",
        phase=2,
        duration_ms=int((time.monotonic() - t0) * 1000),
        warn_only=(status == "warn"),
        hardware_required=True,
    )


async def _check_ap_profile_valid(_ctx: CheckContext) -> CheckResult:
    """The DellDemo-AP nmcli profile must have ``key-mgmt`` set so
    per-meeting rotation doesn't fail (the 2026-05-14 demo failure).

    Phase 3.1 self-heals this at rotation time, but the demo gate
    flags it loudly so the operator knows to expect a one-time
    delete-recreate on the first meeting after setup."""
    t0 = time.monotonic()
    try:
        from meeting_scribe import wifi as _wifi

        proc = subprocess.run(
            [
                "sudo",
                "nmcli",
                "-t",
                "-f",
                "802-11-wireless-security.key-mgmt",
                "con",
                "show",
                _wifi.AP_CON_NAME,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return CheckResult(
            name="ap_profile_valid",
            passed=False,
            detail=f"nmcli probe raised: {exc!s}",
            phase=2,
            duration_ms=int((time.monotonic() - t0) * 1000),
            hardware_required=True,
        )
    out = (proc.stdout or "").strip()
    # ``-t`` terse format is ``field:value``; an empty value means the
    # property is unset, which is the broken state.
    key_mgmt = ""
    for line in out.splitlines():
        if line.startswith("802-11-wireless-security.key-mgmt:"):
            key_mgmt = line.split(":", 1)[1].strip()
            break
    passed = bool(key_mgmt)
    return CheckResult(
        name="ap_profile_valid",
        passed=passed,
        detail=(
            f"key-mgmt='{key_mgmt}'"
            if passed
            else "key-mgmt is unset — rotation will fail until the profile is recreated"
        ),
        phase=2,
        duration_ms=int((time.monotonic() - t0) * 1000),
        hardware_required=True,
    )


async def _check_hotspot_up(_ctx: CheckContext) -> CheckResult:
    """AP active AND live SSID matches the rotation target.

    Reads the cached rotation state Phase 3 records; falls through to
    a live nmcli probe when no rotation has happened in this process
    yet (e.g. demo-smoke called right after a fresh boot)."""
    t0 = time.monotonic()
    try:
        from meeting_scribe import wifi as _wifi
        from meeting_scribe.hotspot import ap_control

        snap = ap_control.get_last_rotation_state()
        if snap["last_rotation_at"] is not None:
            if not snap["rotation_ok"]:
                return CheckResult(
                    name="hotspot_up",
                    passed=False,
                    detail=(
                        f"last rotation ssid mismatch — target={snap['target_ssid']!r} "
                        f"live={snap['live_ssid']!r}"
                    ),
                    phase=2,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    hardware_required=True,
                )
            return CheckResult(
                name="hotspot_up",
                passed=True,
                detail=f"rotation_ok target={snap['target_ssid']!r}",
                phase=2,
                duration_ms=int((time.monotonic() - t0) * 1000),
                hardware_required=True,
            )
        active = _wifi._nmcli_ap_is_active(bypass_cache=True)
    except Exception as exc:
        return CheckResult(
            name="hotspot_up",
            passed=False,
            detail=f"nmcli probe raised: {exc!s}",
            phase=2,
            duration_ms=int((time.monotonic() - t0) * 1000),
            hardware_required=True,
        )
    return CheckResult(
        name="hotspot_up",
        passed=bool(active),
        detail="AP active" if active else "AP not active",
        phase=2,
        duration_ms=int((time.monotonic() - t0) * 1000),
        hardware_required=True,
    )


DEMO_CHECKS = (
    _check_mic_bound_and_live,
    _check_sp325_compliance,
    _check_ap_profile_valid,
    _check_hotspot_up,
)


async def run_demo_checks(ctx: CheckContext, *, skip_hardware: bool) -> list[CheckResult]:
    """Run the Phase 6 demo gate.

    ``skip_hardware=True`` short-circuits every ``hardware_required``
    check to ``passed=True, detail="skipped (--skip-hardware)"``. CI
    runs use this; the GB10 operator workflow does not.
    """
    results: list[CheckResult] = []
    for check in DEMO_CHECKS:
        if skip_hardware:
            results.append(
                CheckResult(
                    name=check.__name__.lstrip("_").removeprefix("check_"),
                    passed=True,
                    detail="skipped (--skip-hardware)",
                    phase=2,
                    duration_ms=0,
                    hardware_required=True,
                )
            )
            continue
        results.append(await check(ctx))
    return results


def cmd_demo(wait_seconds: float, *, skip_hardware: bool = False) -> int:
    """``preflight --mode=demo`` — comprehensive pre-demo gate.

    Runs Phase 0 + Phase 1 + Phase 2 (the regular preflight) AND the
    Phase 6 hardware checks (mic capturing, SP325 compliance, AP
    profile valid, hotspot up). ``meeting-scribe demo-smoke`` calls
    this before its end-to-end probe; ``scripts/ci_local.py`` calls
    it with ``--skip-hardware`` so CI can exercise the gate without
    a GB10.
    """
    ctx = make_context(wait_seconds)
    results = asyncio.run(run_all(ctx, skip_blocker=True))
    # If the regular preflight failed, demo can't possibly pass. Append
    # placeholders so the operator sees the demo checks were skipped
    # downstream of the regular failure.
    if not any((not r.passed) and not r.warn_only for r in results):
        results += asyncio.run(run_demo_checks(ctx, skip_hardware=skip_hardware))
    print("meeting-scribe preflight (demo)")
    print(format_report(results))
    code = classify_exit(results)
    write_audit(results, mode="demo")
    if code == EXIT_OK:
        print("Result: GREEN — ready for customer demo")
    else:
        print(f"Result: FAIL (exit {code}) — fix the above before starting a demo")
    return code


def cmd_shutdown() -> int:
    """Static-only subset. Called by ExecStopPost during real shutdowns.

    Runs Phase 0 checks that don't depend on Docker/network/user-session
    state. Writes the audit JSON but never blocks shutdown itself.
    """
    ctx = make_context(wait_seconds=10.0)
    # Skip no_boot_blocker — we're writing, not reading, the blocker.
    results = [c(ctx) for c in PHASE_0_CHECKS if c is not _check_no_boot_blocker]
    write_audit(results, mode="shutdown")
    for r in results:
        tag = "PASS" if r.passed else ("WARN" if r.warn_only else "FAIL")
        print(f"[{tag}] {r.name}: {r.detail}", file=sys.stderr)
    if any(not r.passed and not r.warn_only for r in results):
        write_boot_blocked(
            "static preflight failed at shutdown: "
            + "; ".join(r.detail for r in results if not r.passed)
        )
    return EXIT_OK  # never block shutdown
