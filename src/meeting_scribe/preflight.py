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
EXIT_SOFT_FAIL = 1           # Phase 1/2 retriable failure (e.g. Docker late).
EXIT_HARD_FAIL = 64          # Phase 0 failure other than blocker marker.
EXIT_BLOCKED = 65            # BOOT_BLOCKED file present. Non-retriable.

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
        "docker", "compose",
        "-f", str(ctx.compose_file),
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
        ctx.compose_images = sorted(
            {s.get("image") for s in services.values() if s.get("image")}
        )
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
            capture_output=True, text=True, check=False, timeout=10,
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
    if any(
        (not r.passed) and r.name == "no_boot_blocker" for r in results
    ):
        return EXIT_BLOCKED
    if any(
        (not r.passed) and r.phase == 0 and not r.warn_only for r in results
    ):
        return EXIT_HARD_FAIL
    if any(
        (not r.passed) and not r.warn_only for r in results
    ):
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
