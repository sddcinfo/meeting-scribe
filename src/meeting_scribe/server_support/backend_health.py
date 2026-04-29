"""Backend-failure tracking, container auto-restart, and deep health probes.

Two layers of health surveillance live here:

1. **Failure-counter / degraded flag**. Every backend call passes
   through ``_record_backend_failure`` / ``_record_backend_success``;
   after ``BACKEND_FAILURE_THRESHOLD`` consecutive failures the backend
   is marked "degraded" so ``/api/status`` can render a red dot even
   when the in-process backend object is alive but every request is
   failing (typical CUDA-context corruption pattern).

2. **Container auto-restart**. A cooldown-bounded ``docker restart``
   issued by the server when a backend is degraded. Skipped for
   blacklisted containers whose model load takes minutes
   (``scribe-asr``, ``autosre-vllm-local``).

3. **Deep health probes** that hit each backend container's ``/health``
   or ``/v1/models`` endpoint, with a 30 s cache so ``/api/status`` polls
   don't burst-fire concurrent TLS probes during a meeting.

Pulled out of ``server.py`` so the upcoming ``routes/status.py`` and
``runtime/health.py`` modules can both reach these without circling
back through the server module.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import httpx

from meeting_scribe.runtime import state

logger = logging.getLogger(__name__)


# ── Failure tracking ────────────────────────────────────────────
_backend_failure_counts: dict[str, int] = {"asr": 0, "diarize": 0, "translate": 0}
_backend_last_errors: dict[str, str] = {}
BACKEND_FAILURE_THRESHOLD = 3  # 3 consecutive failures → mark degraded


def _record_backend_failure(name: str, err: str) -> None:
    _backend_failure_counts[name] = _backend_failure_counts.get(name, 0) + 1
    _backend_last_errors[name] = err[:200]
    if _backend_failure_counts[name] >= BACKEND_FAILURE_THRESHOLD:
        logger.error(
            "Backend '%s' marked degraded after %d consecutive failures: %s",
            name,
            _backend_failure_counts[name],
            err[:200],
        )


def _record_backend_success(name: str) -> None:
    if _backend_failure_counts.get(name, 0) > 0:
        logger.info("Backend '%s' recovered after %d failures", name, _backend_failure_counts[name])
    _backend_failure_counts[name] = 0
    _backend_last_errors.pop(name, None)


def _backend_is_degraded(name: str) -> bool:
    return _backend_failure_counts.get(name, 0) >= BACKEND_FAILURE_THRESHOLD


# ── Container auto-restart ──────────────────────────────────────
#
# When a container's CUDA context is corrupted, only a restart clears
# it. This is a second line of defense behind Docker's autoheal sidecar.
_container_restart_cooldown: dict[str, float] = {}
_RESTART_COOLDOWN_S = 120.0
# Never auto-restart vLLM containers — model reload takes 3+ minutes.
_RESTART_BLACKLIST = frozenset({"scribe-asr", "autosre-vllm-local"})


async def _restart_container(container_name: str) -> bool:
    """Restart a Docker container if not on cooldown. Returns True if issued."""
    if container_name in _RESTART_BLACKLIST:
        logger.warning(
            "Skipping auto-restart of %s (blacklisted — slow model load)",
            container_name,
        )
        return False

    now = time.monotonic()
    last = _container_restart_cooldown.get(container_name, 0)
    if now - last < _RESTART_COOLDOWN_S:
        return False

    _container_restart_cooldown[container_name] = now
    logger.warning("Auto-restarting degraded container: %s", container_name)

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "restart",
            container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            logger.info("Container %s restart issued successfully", container_name)
            return True
        logger.error(
            "Container %s restart failed (rc=%d): %s",
            container_name,
            proc.returncode,
            stderr.decode()[:200],
        )
    except Exception as e:
        logger.error("Failed to restart container %s: %s", container_name, e)
    return False


# ── Deep backend health (verifies real inference, not just object existence) ──
#
# Each backend is probed with a real lightweight request to confirm it
# can actually serve work. Results are cached to avoid burning resources
# on every ``/api/status`` call.
_deep_health_cache: dict[str, tuple[float, dict]] = {}
# Stale-while-revalidate: after this many seconds the entry is "stale"
# and a background refresh is fired, but the stale value is returned
# immediately so /api/status never blocks on backend probes after the
# first cold-start request. This eliminates the 2 s wall-clock spike
# that fired on the first poll past the previous 30 s TTL while the
# TTS/translate vLLMs were busy.
_DEEP_HEALTH_TTL = 30.0
# In-flight refresh dedup: a backend name is in this set while its
# background refresh task runs, so concurrent /api/status callers
# don't fire N parallel probes against the same endpoint.
_deep_health_refreshing: set[str] = set()


async def _deep_check_asr() -> dict:
    """Actually call the ASR vLLM endpoint — catches model-loading/crashed state."""
    if state.asr_backend is None:
        return {"ready": False, "detail": "not initialized"}
    if getattr(state.asr_backend, "degraded", False):
        return {"ready": False, "detail": getattr(state.asr_backend, "last_error", "degraded")}
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{state.config.asr_vllm_url}/v1/models")
            if r.status_code != 200:
                return {"ready": False, "detail": f"/v1/models HTTP {r.status_code}"}
            data = r.json().get("data", [])
            if not data:
                return {"ready": False, "detail": "no models loaded"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_translate() -> dict:
    """Actually call the translate vLLM endpoint — catches model loading."""
    if state.translate_backend is None:
        return {"ready": False, "detail": "not initialized"}
    # Reject if the retry counter has been tripping
    if _backend_is_degraded("translate"):
        return {"ready": False, "detail": _backend_last_errors.get("translate", "degraded")}
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{state.config.translate_vllm_url}/v1/models")
            if r.status_code != 200:
                return {"ready": False, "detail": f"/v1/models HTTP {r.status_code}"}
            data = r.json().get("data", [])
            if not data:
                return {"ready": False, "detail": "no models loaded"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_diarize() -> dict:
    """Actually call the diarize container /health — catches CUDA corruption."""
    if state.diarize_backend is None:
        return {"ready": False, "detail": "not initialized"}
    if getattr(state.diarize_backend, "degraded", False):
        return {
            "ready": False,
            "detail": getattr(state.diarize_backend, "last_error", None) or "degraded",
        }
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{state.config.diarize_url}/health")
            if r.status_code != 200:
                return {"ready": False, "detail": f"HTTP {r.status_code}"}
            body = r.json()
            if body.get("status") != "ok":
                return {"ready": False, "detail": str(body)[:80]}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_tts() -> dict:
    """Actually call the TTS container /health — catches CUDA corruption."""
    if state.tts_backend is None:
        return {"ready": False, "detail": "not initialized"}
    if getattr(state.tts_backend, "degraded", False):
        return {
            "ready": False,
            "detail": getattr(state.tts_backend, "last_error", None) or "degraded",
        }
    url = (getattr(state.tts_backend, "_vllm_url", None) or state.config.tts_vllm_url or "").strip()
    if not url:
        return {"ready": False, "detail": "no TTS endpoint configured"}

    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            try:
                r = await c.get(f"{url}/health")
            except Exception as e:
                return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:60]}"}
            if r.status_code != 200:
                return {"ready": False, "detail": f"HTTP {r.status_code}"}
            # /v1/models is informational only. The faster-qwen3-tts server
            # returns 404 because it's not a full OpenAI-compatible endpoint
            # — that is expected and must NOT surface as "TTS not ready"
            # in /api/status.
            body = None
            with suppress(Exception):
                body = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else None
                )
            if isinstance(body, dict):
                status = body.get("status", "unknown")
                if status and status != "healthy":
                    return {"ready": False, "detail": f"health status={status}"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _deep_check_furigana() -> dict:
    """Verify pykakasi is loaded and produces ruby output for a kanji probe."""
    if state.furigana_backend is None:
        return {"ready": False, "detail": "not initialized (pykakasi import failed?)"}
    if getattr(state.furigana_backend, "_kks", None) is None:
        return {"ready": False, "detail": "pykakasi dictionary not loaded"}
    try:
        html = await state.furigana_backend.annotate("会議")
        if not html or "<ruby>" not in html:
            return {"ready": False, "detail": "probe returned no ruby markup"}
        return {"ready": True, "detail": None}
    except Exception as e:
        return {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}


async def _refresh_deep_health(name: str, fn: Callable[[], Awaitable[dict]]) -> None:
    """Background refresh for a single backend probe. Updates the cache atomically."""
    try:
        result = await fn()
    except Exception as e:
        result = {"ready": False, "detail": f"{type(e).__name__}: {str(e)[:80]}"}
    _deep_health_cache[name] = (time.monotonic(), result)
    _deep_health_refreshing.discard(name)


async def _deep_backend_health(force: bool = False) -> dict[str, dict]:
    """Stale-while-revalidate cache over per-backend health probes.

    Returns ``{backend_name: {"ready": bool, "detail": str|None}}``.

    Cold start (no cached entry) blocks on the probe so callers see real
    data. Subsequent calls return the cached value immediately; if the
    entry is older than ``_DEEP_HEALTH_TTL`` a background refresh is
    fired (deduped via ``_deep_health_refreshing``).

    ``force=True`` bypasses the cache entirely (e.g. ``/api/meeting/start``
    where staleness would let a meeting start against a dead backend).
    """
    now = time.monotonic()
    results: dict[str, dict] = {}
    cold_start: list[tuple[str, Callable[[], Awaitable[dict]]]] = []
    checks = {
        "asr": _deep_check_asr,
        "translate": _deep_check_translate,
        "diarize": _deep_check_diarize,
        "tts": _deep_check_tts,
        "furigana": _deep_check_furigana,
    }
    for name, fn in checks.items():
        if force:
            cold_start.append((name, fn))
            continue
        cached = _deep_health_cache.get(name)
        if cached is None:
            cold_start.append((name, fn))
            continue
        ts, value = cached
        results[name] = value
        if now - ts >= _DEEP_HEALTH_TTL and name not in _deep_health_refreshing:
            _deep_health_refreshing.add(name)
            asyncio.create_task(_refresh_deep_health(name, fn))

    if cold_start:
        fresh = await asyncio.gather(*[fn() for _, fn in cold_start])
        for (name, _), result in zip(cold_start, fresh):
            _deep_health_cache[name] = (now, result)
            results[name] = result

    return results
