"""Server status, language list, and Prometheus metrics endpoints.

Three read-only routes that the admin UI polls (``/api/status``
~10 s, ``/api/languages`` once on load) plus the Prometheus
``/metrics`` text-format scrape endpoint.

``_probe_vllm_status`` lives here too because nothing else uses it
— it's the per-backend container-loading-progress probe that
``/api/status`` calls when a backend is reported "not ready" via
the deep-health pass.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess

import fastapi
import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from meeting_scribe.runtime import state
from meeting_scribe.server_support.backend_health import _deep_backend_health
from meeting_scribe.server_support.request_scope import _is_hotspot_client
from meeting_scribe.server_support.settings_store import _is_dev_mode

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/languages")
async def get_languages() -> JSONResponse:
    """Return the selectable language list + default pair for the setup UI."""
    from meeting_scribe.languages import to_api_response

    return JSONResponse(to_api_response())


async def _probe_vllm_status(url: str) -> dict:
    """Probe a vLLM endpoint for detailed loading status.

    Returns: {"ready": bool, "status": str, "detail": str|None}
    """
    base = url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/health")
            if r.status_code == 200:
                # Healthy — try to get model name
                try:
                    mr = await client.get(f"{base}/v1/models")
                    if mr.status_code == 200:
                        models = mr.json().get("data", [])
                        model_id = models[0]["id"] if models else "unknown"
                        return {"ready": True, "status": "active", "detail": model_id}
                except Exception:
                    pass
                return {"ready": True, "status": "active", "detail": None}
    except Exception:
        pass

    # Not healthy — check if container is running and parse loading progress
    port = base.split(":")[-1].split("/")[0]
    container_map = {
        "8010": "autosre-vllm-local",
        "8003": "scribe-asr",
        "8002": "scribe-tts",
        "8001": "scribe-diarization",
    }
    container = container_map.get(port)
    if not container:
        return {"ready": False, "status": "down", "detail": None}

    try:
        ps = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if ps.returncode != 0:
            return {"ready": False, "status": "not started", "detail": "Container not found"}

        docker_state = ps.stdout.strip()
        if docker_state == "restarting":
            return {"ready": False, "status": "restarting", "detail": "Container restarting"}
        if docker_state != "running":
            return {"ready": False, "status": docker_state, "detail": None}

        # Container running but health check failed — parse logs for loading progress
        logs = subprocess.run(
            ["docker", "logs", container, "--tail", "50"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        log_text = (logs.stdout or "") + (logs.stderr or "")

        # vLLM loading stages (same as autosre TUI)
        stages = [
            ("Application startup complete", "Starting API..."),
            ("Available KV cache memory", "Allocating KV cache..."),
            ("Model loading took", "Profiling..."),
            ("Initializing", "Initializing engine..."),
            ("non-default args", "Initializing engine..."),
            ("INT8 LM Head", "Applying patches..."),
            ("Starting vLLM", "Starting vLLM..."),
        ]

        # Weight loading progress
        matches = re.findall(
            r"Loading.*safetensors.*?(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)", log_text
        )
        if matches:
            pct, done, total = matches[-1]
            return {
                "ready": False,
                "status": "loading",
                "detail": f"Loading weights: {pct}% ({done}/{total})",
            }

        for marker, msg in stages:
            if marker in log_text:
                return {"ready": False, "status": "loading", "detail": msg}

        return {"ready": False, "status": "loading", "detail": "Starting container..."}
    except Exception:
        return {"ready": False, "status": "unknown", "detail": None}


@router.get("/metrics")
async def get_metrics() -> fastapi.Response:
    """Prometheus-compatible text metrics [Phase 2 optional].

    Exposes the same TTS counters and latency histograms as ``/api/status``
    but in a scrape-friendly text format. No prometheus_client dependency —
    the format is trivial and stable. Gauges only (no histograms); the
    rolling percentiles are exposed as separate gauges because the
    underlying windows are bounded deques, not Prometheus histograms.
    """
    lines: list[str] = []

    def _g(name: str, value: float | int, help_text: str = "") -> None:
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    def _pct(prefix: str, d: dict) -> None:
        sc = d.get("sample_count", 0)
        _g(f"{prefix}_sample_count", sc)
        for q in ("p50", "p95", "p99"):
            v = d.get(q)
            if v is not None:
                _g(f"{prefix}_{q}", v)

    snap = state.metrics.to_dict()
    tts = snap.get("tts", {})
    listener = snap.get("listener", {})
    loop_lag = snap.get("loop_lag_ms", {})

    _g("scribe_meeting_elapsed_s", snap.get("elapsed_s", 0))
    _g("scribe_asr_events_total", snap.get("asr_events", 0))
    _g("scribe_translations_completed_total", snap.get("translations_completed", 0))
    _g("scribe_translations_failed_total", snap.get("translations_failed", 0))

    _g("scribe_tts_queue_depth", tts.get("queue_depth", 0))
    _g("scribe_tts_queue_maxsize", tts.get("queue_maxsize", 0))
    _g("scribe_tts_workers_busy", tts.get("workers_busy", 0))
    _g("scribe_tts_container_concurrency", tts.get("container_concurrency", 0))
    _g("scribe_tts_submitted_total", tts.get("submitted", 0))
    _g("scribe_tts_delivered_total", tts.get("delivered", 0))
    _g("scribe_tts_timeouts_total", tts.get("timeouts", 0))
    _g("scribe_tts_oldest_inflight_age_ms", tts.get("oldest_inflight_age_ms", 0))
    for k, v in (tts.get("drops") or {}).items():
        _g(f"scribe_tts_drops_{k}_total", v)
    _pct("scribe_tts_synth_ms", tts.get("synth_ms") or {})
    _pct("scribe_tts_upstream_lag_ms", tts.get("upstream_lag_ms") or {})
    _pct("scribe_tts_post_translation_lag_ms", tts.get("tts_post_translation_lag_ms") or {})
    _pct("scribe_tts_end_to_end_lag_ms", tts.get("end_to_end_lag_ms") or {})

    health = {"healthy": 0, "degraded": 1, "stalled": 2}.get(tts.get("health", "healthy"), 0)
    _g("scribe_tts_health_state", health, help_text="0=healthy 1=degraded 2=stalled")

    _g("scribe_listener_connected", listener.get("connected", 0))
    _g("scribe_listener_deliveries_total", listener.get("deliveries", 0))
    _g("scribe_listener_send_failed_total", listener.get("send_failed", 0))
    _pct("scribe_listener_send_ms", listener.get("send_ms") or {})

    _pct("scribe_loop_lag_ms", loop_lag)

    crash = snap.get("crash")
    _g("scribe_crash_state", 1 if crash else 0)

    return fastapi.Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


@router.get("/api/status")
async def get_status(request: fastapi.Request) -> JSONResponse:
    """Current server and meeting status."""
    from meeting_scribe.gpu_monitor import get_system_resources, get_vram_usage

    gpu_data = None
    vram = get_vram_usage()
    if vram:
        gpu_data = {
            "vram_used_mb": vram.used_mb,
            "vram_total_mb": vram.total_mb,
            "vram_pct": round(vram.pct, 1),
        }

    # ROOT CAUSE, 2026-04-15: `get_system_resources()` internally calls
    # `subprocess.run(["docker", "stats", "--no-stream", ...], timeout=8)`
    # to populate container state.metrics. `docker stats` blocks 1.5–2.5 s even
    # with --no-stream because the daemon still samples every container
    # once. The previous code called this SYNCHRONOUSLY from this async
    # handler, freezing the entire event loop for the duration of the
    # subprocess. Under a 10-sec admin poll, that showed up as clockwork
    # 2.5 s event-loop stalls every 10–12 s — which was what was killing
    # the hotspot listener's WebSocket ping. Running the whole thing in
    # a thread via run_in_executor means the loop stays responsive; the
    # existing 5 s sys + 10 s container caches still limit how often we
    # actually shell out to `docker`.
    sys_res = await asyncio.get_event_loop().run_in_executor(None, get_system_resources)
    system_data = None
    if sys_res:
        system_data = {
            "cpu_pct": sys_res.cpu_pct,
            "mem_used_mb": sys_res.mem_used_mb,
            "mem_total_mb": sys_res.mem_total_mb,
            "mem_pct": sys_res.mem_pct,
            "load": [sys_res.load_1m, sys_res.load_5m, sys_res.load_15m],
            "uptime_s": sys_res.uptime_s,
            "containers": sys_res.containers,
        }

    # DEEP backend health — every "ready" status reflects an actual live
    # inference check (or a recent cached one), not just "the Python object
    # exists in memory". Without this, /api/status used to lie: if the
    # backend object was constructed but its container hit CUDA errors,
    # we'd still report "active".
    deep = await _deep_backend_health()

    def _backend_extra(obj) -> dict:
        """Extract model/url/failures from a backend object for status."""
        extra: dict = {}
        for attr in ("_model", "_vllm_model", "_model_name"):
            m = getattr(obj, attr, None)
            if m:
                extra["model"] = m
                break
        for attr in ("_base_url", "_vllm_url", "_url"):
            u = getattr(obj, attr, None)
            if u:
                extra["url"] = str(u)
                break
        cf = getattr(obj, "_consecutive_failures", None)
        if cf is not None:
            extra["consecutive_failures"] = cf
        return extra

    backend_details: dict[str, dict] = {}
    loading_probes: list[tuple[str, str]] = []
    # TTS URL may be a comma-separated pool — the loading probe only needs
    # one live endpoint, so take the first. The live pool state is already
    # reflected in deep-health ("ready") via the backend's own tracking.
    tts_probe_url = (state.config.tts_vllm_url or "").split(",")[0].strip()
    for name, backend_obj, url in [
        ("asr", state.asr_backend, state.config.asr_vllm_url),
        ("translate", state.translate_backend, state.config.translate_vllm_url),
        ("diarize", state.diarize_backend, state.config.diarize_url),
        ("tts", state.tts_backend, tts_probe_url),
        ("furigana", state.furigana_backend, ""),
    ]:
        dh = deep.get(name) or {}
        extra = _backend_extra(backend_obj) if backend_obj else {}
        if dh.get("ready"):
            backend_details[name] = {
                "ready": True,
                "status": "active",
                "detail": None,
                **extra,
            }
            continue

        # Not deep-ready — distinguish between "degraded" (had a real error)
        # and "loading/starting" (container not finished initializing).
        if backend_obj is not None and getattr(backend_obj, "degraded", False):
            backend_details[name] = {
                "ready": False,
                "status": "error",
                "detail": (
                    getattr(backend_obj, "last_error", None)
                    or dh.get("detail")
                    or "Backend degraded"
                ),
                **extra,
            }
            continue

        # Look up container state to produce a meaningful loading status
        loading_probes.append((name, url))

    # Probe unready backends in parallel for loading progress
    if loading_probes:
        results = await asyncio.gather(*[_probe_vllm_status(url) for _, url in loading_probes])
        for (name, _), result in zip(loading_probes, results):
            # If the container probe says "ready" but deep health failed,
            # the backend is up but scribe-side wiring isn't complete.
            if result.get("ready"):
                detail = deep.get(name, {}).get("detail") or "Connecting..."
                backend_details[name] = {
                    "ready": False,
                    "status": "loading",
                    "detail": detail,
                }
            else:
                backend_details[name] = result

    return JSONResponse(
        {
            "meeting": {
                "id": state.current_meeting.meeting_id if state.current_meeting else None,
                "state": state.current_meeting.state.value if state.current_meeting else None,
            },
            "backends": {
                "asr": backend_details.get("asr", {}).get("ready", False),
                "diarize": backend_details.get("diarize", {}).get("ready", False),
                "translate": backend_details.get("translate", {}).get("ready", False),
                "tts": backend_details.get("tts", {}).get("ready", False),
                "furigana": backend_details.get("furigana", {}).get("ready", False),
            },
            "backend_details": backend_details,
            "language_correction": (
                __import__(
                    "meeting_scribe.language_correction", fromlist=["correction_stats"]
                ).correction_stats.snapshot()
            ),
            "connections": len(state.ws_connections),
            "audio_out_connections": len(state._audio_out_clients),
            "terminal": state._terminal_registry.summary(),
            "metrics": state.metrics.to_dict(),
            "gpu": gpu_data,
            "system": system_data,
            "guest": _is_hotspot_client(request),
            "dev_mode": _is_dev_mode(),
        }
    )
