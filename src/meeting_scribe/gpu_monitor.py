"""GPU and system resource monitoring for GB10.

Reads VRAM usage via pynvml (fast, no subprocess) with nvidia-smi fallback.
Also provides CPU, system memory, and per-container resource snapshots.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VRAMUsage:
    """GPU VRAM usage snapshot."""

    used_mb: int
    total_mb: int

    @property
    def pct(self) -> float:
        return (self.used_mb / self.total_mb * 100) if self.total_mb > 0 else 0.0

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def get_vram_usage() -> VRAMUsage | None:
    """Read current GPU VRAM usage. Returns None if unavailable."""
    # Try pynvml first (fast, no subprocess)
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return VRAMUsage(
            used_mb=int(info.used / 1024 / 1024),
            total_mb=int(info.total / 1024 / 1024),
        )
    except Exception:
        pass

    # Fallback: parse nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            used = parts[0].strip()
            total = parts[1].strip()
            if used != "[N/A]" and total != "[N/A]":
                return VRAMUsage(used_mb=int(used), total_mb=int(total))
    except Exception:
        pass

    # GB10 unified memory: use system memory as proxy via /proc/meminfo
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "GB10" in result.stdout:
            # GB10 uses unified memory — report system RAM usage as best approximation
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            used_kb = total_kb - avail_kb
            return VRAMUsage(
                used_mb=used_kb // 1024,
                total_mb=total_kb // 1024,
            )
    except Exception:
        pass

    return None


@dataclass
class SystemResources:
    """System-wide resource snapshot."""

    cpu_pct: float  # Overall CPU usage (0-100)
    mem_used_mb: int
    mem_total_mb: int
    mem_pct: float
    load_1m: float
    load_5m: float
    load_15m: float
    uptime_s: int
    containers: list[dict] = field(default_factory=list)


# Cache for container stats (expensive subprocess call)
_container_cache: list | None = None
_container_cache_time: float = 0.0
_CONTAINER_CACHE_TTL = 10.0
# Stale-while-revalidate guard: True while a background `docker stats`
# call is running so concurrent /api/status callers don't spawn N
# parallel subprocesses against the docker daemon.
_container_refresh_in_flight: bool = False
_container_refresh_lock = threading.Lock()

# Cache for system resources (avoids 50ms CPU sample + docker stats per call)
_sys_cache: SystemResources | None = None
_sys_cache_time: float = 0.0
_SYS_CACHE_TTL = 5.0

# Persistent CPU counters — compute delta between calls instead of sleeping
_cpu_prev: tuple[int, int] | None = None  # (total, idle)


def _read_cpu_usage() -> float:
    """Read CPU usage from /proc/stat using delta from previous call."""
    global _cpu_prev
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = [int(x) for x in line.split()[1:]]
        idle = vals[3]
        total = sum(vals)

        if _cpu_prev is None:
            _cpu_prev = (total, idle)
            return 0.0

        prev_total, prev_idle = _cpu_prev
        _cpu_prev = (total, idle)

        d_total = total - prev_total
        d_idle = idle - prev_idle
        if d_total == 0:
            return 0.0
        return round((1.0 - d_idle / d_total) * 100, 1)
    except Exception:
        return 0.0


def _docker_stats_blocking() -> list[dict] | None:
    """Run `docker stats --no-stream` and parse. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.PIDs}}",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode != 0:
            return None

        containers = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            name = parts[0]
            if not name.startswith("scribe-") and not name.startswith("autosre-"):
                continue
            cpu = parts[1].rstrip("%")
            mem_raw = parts[2]  # e.g. "4.5GiB / 128GiB"
            pids = parts[3]

            mem_mb = 0
            try:
                used_str = mem_raw.split("/")[0].strip()
                if "GiB" in used_str:
                    mem_mb = int(float(used_str.replace("GiB", "").strip()) * 1024)
                elif "MiB" in used_str:
                    mem_mb = int(float(used_str.replace("MiB", "").strip()))
            except Exception:
                pass

            containers.append(
                {
                    "name": name,
                    "cpu_pct": float(cpu) if cpu else 0.0,
                    "mem_mb": mem_mb,
                    "pids": int(pids) if pids.isdigit() else 0,
                }
            )
        return containers
    except Exception:
        return None


def _refresh_container_cache_async() -> None:
    """Background-thread refresh of the container stats cache."""
    global _container_cache, _container_cache_time, _container_refresh_in_flight
    try:
        fresh = _docker_stats_blocking()
        if fresh is not None:
            _container_cache = fresh
            _container_cache_time = time.monotonic()
    finally:
        with _container_refresh_lock:
            _container_refresh_in_flight = False


def _get_container_stats() -> list[dict]:
    """Get resource usage for running scribe containers via docker stats.

    Stale-while-revalidate: a populated cache is returned immediately,
    even if older than the TTL. Once stale, a background thread refreshes
    it (deduped via ``_container_refresh_in_flight``) so the next call
    sees fresh data without anyone paying the 1.5–2 s ``docker stats``
    cost on the request path.
    """
    global _container_cache, _container_cache_time, _container_refresh_in_flight

    now = time.monotonic()
    fresh_enough = (
        _container_cache is not None and now - _container_cache_time < _CONTAINER_CACHE_TTL
    )
    if fresh_enough:
        return _container_cache  # type: ignore[return-value]

    if _container_cache is not None:
        with _container_refresh_lock:
            if not _container_refresh_in_flight:
                _container_refresh_in_flight = True
                threading.Thread(
                    target=_refresh_container_cache_async,
                    name="container-stats-refresh",
                    daemon=True,
                ).start()
        return _container_cache

    fresh = _docker_stats_blocking()
    if fresh is None:
        return []
    _container_cache = fresh
    _container_cache_time = now
    return fresh


def get_system_resources() -> SystemResources | None:
    """Collect system-wide resource metrics. Cached for 5s."""
    global _sys_cache, _sys_cache_time
    now = time.monotonic()
    if _sys_cache is not None and now - _sys_cache_time < _SYS_CACHE_TTL:
        return _sys_cache
    try:
        # Memory from /proc/meminfo (no dependencies)
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        mem_total_mb = total_kb // 1024
        mem_used_mb = used_kb // 1024
        mem_pct = round(used_kb / total_kb * 100, 1) if total_kb > 0 else 0.0

        # Load average
        load_1, load_5, load_15 = os.getloadavg()

        # Uptime
        with open("/proc/uptime") as f:
            uptime_s = int(float(f.read().split()[0]))

        # CPU usage
        cpu_pct = _read_cpu_usage()

        # Container stats (cached)
        containers = _get_container_stats()

        result = SystemResources(
            cpu_pct=cpu_pct,
            mem_used_mb=mem_used_mb,
            mem_total_mb=mem_total_mb,
            mem_pct=mem_pct,
            load_1m=round(load_1, 2),
            load_5m=round(load_5, 2),
            load_15m=round(load_15, 2),
            uptime_s=uptime_s,
            containers=containers,
        )
        _sys_cache = result
        _sys_cache_time = now
        return result
    except Exception as e:
        logger.debug("System resource collection failed: %s", e)
        return _sys_cache
