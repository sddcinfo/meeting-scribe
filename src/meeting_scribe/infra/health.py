"""Service health checking with retry and timeout.

Polls HTTP health endpoints until they respond or timeout is reached.
Pattern adapted from auto-sre's _wait_for_vllm().
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Default service ports (host networking, no port mapping)
SERVICE_PORTS = {
    "translation": 8010,
    "diarization": 8001,
    "tts": 8002,
    "asr": 8003,
}


@dataclass
class ServiceStatus:
    """Health status of a single service."""

    name: str
    url: str
    healthy: bool
    model: str | None = None
    error: str | None = None


async def check_service(
    url: str,
    *,
    timeout: float = 5.0,
    total_timeout: float = 300.0,
    poll_interval: float = 5.0,
    wait: bool = False,
) -> bool:
    """Check if a service is healthy, optionally waiting for it.

    Args:
        url: Health endpoint URL (e.g., "http://192.168.1.100:8000/health").
        timeout: Per-request timeout in seconds.
        total_timeout: Maximum time to wait if wait=True.
        poll_interval: Seconds between retries if wait=True.
        wait: If True, poll until healthy or total_timeout.

    Returns:
        True if the service is healthy.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        deadline = asyncio.get_event_loop().time() + total_timeout

        while True:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
                logger.debug("Health check %s: status %d", url, resp.status_code)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, OSError) as e:
                logger.debug("Health check %s: %s", url, e)

            if not wait or asyncio.get_event_loop().time() >= deadline:
                return False

            await asyncio.sleep(poll_interval)


async def _check_one_service(
    name: str,
    port: int,
    host: str,
    *,
    wait: bool,
    total_timeout: float,
) -> tuple[str, ServiceStatus]:
    """Check a single service and return its status."""
    health_url = f"http://{host}:{port}/health"
    model_url = f"http://{host}:{port}/v1/models"

    healthy = await check_service(health_url, wait=wait, total_timeout=total_timeout)

    model = None
    if healthy:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(model_url)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data:
                        model = data[0].get("id")
        except (httpx.HTTPError, KeyError, IndexError):
            pass

    error = None if healthy else f"Service not responding at {health_url}"
    return name, ServiceStatus(
        name=name,
        url=f"http://{host}:{port}",
        healthy=healthy,
        model=model,
        error=error,
    )


async def check_all_services(
    host: str,
    *,
    ports: dict[str, int] | None = None,
    wait: bool = False,
    total_timeout: float = 300.0,
) -> dict[str, ServiceStatus]:
    """Check health of all meeting-scribe services on a GB10 node.

    When ``wait=True``, all services are polled **concurrently** with a
    shared ``total_timeout`` deadline. Total wait = max(slowest_service)
    instead of sum(all_services).

    Args:
        host: GB10 IP or hostname.
        ports: Override default service ports.
        wait: If True, wait for each service to become healthy.
        total_timeout: Max wait time (shared across all services when
                       wait=True).

    Returns:
        Dict mapping service name to ServiceStatus.
    """
    svc_ports = ports or SERVICE_PORTS

    tasks = [
        _check_one_service(name, port, host, wait=wait, total_timeout=total_timeout)
        for name, port in svc_ports.items()
    ]
    pairs = await asyncio.gather(*tasks)
    return {name: status for name, status in pairs}
