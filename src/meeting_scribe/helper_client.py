"""Client for the privileged helper daemon.

Used by both the web service (to invoke wifi/firewall/regdomain ops
without ``sudo``) and the CLI (so ``sudo meeting-scribe wifi up`` and
``meeting-scribe wifi up`` from an admin UI route share one code path).

Wire protocol matches :mod:`meeting_scribe_helper.protocol` — one
JSON request per connection, one JSON response, close. Failures
(``invalid_args``, ``cli_only_verb``, ``unknown_verb``, ``uid_not_allowed``,
``binary_not_found``, ``subprocess_timeout``, etc.) raise
:class:`HelperError` carrying the structured error code.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SOCKET = Path("/run/meeting-scribe/helper.sock")


@dataclass
class HelperError(RuntimeError):
    """Wraps a JSON-RPC error from the helper.

    ``code`` is the on-the-wire error string (one of the structured
    codes the daemon emits); callers compare on it directly to branch.
    """

    code: str

    def __str__(self) -> str:
        return f"helper_error: {self.code}"


def _socket_path() -> Path:
    """Resolve the helper socket path. Honors ``SCRIBE_HELPER_SOCKET``
    so dev/CI runs and test fixtures can point elsewhere."""
    override = os.environ.get("SCRIBE_HELPER_SOCKET", "").strip()
    return Path(override) if override else DEFAULT_SOCKET


async def call(
    verb: str,
    args: dict[str, Any] | None = None,
    *,
    socket_path: Path | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Invoke ``verb`` with ``args``; return the helper's ``result`` dict.

    Raises :class:`HelperError` on any structured failure (verb-side
    ``invalid_args``, gating denial, etc.) or :class:`OSError` on
    socket-level failure (helper down, perms wrong).
    """
    path = socket_path or _socket_path()
    request_id = secrets.token_hex(8)
    request_line = (
        json.dumps(
            {
                "verb": verb,
                "args": args or {},
                "request_id": request_id,
            }
        )
        + "\n"
    ).encode("utf-8")

    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(path)),
        timeout=timeout,
    )
    try:
        writer.write(request_line)
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 — best-effort socket teardown
            pass

    try:
        body = json.loads(raw.decode("utf-8").strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HelperError(code="bad_response") from exc
    if not isinstance(body, dict) or "ok" not in body:
        raise HelperError(code="bad_response")
    if not body["ok"]:
        raise HelperError(code=body.get("error") or "unspecified_error")
    return body.get("result") or {}


# ── Convenience wrappers (one per verb) ──────────────────────────


async def wifi_up(
    *,
    mode: str,
    ssid: str,
    password: str,
    band: str,
    channel: int,
) -> dict[str, Any]:
    return await call(
        "wifi.up",
        {
            "mode": mode,
            "ssid": ssid,
            "password": password,
            "band": band,
            "channel": channel,
        },
    )


async def wifi_down(*, mode: str | None = None) -> dict[str, Any]:
    return await call("wifi.down", {"mode": mode} if mode else {})


async def wifi_status() -> dict[str, Any]:
    return await call("wifi.status", {})


async def firewall_apply(
    *,
    mode: str,
    cidr: str,
    sta_iface_present: bool = False,
) -> dict[str, Any]:
    return await call(
        "firewall.apply",
        {"mode": mode, "cidr": cidr, "sta_iface_present": sta_iface_present},
    )


async def firewall_status() -> dict[str, Any]:
    return await call("firewall.status", {})


async def regdomain_set(*, country: str) -> dict[str, Any]:
    return await call("regdomain.set", {"country": country})
