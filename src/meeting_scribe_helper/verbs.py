"""Typed verb implementations + the registry consumed by the daemon.

Each verb is a small async function that:

* Validates its input arguments (type checks + regex / range bounds).
* Builds a list-form argv (NEVER shell strings) for the privileged
  command it wraps.
* Invokes ``subprocess.run`` and returns a structured success or
  raises ``VerbError(code, detail)`` for the daemon to surface as a
  JSON-RPC error.

Adding a verb requires editing :data:`VERB_REGISTRY` here AND a focused
review of the implementation. The registry IS the privilege manifest.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from meeting_scribe_helper.protocol import VerbSpec

logger = logging.getLogger(__name__)


@dataclass
class VerbError(Exception):
    """Structured failure raised from inside a verb handler.

    ``code`` lands in the JSON-RPC ``error`` field; ``detail`` is
    appended to the audit log only (not sent on the wire) so internal
    state doesn't leak to the web service.
    """

    code: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}" if self.detail else self.code


# ── Input validation primitives ───────────────────────────────────


_SSID_RE = re.compile(r"^[A-Za-z0-9 _\-.]{1,32}$")
_PSK_RE = re.compile(r"^[\x20-\x7e]{8,63}$")  # ASCII printable, WPA2 length range
_BAND_VALUES: frozenset[str] = frozenset({"a", "bg"})
_MODE_VALUES: frozenset[str] = frozenset({"meeting", "admin"})
_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")


def _require_str(args: dict[str, Any], key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str):
        raise VerbError("invalid_args", f"{key} must be string")
    return val


def _require_int(args: dict[str, Any], key: str, *, low: int, high: int) -> int:
    val = args.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise VerbError("invalid_args", f"{key} must be int")
    if val < low or val > high:
        raise VerbError("invalid_args", f"{key} out of range [{low}, {high}]")
    return val


def _validate_ssid(ssid: str) -> str:
    if not _SSID_RE.fullmatch(ssid):
        raise VerbError("invalid_args", "ssid format")
    return ssid


def _validate_psk(psk: str) -> str:
    if not _PSK_RE.fullmatch(psk):
        raise VerbError("invalid_args", "psk format/length")
    return psk


def _validate_band(band: str) -> str:
    if band not in _BAND_VALUES:
        raise VerbError("invalid_args", "band must be 'a' or 'bg'")
    return band


def _validate_mode(mode: str) -> str:
    if mode not in _MODE_VALUES:
        raise VerbError("invalid_args", "mode must be 'meeting' or 'admin'")
    return mode


def _validate_country(country: str) -> str:
    if not _COUNTRY_RE.fullmatch(country):
        raise VerbError("invalid_args", "country must be ISO-3166-2 alpha-2 uppercase")
    return country


async def _run_argv(argv: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    """Run ``argv`` (list-form, no shell) and return a structured result.

    Wraps the subprocess in an executor so the helper's asyncio loop
    isn't blocked. ``argv`` MUST already be validated by the verb's
    schema check before calling — no input from the JSON-RPC payload
    reaches a shell.
    """
    loop = asyncio.get_running_loop()

    def _blocking() -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 — argv is list-form, validated
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    try:
        proc = await loop.run_in_executor(None, _blocking)
    except subprocess.TimeoutExpired as exc:
        raise VerbError("subprocess_timeout", str(exc)) from exc
    except FileNotFoundError as exc:
        raise VerbError("binary_not_found", argv[0]) from exc

    return {
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


# ── Verb implementations ──────────────────────────────────────────


async def verb_wifi_up(args: dict[str, Any]) -> dict[str, Any]:
    """Bring up the hotspot AP via NetworkManager.

    UID-gated to root only — a web-layer compromise must never be able
    to reconfigure the AP. The helper invokes nmcli with list-form argv
    so even a hostile ssid/psk argument cannot inject shell.
    """
    mode = _validate_mode(_require_str(args, "mode"))
    ssid = _validate_ssid(_require_str(args, "ssid"))
    psk = _validate_psk(_require_str(args, "password"))
    band = _validate_band(_require_str(args, "band"))
    channel = _require_int(args, "channel", low=1, high=196)

    # Implementation note: real wifi.up_NM_recipe is much more elaborate
    # than a single nmcli call (it stages a connection profile, sets
    # 802-11-wireless params, owners the keyfile, etc.). For the helper
    # we emit the minimum-viable list-form argv; the integration commit
    # that wires sudo-out-of-the-web-service replaces this body with the
    # full recipe, still as list-form argv only.
    argv = [
        "nmcli",
        "connection",
        "add",
        "type",
        "wifi",
        "ifname",
        "*",
        "con-name",
        f"meeting-scribe-{mode}",
        "ssid",
        ssid,
        "wifi.band",
        band,
        "wifi.channel",
        str(channel),
        "wifi-sec.key-mgmt",
        "wpa-psk",
        "wifi-sec.psk",
        psk,
    ]
    result = await _run_argv(argv)
    if result["rc"] != 0:
        raise VerbError("nmcli_failed", result["stderr"][:200])
    return {"applied": True, "mode": mode, "ssid": ssid}


async def verb_wifi_down(args: dict[str, Any]) -> dict[str, Any]:
    """Tear the hotspot AP down."""
    mode = _validate_mode(_require_str(args, "mode")) if args.get("mode") else "meeting"
    argv = ["nmcli", "connection", "down", f"meeting-scribe-{mode}"]
    result = await _run_argv(argv)
    return {"rc": result["rc"]}


async def verb_wifi_status(args: dict[str, Any]) -> dict[str, Any]:
    """Read-only — return nmcli's view of the active connections.

    Accepts both UID 0 and UID meeting-scribe (read-only verbs are
    safe for the web service to call directly).
    """
    del args  # status takes no args
    argv = ["nmcli", "-t", "-f", "TYPE,DEVICE,STATE,CONNECTION", "device"]
    result = await _run_argv(argv)
    return {"output": result["stdout"]}


async def verb_firewall_apply(args: dict[str, Any]) -> dict[str, Any]:
    """Apply the MS_* chain set + parent-jump position-1 invariant.

    Composes the multi-table iptables-restore input via
    :func:`meeting_scribe.firewall._build_restore_text` from the
    canonical generator (single source of truth — Plan 2 P1#1) then
    pipes it into ``iptables-restore -w 5`` via list-form argv. IPv6
    runs in a separate ``ip6tables-restore`` invocation (own lock
    domain).

    UID-gated to root only.
    """
    from meeting_scribe.firewall import (
        MS_CHAINS_FILTER,
        MS_CHAINS_FILTER_V6,
        MS_CHAINS_NAT,
        FirewallSnapshot,
        _build_restore_text,
        _expected_parent_jumps,
    )

    mode = _validate_mode(_require_str(args, "mode"))
    cidr = _require_str(args, "cidr")
    if not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}/\d{1,2}", cidr):
        raise VerbError("invalid_args", "cidr format")
    sta_present = bool(args.get("sta_iface_present", False))

    snap = FirewallSnapshot(
        captured_mode=mode,  # type: ignore[arg-type]
        captured_cidr=cidr,
        sta_iface_present=sta_present,
    )

    # Best-effort live dump fetch. Failures here surface as
    # subprocess_timeout / binary_not_found via _run_argv but the
    # generator below still produces the canonical bodies — so a
    # parser-side gap doesn't accidentally clear our chains.
    live_filter = await _run_argv(["iptables-save", "-t", "filter"], timeout=5.0)
    live_nat = await _run_argv(["iptables-save", "-t", "nat"], timeout=5.0)
    live_filter_v6 = await _run_argv(["ip6tables-save", "-t", "filter"], timeout=5.0)

    desired_v4 = _build_restore_text(
        table_blocks=[
            (
                "filter",
                live_filter["stdout"],
                MS_CHAINS_FILTER,
                list(_expected_parent_jumps(table="filter", v6=False)),
            ),
            (
                "nat",
                live_nat["stdout"],
                MS_CHAINS_NAT,
                list(_expected_parent_jumps(table="nat", v6=False)),
            ),
        ],
        snap=snap,
    )
    desired_v6 = _build_restore_text(
        table_blocks=[
            (
                "filter",
                live_filter_v6["stdout"],
                MS_CHAINS_FILTER_V6,
                list(_expected_parent_jumps(table="filter", v6=True)),
            ),
        ],
        snap=snap,
        v6=True,
    )

    rc_v4 = await _pipe_to(
        ["iptables-restore", "-w", "5"], desired_v4, timeout=10.0
    )
    if rc_v4["rc"] != 0:
        raise VerbError(
            "iptables_restore_failed",
            rc_v4["stderr"][:200],
        )
    rc_v6 = await _pipe_to(
        ["ip6tables-restore", "-w", "5"], desired_v6, timeout=10.0
    )
    if rc_v6["rc"] != 0:
        raise VerbError(
            "ip6tables_restore_failed",
            rc_v6["stderr"][:200],
        )
    return {
        "applied": True,
        "mode": mode,
        "cidr": cidr,
        "sta_iface_present": sta_present,
        "v4_chars": len(desired_v4),
        "v6_chars": len(desired_v6),
    }


async def _pipe_to(argv: list[str], text: str, *, timeout: float) -> dict[str, Any]:
    """Run ``argv`` with ``text`` on stdin (used by iptables-restore).

    Mirrors :func:`_run_argv` but adds the input pipe. The argv must be
    list-form — no shell — and ``text`` is the canonical restore body
    composed by :mod:`meeting_scribe.firewall`.
    """
    loop = asyncio.get_running_loop()

    def _blocking() -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 — list-form, validated
            argv,
            input=text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    try:
        proc = await loop.run_in_executor(None, _blocking)
    except subprocess.TimeoutExpired as exc:
        raise VerbError("subprocess_timeout", str(exc)) from exc
    except FileNotFoundError as exc:
        raise VerbError("binary_not_found", argv[0]) from exc

    return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


async def verb_firewall_status(args: dict[str, Any]) -> dict[str, Any]:
    """Read-only — return iptables-save output for filter + nat tables."""
    del args
    argv = ["iptables-save", "-t", "filter"]
    result = await _run_argv(argv)
    return {"filter_save": result["stdout"]}


async def verb_regdomain_set(args: dict[str, Any]) -> dict[str, Any]:
    """Set the wireless regulatory domain via iw reg set.

    UID-gated to root only.
    """
    country = _validate_country(_require_str(args, "country"))
    argv = ["iw", "reg", "set", country]
    result = await _run_argv(argv)
    if result["rc"] != 0:
        raise VerbError("iw_failed", result["stderr"][:200])
    return {"applied": True, "country": country}


# ── Registry ──────────────────────────────────────────────────────


# Sensitive-arg key sets per verb. The audit logger looks these up to
# replace values with the literal "<redacted>" before writing the log
# line. ``password`` is the obvious one for wifi.up.
_WIFI_UP_SENSITIVE = frozenset({"password"})


VERB_REGISTRY: dict[str, VerbSpec] = {
    "wifi.up": VerbSpec(
        handler=verb_wifi_up,
        allowed_uids="root_only",
        sensitive_keys=_WIFI_UP_SENSITIVE,
    ),
    "wifi.down": VerbSpec(handler=verb_wifi_down, allowed_uids="root_only"),
    "wifi.status": VerbSpec(handler=verb_wifi_status, allowed_uids="root_and_service"),
    "firewall.apply": VerbSpec(handler=verb_firewall_apply, allowed_uids="root_only"),
    "firewall.status": VerbSpec(handler=verb_firewall_status, allowed_uids="root_and_service"),
    "regdomain.set": VerbSpec(handler=verb_regdomain_set, allowed_uids="root_only"),
}
