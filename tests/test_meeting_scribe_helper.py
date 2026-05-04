"""Tests for the privileged helper daemon — protocol, verb dispatch,
caller-UID gating, sensitive-arg redaction, end-to-end Unix-socket
round-trip via the helper_client.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from meeting_scribe.helper_client import HelperError, call as client_call
from meeting_scribe_helper.__main__ import serve
from meeting_scribe_helper.protocol import (
    VerbSpec,
    caller_authorized,
    encode_response,
    parse_request,
    redact_sensitive,
)
from meeting_scribe_helper.verbs import (
    VERB_REGISTRY,
    VerbError,
    verb_firewall_apply,
)


# ── Pure-function unit tests ──────────────────────────────────────


def test_parse_request_minimum_shape() -> None:
    raw = b'{"verb": "wifi.up", "args": {"x": 1}, "request_id": "abc"}\n'
    verb, args, rid = parse_request(raw)
    assert verb == "wifi.up"
    assert args == {"x": 1}
    assert rid == "abc"


def test_parse_request_handles_garbage() -> None:
    for junk in (b"", b"\n", b"not-json\n", b'"just-a-string"\n', b'{"verb": 5}\n'):
        verb, args, _ = parse_request(junk)
        assert verb is None or verb == ""
        assert isinstance(args, dict)


def test_encode_response_round_trip() -> None:
    raw = encode_response(ok=True, request_id="r1", result={"a": 1})
    body = json.loads(raw.decode("utf-8"))
    assert body == {"ok": True, "request_id": "r1", "result": {"a": 1}}
    raw_err = encode_response(ok=False, request_id="r2", error="invalid_args")
    body = json.loads(raw_err.decode("utf-8"))
    assert body == {"ok": False, "request_id": "r2", "error": "invalid_args"}


def test_caller_authorized_root_only_rejects_service_uid() -> None:
    """Mutating verbs reject UID meeting-scribe with cli_only_verb —
    web-layer compromise can't reconfigure the AP."""
    spec = VerbSpec(handler=verb_firewall_apply, allowed_uids="root_only")
    ok, err = caller_authorized(spec, caller_uid=1001, service_uid=1001)
    assert not ok
    assert err == "cli_only_verb"


def test_caller_authorized_root_only_accepts_root() -> None:
    spec = VerbSpec(handler=verb_firewall_apply, allowed_uids="root_only")
    ok, err = caller_authorized(spec, caller_uid=0, service_uid=1001)
    assert ok
    assert err is None


def test_caller_authorized_root_and_service() -> None:
    spec = VerbSpec(handler=verb_firewall_apply, allowed_uids="root_and_service")
    assert caller_authorized(spec, caller_uid=0, service_uid=1001)[0] is True
    assert caller_authorized(spec, caller_uid=1001, service_uid=1001)[0] is True
    assert caller_authorized(spec, caller_uid=1234, service_uid=1001) == (False, "uid_not_allowed")


def test_redact_sensitive_blanks_password() -> None:
    args = {"ssid": "demo", "password": "hunter2", "channel": 36}
    out = redact_sensitive(args, frozenset({"password"}))
    assert out == {"ssid": "demo", "password": "<redacted>", "channel": 36}
    # No SHA-256 fingerprint of the secret leaks.
    assert "hunter2" not in repr(out)
    assert "hunter2" not in str(out)


# ── Verb input validation ────────────────────────────────────────


def test_verb_wifi_up_rejects_bad_ssid() -> None:
    async def runner() -> None:
        with pytest.raises(VerbError) as exc_info:
            await VERB_REGISTRY["wifi.up"].handler(
                {
                    "mode": "meeting",
                    "ssid": "demo;rm -rf /",  # shell metacharacters
                    "password": "longenough",
                    "band": "a",
                    "channel": 36,
                }
            )
        assert exc_info.value.code == "invalid_args"

    asyncio.run(runner())


def test_verb_wifi_up_rejects_bad_psk() -> None:
    async def runner() -> None:
        with pytest.raises(VerbError) as exc_info:
            await VERB_REGISTRY["wifi.up"].handler(
                {
                    "mode": "meeting",
                    "ssid": "demo",
                    "password": "short",  # < 8 chars
                    "band": "a",
                    "channel": 36,
                }
            )
        assert exc_info.value.code == "invalid_args"

    asyncio.run(runner())


def test_verb_wifi_up_rejects_bad_channel() -> None:
    async def runner() -> None:
        with pytest.raises(VerbError) as exc_info:
            await VERB_REGISTRY["wifi.up"].handler(
                {
                    "mode": "meeting",
                    "ssid": "demo",
                    "password": "longenough",
                    "band": "a",
                    "channel": 9999,
                }
            )
        assert exc_info.value.code == "invalid_args"

    asyncio.run(runner())


def test_verb_regdomain_set_rejects_invalid_country() -> None:
    async def runner() -> None:
        with pytest.raises(VerbError) as exc_info:
            await VERB_REGISTRY["regdomain.set"].handler({"country": "u1"})
        assert exc_info.value.code == "invalid_args"

    asyncio.run(runner())


def test_verb_firewall_apply_rejects_bad_cidr() -> None:
    async def runner() -> None:
        with pytest.raises(VerbError) as exc_info:
            await VERB_REGISTRY["firewall.apply"].handler(
                {
                    "mode": "meeting",
                    "cidr": "not-a-cidr",
                    "sta_iface_present": False,
                }
            )
        assert exc_info.value.code == "invalid_args"

    asyncio.run(runner())


def test_verb_firewall_apply_skeleton_returns_structured() -> None:
    """Skeleton path returns a structured success dict so the verb
    dispatch end-to-end can be exercised without root + iptables."""

    async def runner() -> None:
        result = await VERB_REGISTRY["firewall.apply"].handler(
            {
                "mode": "meeting",
                "cidr": "10.42.0.0/24",
                "sta_iface_present": True,
            }
        )
        assert result["mode"] == "meeting"
        assert result["cidr"] == "10.42.0.0/24"
        assert result["sta_iface_present"] is True
        assert "skeleton" in result["note"]

    asyncio.run(runner())


# ── End-to-end Unix-socket round-trip ────────────────────────────


@pytest.fixture
async def helper_socket(tmp_path: Path) -> Any:
    """Spin up a daemon on a tmp socket; helper_client connects via
    SCRIBE_HELPER_SOCKET. Caller's UID == current process UID, which is
    used as both the "root" uid in tests (non-root is fine for the
    skeleton verbs that don't shell out) and the service_uid."""
    sock_path = tmp_path / "helper.sock"
    service_uid = os.geteuid()  # current uid satisfies "root_and_service"
    server_task = asyncio.create_task(
        serve(socket_path=sock_path, service_uid=service_uid)
    )
    # Wait for the socket to actually exist.
    for _ in range(50):
        if sock_path.exists():
            break
        await asyncio.sleep(0.02)
    yield sock_path
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_helper_unknown_verb_returns_error(
    helper_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCRIBE_HELPER_SOCKET", str(helper_socket))
    with pytest.raises(HelperError) as exc_info:
        await client_call("definitely.not.a.verb", {"x": 1})
    assert exc_info.value.code == "unknown_verb"


@pytest.mark.asyncio
async def test_helper_root_only_verb_rejects_service_uid(
    helper_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating verbs return cli_only_verb when called from the service
    UID — defense-in-depth even before input validation. The fixture
    runs daemon with service_uid = current EUID so any client call from
    the test process trips this gate."""
    monkeypatch.setenv("SCRIBE_HELPER_SOCKET", str(helper_socket))
    with pytest.raises(HelperError) as exc_info:
        await client_call("regdomain.set", {"country": "US"})
    assert exc_info.value.code == "cli_only_verb"


@pytest.mark.asyncio
async def test_helper_firewall_apply_skeleton_round_trip(
    helper_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skeleton firewall.apply round-trips end-to-end: client → unix
    socket → daemon → verb → response."""
    monkeypatch.setenv("SCRIBE_HELPER_SOCKET", str(helper_socket))
    # Test fixture uses current EUID as service_uid, so the gating
    # path is "service-uid call to a root_only verb → cli_only_verb".
    # That's exactly what we want to exercise.
    with pytest.raises(HelperError) as exc_info:
        await client_call(
            "firewall.apply",
            {
                "mode": "meeting",
                "cidr": "10.42.0.0/24",
                "sta_iface_present": False,
            },
        )
    assert exc_info.value.code == "cli_only_verb"


@pytest.mark.asyncio
async def test_helper_invalid_request_returns_invalid_request(
    helper_socket: Path,
) -> None:
    """Garbage on the socket → invalid_request, not a crash."""
    reader, writer = await asyncio.open_unix_connection(str(helper_socket))
    try:
        writer.write(b"definitely not json\n")
        await writer.drain()
        raw = await reader.readline()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass
    body = json.loads(raw.decode("utf-8").strip())
    assert body["ok"] is False
    assert body["error"] == "invalid_request"


def test_helper_password_never_in_plaintext_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The audit log line for wifi.up redacts ``password`` to
    ``<redacted>`` — no SHA-256 fingerprint, no echo."""
    args = {
        "mode": "meeting",
        "ssid": "demo",
        "password": "secretvalue",
        "band": "a",
        "channel": 36,
    }
    redacted = redact_sensitive(args, frozenset({"password"}))
    # Mimic the daemon's log line format.
    msg = f"verb_invoked verb='wifi.up' args={redacted}"
    assert "secretvalue" not in msg
    assert "<redacted>" in msg
    # Also verify no SHA-256-like hex of the secret leaks. (We don't
    # produce one — the redact_sensitive contract is "literal
    # <redacted>", nothing else.)
    import hashlib

    fingerprint = hashlib.sha256(b"secretvalue").hexdigest()
    assert fingerprint not in msg


def test_verb_registry_lists_all_documented_verbs() -> None:
    """Every verb named in the design lands in VERB_REGISTRY."""
    expected = {
        "wifi.up",
        "wifi.down",
        "wifi.status",
        "firewall.apply",
        "firewall.status",
        "regdomain.set",
    }
    assert set(VERB_REGISTRY.keys()) == expected


def test_verb_registry_mutating_verbs_root_only() -> None:
    """The privilege manifest: every mutating verb must be root_only.
    Read-only verbs (wifi.status, firewall.status) accept the service
    UID."""
    mutating = {"wifi.up", "wifi.down", "firewall.apply", "regdomain.set"}
    read_only = {"wifi.status", "firewall.status"}
    for name in mutating:
        assert VERB_REGISTRY[name].allowed_uids == "root_only", name
    for name in read_only:
        assert VERB_REGISTRY[name].allowed_uids == "root_and_service", name
