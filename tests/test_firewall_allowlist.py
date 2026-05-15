"""firewall_allowlist — ipset CRUD via an injectable subprocess seam.

All shellouts replaced by a FakeRunner; no real ipset calls.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

import meeting_scribe.server_support.firewall_allowlist as fwa


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Default: every command rc=0 with empty stdout.
        self._responses: list[tuple[Callable[[list[str]], bool], subprocess.CompletedProcess]] = []

    def respond(
        self, match: Callable[[list[str]], bool], *, rc: int = 0, stdout: str = "", stderr: str = ""
    ) -> None:
        self._responses.append(
            (
                match,
                subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr),
            )
        )

    def respond_prefix(self, prefix: list[str], **kwargs: Any) -> None:
        def _matches(argv: list[str]) -> bool:
            return argv[: len(prefix)] == prefix

        self.respond(_matches, **kwargs)

    async def __call__(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        for match, resp in self._responses:
            if match(argv):
                return subprocess.CompletedProcess(
                    args=argv, returncode=resp.returncode, stdout=resp.stdout, stderr=resp.stderr
                )
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")


@pytest.fixture
def fake_runner(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(fwa, "IPSET_RUNNER", runner)
    # Pretend the binary is on PATH even when ipset isn't installed
    # in the test environment.
    monkeypatch.setattr(fwa, "is_available", lambda: True)
    return runner


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


# ─── _is_valid_v4 ─────────────────────────────────────────────


def test_is_valid_v4_accepts_dotted_quad() -> None:
    assert fwa._is_valid_v4("10.42.0.42") is True
    assert fwa._is_valid_v4("192.168.8.153") is True


def test_is_valid_v4_rejects_v6_garbage_empty() -> None:
    assert fwa._is_valid_v4("::1") is False  # v4 only
    assert fwa._is_valid_v4("not-an-ip") is False
    assert fwa._is_valid_v4("") is False
    # Bytes accidentally typed as str
    assert fwa._is_valid_v4("256.0.0.1") is False


# ─── Set lifecycle ────────────────────────────────────────────


def test_ensure_sets_creates_both_with_exist(fake_runner) -> None:
    assert _run(fwa.ensure_sets()) is True
    create_calls = [c for c in fake_runner.calls if c[:2] == ["ipset", "create"]]
    assert len(create_calls) == 2
    names = {c[2] for c in create_calls}
    assert names == {fwa.ADMIN_SET, fwa.GUEST_SET}
    # ``-exist`` flag is present on every create so re-invocations are
    # idempotent across AP bring-ups.
    for c in create_calls:
        assert "-exist" in c
        # hash:ip is the only set type we support — guards against an
        # operator dropping a hash:net or similar.
        assert c[3] == "hash:ip"


def test_ensure_sets_returns_false_on_create_failure(fake_runner) -> None:
    fake_runner.respond_prefix(["ipset", "create"], rc=1, stderr="kernel error")
    assert _run(fwa.ensure_sets()) is False


def test_destroy_sets_emits_both_destroys(fake_runner) -> None:
    _run(fwa.destroy_sets())
    destroy_calls = [c for c in fake_runner.calls if c[:2] == ["ipset", "destroy"]]
    assert {c[2] for c in destroy_calls} == {fwa.ADMIN_SET, fwa.GUEST_SET}


# ─── Add / remove ─────────────────────────────────────────────


def test_add_admin_emits_ipset_add(fake_runner) -> None:
    assert _run(fwa.add_admin("10.42.0.42")) is True
    add_calls = [c for c in fake_runner.calls if c[:2] == ["ipset", "add"]]
    assert len(add_calls) == 1
    assert add_calls[0][2] == fwa.ADMIN_SET
    assert add_calls[0][3] == "10.42.0.42"
    assert "-exist" in add_calls[0]


def test_add_admin_rejects_invalid_ip(fake_runner) -> None:
    assert _run(fwa.add_admin("not-an-ip")) is False
    # Crucially: the bad IP never reaches subprocess.
    assert fake_runner.calls == []


def test_add_guest_uses_guest_set(fake_runner) -> None:
    assert _run(fwa.add_guest("10.42.0.55")) is True
    add_calls = [c for c in fake_runner.calls if c[:2] == ["ipset", "add"]]
    assert add_calls[0][2] == fwa.GUEST_SET


def test_remove_admin_emits_del(fake_runner) -> None:
    assert _run(fwa.remove_admin("10.42.0.42")) is True
    del_calls = [c for c in fake_runner.calls if c[:2] == ["ipset", "del"]]
    assert len(del_calls) == 1
    assert del_calls[0][2] == fwa.ADMIN_SET
    # ``-exist`` makes del idempotent (not-present → ok).
    assert "-exist" in del_calls[0]


def test_remove_unknown_ip_swallows_failure(fake_runner) -> None:
    fake_runner.respond_prefix(["ipset", "del"], rc=1, stderr="not present")
    assert _run(fwa.remove_admin("10.42.0.42")) is False


# ─── list_* ──────────────────────────────────────────────────


_SAMPLE_SAVE_OUTPUT = """\
create ms-allowed-admins hash:ip family inet hashsize 1024 maxelem 65536
add ms-allowed-admins 10.42.0.42
add ms-allowed-admins 10.42.0.55
"""


def test_list_admins_parses_save_output(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["ipset", "list", fwa.ADMIN_SET, "-o", "save"], stdout=_SAMPLE_SAVE_OUTPUT
    )
    members = _run(fwa.list_admins())
    assert members == {"10.42.0.42", "10.42.0.55"}


def test_list_admins_empty_on_command_failure(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["ipset", "list", fwa.ADMIN_SET, "-o", "save"], rc=1, stderr="No such set"
    )
    assert _run(fwa.list_admins()) == set()


# ─── Availability fallback ────────────────────────────────────


def test_no_op_when_ipset_not_available(monkeypatch) -> None:
    """Every CRUD call returns False / skips subprocess when ipset is missing."""
    calls: list[list[str]] = []

    async def _runner(argv, timeout):
        calls.append(list(argv))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(fwa, "IPSET_RUNNER", _runner)
    monkeypatch.setattr(fwa, "is_available", lambda: False)
    monkeypatch.setattr(fwa, "_AVAILABILITY_WARNED", True)  # silence the warn

    assert _run(fwa.ensure_sets()) is False
    assert _run(fwa.add_admin("10.42.0.42")) is False
    assert _run(fwa.add_guest("10.42.0.55")) is False
    assert _run(fwa.remove_admin("10.42.0.42")) is False
    assert _run(fwa.list_admins()) == set()
    # ``destroy_sets()`` is best-effort; it should also short-circuit.
    _run(fwa.destroy_sets())
    assert calls == []


# ─── Lease parser ────────────────────────────────────────────


def test_parse_dnsmasq_leases_extracts_v4() -> None:
    lease_text = (
        "1715600000 aa:bb:cc:dd:ee:ff 10.42.0.42 phone *\n"
        "1715603600 11:22:33:44:55:66 10.42.0.55 laptop *\n"
        "\n"  # blank line
        "bogus line\n"  # malformed, must skip
    )
    leases = fwa.parse_dnsmasq_leases(lease_text)
    assert leases == {"10.42.0.42", "10.42.0.55"}


def test_parse_dnsmasq_leases_skips_malformed_quietly() -> None:
    lease_text = "not a real line\n"
    assert fwa.parse_dnsmasq_leases(lease_text) == set()


# ─── gc_once ─────────────────────────────────────────────────


def test_gc_once_prunes_unleased_ipset_entries(tmp_path, fake_runner) -> None:
    """An ipset entry whose IP no longer holds a lease is pruned."""
    leases_path = tmp_path / "dnsmasq.leases"
    # 42 has a lease, 99 does not.
    leases_path.write_text("1715600000 aa:bb:cc:dd:ee:ff 10.42.0.42 phone *\n")

    fake_runner.respond_prefix(
        ["ipset", "list", fwa.ADMIN_SET, "-o", "save"],
        stdout="add ms-allowed-admins 10.42.0.42\nadd ms-allowed-admins 10.42.0.99\n",
    )
    fake_runner.respond_prefix(
        ["ipset", "list", fwa.GUEST_SET, "-o", "save"],
        stdout="",
    )

    result = _run(fwa.gc_once(leases_path=str(leases_path)))
    assert result == {"admins_pruned": 1, "guests_pruned": 0}
    # The pruned IP got an explicit del; the still-leased one did NOT.
    del_calls = [c for c in fake_runner.calls if c[:2] == ["ipset", "del"]]
    pruned_ips = {c[3] for c in del_calls}
    assert pruned_ips == {"10.42.0.99"}
    assert "10.42.0.42" not in pruned_ips


def test_gc_once_skips_when_leases_file_missing(tmp_path, fake_runner) -> None:
    """A missing leases file MUST NOT drain the ipsets — the GC tick is
    skipped instead, leaving stale entries until the next successful read."""
    fake_runner.respond_prefix(
        ["ipset", "list", fwa.ADMIN_SET, "-o", "save"],
        stdout="add ms-allowed-admins 10.42.0.42\n",
    )
    result = _run(fwa.gc_once(leases_path=str(tmp_path / "nope.leases")))
    assert result == {"admins_pruned": 0, "guests_pruned": 0}
    # The ipset MUST not be modified.
    assert not [c for c in fake_runner.calls if c[:2] == ["ipset", "del"]]


def test_gc_once_noop_when_ipset_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fwa, "is_available", lambda: False)
    monkeypatch.setattr(fwa, "_AVAILABILITY_WARNED", True)
    result = _run(fwa.gc_once(leases_path=str(tmp_path / "x.leases")))
    assert result == {"admins_pruned": 0, "guests_pruned": 0}


# ─── Sync convenience: is_admin / is_guest ───────────────────


def test_is_admin_returns_true_when_in_set(monkeypatch) -> None:
    """The captive sub-app uses the sync flavor in its hot path."""
    monkeypatch.setattr(fwa, "is_available", lambda: True)
    monkeypatch.setattr(
        fwa,
        "_sync_list_members",
        lambda name: {"10.42.0.42"} if name == fwa.ADMIN_SET else set(),
    )
    assert fwa.is_admin("10.42.0.42") is True
    assert fwa.is_admin("10.42.0.99") is False


def test_is_admin_rejects_invalid_ip(monkeypatch) -> None:
    """Garbage in returns False without consulting the ipset."""
    monkeypatch.setattr(fwa, "is_available", lambda: True)
    called = {"count": 0}

    def _spy(name):
        called["count"] += 1
        return set()

    monkeypatch.setattr(fwa, "_sync_list_members", _spy)
    assert fwa.is_admin("not-an-ip") is False
    assert called["count"] == 0
