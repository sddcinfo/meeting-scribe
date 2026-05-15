"""server_support.mdns — avahi-publish-address driver.

The real publication needs a running avahi-daemon and an interface
with the target IP; these tests stub the subprocess so they can run
in any sandbox.
"""

from __future__ import annotations

import asyncio
from typing import Any

from meeting_scribe.server_support import mdns


class _FakeProcess:
    """Minimal stand-in for ``asyncio.subprocess.Process`` covering
    only what :mod:`mdns` reaches for."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.returncode: int | None = None
        self._terminated = False
        self._killed = False
        # ``wait()`` would return only after the publisher exits — we
        # simulate "stays alive forever" by suspending until terminate
        # is called.
        self._exit_event = asyncio.Event()
        self.stderr = None

    async def wait(self) -> int:
        await self._exit_event.wait()
        return self.returncode if self.returncode is not None else 0

    def terminate(self) -> None:
        self._terminated = True
        self.returncode = 0
        self._exit_event.set()

    def kill(self) -> None:
        self._killed = True
        self.returncode = -9
        self._exit_event.set()


def test_publisher_argv_uses_avahi_publish_address() -> None:
    """Stable argv shape so the lifespan hook is grep-able."""
    argv = mdns._publisher_argv("meeting-1618.local", "10.42.0.1")
    assert argv[0] == "avahi-publish-address"
    assert "meeting-1618.local" in argv
    assert "10.42.0.1" in argv


def test_publish_aliases_spawns_one_per_name(monkeypatch) -> None:
    spawned: list[list[str]] = []
    fakes: list[_FakeProcess] = []

    async def _fake_create(*argv: str, **_kwargs: Any) -> _FakeProcess:
        spawned.append(list(argv))
        f = _FakeProcess(argv[-2])  # the name arg
        fakes.append(f)
        return f

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create)
    names = ["meeting-scribe-5cb2.local", "meeting-1618.local"]

    publishers = asyncio.run(mdns.publish_aliases(names))
    assert len(publishers) == 2
    assert {a[-2] for a in spawned} == set(names)
    for argv in spawned:
        assert argv[0] == "avahi-publish-address"
        assert argv[-1] == mdns._DEFAULT_AP_IP


def test_publish_aliases_swallows_spawn_failure(monkeypatch, caplog) -> None:
    """A spawn failure for one name MUST NOT block the rest."""

    async def _fail_spawn(*argv: str, **_kwargs: Any) -> _FakeProcess:
        raise FileNotFoundError("avahi-publish-address not on PATH")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fail_spawn)
    with caplog.at_level("WARNING"):
        publishers = asyncio.run(mdns.publish_aliases(["meeting-1618.local"]))
    assert publishers == []
    assert any("avahi-publish-address spawn failed" in m for m in caplog.messages)


def test_stop_aliases_terminates_running_publishers() -> None:
    """terminate() on every still-running publisher; wait() completes."""

    p1 = _FakeProcess("a.local")
    p2 = _FakeProcess("b.local")

    async def _go() -> None:
        await mdns.stop_aliases([p1, p2])

    asyncio.run(_go())
    assert p1._terminated and p2._terminated


def test_stop_aliases_is_idempotent_for_dead_children() -> None:
    """Already-exited publisher must not be re-terminated."""

    dead = _FakeProcess("a.local")
    dead.returncode = 0  # already done
    asyncio.run(mdns.stop_aliases([dead]))
    assert dead._terminated is False  # never asked to die again


def test_required_leaf_dns_sans_includes_pin_form(monkeypatch) -> None:
    """The cert SAN list must include both the engineering id-form and
    the operator-friendly ``meeting-<pin>.local`` form so a fresh
    cert covers both mDNS names without a browser warning."""
    from meeting_scribe.cli import _common

    monkeypatch.setattr(_common, "_read_or_mint_appliance_id", lambda: "5cb2f6f7594bd386")
    sans = _common._required_leaf_dns_sans()
    assert "meeting-scribe-5cb2.local" in sans
    pin = _common.appliance_pin()
    assert f"meeting-{pin}.local" in sans
    # Two distinct names — never collapse them.
    assert len(sans) == 2
