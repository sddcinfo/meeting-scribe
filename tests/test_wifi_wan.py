"""Phase B — wifi_wan.py orchestrator.

Tests mock the nmcli/iw/curl/sysctl shell-outs via the
:data:`NMCLI_RUNNER` seam. No network access, no real subprocess.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

import meeting_scribe.wifi_wan as wifi_wan
from meeting_scribe.server_support import secrets

# ─── Common fixtures ──────────────────────────────────────────


@pytest.fixture
def fresh_settings(tmp_path: Path, monkeypatch):
    """Pin settings_store to a per-test JSON file."""
    settings_path = tmp_path / "settings.json"
    import meeting_scribe.server_support.settings_store as store

    monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
    monkeypatch.setattr(store, "_settings_cache", None)
    monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
    monkeypatch.setattr(store, "_legacy_migration_attempted", False)
    return store


@pytest.fixture
def stub_psk_resolver(monkeypatch):
    """Hand back a deterministic PSK without invoking decrypt-creds.sh."""

    def _resolve(psk_ref: str) -> str:
        return f"plaintext-for-{psk_ref}"

    monkeypatch.setattr(secrets, "resolve_psk", _resolve)
    return _resolve


_VALID_UUID = "12345678-1234-4567-8901-123456789abc"
_VALID_UUID_2 = "abcdef01-2345-4789-89ab-cdef01234567"


def _build_profile(profile_id: str = _VALID_UUID, **overrides: Any) -> dict:
    base = {
        "id": profile_id,
        "ssid": "Yunomotocho",
        "bssid": None,
        "band": "auto",
        "psk_ref": "YUNOMOTOCHO_PSK",
        "regdomain": "JP",
        "last_seen": None,
    }
    base.update(overrides)
    return base


class FakeRunner:
    """Records argv calls and returns staged responses keyed by argv-prefix."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Default stub: rc=0, empty output. Specific responders below.
        self._responders: list[tuple[Callable[[list[str]], bool], subprocess.CompletedProcess]] = []
        # Simulated NM connection set (con-name -> True). add/delete update it.
        self.nm_connections: set[str] = set()

    def respond(
        self,
        match: Callable[[list[str]], bool],
        *,
        rc: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        proc = subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)
        self._responders.append((match, proc))

    def respond_prefix(self, prefix: list[str], **kwargs: Any) -> None:
        def _matches(argv: list[str]) -> bool:
            return argv[: len(prefix)] == prefix

        self.respond(_matches, **kwargs)

    async def __call__(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))

        # Auto-track simulated NM state for show/add/delete.
        if argv[:4] == ["nmcli", "-t", "-f", "NAME"] and argv[4:6] == ["con", "show"]:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="\n".join(sorted(self.nm_connections)) + "\n",
                stderr="",
            )
        if argv[:3] == ["nmcli", "con", "add"]:
            try:
                name_idx = argv.index("con-name") + 1
                self.nm_connections.add(argv[name_idx])
            except ValueError:
                pass
        if argv[:3] == ["nmcli", "con", "delete"]:
            name = argv[3] if len(argv) > 3 else ""
            self.nm_connections.discard(name)

        # Then any custom responder.
        for match, resp in self._responders:
            if match(argv):
                return subprocess.CompletedProcess(
                    args=argv, returncode=resp.returncode, stdout=resp.stdout, stderr=resp.stderr
                )
        # Default: rc=0, empty output.
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")


@pytest.fixture
def fake_runner(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(wifi_wan, "NMCLI_RUNNER", runner)
    return runner


@pytest.fixture
def stub_sta_iface_ensure(monkeypatch):
    """Pretend the wlan_sta virtual interface is always available."""

    async def _ensure(**_kwargs):
        return True

    monkeypatch.setattr("meeting_scribe.wifi_sta.sta_iface_ensure", _ensure)


@pytest.fixture
def stub_reconcile(monkeypatch):
    """Capture reconcile-hook invocations without running the real fn."""

    counter = {"calls": 0}

    async def _stub():
        counter["calls"] += 1

    monkeypatch.setattr(wifi_wan, "RECONCILE_HOOK", _stub)
    return counter


# ─── wan_up: fresh state ──────────────────────────────────────


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


def test_wan_up_fresh_state_adds_profile(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile())

    _run(wifi_wan.wan_up(_VALID_UUID))

    # Argv-level assertions
    add_calls = [c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"]]
    assert len(add_calls) == 1, f"expected exactly one add, got {add_calls!r}"
    add = add_calls[0]
    name = f"meeting-scribe-sta-{_VALID_UUID}"
    assert name in add
    assert "ipv6.method" in add and add[add.index("ipv6.method") + 1] == "disabled"
    assert "wifi-sec.psk" in add
    psk = add[add.index("wifi-sec.psk") + 1]
    assert psk == "plaintext-for-YUNOMOTOCHO_PSK"
    assert "connection.autoconnect" in add
    assert add[add.index("connection.autoconnect") + 1] == "no"
    assert "ipv4.route-metric" in add
    assert add[add.index("ipv4.route-metric") + 1] == "600"

    # Settings updated
    assert fresh_settings._effective_wan_active_profile_id() == _VALID_UUID
    # Reconcile fired
    assert stub_reconcile["calls"] >= 1


def test_wan_up_creates_owned_profile_with_ipv6_disabled(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile())
    _run(wifi_wan.wan_up(_VALID_UUID))
    add = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"])
    # Plan hard-stop: must be "disabled", not "ignore".
    assert add[add.index("ipv6.method") + 1] == "disabled"


def test_wan_up_sets_per_iface_v6_sysctl_before_up(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile())
    _run(wifi_wan.wan_up(_VALID_UUID))
    sysctl_calls = [c for c in fake_runner.calls if c[:3] == ["sudo", "sysctl", "-w"]]
    up_calls = [c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "up"]]
    assert any("net.ipv6.conf.wlan_sta.disable_ipv6=1" in c[3] for c in sysctl_calls)
    # The sysctl write must come BEFORE any nmcli con up.
    first_sysctl = fake_runner.calls.index(sysctl_calls[0])
    first_up = fake_runner.calls.index(up_calls[0])
    assert first_sysctl < first_up


# ─── wan_up: preserved profile (post-reboot recovery) ────────


def test_wan_up_preserved_profile_modifies_in_place(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    """Critical regression: post-reboot the NM keyfile still exists on
    disk. wan_up must NOT run ``con add`` (would error or duplicate)
    — it must ``con modify`` in place. Plan hard-stop #6.
    """
    fresh_settings._save_wan_profile(_build_profile())
    # Simulate post-reboot: profile already on disk, settings already
    # point at it.
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)

    _run(wifi_wan.wan_up(_VALID_UUID))

    add_calls = [c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"]]
    modify_calls = [c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "modify"]]
    assert add_calls == [], "must not con-add a profile that already exists"
    assert len(modify_calls) >= 1
    assert any(f"meeting-scribe-sta-{_VALID_UUID}" in c for c in modify_calls)

    # Final invariant: exactly one owned profile, named correctly.
    owned = [n for n in fake_runner.nm_connections if n.startswith("meeting-scribe-sta-")]
    assert owned == [f"meeting-scribe-sta-{_VALID_UUID}"]


def test_wan_up_profile_id_switch_deletes_old(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    """Switching from active=A to active=B must leave exactly one owned profile."""
    fresh_settings._save_wan_profile(_build_profile(_VALID_UUID))
    fresh_settings._save_wan_profile(_build_profile(_VALID_UUID_2, ssid="Other"))
    # Simulate A currently active + on disk.
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)

    _run(wifi_wan.wan_up(_VALID_UUID_2))

    owned = sorted(n for n in fake_runner.nm_connections if n.startswith("meeting-scribe-sta-"))
    assert owned == [f"meeting-scribe-sta-{_VALID_UUID_2}"]
    assert fresh_settings._effective_wan_active_profile_id() == _VALID_UUID_2


def test_wan_up_unknown_id_raises(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    with pytest.raises(ValueError, match="no wan profile"):
        _run(wifi_wan.wan_up(_VALID_UUID))


# ─── wan_down ─────────────────────────────────────────────────


def test_wan_down_removes_only_owned_profiles(
    fresh_settings, fake_runner, stub_psk_resolver, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile())
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fake_runner.nm_connections.add("Wired connection 3")  # system profile, must not delete

    _run(wifi_wan.wan_down())

    assert "Wired connection 3" in fake_runner.nm_connections
    assert f"meeting-scribe-sta-{_VALID_UUID}" not in fake_runner.nm_connections
    assert fresh_settings._effective_wan_active_profile_id() is None


def test_wan_down_idempotent_when_nothing_active(
    fresh_settings, fake_runner, stub_psk_resolver, stub_reconcile
) -> None:
    """Calling wan_down without an active profile is a clean no-op."""
    _run(wifi_wan.wan_down())  # no exception
    assert fresh_settings._effective_wan_active_profile_id() is None


# ─── cleanup_orphan_sta_profiles ─────────────────────────────


def test_cleanup_orphan_sta_profiles_at_boot(fresh_settings, fake_runner, stub_reconcile) -> None:
    """An orphan (id != active) gets deleted at boot."""
    fresh_settings._save_wan_profile(_build_profile())
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID_2}")  # stale orphan
    fake_runner.nm_connections.add("Wired connection 3")  # bystander

    deleted = _run(wifi_wan.cleanup_orphan_sta_profiles())

    assert deleted == 1
    # Active stays. Bystander stays. Orphan gone.
    assert f"meeting-scribe-sta-{_VALID_UUID}" in fake_runner.nm_connections
    assert f"meeting-scribe-sta-{_VALID_UUID_2}" not in fake_runner.nm_connections
    assert "Wired connection 3" in fake_runner.nm_connections


def test_cleanup_preserves_active_profile_across_reboot(
    fresh_settings, fake_runner, stub_reconcile
) -> None:
    """Plan hard-stop boundary: the active profile MUST persist across reboot.

    v1 design accepts the PSK exposure trade-off — admin doesn't have
    to re-``wan up`` after reboot. If cleanup ever deletes the active
    profile, that contract is broken.
    """
    fresh_settings._save_wan_profile(_build_profile())
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")

    deleted = _run(wifi_wan.cleanup_orphan_sta_profiles())

    assert deleted == 0
    assert f"meeting-scribe-sta-{_VALID_UUID}" in fake_runner.nm_connections


def test_cleanup_with_no_active_profile_wipes_all_owned(
    fresh_settings, fake_runner, stub_reconcile
) -> None:
    """If no profile is active, every owned profile is an orphan."""
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID_2}")
    deleted = _run(wifi_wan.cleanup_orphan_sta_profiles())
    assert deleted == 2
    owned = [n for n in fake_runner.nm_connections if n.startswith("meeting-scribe-sta-")]
    assert owned == []


# ─── claim_wired_profile ─────────────────────────────────────


def test_claim_wired_profile_returns_active_name(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        stdout="lo:lo\nWired connection 3:enP7s7\ndocker0:docker0\n",
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "ipv4.route-metric", "con", "show", "Wired connection 3"],
        stdout="ipv4.route-metric:-1\n",
    )
    claim = _run(wifi_wan.claim_wired_profile())
    assert claim is not None
    assert claim.name == "Wired connection 3"
    # NM convention: -1 = unset (auto). We surface as None.
    assert claim.prior_metric is None


def test_claim_wired_profile_returns_none_when_no_active(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        stdout="lo:lo\n",
    )
    assert _run(wifi_wan.claim_wired_profile()) is None


def test_claim_wired_profile_idempotent(fake_runner) -> None:
    """Two consecutive claims must yield identical results."""
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        stdout="Wired connection 3:enP7s7\n",
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "ipv4.route-metric", "con", "show", "Wired connection 3"],
        stdout="ipv4.route-metric:600\n",
    )
    first = _run(wifi_wan.claim_wired_profile())
    second = _run(wifi_wan.claim_wired_profile())
    assert first == second
    assert first is not None and first.prior_metric == 600


def test_enforce_wired_metric_skips_when_already_at_target(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "ipv4.route-metric", "con", "show", "Wired connection 3"],
        stdout="ipv4.route-metric:100\n",
    )
    changed = _run(wifi_wan.enforce_wired_metric("Wired connection 3", target=100))
    assert changed is False
    # No con modify issued.
    assert not any(c[:3] == ["nmcli", "con", "modify"] for c in fake_runner.calls)


def test_enforce_wired_metric_applies_when_drifted(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "ipv4.route-metric", "con", "show", "Wired connection 3"],
        stdout="ipv4.route-metric:600\n",
    )
    changed = _run(wifi_wan.enforce_wired_metric("Wired connection 3", target=100))
    assert changed is True
    modify_calls = [c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "modify"]]
    assert any(
        "ipv4.route-metric" in c and c[c.index("ipv4.route-metric") + 1] == "100"
        for c in modify_calls
    )


# ─── wan_status: per-iface independence (P1 captive-mask fix) ──


def test_status_per_iface_connectivity(
    fresh_settings, fake_runner, stub_reconcile, monkeypatch
) -> None:
    """Wired-full + wlan_sta-portal must be reported correctly per-iface."""
    fake_runner.respond_prefix(
        ["ip", "-br", "link", "show", "enP7s7"],
        stdout="enP7s7           UP             4c:c5:d9:bd:f4:26\n",
    )
    fake_runner.respond_prefix(
        ["ip", "-br", "link", "show", "wlan_sta"],
        stdout="wlan_sta         UP             50:bb:b5:50:ba:62\n",
    )
    fake_runner.respond_prefix(
        ["ip", "-4", "-br", "addr", "show", "enP7s7"],
        stdout="enP7s7           UP             192.168.8.153/24\n",
    )
    fake_runner.respond_prefix(
        ["ip", "-4", "-br", "addr", "show", "wlan_sta"],
        stdout="wlan_sta         UP             10.0.0.42/24\n",
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"],
        stdout="Wired connection 3:enP7s7\nmeeting-scribe-sta-12345678-1234-4567-8901-123456789abc:wlan_sta\n",
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "ipv4.route-metric", "con", "show", "Wired connection 3"],
        stdout="ipv4.route-metric:100\n",
    )
    fake_runner.respond_prefix(
        [
            "nmcli",
            "-t",
            "-f",
            "ipv4.route-metric",
            "con",
            "show",
            "meeting-scribe-sta-12345678-1234-4567-8901-123456789abc",
        ],
        stdout="ipv4.route-metric:600\n",
    )
    fake_runner.respond_prefix(
        ["ip", "-4", "route", "show", "default"],
        stdout=(
            "default via 192.168.8.1 dev enP7s7 proto dhcp metric 100\n"
            "default via 10.0.0.1 dev wlan_sta proto dhcp metric 600\n"
        ),
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "IP4.CAPTIVE-PORTAL", "dev", "show", "wlan_sta"],
        stdout="IP4.CAPTIVE-PORTAL:https://venue.example/login\n",
    )
    fake_runner.respond_prefix(
        ["iw", "dev", "wlan_sta", "link"],
        stdout=(
            "Connected to aa:bb:cc:dd:ee:ff (on wlan_sta)\n\tSSID: Yunomotocho\n\tsignal: -55 dBm\n"
        ),
    )
    # Curl probe returns a 30x to the portal — per-iface captive.
    fake_runner.respond_prefix(
        ["curl", "--interface", "wlan_sta"],
        stdout="\n302\nhttps://venue.example/login\n",
    )

    # Simulate post-`wan up` state.
    fresh_settings._save_wan_profile(_build_profile())
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)
    # And settings has the active set so wifi.profile_name resolves.
    fake_runner.nm_connections.add("meeting-scribe-sta-12345678-1234-4567-8901-123456789abc")

    monkeypatch.setattr(
        wifi_wan.Path,
        "exists",
        lambda self: str(self) == "/sys/class/net/wlan_sta" or True,
    )
    status = _run(wifi_wan.wan_status())

    assert status["wired"]["up"] is True
    assert status["wired"]["lease"] == "192.168.8.153"
    assert status["wired"]["default_route"] is True

    assert status["wifi"]["up"] is True
    assert status["wifi"]["lease"] == "10.0.0.42"
    # Per-iface captive — wired being full must NOT mask wlan_sta portal.
    assert status["wifi"]["connectivity"] == "portal"
    assert status["wifi"]["portal_url"] == "https://venue.example/login"
    assert status["wifi"]["default_route"] is False

    assert status["active_default"] == "enP7s7"


def test_captive_portal_url_parsed_from_nmcli_dev_show(fake_runner) -> None:
    """Always read CAPPORT via per-device ``IP4.CAPTIVE-PORTAL``."""
    fake_runner.respond_prefix(
        ["ip", "-br", "link", "show", "wlan_sta"],
        stdout="wlan_sta UP\n",
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "IP4.CAPTIVE-PORTAL", "dev", "show", "wlan_sta"],
        stdout="IP4.CAPTIVE-PORTAL:https://capport.example/portal\n",
    )
    fake_runner.respond_prefix(
        ["curl", "--interface", "wlan_sta"],
        stdout="NetworkManager is online\n200\n\n",
    )
    conn, url = _run(wifi_wan.probe_wlan_sta_connectivity())
    # 200 + expected body wins regardless of CAPPORT URL.
    assert conn == "full"
    assert url is None


def test_probe_returns_full_for_200_with_expected_body(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["ip", "-br", "link", "show", "wlan_sta"],
        stdout="wlan_sta UP\n",
    )
    fake_runner.respond_prefix(
        ["nmcli", "-t", "-f", "IP4.CAPTIVE-PORTAL", "dev", "show", "wlan_sta"],
        stdout="IP4.CAPTIVE-PORTAL:--\n",
    )
    fake_runner.respond_prefix(
        ["curl", "--interface", "wlan_sta"],
        stdout="NetworkManager is online\n\n200\n\n",
    )
    conn, url = _run(wifi_wan.probe_wlan_sta_connectivity())
    assert conn == "full"
    assert url is None


def test_probe_returns_none_when_link_down(fake_runner) -> None:
    fake_runner.respond_prefix(
        ["ip", "-br", "link", "show", "wlan_sta"],
        stdout="wlan_sta DOWN\n",
    )
    conn, url = _run(wifi_wan.probe_wlan_sta_connectivity())
    assert conn == "none"
    assert url is None


# ─── nmcli-based scan ──────────────────────────────────────────


def test_nmcli_terse_split_handles_escaped_colons() -> None:
    """``nmcli -t`` escapes colons inside fields with ``\\:`` — the parser
    must reassemble BSSIDs (which contain colons) correctly."""
    line = r"DellEfficiency_Guest:CC\:DB\:93\:C0\:4F\:A1:89:1:"
    fields = wifi_wan._split_nmcli_terse(line)
    assert fields == ["DellEfficiency_Guest", "CC:DB:93:C0:4F:A1", "89", "1", ""]


def test_parse_nmcli_wifi_list_yields_scan_entries() -> None:
    sample = (
        "DellEfficiency_Guest:CC\\:DB\\:93\\:C0\\:4F\\:A1:89:1:\n"
        "DellEfficiency_Corp:CC\\:DB\\:93\\:C0\\:7C\\:8F:57:36:WPA2 802.1X\n"
        ":FF\\:FF\\:FF\\:FF\\:FF\\:FF:10:1:\n"
    )
    entries = wifi_wan._parse_nmcli_wifi_list(sample)
    assert len(entries) == 2
    guest = entries[0]
    assert guest.ssid == "DellEfficiency_Guest"
    assert guest.bssid == "cc:db:93:c0:4f:a1"
    assert guest.channel == 1
    assert guest.rsn_present is False
    corp = entries[1]
    assert corp.ssid == "DellEfficiency_Corp"
    assert corp.channel == 36
    assert corp.rsn_present is True


def test_scan_upstream_uses_wlp9s9_when_sta_iface_missing(monkeypatch, fake_runner) -> None:
    """When ``wlan_sta`` doesn't exist, scan against the AP iface (wlP9s9)."""
    monkeypatch.setattr(
        "meeting_scribe.wifi_wan.Path",
        type(
            "FakePath",
            (),
            {
                "__init__": lambda self, p: setattr(self, "_p", p),
                "exists": lambda self: not str(self._p).endswith("/wlan_sta"),
            },
        ),
    )
    fake_runner.respond_prefix(
        [
            "nmcli",
            "-t",
            "-f",
            "SSID,BSSID,SIGNAL,CHAN,SECURITY",
            "dev",
            "wifi",
            "list",
            "ifname",
            "wlP9s9",
        ],
        stdout="DellEfficiency_Guest:AA\\:BB\\:CC\\:DD\\:EE\\:FF:80:36:\n",
    )
    entries = _run(wifi_wan.scan_upstream())
    assert len(entries) == 1
    assert entries[0].ssid == "DellEfficiency_Guest"
    list_call = next(c for c in fake_runner.calls if "list" in c and "ifname" in c)
    assert list_call[list_call.index("ifname") + 1] == "wlP9s9"


# ─── multi-BSSID profile disambiguation ──────────────────────


def test_profile_id_disambiguates_dup_ssid(fresh_settings) -> None:
    """Two profiles can share an SSID but differ in BSSID; CLI/REST
    resolves by uuid4 id, never SSID alone.
    """
    fresh_settings._save_wan_profile(_build_profile(_VALID_UUID))
    fresh_settings._save_wan_profile(_build_profile(_VALID_UUID_2, bssid="aa:bb:cc:dd:ee:ff"))
    by_ssid = fresh_settings._find_wan_profiles_by_ssid("Yunomotocho")
    assert {p["id"] for p in by_ssid} == {_VALID_UUID, _VALID_UUID_2}
    # Lookup by id is unambiguous.
    assert fresh_settings._find_wan_profile_by_id(_VALID_UUID)["bssid"] is None
    assert fresh_settings._find_wan_profile_by_id(_VALID_UUID_2)["bssid"] == "aa:bb:cc:dd:ee:ff"


# ─── Band classification + scan consolidation ─────────────────


def test_channel_to_band_24ghz() -> None:
    for ch in (1, 6, 11, 13, 14):
        assert wifi_wan.channel_to_band(ch) == "bg"


def test_channel_to_band_5ghz() -> None:
    for ch in (36, 64, 100, 149, 165):
        assert wifi_wan.channel_to_band(ch) == "a"


def test_consolidate_scan_groups_by_ssid_and_security() -> None:
    from meeting_scribe.wifi_sta import ScanEntry

    entries = [
        ScanEntry(
            bssid="aa:00:00:00:00:01", ssid="Foo", channel=36, signal_dbm=-50.0, rsn_present=True
        ),
        ScanEntry(
            bssid="aa:00:00:00:00:02", ssid="Foo", channel=1, signal_dbm=-72.0, rsn_present=True
        ),
        ScanEntry(
            bssid="aa:00:00:00:00:03", ssid="Foo", channel=149, signal_dbm=-65.0, rsn_present=True
        ),
        # Same SSID but OPEN — treated as a separate network.
        ScanEntry(
            bssid="bb:00:00:00:00:01", ssid="Foo", channel=6, signal_dbm=-80.0, rsn_present=False
        ),
        ScanEntry(
            bssid="cc:00:00:00:00:01", ssid="Bar", channel=6, signal_dbm=-40.0, rsn_present=True
        ),
    ]
    groups = wifi_wan.consolidate_scan(entries)
    by_key = {(g.ssid, g.rsn_present): g for g in groups}

    foo_secure = by_key[("Foo", True)]
    assert foo_secure.ap_count == 3
    assert foo_secure.bands == ("a", "bg")  # sorted; both bands present
    assert foo_secure.best_signal_dbm == -50.0
    assert foo_secure.best_signal_band == "a"  # the -50 BSS is on channel 36
    assert foo_secure.channels == (1, 36, 149)

    foo_open = by_key[("Foo", False)]
    assert foo_open.ap_count == 1
    assert foo_open.bands == ("bg",)

    # Output is sorted by best signal descending — Bar (-40) > Foo-secure (-50).
    assert groups[0].ssid == "Bar"


def test_consolidate_scan_drops_empty_ssid() -> None:
    from meeting_scribe.wifi_sta import ScanEntry

    entries = [
        ScanEntry(
            bssid="aa:00:00:00:00:01", ssid="", channel=6, signal_dbm=-60.0, rsn_present=True
        ),
        ScanEntry(
            bssid="aa:00:00:00:00:02", ssid="Real", channel=6, signal_dbm=-60.0, rsn_present=True
        ),
    ]
    groups = wifi_wan.consolidate_scan(entries)
    assert [g.ssid for g in groups] == ["Real"]


# ─── Band wiring into NM profile add/modify ───────────────────


def test_wan_up_with_5ghz_band_sets_wifi_band_a(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile(band="a"))
    _run(wifi_wan.wan_up(_VALID_UUID))
    add = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"])
    assert "wifi.band" in add
    assert add[add.index("wifi.band") + 1] == "a"


def test_wan_up_with_2ghz_band_sets_wifi_band_bg(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile(band="bg"))
    _run(wifi_wan.wan_up(_VALID_UUID))
    add = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"])
    assert "wifi.band" in add
    assert add[add.index("wifi.band") + 1] == "bg"


def test_wan_up_with_auto_band_omits_wifi_band(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    fresh_settings._save_wan_profile(_build_profile(band="auto"))
    _run(wifi_wan.wan_up(_VALID_UUID))
    add = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"])
    assert "wifi.band" not in add


def test_wan_up_preserved_profile_clears_wifi_band_on_auto(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    """Switching a preserved profile from a fixed band back to auto
    must emit ``-wifi.band ""`` so NM clears any prior pin."""
    fresh_settings._save_wan_profile(_build_profile(band="auto"))
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)
    _run(wifi_wan.wan_up(_VALID_UUID))
    mod = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "modify"])
    assert "-wifi.band" in mod


# ─── Open-network support ──────────────────────────────────────


def test_wan_up_open_network_omits_wifi_sec(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    """OPEN profile must NOT pass wifi-sec.* to nmcli con add."""
    fresh_settings._save_wan_profile(_build_profile(ssid="DellEfficiency_Guest", psk_ref=None))
    _run(wifi_wan.wan_up(_VALID_UUID))
    add = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"])
    assert "wifi-sec.key-mgmt" not in add
    assert "wifi-sec.psk" not in add


def test_wan_up_open_network_skips_psk_resolver(
    fresh_settings, fake_runner, stub_sta_iface_ensure, stub_reconcile, monkeypatch
) -> None:
    """resolve_psk MUST NOT be called for an OPEN profile."""
    called = {"count": 0}

    def _spy(psk_ref):
        called["count"] += 1
        raise AssertionError(f"resolve_psk called for OPEN profile (ref={psk_ref!r})")

    monkeypatch.setattr(secrets, "resolve_psk", _spy)
    fresh_settings._save_wan_profile(_build_profile(ssid="DellEfficiency_Guest", psk_ref=None))
    _run(wifi_wan.wan_up(_VALID_UUID))
    assert called["count"] == 0


def test_wan_up_open_preserved_profile_clears_wifi_sec(
    fresh_settings, fake_runner, stub_sta_iface_ensure, stub_reconcile, monkeypatch
) -> None:
    """Switching a preserved profile from WPA-PSK to OPEN must emit
    ``-wifi-sec.key-mgmt`` and ``-wifi-sec.psk`` to drop the prior
    security config."""
    fresh_settings._save_wan_profile(_build_profile(ssid="DellEfficiency_Guest", psk_ref=None))
    fake_runner.nm_connections.add(f"meeting-scribe-sta-{_VALID_UUID}")
    fresh_settings._set_wan_active_profile_id(_VALID_UUID)
    monkeypatch.setattr(secrets, "resolve_psk", lambda ref: "must-not-be-called")
    _run(wifi_wan.wan_up(_VALID_UUID))
    mod = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "modify"])
    assert "-wifi-sec.key-mgmt" in mod
    assert "-wifi-sec.psk" in mod
    # Positive sec args must NOT be present.
    assert "wpa-psk" not in mod


def test_wan_up_wpa_network_still_sends_psk(
    fresh_settings, fake_runner, stub_psk_resolver, stub_sta_iface_ensure, stub_reconcile
) -> None:
    """Regression: the OPEN branch must not break the existing WPA path."""
    fresh_settings._save_wan_profile(_build_profile())  # has psk_ref by default
    _run(wifi_wan.wan_up(_VALID_UUID))
    add = next(c for c in fake_runner.calls if c[:3] == ["nmcli", "con", "add"])
    assert "wifi-sec.key-mgmt" in add
    assert add[add.index("wifi-sec.key-mgmt") + 1] == "wpa-psk"
    assert "wifi-sec.psk" in add
