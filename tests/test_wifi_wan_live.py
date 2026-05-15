"""Phase F — live WAN tests, gated to real GB10 hardware.

These tests are auto-skipped without the ``real_gb10`` marker selection.
They run the actual ``nmcli`` / ``ip`` / ``iptables`` commands so they
need to be executed on a GB10 with the MT7925 radio + a known-reachable
upstream WiFi (Yunomotocho test SSID with PSK in the age store).

Run on hardware via:

    pytest -m real_gb10 tests/test_wifi_wan_live.py

Each test prints its assertion failure with an actionable hint mapped
back to the manual checklist in tests/manual/wifi_wan_cutover.md so an
operator running the suite can correlate a Python failure to a manual
step.
"""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

requires_gb10 = pytest.mark.real_gb10

# Sentinel env var so the test can opt-in to destructive WAN cycling on
# the live box. By default these tests run in "inspect only" mode: they
# observe current state but don't bring up/down WAN. CI never sees this
# env var; on the GB10 the operator sets it for the cutover dry-run.
LIVE_CYCLE = os.environ.get("WIFI_WAN_LIVE_CYCLE") == "1"


@requires_gb10
def test_no_v6_address_on_wlan_sta_when_wan_up() -> None:
    """Hard-stop: wlan_sta MUST have zero v6 addresses while WAN is up.

    Maps to manual checklist §2 (third checkbox).
    """
    proc = subprocess.run(
        ["ip", "-6", "addr", "show", "wlan_sta"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("wlan_sta not present — bring it up first via wan_up")
    # We expect ONLY `inet6` headers, no actual addresses. If the
    # interface has any inet6 entry, the v6 disable failed.
    lines = [line for line in proc.stdout.splitlines() if "inet6" in line]
    assert lines == [], f"v6 addresses present on wlan_sta — hard stop: {lines}"


@requires_gb10
def test_ip6tables_forward_policy_is_drop() -> None:
    """Hard-stop: v6 FORWARD policy must be DROP under gateway apply.

    Maps to manual checklist §5.
    """
    proc = subprocess.run(
        ["sudo", "ip6tables", "-L", "FORWARD", "-n"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("ip6tables not callable in this environment")
    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    assert "policy DROP" in first_line, f"ip6tables FORWARD policy != DROP: {first_line!r}"


@requires_gb10
def test_no_v6_nat_rules_in_ip6tables() -> None:
    """v6 NAT66 must NOT be emitted (we don't forward v6 in v1)."""
    proc = subprocess.run(
        ["sudo", "ip6tables", "-t", "nat", "-S", "POSTROUTING"],
        capture_output=True,
        text=True,
        check=False,
    )
    # rc != 0 typically means the v6 nat table isn't loaded; that's
    # acceptable. The bad case is loaded AND has MASQUERADE rules.
    if proc.returncode != 0:
        return
    for line in proc.stdout.splitlines():
        assert "MASQUERADE" not in line, f"unexpected v6 MASQUERADE: {line!r}"


@requires_gb10
def test_sysctl_v4_forward_v6_off_when_gateway() -> None:
    """Persistent sysctl conf must reflect the egress_mode contract."""
    from pathlib import Path

    conf = Path("/etc/sysctl.d/99-meeting-scribe-gateway.conf")
    if not conf.exists():
        pytest.skip(f"{conf} not yet written — bring up WAN once first")
    body = conf.read_text()
    assert "net.ipv6.conf.all.forwarding = 0" in body
    # v4_forward depends on current egress_mode — both 0 and 1 are valid
    # but the *line* must be present.
    assert "net.ipv4.ip_forward" in body


@requires_gb10
def test_ms_fw_post_routing_present_when_gateway() -> None:
    """If egress_mode is gateway, MASQUERADE for the hotspot subnet exists."""
    from meeting_scribe.server_support import settings_store

    if settings_store._effective_wan_egress_mode() != "gateway":
        pytest.skip("egress_mode is not gateway — skip POSTROUTING assertion")
    proc = subprocess.run(
        ["sudo", "iptables", "-t", "nat", "-S", "POSTROUTING"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    masq_lines = [l for l in proc.stdout.splitlines() if "ms-fw" in l and "MASQUERADE" in l]
    assert masq_lines, "expected at least one ms-fw MASQUERADE rule in gateway mode"


@requires_gb10
def test_exactly_one_owned_sta_profile_when_active() -> None:
    """Final invariant: exactly one ``meeting-scribe-sta-<id>`` profile.

    Maps to manual checklist §2 (first checkbox) and §8 (final checkbox).
    """
    from meeting_scribe.server_support import settings_store

    active = settings_store._effective_wan_active_profile_id()
    if active is None:
        pytest.skip("no active WAN profile — nothing to count")
    proc = subprocess.run(
        ["nmcli", "-t", "-f", "NAME", "con", "show"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    owned = [line for line in proc.stdout.splitlines() if line.startswith("meeting-scribe-sta-")]
    assert owned == [f"meeting-scribe-sta-{active}"], (
        f"expected exactly one owned profile matching the active id, got {owned}"
    )


@requires_gb10
def test_reconcile_idempotence_live() -> None:
    """Five sequential reconciles must leave iptables-save unchanged."""
    import meeting_scribe.wifi as wifi

    snapshots: list[str] = []
    for _ in range(5):
        asyncio.run(wifi.reconcile_network_state())
        proc = subprocess.run(
            ["sudo", "iptables-save"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Filter to ms-fw-tagged rules so kernel auto-bookkeeping (counter
        # bytes, accept/drop counters) doesn't trip the assertion.
        filtered = sorted(line for line in proc.stdout.splitlines() if "ms-fw" in line)
        snapshots.append("\n".join(filtered))
    for i, snap in enumerate(snapshots[1:], start=1):
        assert snap == snapshots[0], f"reconcile drift at iteration {i}"


@requires_gb10
@pytest.mark.skipif(not LIVE_CYCLE, reason="WIFI_WAN_LIVE_CYCLE=1 not set")
def test_post_reboot_wan_up_does_not_duplicate_profile() -> None:
    """Plan hard-stop §6 — modifies-in-place over a preserved profile.

    DESTRUCTIVE: brings the WAN down + back up. Gated behind
    ``WIFI_WAN_LIVE_CYCLE=1`` env so accidental runs don't cycle a
    production STA association.
    """
    from meeting_scribe import wifi_wan
    from meeting_scribe.server_support import settings_store

    active = settings_store._effective_wan_active_profile_id()
    if active is None:
        pytest.skip("no active WAN profile to exercise")
    # Bring it up once — confirms no add-collision when the profile
    # already exists on disk (post-reboot recovery path).
    asyncio.run(wifi_wan.wan_up(active))
    proc = subprocess.run(
        ["nmcli", "-t", "-f", "NAME", "con", "show"],
        capture_output=True,
        text=True,
        check=False,
    )
    owned = [
        line for line in proc.stdout.splitlines() if line.startswith(f"meeting-scribe-sta-{active}")
    ]
    assert len(owned) == 1, f"duplicate or missing owned profile: {owned}"
