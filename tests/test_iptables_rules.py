"""Static-analysis smoke test of the Phase-H firewall emit site.

Phase H replaced the per-mode meeting/admin firewalls (which used
``REJECT --reject-with tcp-reset`` to dodge the iOS 30-s hotspot
hang on port 443) with a single AP-iface-scoped INPUT + FORWARD
allowlist. Ports 80 + 443 are explicitly ACCEPT'd on the AP iface
so the wizard + captive sub-app are reachable; everything else
falls through to ``-j DROP``. iOS no longer probes a rejected 443
because there is no rejection — port 443 is now the wizard.

These tests verify the static layout of ``_apply_simple_firewall``
hasn't drifted: both ``iptables`` and ``ip6tables`` must be
invoked, the AP iface must appear in the rules, ports 80 + 443 must
both be in ACCEPT lists, and ``--comment ms-fw`` must tag every
rule for clean removal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WIFI_PY = REPO_ROOT / "src" / "meeting_scribe" / "wifi.py"


def _wifi_source() -> str:
    if not WIFI_PY.exists():
        pytest.skip(f"missing {WIFI_PY}")
    return WIFI_PY.read_text()


def test_simple_firewall_drives_both_v4_and_v6() -> None:
    """``_apply_simple_firewall`` must invoke BOTH ``iptables`` AND
    ``ip6tables`` so AP clients can't reach a daemon listening on
    ``::`` via IPv6 link-local."""
    src = _wifi_source()
    assert '"iptables", "ip6tables"' in src or '"iptables"' in src
    assert "ip6tables" in src, "Phase H requires ip6tables coverage"


def test_simple_firewall_tags_with_ms_fw_comment() -> None:
    """Every rule must carry ``--comment ms-fw`` so removal is
    exact (no MS_* chains, no positional invariants — just a tag-
    based delete sweep)."""
    src = _wifi_source()
    assert "ms-fw" in src
    assert "_MS_FW_COMMENT" in src


def test_simple_firewall_accepts_443_and_80_on_ap_iface() -> None:
    """The wizard runs on 443; the captive sub-app on 80. Both
    must be ACCEPT'd on the AP iface — the rest of the AP-iface
    traffic falls through to DROP."""
    src = _wifi_source()
    assert '"443"' in src
    assert '"80"' in src
    assert "ACCEPT" in src
    assert "ap_iface" in src or "WIFI_IFACE" in src


def test_simple_firewall_drops_unmatched_ap_traffic() -> None:
    """Catch-all on the AP iface is ``-j DROP`` (silent drop is
    correct here — Phase H trades the iOS-hang-on-REJECT story
    away because there's no longer a 443-reject rule that AP
    clients would hit)."""
    src = _wifi_source()
    assert "DROP" in src


def test_simple_firewall_blocks_ap_to_ap_forwarding() -> None:
    """AP client isolation at L3 — defense-in-depth on top of
    NM ``wifi.ap-isolation=true`` at L2."""
    src = _wifi_source()
    # ``-i ap_iface -o ap_iface ... DROP`` is the L3 isolation rule.
    assert "FORWARD" in src
    # The rule body uses the same iface arg twice — search for that
    # pattern in the rule list.
    assert src.count("ap_iface") >= 4
