"""Static-analysis test of the hotspot iptables rule emit sites.

The bug class this catches: a long-standing iOS 30-second hotspot-
join hang was caused by hotspot iptables rules using ``-j DROP``
instead of ``-j REJECT --reject-with tcp-reset``. ``DROP`` makes the
client wait the full TCP timeout before failing over; ``REJECT``
returns an immediate RST so the OS captive-portal sheet pops in
under a second.

A pure-Python check by static analysis: walk the source of
``meeting_scribe/wifi.py``, find every iptables rule string, and
assert that every blocking rule uses ``REJECT`` (not ``DROP``).

Why static analysis: the rules are inlined inside ``_apply_*_firewall``
functions that subprocess out to ``sudo iptables``; they're not in a
pure emit function we can call. Refactoring is out of scope for the
QA uplift; static grep gives 95% of the value at 0% of the risk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WIFI_PY = REPO_ROOT / "src" / "meeting_scribe" / "wifi.py"


def _iptables_rule_strings() -> list[str]:
    """Return every f-string literal in wifi.py that looks like an
    iptables rule (starts with ``-I`` or ``-A`` or ``-t``)."""
    if not WIFI_PY.exists():
        pytest.skip(f"missing {WIFI_PY}")
    src = WIFI_PY.read_text()
    # Pull every f-string body. Loose: any "-I "/"-A "/"-t " inside
    # f-string-like quotes. Good enough — wifi.py is the only emitter.
    rules: list[str] = []
    for m in re.finditer(r'f["\']([^"\']*(?:-I|-A|-t)\s+[^"\']*)["\']', src):
        body = m.group(1)
        if "-j " in body or "PREROUTING" in body:
            rules.append(body)
    return rules


def test_at_least_one_iptables_rule_found():
    rules = _iptables_rule_strings()
    assert rules, (
        f"static analysis found no iptables rules in {WIFI_PY.name}. "
        f"Either the file moved or the regex needs an update."
    )


def test_no_blocking_rule_uses_drop():
    """Every blocking rule must use REJECT, not DROP.

    DROP makes iOS wait the full TCP timeout (~30 s) before failing.
    REJECT --reject-with tcp-reset returns an immediate RST so the
    OS captive-portal sheet pops within ~1 s.
    """
    rules = _iptables_rule_strings()
    drop_rules = [r for r in rules if " -j DROP" in r or "\tj DROP" in r]
    assert not drop_rules, (
        f"hotspot rules using -j DROP (must be REJECT --reject-with tcp-reset "
        f"or icmp-port-unreachable): {drop_rules}"
    )


def test_https_rejection_uses_tcp_reset():
    """The 443/admin-port REJECT rules must specify
    ``--reject-with tcp-reset`` so the client gets an instant RST and
    the OS retries on port 80 (which the redirect catches)."""
    rules = _iptables_rule_strings()
    # Find rules that REJECT TCP traffic.
    https_rules = [
        r
        for r in rules
        if "-j REJECT" in r
        and "-p tcp" in r
        and ("--dport 443" in r or "--dport {admin_port}" in r)
    ]
    assert https_rules, "no TCP-REJECT rule found for ports 443/admin"
    for r in https_rules:
        assert "--reject-with tcp-reset" in r, (
            f"TCP REJECT rule missing --reject-with tcp-reset (would silently "
            f"hang iOS for ~30s): {r}"
        )


def test_default_deny_uses_icmp_port_unreachable():
    """The catch-all default-deny REJECT rules should use
    ``--reject-with icmp-port-unreachable`` (proper protocol response,
    not silent drop)."""
    rules = _iptables_rule_strings()
    default_deny = [
        r
        for r in rules
        if "-j REJECT" in r and "-p tcp" not in r and "-p udp" not in r and "-p icmp" not in r
    ]
    # We may have several catch-all REJECT rules (INPUT and FORWARD)
    assert default_deny, "no default-deny REJECT rule found"
    for r in default_deny:
        assert "--reject-with icmp-port-unreachable" in r, (
            f"default-deny REJECT rule missing --reject-with icmp-port-unreachable: {r}"
        )
