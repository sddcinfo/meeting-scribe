"""Unit tests for the MS_* chain firewall layer (Plan 1 + Plan 2 v36 merge).

Pure-code coverage — every helper takes its iptables-save text as a
string argument so we can drive scenarios without root or live iptables.
Hardware-side end-to-end verification (apply → save → assert position-1)
is a hardware-gated follow-up phase.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from meeting_scribe.firewall import (
    MS_CHAINS_FILTER,
    MS_CHAINS_FILTER_V6,
    MS_CHAINS_NAT,
    MS_FW_COMMENT,
    _build_restore_text,
    _emit_table_section,
    _expected_chain_rules,
    _expected_jump_save_line,
    _expected_parent_jumps,
    _extract_managed_jumps,
    _extract_target,
    _is_managed_jump_at_position_1,
    _normalize_save_line,
    _validate_snapshot_complete,
    take_snapshot,
)


# ── Synthetic iptables-save dumps ─────────────────────────────────


def _good_filter_dump(*, sta: bool = False) -> str:
    """Produce a well-formed filter dump matching the canonical
    generator output for ``mode="meeting"``, AP IP ``10.42.0.0/24``."""
    chain_rules = {
        "MS_INPUT": _expected_chain_rules(
            table="filter",
            v6=False,
            chain="MS_INPUT",
            mode="meeting",
            cidr="10.42.0.0/24",
            sta_iface_present=sta,
        ),
        "MS_FWD": _expected_chain_rules(
            table="filter",
            v6=False,
            chain="MS_FWD",
            mode="meeting",
            cidr="10.42.0.0/24",
            sta_iface_present=sta,
        ),
    }
    lines = [
        "*filter",
        ":INPUT ACCEPT [0:0]",
        ":FORWARD ACCEPT [0:0]",
        ":OUTPUT ACCEPT [0:0]",
        ":MS_INPUT - [0:0]",
        ":MS_FWD - [0:0]",
        # Position-1 managed jumps in each parent.
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT",
        f"-A FORWARD -m comment --comment {MS_FW_COMMENT} -j MS_FWD",
        # MS_INPUT body
        *chain_rules["MS_INPUT"],
        # MS_FWD body
        *chain_rules["MS_FWD"],
        "COMMIT",
    ]
    return "\n".join(lines) + "\n"


def _good_nat_dump(*, sta: bool = False) -> str:
    chain_rules = {
        "MS_PRE": _expected_chain_rules(
            table="nat",
            v6=False,
            chain="MS_PRE",
            mode="meeting",
            cidr="10.42.0.0/24",
            sta_iface_present=sta,
        ),
        "MS_POST": _expected_chain_rules(
            table="nat",
            v6=False,
            chain="MS_POST",
            mode="meeting",
            cidr="10.42.0.0/24",
            sta_iface_present=sta,
        ),
    }
    lines = [
        "*nat",
        ":PREROUTING ACCEPT [0:0]",
        ":POSTROUTING ACCEPT [0:0]",
        ":OUTPUT ACCEPT [0:0]",
        ":MS_PRE - [0:0]",
        ":MS_POST - [0:0]",
        f"-A PREROUTING -m comment --comment {MS_FW_COMMENT} -j MS_PRE",
        f"-A POSTROUTING -m comment --comment {MS_FW_COMMENT} -j MS_POST",
        *chain_rules["MS_PRE"],
        *chain_rules["MS_POST"],
        "COMMIT",
    ]
    return "\n".join(lines) + "\n"


def _good_filter_v6_dump() -> str:
    chain_rules = _expected_chain_rules(
        table="filter",
        v6=True,
        chain="MS_INPUT6",
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
    )
    lines = [
        "*filter",
        ":INPUT ACCEPT [0:0]",
        ":FORWARD ACCEPT [0:0]",
        ":OUTPUT ACCEPT [0:0]",
        ":MS_INPUT6 - [0:0]",
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT6",
        *chain_rules,
        "COMMIT",
    ]
    return "\n".join(lines) + "\n"


# ── _extract_target ────────────────────────────────────────────────


def test_extract_target_basic() -> None:
    line = (
        "-A INPUT -i wlP9s9 -s 10.42.0.0/24 -p tcp --dport 443 "
        "-m comment --comment ms-fw-managed -j ACCEPT"
    )
    assert _extract_target(line) == "ACCEPT"


def test_extract_target_no_jump() -> None:
    assert _extract_target("-A FORWARD -s 10.42.0.0/24") is None


# ── Position-1 invariant ─────────────────────────────────────────


def test_position1_invariant_holds_when_first_rule_is_managed_jump() -> None:
    dump = (
        "*filter\n"
        ":INPUT ACCEPT [0:0]\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        # Docker-style rule at position 2 — DOES NOT trip the invariant.
        "-A INPUT -i docker0 -j ACCEPT\n"
        "COMMIT\n"
    )
    assert _is_managed_jump_at_position_1(dump, "INPUT", "MS_INPUT") is True


def test_position1_invariant_fails_when_displaced() -> None:
    """External actor inserts a rule at position 1 → invariant fails."""
    dump = (
        "*filter\n"
        ":INPUT ACCEPT [0:0]\n"
        # Foreign actor displaced our jump.
        "-A INPUT -s 192.0.2.0/24 -j ACCEPT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        "COMMIT\n"
    )
    assert _is_managed_jump_at_position_1(dump, "INPUT", "MS_INPUT") is False


def test_position1_invariant_fails_with_wrong_target() -> None:
    """First rule is marker-tagged but jumps somewhere else."""
    dump = (
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j ACCEPT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
    )
    assert _is_managed_jump_at_position_1(dump, "INPUT", "MS_INPUT") is False


def test_position1_invariant_fails_when_parent_empty() -> None:
    assert _is_managed_jump_at_position_1("", "INPUT", "MS_INPUT") is False


def test_position1_invariant_fails_unmarked_first_rule() -> None:
    """First rule is a non-marker-tagged jump to MS_INPUT (someone
    else's accidental jump). Refuse — only OUR marker-tagged jump
    counts."""
    dump = (
        "-A INPUT -j MS_INPUT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
    )
    assert _is_managed_jump_at_position_1(dump, "INPUT", "MS_INPUT") is False


# ── Managed-jump extraction ──────────────────────────────────────


def test_extract_managed_jumps_returns_in_dump_order() -> None:
    """Multiple marker-tagged jumps are returned in dump order — caller
    compares as exact ordered list."""
    dump = (
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        # Stranger's tagged-but-different-target jump — filtered out.
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j ACCEPT\n"
        # Stranger's NM-managed jump — NOT marker-tagged, filtered out.
        "-A INPUT -m comment --comment 'nm.routing.rules' -j MS_INPUT\n"
    )
    extracted = _extract_managed_jumps(dump, "INPUT", {"MS_INPUT", "MS_FWD"})
    assert len(extracted) == 1
    assert "MS_INPUT" in extracted[0]
    assert MS_FW_COMMENT in extracted[0]


def test_extract_managed_jumps_catches_duplicates() -> None:
    """Two marker-tagged jumps to the same MS_INPUT (someone re-applied
    the firewall without scrubbing first) — both surface."""
    dump = (
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
    )
    extracted = _extract_managed_jumps(dump, "INPUT", {"MS_INPUT"})
    assert len(extracted) == 2


# ── Snapshot validation ──────────────────────────────────────────


def test_snapshot_validates_clean() -> None:
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=_good_filter_dump(),
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    ok, reason = _validate_snapshot_complete(snap)
    assert ok, reason


def test_snapshot_validates_with_sta_present() -> None:
    """STA-active snapshot adds the egress-isolation rules in
    MS_FWD/MS_POST; canonical generator emits them, snapshot validates."""
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=True,
        filter_dump=_good_filter_dump(sta=True),
        nat_dump=_good_nat_dump(sta=True),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    ok, reason = _validate_snapshot_complete(snap)
    assert ok, reason


def test_snapshot_fails_when_chain_undeclared() -> None:
    """Missing chain decl → validation refuses."""
    bad = _good_filter_dump().replace(":MS_FWD - [0:0]\n", "")
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=bad,
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    ok, reason = _validate_snapshot_complete(snap)
    assert not ok
    assert "MS_FWD not declared" in reason


def test_snapshot_fails_with_extra_jump_in_parent() -> None:
    """Plan 2 P2 #1: duplicate marker-tagged jumps in the parent → fail."""
    dump = _good_filter_dump()
    extra_jump = f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
    bad = dump.replace(
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n",
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n{extra_jump}",
        1,
    )
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=bad,
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    ok, reason = _validate_snapshot_complete(snap)
    assert not ok
    assert "managed jumps mismatch" in reason


def test_snapshot_fails_with_chain_body_corruption() -> None:
    """Plan 2 P1 #1: the captured chain body diverges from the canonical
    generator → validation refuses (catches an external actor that
    poked around inside MS_INPUT)."""
    dump = _good_filter_dump()
    # Append a stray rule to MS_INPUT that the canonical generator does
    # NOT emit.
    bad = dump.replace(
        "COMMIT\n",
        f"-A MS_INPUT -p tcp --dport 22 -m comment --comment {MS_FW_COMMENT} -j ACCEPT\nCOMMIT\n",
    )
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=bad,
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    ok, reason = _validate_snapshot_complete(snap)
    assert not ok
    assert "captured != expected" in reason


def test_snapshot_position1_pos1_flag_reflects_displacement() -> None:
    """The ``filter_managed_jump_at_pos1`` flag is set by the snapshot
    capture; restore re-checks live state at the final under-lock
    revalidation step."""
    dump = _good_filter_dump().replace(
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n",
        "-A INPUT -s 192.0.2.0/24 -j ACCEPT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n",
    )
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=dump,
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    assert snap.filter_managed_jump_at_pos1["INPUT"] is False
    assert snap.filter_managed_jump_at_pos1["FORWARD"] is True


# ── Canonical generator ──────────────────────────────────────────


def test_canonical_generator_no_sta_no_egress_isolation() -> None:
    """Without STA, MS_FWD only has the cross-direction DROPs."""
    rules = _expected_chain_rules(
        table="filter",
        v6=False,
        chain="MS_FWD",
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
    )
    assert len(rules) == 2
    assert all("wlan_sta" not in r for r in rules)


def test_canonical_generator_sta_adds_egress_drop() -> None:
    """With STA, MS_FWD adds the explicit hotspot → STA drop."""
    rules = _expected_chain_rules(
        table="filter",
        v6=False,
        chain="MS_FWD",
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=True,
    )
    assert len(rules) == 3
    assert any("wlan_sta" in r for r in rules)


def test_canonical_generator_v6_input_has_drop_terminal() -> None:
    """MS_INPUT6 ends with a terminal DROP — IPv6 ingress is denied
    except loopback + ND."""
    rules = _expected_chain_rules(
        table="filter",
        v6=True,
        chain="MS_INPUT6",
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
    )
    assert rules[-1].endswith("-j DROP")


def test_canonical_generator_unknown_chain_raises() -> None:
    with pytest.raises(ValueError, match="unknown chain spec"):
        _expected_chain_rules(
            table="filter",
            v6=False,
            chain="MS_NOPE",
            mode="meeting",
            cidr="10.42.0.0/24",
            sta_iface_present=False,
        )


def test_canonical_generator_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _expected_chain_rules(
            table="filter",
            v6=False,
            chain="MS_INPUT",
            mode="bogus",  # type: ignore[arg-type]
            cidr="10.42.0.0/24",
            sta_iface_present=False,
        )


# ── Restore-text composer ────────────────────────────────────────


def test_restore_text_emits_canonical_chain_bodies() -> None:
    """Even if the live dump's MS_INPUT body has been corrupted, the
    composed restore text contains the canonical generator's output —
    not the live dump's stray rules. Plan 2 P1 #1 single source of
    truth.
    """
    corrupted_filter = _good_filter_dump().replace(
        "COMMIT\n",
        f"-A MS_INPUT -p tcp --dport 22 -m comment --comment {MS_FW_COMMENT} -j ACCEPT\nCOMMIT\n",
    )
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=corrupted_filter,
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    restore_text = _build_restore_text(
        table_blocks=[
            (
                "filter",
                corrupted_filter,
                MS_CHAINS_FILTER,
                list(_expected_parent_jumps(table="filter", v6=False)),
            ),
            (
                "nat",
                _good_nat_dump(),
                MS_CHAINS_NAT,
                list(_expected_parent_jumps(table="nat", v6=False)),
            ),
        ],
        snap=snap,
    )
    # The corrupted port-22 rule is NOT in the restore output.
    assert "--dport 22" not in restore_text
    # MS_INPUT canonical body (port 443 ACCEPT) IS in the output.
    assert "--dport 443" in restore_text
    # Combined filter+nat in one string (single iptables-restore).
    assert restore_text.count("*filter") == 1
    assert restore_text.count("*nat") == 1
    assert restore_text.count("COMMIT") == 2


def test_restore_text_inserts_jump_at_position_1() -> None:
    """The composed parent chain has the MS_* jump at position 1, even
    if the live dump had it elsewhere or had duplicates."""
    # Live dump where MS_INPUT jump is at position 2 with a foreign
    # rule at position 1.
    live = (
        "*filter\n"
        ":INPUT ACCEPT [0:0]\n"
        ":FORWARD ACCEPT [0:0]\n"
        ":MS_INPUT - [0:0]\n"
        ":MS_FWD - [0:0]\n"
        "-A INPUT -s 192.0.2.0/24 -j ACCEPT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        f"-A FORWARD -m comment --comment {MS_FW_COMMENT} -j MS_FWD\n"
        "COMMIT\n"
    )
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=_good_filter_dump(),  # canonical, for snap.captured_*
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    section = _emit_table_section(
        table="filter",
        live_dump=live,
        ms_chains=MS_CHAINS_FILTER,
        parent_jumps=list(_expected_parent_jumps(table="filter", v6=False)),
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        v6=False,
    )
    lines = [_normalize_save_line(line) for line in section.splitlines()]
    # Find the FIRST -A INPUT rule in the output.
    input_rules = [line for line in lines if line.startswith("-A INPUT ")]
    assert input_rules, lines
    first_input = input_rules[0]
    assert MS_FW_COMMENT in first_input
    assert _extract_target(first_input) == "MS_INPUT"
    # The foreign 192.0.2.0/24 rule is preserved at position 2+ (no scrub).
    assert any("192.0.2.0/24" in line for line in lines)


def test_restore_text_strips_duplicate_managed_jumps() -> None:
    """If the live dump has two marker-tagged jumps to MS_INPUT (someone
    re-applied without scrubbing), the composed restore output emits
    exactly ONE jump, at position 1."""
    live = (
        "*filter\n"
        ":INPUT ACCEPT [0:0]\n"
        ":FORWARD ACCEPT [0:0]\n"
        ":MS_INPUT - [0:0]\n"
        ":MS_FWD - [0:0]\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT\n"
        f"-A FORWARD -m comment --comment {MS_FW_COMMENT} -j MS_FWD\n"
        "COMMIT\n"
    )
    snap = take_snapshot(
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        filter_dump=_good_filter_dump(),
        nat_dump=_good_nat_dump(),
        filter_v6_dump=_good_filter_v6_dump(),
    )
    section = _emit_table_section(
        table="filter",
        live_dump=live,
        ms_chains=MS_CHAINS_FILTER,
        parent_jumps=list(_expected_parent_jumps(table="filter", v6=False)),
        mode="meeting",
        cidr="10.42.0.0/24",
        sta_iface_present=False,
        v6=False,
    )
    # Exactly one managed jump to MS_INPUT in the INPUT chain.
    input_managed = [
        line
        for line in section.splitlines()
        if line.startswith("-A INPUT ")
        and MS_FW_COMMENT in line
        and _extract_target(line) == "MS_INPUT"
    ]
    assert len(input_managed) == 1


def test_normalize_save_line_collapses_whitespace() -> None:
    a = "-A INPUT  -i  wlP9s9  -j  ACCEPT"
    b = "-A INPUT -i wlP9s9 -j ACCEPT"
    assert _normalize_save_line(a) == _normalize_save_line(b) == b


def test_expected_jump_save_line_matches_canonical_form() -> None:
    line = _expected_jump_save_line(parent="INPUT", target="MS_INPUT")
    assert line == f"-A INPUT -m comment --comment {MS_FW_COMMENT} -j MS_INPUT"
