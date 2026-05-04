"""MS_* chain firewall layer — combined Plan 1 + Plan 2 (v36) implementation.

Single source of truth for the appliance's firewall posture. Owns the
``MS_INPUT`` / ``MS_FWD`` / ``MS_PRE`` / ``MS_POST`` / ``MS_INPUT6`` chain
set, the position-1 drift invariant, the canonical chain-body generator,
and the atomic-restore snapshot machinery.

Why a new module:

* Plan 1 (single-hotspot v1.0) wants a dedicated ``MEETING_SCRIBE_INPUT``
  chain at INPUT position 1 with interface-scoped accept rules so that
  pre-existing distro rules don't pre-empt the appliance's deny posture.
* Plan 2 (concurrent STA + AP, v36) wants a five-chain set with the same
  position-1 invariant per parent, an atomic combined filter+nat
  restore, and a canonical generator so build/validate/restore agree on
  the chain bodies.

The merge follows Plan 2's chain naming + position-1 invariant + atomic
restore, and embeds Plan 1's interface-scoped accept rules inside the
``_expected_chain_rules`` bodies.

This module is **pure code** — every helper takes its
``iptables-save`` text as a string argument so the unit tests can drive
it via subprocess mocks; the actual iptables/iptables-restore shell-out
lives in :mod:`meeting_scribe.wifi`. Hardware-side end-to-end
verification (apply → save → assert position-1) is a hardware-gated
follow-up phase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ── Constants ─────────────────────────────────────────────────────


# Marker comment on every managed rule. Snapshot validation extracts
# only rules tagged with this comment; everything else (Docker,
# firewalld, distro defaults) is preserved verbatim through restore.
MS_FW_COMMENT = "ms-fw-managed"

# IPv4 chain set (filter + nat tables).
MS_CHAINS_FILTER: tuple[str, ...] = ("MS_INPUT", "MS_FWD")
MS_CHAINS_NAT: tuple[str, ...] = ("MS_PRE", "MS_POST")

# IPv6 chain set (filter table only — we only filter ingress on v6).
MS_CHAINS_FILTER_V6: tuple[str, ...] = ("MS_INPUT6",)

# Parent-chain jumps emitted at position 1 in each parent.
_FILTER_PARENT_JUMPS: tuple[tuple[str, str], ...] = (
    ("INPUT", "MS_INPUT"),
    ("FORWARD", "MS_FWD"),
)
_NAT_PARENT_JUMPS: tuple[tuple[str, str], ...] = (
    ("PREROUTING", "MS_PRE"),
    ("POSTROUTING", "MS_POST"),
)
_FILTER_V6_PARENT_JUMPS: tuple[tuple[str, str], ...] = (("INPUT", "MS_INPUT6"),)


def _expected_parent_jumps(
    *,
    table: Literal["filter", "nat"],
    v6: bool,
) -> tuple[tuple[str, str], ...]:
    """Return the (parent_chain, target_chain) tuples for ``table``.

    Used by :func:`_validate_snapshot_complete` to assert exact ordered
    list match in each parent chain, and by the restore layer to compose
    the position-1 jump rules.
    """
    if v6:
        if table != "filter":
            return ()
        return _FILTER_V6_PARENT_JUMPS
    if table == "filter":
        return _FILTER_PARENT_JUMPS
    if table == "nat":
        return _NAT_PARENT_JUMPS
    return ()


# ── iptables-save line parsing ─────────────────────────────────────


_TARGET_RE = re.compile(r"\s-j\s+(\S+)")
_PARENT_RE_TPL = r"^-A\s+{parent}(\s|$)"
_COMMENT_RE = re.compile(r'(?:--comment\s+(?:"([^"]*)"|(\S+)))')


def _extract_target(save_line: str) -> str | None:
    """Return the ``-j <TARGET>`` target name from one ``-A`` rule.

    Returns ``None`` if the line has no jump target (e.g. a return rule
    with no -j, or a malformed line). Used by the position-1 invariant
    check + the managed-jump scrubber in restore composition.
    """
    match = _TARGET_RE.search(save_line)
    return match.group(1) if match else None


def _normalize_save_line(line: str) -> str:
    """Whitespace-normalize one iptables-save line for exact comparison.

    iptables-save output can carry trailing whitespace, repeated spaces,
    or different ordering inside ``-m comment``. We split on whitespace
    and rejoin so two semantically equivalent lines compare equal as
    Python strings — same primitive Plan 2 §exact-list-match relies on.
    """
    return " ".join(line.split())


def _expected_jump_save_line(*, parent: str, target: str) -> str:
    """Canonical form of the marker-tagged jump from ``parent`` → ``target``.

    Single source of truth — used both by build (to insert the jump at
    position 1 during restore) and by validate (to compare against what
    the snapshot captured). The comment marker is non-quoted so the
    iptables-save round-trip preserves it.
    """
    return f"-A {parent} -m comment --comment {MS_FW_COMMENT} -j {target}"


def _is_managed_jump_at_position_1(
    dump: str,
    parent: str,
    expected_target: str,
) -> bool:
    """Position-1 invariant per Plan 2 P1 #2.

    Returns True iff the FIRST ``-A <parent> ...`` line in the dump
    carries the ``ms-fw-managed`` comment AND jumps to
    ``expected_target``. Tolerates Docker/etc. rules at positions 2+
    (they don't displace our REJECT). Refuses if some external actor
    moved our jump out of position 1.
    """
    parent_re = re.compile(_PARENT_RE_TPL.format(parent=re.escape(parent)))
    seen_first = False
    for raw in dump.splitlines():
        if not parent_re.search(raw):
            continue
        if seen_first:
            return False
        seen_first = True
        if MS_FW_COMMENT not in raw:
            return False
        return _extract_target(raw) == expected_target
    # No -A lines at all for this parent → invariant cannot hold.
    return False


def _extract_managed_jumps(
    dump: str,
    parent: str,
    valid_targets: set[str],
) -> list[str]:
    """Extract our marker-tagged jumps from ``parent`` (in dump order).

    Returns the normalized save_line list of every ``-A <parent>`` rule
    that:

    1. carries the ``ms-fw-managed`` comment, AND
    2. jumps to one of ``valid_targets``.

    Plan 2 P2 #1 uses this output to compare the snapshot's per-parent
    list against ``_expected_parent_jumps`` as an exact-ordered list,
    catching duplicates / extras / wrong-order / missing.
    """
    parent_re = re.compile(_PARENT_RE_TPL.format(parent=re.escape(parent)))
    out: list[str] = []
    for raw in dump.splitlines():
        if not parent_re.search(raw):
            continue
        if MS_FW_COMMENT not in raw:
            continue
        target = _extract_target(raw)
        if target in valid_targets:
            out.append(_normalize_save_line(raw))
    return out


def _extract_chain_decls(dump: str, names: set[str]) -> set[str]:
    """Return the subset of ``names`` that have a ``:CHAIN POLICY [c:p]``
    declaration in ``dump``.

    Catches the "chain decl exists but no rules" empty-vs-missing case
    that Plan 2 §13 distinguishes.
    """
    decl_re = re.compile(r"^:(\S+)\s")
    declared: set[str] = set()
    for line in dump.splitlines():
        match = decl_re.search(line)
        if match and match.group(1) in names:
            declared.add(match.group(1))
    return declared


def _extract_chain_rules(dump: str, chain: str) -> list[str]:
    """Return the ``-A <chain> ...`` rule list (in dump order, normalized).

    Snapshot stores this for VALIDATION ONLY — restore reconstructs the
    chain body from the canonical generator, never from these captured
    lines. Plan 2 P1 #1 single-source-of-truth.
    """
    chain_re = re.compile(rf"^-A\s+{re.escape(chain)}(\s|$)")
    out: list[str] = []
    for line in dump.splitlines():
        if chain_re.search(line):
            out.append(_normalize_save_line(line))
    return out


# ── Canonical chain-body generator ────────────────────────────────


def _expected_chain_rules(
    *,
    table: Literal["filter", "nat"],
    v6: bool,
    chain: str,
    mode: Literal["meeting", "admin"],
    cidr: str,
    sta_iface_present: bool,
    iface: str = "wlP9s9",
    sta_iface: str = "wlan_sta",
) -> list[str]:
    """Single source of truth for what each MS_* chain should contain.

    Build, validate, and restore all consult this generator so they
    cannot disagree (Plan 2 P1 #1). The body embeds Plan 1's
    interface-scoped accept rules: traffic from the AP interface +
    hotspot subnet gets ACCEPT on 443/80; everything else on those
    ports falls into a terminal DROP.

    Arguments
    ---------
    table, v6 : which iptables table this chain lives in.
    chain : the MS_* chain name we're emitting body for.
    mode : ``"meeting"`` is the v1.0 default — admin and guest share
        the single hotspot. ``"admin"`` is reserved for a future
        operator-only mode (currently unused; kept for snapshot
        round-trip).
    cidr : hotspot subnet CIDR, e.g. ``"10.42.0.0/24"``.
    sta_iface_present : True only when concurrent STA + AP is active.
        Toggles the egress-isolation rules in MS_FWD/MS_POST that
        block hotspot → upstream WiFi traffic.
    iface : AP interface name (default ``"wlP9s9"`` — GB10 hardware).
    sta_iface : STA virtual interface name (default ``"wlan_sta"``).

    Returns the list of normalized save lines that should appear in the
    chain, in order.
    """
    if mode not in ("meeting", "admin"):
        raise ValueError(f"mode must be 'meeting' or 'admin'; got {mode!r}")

    def _line(parts: list[str]) -> str:
        return _normalize_save_line(" ".join(parts))

    cmt = ["-m", "comment", "--comment", MS_FW_COMMENT]

    if not v6 and table == "filter" and chain == "MS_INPUT":
        # AP + STA ingress rules. Loopback and the hotspot subnet on the
        # AP interface get ACCEPT for HTTPS/HTTP. Anything else into the
        # appliance on those ports is dropped — pre-existing distro
        # ACCEPT rules upstream of MS_INPUT cannot shadow this because
        # MS_INPUT runs at INPUT position 1.
        return [
            _line(["-A", chain, "-i", "lo", *cmt, "-j", "ACCEPT"]),
            _line(
                [
                    "-A",
                    chain,
                    "-i",
                    iface,
                    "-s",
                    cidr,
                    "-p",
                    "tcp",
                    "--dport",
                    "443",
                    *cmt,
                    "-j",
                    "ACCEPT",
                ]
            ),
            _line(
                [
                    "-A",
                    chain,
                    "-i",
                    iface,
                    "-s",
                    cidr,
                    "-p",
                    "tcp",
                    "--dport",
                    "80",
                    *cmt,
                    "-j",
                    "ACCEPT",
                ]
            ),
            _line(
                [
                    "-A",
                    chain,
                    "-i",
                    iface,
                    "-s",
                    cidr,
                    "-p",
                    "udp",
                    "--dport",
                    "53",
                    *cmt,
                    "-j",
                    "ACCEPT",
                ]
            ),
            _line(
                [
                    "-A",
                    chain,
                    "-i",
                    iface,
                    "-s",
                    cidr,
                    "-p",
                    "udp",
                    "--dport",
                    "67",
                    *cmt,
                    "-j",
                    "ACCEPT",
                ]
            ),
            _line(
                [
                    "-A",
                    chain,
                    "-i",
                    iface,
                    "-s",
                    cidr,
                    "-p",
                    "icmp",
                    *cmt,
                    "-j",
                    "ACCEPT",
                ]
            ),
            # Terminal drop on 443/80 for any source not already ACCEPTed.
            _line(["-A", chain, "-p", "tcp", "--dport", "443", *cmt, "-j", "DROP"]),
            _line(["-A", chain, "-p", "tcp", "--dport", "80", *cmt, "-j", "DROP"]),
        ]

    if not v6 and table == "filter" and chain == "MS_FWD":
        # Block any forwarding into or out of the hotspot subnet — the
        # appliance is not a router. When STA is active, also explicitly
        # drop hotspot → STA forwarding so guests cannot egress upstream.
        rules = [
            _line(["-A", chain, "-s", cidr, *cmt, "-j", "DROP"]),
            _line(["-A", chain, "-d", cidr, *cmt, "-j", "DROP"]),
        ]
        if sta_iface_present:
            rules.append(
                _line(
                    [
                        "-A",
                        chain,
                        "-i",
                        iface,
                        "-o",
                        sta_iface,
                        *cmt,
                        "-j",
                        "DROP",
                    ]
                )
            )
        return rules

    if not v6 and table == "nat" and chain == "MS_PRE":
        # PREROUTING: redirect captive-portal HTTP probes to local 80.
        return [
            _line(
                [
                    "-A",
                    chain,
                    "-i",
                    iface,
                    "-s",
                    cidr,
                    "-p",
                    "tcp",
                    "--dport",
                    "80",
                    *cmt,
                    "-j",
                    "REDIRECT",
                    "--to-port",
                    "80",
                ]
            ),
        ]

    if not v6 and table == "nat" and chain == "MS_POST":
        # POSTROUTING: never NAT hotspot traffic onto the upstream
        # interface. When STA is active this is the rule that strips
        # any masquerade an external actor (Docker etc.) might have
        # set up so guest egress stays blocked.
        if sta_iface_present:
            return [
                _line(["-A", chain, "-s", cidr, "-o", sta_iface, *cmt, "-j", "RETURN"]),
            ]
        return []

    if v6 and table == "filter" and chain == "MS_INPUT6":
        # Mirror MS_INPUT but for IPv6. Hotspot is v4-only; v6 ingress
        # is dropped except for loopback + ICMPv6 neighbor-discovery
        # which the kernel needs.
        return [
            _line(["-A", chain, "-i", "lo", *cmt, "-j", "ACCEPT"]),
            _line(["-A", chain, "-p", "ipv6-icmp", *cmt, "-j", "ACCEPT"]),
            _line(["-A", chain, *cmt, "-j", "DROP"]),
        ]

    raise ValueError(f"unknown chain spec: table={table!r} v6={v6} chain={chain!r}")


# ── Snapshot dataclass + validation ───────────────────────────────


@dataclass
class FirewallSnapshot:
    """v36 snapshot — captured immediately after AP bring-up.

    Stores both the canonical chain decls/rules (validated against
    ``_expected_chain_rules``) and the position-1 invariant per parent
    chain. The invariant is re-checked at restore time inside the
    xtables wait lock; if drift is detected, restore refuses and writes
    the STA-DEGRADED marker.
    """

    captured_mode: Literal["meeting", "admin"]
    captured_cidr: str
    sta_iface_present: bool

    raw_filter_dump: str = ""
    raw_nat_dump: str = ""
    raw_filter_v6_dump: str = ""

    filter_chain_decls: set[str] = field(default_factory=set)
    nat_chain_decls: set[str] = field(default_factory=set)
    filter_v6_chain_decls: set[str] = field(default_factory=set)

    # For VALIDATION ONLY (Plan 2 P1 #1).
    filter_chain_save_lines: dict[str, list[str]] = field(default_factory=dict)
    nat_chain_save_lines: dict[str, list[str]] = field(default_factory=dict)
    filter_v6_chain_save_lines: dict[str, list[str]] = field(default_factory=dict)

    # Per-parent ordered managed-jump list (Plan 2 P2 #1).
    filter_managed_jumps_per_parent: dict[str, list[str]] = field(default_factory=dict)
    nat_managed_jumps_per_parent: dict[str, list[str]] = field(default_factory=dict)
    filter_v6_managed_jumps_per_parent: dict[str, list[str]] = field(default_factory=dict)

    # Position-1 invariant per parent (Plan 2 P1 #2).
    filter_managed_jump_at_pos1: dict[str, bool] = field(default_factory=dict)
    nat_managed_jump_at_pos1: dict[str, bool] = field(default_factory=dict)
    filter_v6_managed_jump_at_pos1: dict[str, bool] = field(default_factory=dict)


def take_snapshot(
    *,
    mode: Literal["meeting", "admin"],
    cidr: str,
    sta_iface_present: bool,
    filter_dump: str,
    nat_dump: str,
    filter_v6_dump: str,
) -> FirewallSnapshot:
    """Build a :class:`FirewallSnapshot` from raw iptables-save dumps.

    Pure function — callers handle the actual ``iptables-save`` shellout
    and pass the text in. Test fixtures use this directly with synthetic
    dumps to exercise the validation + position-1 logic without root.
    """
    snap = FirewallSnapshot(
        captured_mode=mode,
        captured_cidr=cidr,
        sta_iface_present=sta_iface_present,
        raw_filter_dump=filter_dump,
        raw_nat_dump=nat_dump,
        raw_filter_v6_dump=filter_v6_dump,
        filter_chain_decls=_extract_chain_decls(filter_dump, set(MS_CHAINS_FILTER)),
        nat_chain_decls=_extract_chain_decls(nat_dump, set(MS_CHAINS_NAT)),
        filter_v6_chain_decls=_extract_chain_decls(filter_v6_dump, set(MS_CHAINS_FILTER_V6)),
        filter_chain_save_lines={c: _extract_chain_rules(filter_dump, c) for c in MS_CHAINS_FILTER},
        nat_chain_save_lines={c: _extract_chain_rules(nat_dump, c) for c in MS_CHAINS_NAT},
        filter_v6_chain_save_lines={
            c: _extract_chain_rules(filter_v6_dump, c) for c in MS_CHAINS_FILTER_V6
        },
        filter_managed_jumps_per_parent={
            p: _extract_managed_jumps(filter_dump, p, set(MS_CHAINS_FILTER))
            for p, _t in _FILTER_PARENT_JUMPS
        },
        nat_managed_jumps_per_parent={
            p: _extract_managed_jumps(nat_dump, p, set(MS_CHAINS_NAT))
            for p, _t in _NAT_PARENT_JUMPS
        },
        filter_v6_managed_jumps_per_parent={
            p: _extract_managed_jumps(filter_v6_dump, p, set(MS_CHAINS_FILTER_V6))
            for p, _t in _FILTER_V6_PARENT_JUMPS
        },
        filter_managed_jump_at_pos1={
            p: _is_managed_jump_at_position_1(filter_dump, p, t)
            for p, t in _FILTER_PARENT_JUMPS
        },
        nat_managed_jump_at_pos1={
            p: _is_managed_jump_at_position_1(nat_dump, p, t) for p, t in _NAT_PARENT_JUMPS
        },
        filter_v6_managed_jump_at_pos1={
            p: _is_managed_jump_at_position_1(filter_v6_dump, p, t)
            for p, t in _FILTER_V6_PARENT_JUMPS
        },
    )
    return snap


def _validate_snapshot_complete(snap: FirewallSnapshot) -> tuple[bool, str]:
    """Return ``(ok, reason)`` — exact ordered match per chain + parent.

    Plan 2 P1#1 + P2#1 in one pass:

    * Every MS_* chain is declared.
    * Every MS_* chain's body equals what ``_expected_chain_rules``
      would emit for the captured ``(mode, cidr, sta_iface_present)``.
    * Every parent chain's managed-jump list (in dump order) equals
      ``_expected_parent_jumps`` (catches duplicates / extras / wrong
      order / missing).

    The bool form drives :func:`_atomic_restore_v36`: validation failure
    aborts the restore and writes the STA-DEGRADED marker.
    """
    mode = snap.captured_mode
    cidr = snap.captured_cidr
    sta = snap.sta_iface_present

    for chain in MS_CHAINS_FILTER:
        if chain not in snap.filter_chain_decls:
            return False, f"filter chain {chain} not declared"
    for chain in MS_CHAINS_NAT:
        if chain not in snap.nat_chain_decls:
            return False, f"nat chain {chain} not declared"
    for chain in MS_CHAINS_FILTER_V6:
        if chain not in snap.filter_v6_chain_decls:
            return False, f"v6 filter chain {chain} not declared"

    def _check_chains(
        actual: dict[str, list[str]],
        *,
        table: Literal["filter", "nat"],
        v6: bool,
    ) -> tuple[bool, str]:
        for chain, captured in actual.items():
            expected = _expected_chain_rules(
                table=table,
                v6=v6,
                chain=chain,
                mode=mode,
                cidr=cidr,
                sta_iface_present=sta,
            )
            captured_norm = [_normalize_save_line(r) for r in captured]
            if captured_norm != expected:
                return False, (
                    f"{('v6 ' if v6 else '')}{table}.{chain}: captured != expected\n"
                    f"  expected: {expected}\n"
                    f"  captured: {captured_norm}"
                )
        return True, ""

    for actual, table, v6 in (
        (snap.filter_chain_save_lines, "filter", False),
        (snap.nat_chain_save_lines, "nat", False),
        (snap.filter_v6_chain_save_lines, "filter", True),
    ):
        ok, reason = _check_chains(actual, table=table, v6=v6)
        if not ok:
            return False, reason

    # Exact ordered managed-jump match per parent chain.
    for jumps_dict, table, v6 in (
        (snap.filter_managed_jumps_per_parent, "filter", False),
        (snap.nat_managed_jumps_per_parent, "nat", False),
        (snap.filter_v6_managed_jumps_per_parent, "filter", True),
    ):
        expected_per_parent: dict[str, list[str]] = {}
        for parent, target in _expected_parent_jumps(table=table, v6=v6):
            expected_per_parent.setdefault(parent, []).append(
                _normalize_save_line(_expected_jump_save_line(parent=parent, target=target))
            )
        for parent, expected_list in expected_per_parent.items():
            actual_list = jumps_dict.get(parent, [])
            if actual_list != expected_list:
                return False, (
                    f"{('v6 ' if v6 else '')}{table}.{parent} managed jumps mismatch:\n"
                    f"  expected: {expected_list}\n"
                    f"  actual:   {actual_list}"
                )

    return True, ""


# ── Restore-text composer ─────────────────────────────────────────


def _build_restore_text(
    *,
    table_blocks: list[
        tuple[
            Literal["filter", "nat"],
            str,  # live_dump
            tuple[str, ...],  # ms_chains
            list[tuple[str, str]],  # parent_jumps
        ]
    ],
    snap: FirewallSnapshot,
    v6: bool = False,
) -> str:
    """Build iptables-restore input that reapplies our MS_* chain set
    while preserving every non-managed rule from the live dumps.

    For each ``(table, live_dump, ms_chains, parent_jumps)`` block:

    * Chain decls are emitted for each MS_* chain (with policy ``-``).
    * Each MS_* chain body is the canonical
      :func:`_expected_chain_rules` output — never the snapshot's
      captured raw lines (single source of truth, Plan 2 P1 #1).
    * For every ``(parent, ms_target)`` in ``parent_jumps``, ALL
      marker-tagged rules in the parent that target ``ms_target`` are
      stripped (removes any duplicates/displaced copies); then the
      canonical position-1 jump is reinserted at index 0.
    * Other rules in parent chains are preserved verbatim from the
      live dump — Docker, firewalld, distro defaults stay intact.

    Concatenation across all table blocks is the input passed to
    ``iptables-restore -w 5`` in a single invocation, giving combined
    filter+nat atomicity per Plan 2 P2 #2. IPv6 callers pass ``v6=True``
    and use ``ip6tables-restore`` separately (its own lock domain).
    """
    sections: list[str] = []
    for table, live_dump, ms_chains, parent_jumps in table_blocks:
        section = _emit_table_section(
            table=table,
            live_dump=live_dump,
            ms_chains=ms_chains,
            parent_jumps=parent_jumps,
            mode=snap.captured_mode,
            cidr=snap.captured_cidr,
            sta_iface_present=snap.sta_iface_present,
            v6=v6,
        )
        sections.append(section)
    return "\n".join(sections) + "\n"


def _emit_table_section(
    *,
    table: Literal["filter", "nat"],
    live_dump: str,
    ms_chains: tuple[str, ...],
    parent_jumps: list[tuple[str, str]],
    mode: Literal["meeting", "admin"],
    cidr: str,
    sta_iface_present: bool,
    v6: bool,
) -> str:
    """Compose one table block of iptables-restore input.

    Pure function over the live dump text; no shellout. Returns text
    starting with ``*<table>`` and ending with ``COMMIT``.
    """
    chain_decls: dict[str, str] = {}  # chain → ":CHAIN POLICY [c:p]"
    rules_per_chain: dict[str, list[str]] = {}
    in_table = False

    for raw in live_dump.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith(f"*{table}"):
            in_table = True
            continue
        if line.startswith("*") and in_table:
            in_table = False
            continue
        if line == "COMMIT":
            in_table = False
            continue
        if not in_table:
            continue
        if line.startswith(":"):
            # Chain decl: ":CHAIN POLICY [c:p]"
            name = line[1:].split()[0]
            chain_decls[name] = line
            rules_per_chain.setdefault(name, [])
            continue
        if line.startswith("-A "):
            chain_name = line.split()[1]
            rules_per_chain.setdefault(chain_name, []).append(line)
            continue
        # Comments / unsupported directives passed through verbatim — they
        # only appear in tests.

    # Ensure each MS_* chain is declared. If iptables-save reported no
    # decl, emit ``:MS_CHAIN - [0:0]`` so iptables-restore creates it.
    for chain in ms_chains:
        chain_decls.setdefault(chain, f":{chain} - [0:0]")
        # Rebuild the chain body from the canonical generator — never
        # from the captured ``rules_per_chain[chain]`` lines.
        rules_per_chain[chain] = _expected_chain_rules(
            table=table,
            v6=v6,
            chain=chain,
            mode=mode,
            cidr=cidr,
            sta_iface_present=sta_iface_present,
        )

    # For each parent jump, scrub all marker-tagged rules that target
    # ms_target, then insert the canonical jump at position 1.
    for parent, ms_target in parent_jumps:
        existing = rules_per_chain.setdefault(parent, [])
        scrubbed = [
            r for r in existing if not (MS_FW_COMMENT in r and _extract_target(r) == ms_target)
        ]
        scrubbed.insert(0, _expected_jump_save_line(parent=parent, target=ms_target))
        rules_per_chain[parent] = scrubbed

    # Reassemble in canonical order: header, decls (sorted), rules
    # (chain decls determine emit order — built-in chains first, then
    # MS_* chains).
    out: list[str] = [f"*{table}"]
    builtin_order = ("INPUT", "FORWARD", "OUTPUT", "PREROUTING", "POSTROUTING")
    seen_chains: set[str] = set()
    for builtin in builtin_order:
        if builtin in chain_decls:
            out.append(chain_decls[builtin])
            seen_chains.add(builtin)
    for chain in ms_chains:
        if chain in chain_decls:
            out.append(chain_decls[chain])
            seen_chains.add(chain)
    for chain, decl in chain_decls.items():
        if chain not in seen_chains:
            out.append(decl)
            seen_chains.add(chain)
    # Emit rules in chain-decl order.
    for chain in [*builtin_order, *ms_chains]:
        if chain in rules_per_chain:
            out.extend(rules_per_chain[chain])
    for chain, rules in rules_per_chain.items():
        if chain not in builtin_order and chain not in ms_chains:
            out.extend(rules)
    out.append("COMMIT")
    return "\n".join(out)
