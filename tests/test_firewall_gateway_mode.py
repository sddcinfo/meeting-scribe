"""Phase C — firewall posture + WAN INPUT deny + reconciliation.

Plan hard-stops covered here:
* block-mode golden test (byte-identical to today)
* WAN-input-isolation for wlan_sta (admin ports DROP'd)
* enP7s7 admin path preserved (admin ports ACCEPT'd)
* ip6tables FORWARD policy DROP
* reconcile idempotence (5x)
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import MagicMock, patch

from meeting_scribe import wifi

# ─── Helpers ────────────────────────────────────────────────


def _mk_completed(rc: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def _captured_argvs(mock_run: MagicMock) -> list[list[str]]:
    """Extract the argv from every subprocess.run call_args entry."""
    out: list[list[str]] = []
    for c in mock_run.call_args_list:
        args = c[0][0] if c[0] else c[1].get("args", [])
        out.append(list(args))
    return out


def _rules_applied(mock_run: MagicMock, binary: str) -> list[list[str]]:
    """Return every argv that targeted ``sudo <binary> ...`` for rule application.

    Excludes the ``-S`` listing reads done by ``_ms_fw_remove``.
    """
    out: list[list[str]] = []
    for argv in _captured_argvs(mock_run):
        if len(argv) < 2 or argv[0] != "sudo" or argv[1] != binary:
            continue
        if "-S" in argv:
            continue
        out.append(argv)
    return out


# ─── Block-mode invariants (regression boundary) ──────────────


class TestBlockModeUnchanged:
    """Block-mode is the existing zero-egress hotspot posture. Hard-stop:
    if the byte-for-byte output of ``_ms_fw_rules`` changes for the
    default no-kwargs call, meeting-privacy is broken."""

    def test_default_call_returns_four_tuple(self) -> None:
        result = wifi._ms_fw_rules("wlP9s9")
        assert isinstance(result, tuple)
        assert len(result) == 4
        _common, _drop, _forward, nat_post = result
        # In block mode: nat_post is empty (no MASQUERADE).
        assert nat_post == []

    def test_block_mode_forward_unchanged(self) -> None:
        """The forward block must match the legacy zero-egress posture exactly."""
        _, _, forward, _ = wifi._ms_fw_rules("wlP9s9")
        # Today's three rules: AP→lo ACCEPT, AP→AP DROP, AP→non-lo DROP.
        assert len(forward) == 3
        assert forward[0][:6] == ["-A", "FORWARD", "-i", "wlP9s9", "-o", "lo"]
        assert forward[0][-1] == "ACCEPT"
        assert forward[1][:6] == ["-A", "FORWARD", "-i", "wlP9s9", "-o", "wlP9s9"]
        assert forward[1][-1] == "DROP"
        assert forward[2][:5] == ["-A", "FORWARD", "-i", "wlP9s9", "!"]
        assert forward[2][-1] == "DROP"

    def test_block_mode_common_input_unchanged(self) -> None:
        """Ports 443/80/53 on the AP iface plus lo + conntrack — same as today."""
        common_input, _, _, _ = wifi._ms_fw_rules("wlP9s9")
        # 5 rules: lo, conntrack, 443, 80, 53.
        assert len(common_input) == 5
        # Spot-check the AP-iface ACCEPT rules.
        ports = [r[r.index("--dport") + 1] for r in common_input if "--dport" in r]
        assert ports == ["443", "80", "53"]

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_default_still_invokes_v4_and_v6(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall()
        bins: set[str] = set()
        for argv in _captured_argvs(mock_run):
            for tok in argv:
                if tok in ("iptables", "ip6tables"):
                    bins.add(tok)
        assert bins == {"iptables", "ip6tables"}

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_default_drops_ap_to_non_lo(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall()
        joined = [" ".join(str(t) for t in argv) for argv in _captured_argvs(mock_run)]
        assert any("FORWARD" in r and "! -o lo" in r and "DROP" in r for r in joined), (
            "block-mode FORWARD egress drop must persist"
        )

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_inserts_forward_rules_above_nm_shared(self, mock_run: MagicMock) -> None:
        """FORWARD rules MUST go via ``-I FORWARD <pos>`` so they sit above
        NetworkManager's ``nm-sh-fw-<iface>`` jump (added by
        ``ipv4.method=shared`` on the AP profile). Appending with ``-A``
        lands our rules below NM's chain, which unconditionally ACCEPTs
        the AP subnet → bypasses the captive ipset gate entirely.

        Regression for GB10 walkthrough 2026-05-13: phone got internet
        immediately after ipset flush because NM's nm-sh-fw-wlP9s9
        ACCEPTed the AP→WAN forward before our REJECT/DROP could match.

        INPUT rules stay on ``-A`` (existing 2026-05-05 ordering fix).
        """
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall(
            egress_mode="captive",
            wan_ifaces=("enP7s7",),
            wan_admin_allowed=("enP7s7",),
        )
        v4_rules = _rules_applied(mock_run, "iptables")
        # Extract rule-apply argvs (skip nat-table NAT POSTROUTING) that
        # mention the FORWARD chain.
        forward_argvs = [
            argv
            for argv in v4_rules
            if "FORWARD" in argv and ("-t" not in argv or argv[argv.index("-t") + 1] != "nat")
        ]
        assert forward_argvs, "expected at least one FORWARD-chain apply"
        # Every FORWARD rule uses -I FORWARD <pos>, not -A FORWARD.
        positions: list[int] = []
        for argv in forward_argvs:
            # argv looks like ['sudo', 'iptables', '-I', 'FORWARD', '<pos>', ...]
            assert "-A" not in argv[:5], (
                f"FORWARD rule used -A (would land below NM's chain): {argv}"
            )
            assert "-I" in argv[:5], f"FORWARD rule must use -I: {argv}"
            i_idx = argv.index("-I")
            assert argv[i_idx + 1] == "FORWARD"
            positions.append(int(argv[i_idx + 2]))
        # Positions are 1, 2, 3, ... in order so the inserted block
        # preserves the rule generator's relative order at the head of
        # FORWARD (admin-ACCEPT before REJECT tcp/443 before catch-all
        # DROP — without this ordering the catch-all DROPs everything).
        assert positions == list(range(1, len(positions) + 1)), (
            f"FORWARD insert positions must be 1..N, got {positions}"
        )

        # INPUT rules must keep using -A so the per-port ACCEPTs come
        # before the catch-all DROP (2026-05-05 invariant).
        input_argvs = [argv for argv in v4_rules if "INPUT" in argv]
        for argv in input_argvs:
            assert "-A" in argv[:5], f"INPUT rule must keep -A: {argv}"


# ─── Gateway-mode posture ─────────────────────────────────────


class TestGatewayMode:
    """The new posture: AP↔WAN forwarding + MASQUERADE + WAN-side INPUT
    asymmetry (wlan_sta admin denied, enP7s7 admin allowed)."""

    def test_gateway_mode_v4_rules_golden(self) -> None:
        _common, _drop, forward, nat_post = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        # MASQUERADE present for both WANs.
        masq_targets = [r[r.index("-o") + 1] for r in nat_post if "MASQUERADE" in r]
        assert sorted(masq_targets) == ["enP7s7", "wlan_sta"]
        # nat_post all live in the ``nat`` table.
        for r in nat_post:
            assert r[:2] == ["-t", "nat"]
        # forward includes AP↔WAN allow + WAN→AP established/related +
        # WAN→AP new state DROP for each WAN. 3 rules per WAN * 2 WANs
        # = 6 + 1 AP↔AP isolation = 7.
        assert len(forward) == 7
        assert any("wlan_sta" in r for r in forward)
        assert any("enP7s7" in r for r in forward)

    def test_wlan_sta_input_explicitly_dropped(self) -> None:
        """Plan hard-stop: wlan_sta MUST drop new inbound on 22/80/443."""
        common, _, _, _ = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        wlan_sta_rules = [r for r in common if "wlan_sta" in r and "--dport" in r]
        # 3 ports * 1 iface = 3
        assert len(wlan_sta_rules) == 3
        ports = sorted(r[r.index("--dport") + 1] for r in wlan_sta_rules)
        assert ports == ["22", "443", "80"]
        # All DROP, not ACCEPT.
        for r in wlan_sta_rules:
            assert r[-1] == "DROP"
            assert "NEW" in r  # conntrack state

    def test_enp7s7_input_admin_allowed(self) -> None:
        """Wired admin path stays open — explicit ACCEPT on 22/80/443."""
        common, _, _, _ = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        wired_rules = [r for r in common if "enP7s7" in r and "--dport" in r]
        assert len(wired_rules) == 3
        for r in wired_rules:
            assert r[-1] == "ACCEPT"

    def test_gateway_mode_no_wan_ifaces_falls_back_to_block(self) -> None:
        """gateway+empty-wans must not silently differ from block."""
        block = wifi._ms_fw_rules("wlP9s9")
        empty_gw = wifi._ms_fw_rules("wlP9s9", egress_mode="gateway", wan_ifaces=())
        assert block == empty_gw

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_gateway_writes_sysctl_v4_on_v6_off(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        # Look for the body string handed to ``sudo tee`` as input.
        tee_inputs = [
            c.kwargs.get("input")
            for c in mock_run.call_args_list
            if c.args and isinstance(c.args[0], list) and "tee" in c.args[0]
        ]
        assert any(
            inp and "net.ipv4.ip_forward = 1" in inp and "net.ipv6.conf.all.forwarding = 0" in inp
            for inp in tee_inputs
        ), f"expected sysctl write with v4=1 v6=0, got {tee_inputs}"

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_block_writes_sysctl_v4_off(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall()
        tee_inputs = [
            c.kwargs.get("input")
            for c in mock_run.call_args_list
            if c.args and isinstance(c.args[0], list) and "tee" in c.args[0]
        ]
        assert any(inp and "net.ipv4.ip_forward = 0" in inp for inp in tee_inputs)

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_gateway_emits_masquerade(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        applied = _rules_applied(mock_run, "iptables")
        masq_calls = [a for a in applied if "MASQUERADE" in a]
        # One per WAN.
        assert len(masq_calls) == 2
        for argv in masq_calls:
            assert "-t" in argv and argv[argv.index("-t") + 1] == "nat"

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_enforces_ip6tables_forward_drop(self, mock_run: MagicMock) -> None:
        """Plan hard-stop: ip6tables FORWARD policy DROP after apply."""
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        applied = _captured_argvs(mock_run)
        policy_drops = [
            a for a in applied if a[:2] == ["sudo", "ip6tables"] and "-P" in a and "DROP" in a
        ]
        assert len(policy_drops) >= 1
        assert policy_drops[0][-2:] == ["FORWARD", "DROP"]

    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_emits_no_v6_nat_rules(self, mock_run: MagicMock) -> None:
        """No NAT66 in v1 — ip6tables nat POSTROUTING must stay untouched."""
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        applied = _rules_applied(mock_run, "ip6tables")
        for argv in applied:
            assert "MASQUERADE" not in argv, f"v6 MASQUERADE emitted: {argv}"
            # No ip6tables POSTROUTING writes either.
            assert "POSTROUTING" not in argv or "-D" in argv


# ─── Reconciliation idempotence ──────────────────────────────


class TestReconcileIdempotence:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_reconcile_idempotent_5x(self, mock_run: MagicMock, tmp_path, monkeypatch) -> None:
        """5 sequential applies produce identical rule sets."""
        mock_run.return_value = _mk_completed()

        # Pretend sta_iface_present + wired are both up.
        monkeypatch.setattr("meeting_scribe.wifi_sta.sta_iface_present", lambda: True)
        # Pin settings so reconcile_network_state reads a stable mode.
        import meeting_scribe.server_support.settings_store as store

        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr(store, "SETTINGS_OVERRIDE_FILE", settings_path)
        monkeypatch.setattr(store, "_settings_cache", None)
        monkeypatch.setattr(store, "_settings_cache_mtime", 0.0)
        store._set_wan_egress_mode("gateway")
        monkeypatch.setattr(
            "meeting_scribe.wifi.Path",
            type(
                "FakePath",
                (),
                {
                    "__init__": lambda self, p: setattr(self, "_p", p),
                    "exists": lambda self: True,
                },
            ),
        )

        snapshots: list[list[list[str]]] = []
        for _ in range(5):
            mock_run.reset_mock()
            asyncio.run(wifi.reconcile_network_state())
            # Filter to only the *apply* rule writes — _ms_fw_remove's
            # ``-S`` listings are non-deterministic across calls because
            # tests don't simulate rule presence between iterations.
            applied = sorted(
                tuple(argv)
                for argv in _captured_argvs(mock_run)
                if argv and argv[0] == "sudo" and "-S" not in argv
            )
            snapshots.append([list(a) for a in applied])

        # Every snapshot equal to the first.
        for i, snap in enumerate(snapshots[1:], start=1):
            assert snap == snapshots[0], f"snapshot {i} drifted from snapshot 0"


# ─── Toggle: block → gateway → block ──────────────────────────


class TestEgressToggle:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_egress_mode_toggle_block_to_gateway_and_back(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()

        wifi._apply_simple_firewall("wlP9s9")  # block
        block_argvs = _captured_argvs(mock_run)

        mock_run.reset_mock()
        wifi._apply_simple_firewall(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        gw_argvs = _captured_argvs(mock_run)

        mock_run.reset_mock()
        wifi._apply_simple_firewall("wlP9s9")  # back to block
        block_again_argvs = _captured_argvs(mock_run)

        # The rule sets differ between block and gateway.
        block_strs = {" ".join(a) for a in block_argvs if "-P" not in a}
        gw_strs = {" ".join(a) for a in gw_argvs if "-P" not in a}
        assert block_strs != gw_strs

        # Block-mode argvs are stable across toggles.
        block_strs_again = {" ".join(a) for a in block_again_argvs if "-P" not in a}
        assert block_strs == block_strs_again


# ─── Phase H: captive mode ────────────────────────────────────


class TestCaptiveMode:
    """Captive mode = gateway shape + ipset gating on FORWARD AP→WAN +
    REDIRECT unauthorized tcp/80 to the local captive sub-app on port 80.

    Block + gateway snapshots must stay byte-identical (regression).
    """

    def test_captive_mode_v4_rules_present(self) -> None:
        _common, _drop, forward, _nat = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="captive",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        # ACCEPT FORWARD rules gate on the admins ipset for both WANs.
        accept_rules = [r for r in forward if "ACCEPT" in r]
        assert len(accept_rules) >= 4  # 2 AP→WAN (admin set), 2 WAN→AP (returns)
        ipset_gated = [
            r
            for r in accept_rules
            if "ms-allowed-admins" in r and r[r.index("--match-set") + 2] == "src"
        ]
        assert len(ipset_gated) == 2  # one per WAN
        # And an explicit DROP after each ipset-gated ACCEPT so an
        # unauthorized client can never slip through on a later catch-all.
        drop_rules = [r for r in forward if "DROP" in r and ap_wan_pair(r)]
        # AP↔AP + AP→WAN-default-deny for each WAN + WAN→AP-default-deny.
        # Exact count is iface-pair-dependent; what matters is that
        # AP→WAN-DROP exists per WAN.
        ap_to_wan_drop = [
            r
            for r in forward
            if "DROP" in r
            and "-i" in r
            and "wlP9s9" in r
            and "-o" in r
            and r[r.index("-o") + 1] in ("enP7s7", "wlan_sta")
            and "--match-set" not in r
        ]
        assert len(ap_to_wan_drop) == 2  # belt + suspenders DROP per WAN

    def test_captive_mode_redirect_unauthorized_http(self) -> None:
        """tcp/80 from a non-admin client gets REDIRECT'd to our
        captive sub-app — including authenticated guests, so iOS'
        background captive probe (captive.apple.com:80) keeps reaching
        our handler. The handler answers with Apple's Success body for
        guests (they're acked via ipset), which is what makes the iOS
        Done button (blue tick) appear post-PIN.

        Regression for GB10 phone test 2026-05-13: previously the
        REDIRECT also excluded ``ms-allowed-guests``, so after PIN
        entry the probe fell through to the FORWARD chain DROP, iOS
        never saw Success, and the Done button never appeared.
        """
        _common, _drop, _forward, nat = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="captive",
            wan_ifaces=("enP7s7",),
            wan_admin_allowed=("enP7s7",),
        )
        redirect_rules = [r for r in nat if "REDIRECT" in r]
        assert len(redirect_rules) == 1
        r = redirect_rules[0]
        # Lives in the nat table on PREROUTING.
        assert r[:4] == ["-t", "nat", "-A", "PREROUTING"]
        # Bound to the AP iface, tcp/80.
        assert "wlP9s9" in r
        assert "80" in r
        # Negation is ONLY on the admins ipset — guests stay in scope
        # so their captive probes still hit our Success handler.
        assert "ms-allowed-admins" in r
        assert r.count("!") == 1, f"expected one '!' (admins only), got rule: {r}"
        assert "ms-allowed-guests" not in r, (
            "guests must NOT be excluded — they need probe interception "
            "for the iOS Done button to appear post-PIN"
        )
        # Redirects to local port 80.
        assert r[-2] == "--to-ports"
        assert r[-1] == "80"

    def test_captive_mode_https_tcp_reset_for_unauthorized(self) -> None:
        """Unauthorized clients trying to FORWARD AP→WAN on tcp/443 get
        a TCP RST back instead of silently being DROP'd. Modern browsers
        default to HTTPS for hostname-typed addresses; without the RST
        the browser TCP-times-out for ~30 s with no feedback. With the
        RST it surfaces "Connection refused" immediately so the user
        notices the captive sheet (already open) or retries via HTTP.

        Admins (ms-allowed-admins ipset) match the earlier ACCEPT rule
        and never reach this REJECT.
        """
        _common, _drop, forward, _nat = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="captive",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        reject_rules = [r for r in forward if "REJECT" in r]
        assert len(reject_rules) == 2, "one tcp-reset per WAN iface"
        for r in reject_rules:
            assert "FORWARD" in r
            assert "-i" in r and r[r.index("-i") + 1] == "wlP9s9"
            assert "-o" in r and r[r.index("-o") + 1] in ("enP7s7", "wlan_sta")
            assert "-p" in r and r[r.index("-p") + 1] == "tcp"
            assert "--dport" in r and r[r.index("--dport") + 1] == "443"
            # Source IP NOT in admins set (the ACCEPT above already
            # covers admin src; this rule fails fast for everyone else).
            assert "--match-set" in r and "ms-allowed-admins" in r
            assert "!" in r, "must be negated match (non-admins)"
            assert "--reject-with" in r
            assert r[r.index("--reject-with") + 1] == "tcp-reset"

    def test_captive_mode_reject_precedes_drop(self) -> None:
        """Order matters: the tcp/443 REJECT must come before the AP→WAN
        catch-all DROP, otherwise the DROP swallows tcp/443 first and
        the reset never fires."""
        _common, _drop, forward, _nat = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="captive",
            wan_ifaces=("enP7s7",),
            wan_admin_allowed=("enP7s7",),
        )
        # Locate per-WAN AP→enP7s7 rules in order.
        ap_to_wan = [
            (i, r)
            for i, r in enumerate(forward)
            if "-i" in r
            and r[r.index("-i") + 1] == "wlP9s9"
            and "-o" in r
            and r[r.index("-o") + 1] == "enP7s7"
        ]
        targets = [(i, r[r.index("-j") + 1]) for i, r in ap_to_wan]
        # Expect: ACCEPT (admins) → REJECT (non-admins, tcp/443) → DROP (catchall).
        ordered = [t for _, t in targets]
        assert ordered == ["ACCEPT", "REJECT", "DROP"], (
            f"AP→WAN ordering must be ACCEPT, REJECT, DROP, got {ordered}"
        )

    def test_gateway_mode_has_no_tcp_reset(self) -> None:
        """tcp-reset is captive-only; gateway must keep the simple
        open-egress shape with no REJECT entries."""
        _, _, forward, _ = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7",),
            wan_admin_allowed=("enP7s7",),
        )
        for rule in forward:
            assert "REJECT" not in rule, "gateway leaked captive tcp-reset"

    def test_captive_mode_masquerade_present(self) -> None:
        """MASQUERADE on each WAN — same as gateway."""
        _, _, _, nat = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="captive",
            wan_ifaces=("enP7s7", "wlan_sta"),
            wan_admin_allowed=("enP7s7",),
        )
        masq = [r for r in nat if "MASQUERADE" in r]
        targets = sorted(r[r.index("-o") + 1] for r in masq)
        assert targets == ["enP7s7", "wlan_sta"]

    def test_block_mode_golden_unchanged_by_captive_addition(self) -> None:
        """Hard stop: adding the captive branch MUST NOT alter block mode."""
        block_default = wifi._ms_fw_rules("wlP9s9")
        block_explicit = wifi._ms_fw_rules("wlP9s9", egress_mode="block")
        assert block_default == block_explicit

    def test_gateway_mode_unchanged_by_captive_addition(self) -> None:
        """Gateway mode must NOT carry the ipset match clause."""
        _, _, forward, nat = wifi._ms_fw_rules(
            "wlP9s9",
            egress_mode="gateway",
            wan_ifaces=("enP7s7",),
            wan_admin_allowed=("enP7s7",),
        )
        for rule in forward:
            assert "ms-allowed-admins" not in rule, "gateway leaked captive clause"
        for rule in nat:
            assert "REDIRECT" not in rule, "gateway leaked captive redirect"


def ap_wan_pair(rule: list[str]) -> bool:
    """Helper: True iff a rule names both an AP iface and a WAN iface."""
    has_ap = "wlP9s9" in rule
    has_wan = "enP7s7" in rule or "wlan_sta" in rule
    return has_ap and has_wan


class TestCaptiveSysctl:
    @patch("meeting_scribe.wifi.subprocess.run")
    def test_apply_captive_enables_v4_forward(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _mk_completed()
        wifi._apply_simple_firewall(
            "wlP9s9",
            egress_mode="captive",
            wan_ifaces=("enP7s7",),
            wan_admin_allowed=("enP7s7",),
        )
        tee_inputs = [
            c.kwargs.get("input")
            for c in mock_run.call_args_list
            if c.args and isinstance(c.args[0], list) and "tee" in c.args[0]
        ]
        assert any(inp and "net.ipv4.ip_forward = 1" in inp for inp in tee_inputs), (
            f"captive must enable v4 forwarding: {tee_inputs}"
        )


class TestFwRemoveSweepsPrerouting:
    """Phase H regression: ``_ms_fw_remove`` MUST include ``nat PREROUTING``
    in its sweep, otherwise the captive REDIRECT rule duplicates on every
    apply (observed live 2026-05-13 on the GB10)."""

    def test_chain_specs_includes_nat_prerouting(self) -> None:
        """Lock the source-of-truth tuple so a future refactor can't
        silently drop the PREROUTING sweep again."""
        import inspect

        src = inspect.getsource(wifi._ms_fw_remove)
        assert '("nat", "PREROUTING")' in src, (
            "_ms_fw_remove must sweep nat PREROUTING — captive REDIRECT lives there"
        )
