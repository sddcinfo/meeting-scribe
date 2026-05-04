# STA + AP Unified Hotspot Implementation (combined plan)

Combines **valiant-strolling-starfish** (single-hotspot meeting-scribe + GB10-owned BT
audio + 1:1 CLI/UI parity, three workstreams A/B/C) with **eventual-forging-church v36**
(concurrent STA + AP + position-1 firewall invariant).

Branch: `feature/sta-ap-unified-2026-05-04`.

## Scope

The combined feature delivers a self-contained demo box that:

1. Boots into a **single hotspot** (`SSID=meeting-scribe`, `https://10.42.0.1/`) where
   admin and guest share one origin discriminated only by the `scribe_admin` cookie.
2. Optionally connects **upstream as STA** to a second SSID (`yunomotocho`) on the same
   MT7925 radio, while continuing to host the hotspot. Hotspot clients still have **zero
   upstream egress**.
3. Owns the **Bluetooth audio bridge** (mic + speaker) so the operator can pair Ray-Ban
   Meta / Mijia glasses without a phone in the loop.
4. Exposes every operator-facing CLI subcommand in the admin web UI; hotspot-mutating ops
   (wifi up/down, regdomain) stay CLI-only and pass through a narrow root-owned
   privileged helper daemon.

## Synthesis points (where the two plans meet)

The plans are independent except for `wifi.py`'s firewall layer, which is touched by
both. Resolution:

- **Chain naming follows Plan 2** (`MS_INPUT`, `MS_FWD`, `MS_PRE`, `MS_POST`,
  `MS_INPUT6`). Plan 1's single `MEETING_SCRIBE_INPUT` is one of the chains in this set.
- **Chain bodies follow Plan 1** (interface-scoped accept rules: `-i lo`,
  `-i ${IFACE} -s 10.42.0.0/24 -p tcp --dport 443/80 -j ACCEPT`, terminal `DROP` for
  any other origin on those ports). They are emitted by Plan 2's
  `_expected_chain_rules(table, v6, chain, mode, cidr, sta_iface_present)` canonical
  generator.
- **Drift detection follows Plan 2** (position-1 invariant per parent chain, exact
  ordered managed-jump match, atomic combined `iptables-restore` for filter+nat in one
  shot, separate ip6tables-restore). Plan 1's "exactly one jump after N applies"
  property is preserved because the canonical generator emits exactly one expected jump
  per parent.
- **STA mode** (Plan 2 only) toggles `sta_iface_present=True` in the snapshot;
  `_expected_chain_rules` then includes the egress-isolation rules that block
  `wlP9s9 → wlan_sta` and tear NAT off `wlan_sta` for hotspot-source traffic.

## Phase order (for implementation)

Each phase is a single coherent commit (or a small set of commits) that leaves the test
suite green. The order is chosen so each phase keeps its blast radius local.

1. **CSP audit + listener protocol generalization.** Move all inline JS in 4 templates +
   bootstrap form to external files. Generalize `audio/output_pipeline.py` to an
   `AudioListener` Protocol so the BT bridge can join `_audio_out_clients` without a
   `WebSocket`. Zero behavior change. Lowest-risk first commit.
2. **Cert/TLS leaf-only.** `scripts/setup_certs.py` produces a per-device self-signed
   leaf (no CA, no intermediate). Persistent appliance-id at
   `/etc/meeting-scribe/appliance-id`. New `runtime/cert_check.py` runs a read-only
   pre-server SAN sanity check that aborts with a remediation message if SANs are
   missing.
3. **Security middlewares.** Host canonicalization, Origin allowlist (fail-closed CSRF),
   cache-headers (exact `/`, `/admin/bootstrap` + prefix `/api/admin/`), CSP injector
   (default-on for every `text/html` response). Wire into `middlewares.py` per the
   ordering in Plan 1 §A.6.
4. **Cookie infrastructure.** `terminal/auth.py:CookieSigner` extended with
   `session_id` (16 random bytes per `/api/admin/authorize`) and per-boot HKDF subkey
   derived from `state.boot_session_id = secrets.token_bytes(32)`. New
   `decode_verified_cookie()` helper. `state._revoked_sessions` and
   `state._admin_ws_by_session` maps. `register_admin_ws()` async context manager.
   Old-format cookies treated as "no cookie" (graceful migration).
5. **HTTP captive sub-app.** New `hotspot/captive_http_app.py` re-registers the four OS
   captive probe handlers, adds a `GET/HEAD` catch-all that 308s to canonical HTTPS
   (query preserved verbatim), and a method-guard middleware that returns
   `426 Upgrade Required` with `Upgrade: TLS/1.2, HTTP/1.1` for any non-GET/HEAD.
6. **Single-listener TLS bind + scope predicate flip.** `server.py:main()` binds the
   main app on `10.42.0.1:443` only (`IP_FREEBIND`) and the captive sub-app on
   `10.42.0.1:80` only. Delete `_detect_management_ip*` and `_serve_dual` from
   `runtime/net.py`; replace with `_serve_two_apps`. `request_scope._is_guest_scope`
   becomes a presentation-only inverse of cookie verification. Delete
   `_is_hotspot_client` and `HOTSPOT_SUBNET`. `admin_guard._require_admin_response`
   stays the sole authorization gate.
7. **Login/logout + bootstrap template.** `POST /api/admin/logout` (303 to `/`, delete
   cookie, revoke session, close WS). Re-auth on `/api/admin/authorize` revokes the
   prior session. Bootstrap form HTML moves to `static/bootstrap.html.j2` with
   `{{leaf_fingerprint}}` placeholder. External `static/js/bootstrap-submit.js` and
   `static/js/admin-signout.js`. Cookie attrs: `Secure; HttpOnly; SameSite=Strict;
   Path=/`.
8. **Firewall rewrite (Plan 1 + Plan 2 merge).** `wifi.py` MS_* chain set with
   `_expected_chain_rules` canonical generator + position-1 invariant + atomic
   combined-table restore + `xtables -w 5` wait. STA-DEGRADED marker. Snapshot
   validation. Idempotent jump install (no piling up after N applies).
9. **Admission control + auth rate limit.** New `server_support/admission.py`. Caps:
   `MAX_GUEST_VIEW_WS=64`, `MAX_GUEST_AUDIO_OUT_WS=32`, `MAX_PER_IP_GUEST_WS=4`.
   Bounded queues (drop-oldest for transcript JSON, immediate-close for audio MSE).
   Multi-layer auth rate limit (per-IP + per-DHCP-MAC + global in-flight semaphore).
   Constant-time failure path. Auth-aware exemption (cookie bypass).
10. **Privileged helper daemon.** New top-level `src/meeting_scribe_helper/` package.
    Unix socket at `/run/meeting-scribe/helper.sock` (mode 0o660, group
    `meeting-scribe`). SO_PEERCRED auth — UID 0 and UID `meeting-scribe`. Strict JSON-RPC
    verb allowlist: `wifi.up/down/status`, `firewall.apply/status`, `regdomain.set`.
    Per-verb caller-UID check (mutating verbs reject web-service caller with
    `cli_only_verb`). List-form argv only — no shell. Secrets-redacted logging with
    per-boot-rotated HMAC.
11. **Migrate sudo callsites to helper_client + remove sudoers.** Replace every
    `sudo -n nmcli/iptables` in `wifi.py` with `helper_client.<verb>(...)`. CLI uses the
    same client. Remove the meeting-scribe sudoers grant. Service unit:
    `User=meeting-scribe`, `AmbientCapabilities=CAP_NET_BIND_SERVICE`,
    `NoNewPrivileges=yes`, `CapabilityBoundingSet=CAP_NET_BIND_SERVICE`.
12. **BT control plane.** New `bt.py` (mirrors `wifi.py` pattern) + `cli/bt.py`. Profile
    discovery via `pactl --format=json list cards`; `bt_choose_profile(want="a2dp"|"hfp")`
    ranks by codec. `bt_resolve_nodes(mac)` via JSON filter on
    `properties.api.bluez5.address`. Settings store extended.
13. **BT data plane.** New `audio/bt_bridge.py`. State machine (Disconnected / Idle
    (A2DP) / MicLive (HFP) / Switching / Scanning). `tracked_node_ids` set ownership.
    Two subprocesses (`pw-record` + `pw-play`) + `pw-mon` watcher. `BTSpeakerListener`
    bound to `audio_format="wav-pcm"` at registration. Cancelable retry timer.
    Single-flight recovery via `_recovery_pending` + `_sm_wakeup_event`. PipeWire
    readiness gate + runtime health watcher.
14. **Lifespan auto-connect + diag endpoint.** `runtime/lifespan.py` schedules
    `bt_bridge.start()` after `wait_for_pipewire(timeout=30)`. `/api/admin/diag/audio`
    reports the live in-process pactl/pipewire result.
15. **Admin UI cards (BT + WiFi-status read-only).** `routes/admin.py`:
    `/api/admin/bt/{status,scan,scan/{id},pair,connect,disconnect,forget,mic}` +
    WS `/api/admin/bt/events`. Read-only `/api/admin/wifi/status`. CSP-clean
    `static/js/admin-bt-card.js` + `admin-card-base.js`. Card markup in
    `static/index.html`.
16. **CLI ↔ UI parity matrix.** `docs/cli-ui-parity.md` enumerates every CLI subcommand
    classified operator-facing / daemon-control / diagnostics. CI lint diffs the matrix
    against the actual `cli/` and admin routes on every PR.
17. **Concurrent STA + AP** (Plan 2's mainline). `wlan_sta` VIF on phy0 (IPv6 off from
    creation). NM profiles pinned. `sta_connect`/`sta_disconnect`/`sta_reconcile`
    transactional flow. BSSID-pinned association + channel preflight + layered egress
    check. STA-DEGRADED + REPAIR-NEEDED markers. WIFI_FLOCK + KEYFILE_LOCK in order.
    Persistent infra (dispatcher hooks, sysctl.d, tmpfiles.d, NM conf.d).
18. **Trust install CLI** (`cli/trust.py`). `meeting-scribe export-cert PATH`,
    `trust-install --from-pem PATH`, `trust-install --confirm-fingerprint`,
    `cert-fingerprint`, `trust-uninstall <appliance_id>` / `--all`. Per-appliance
    Subject CN match for replace-not-add semantics.
19. **Tests + golden-path verification.** Every test file listed in Plan 1 §Tests
    + Plan 2's `tests/test_wifi.py` / `tests/test_wifi_sta.py` additions. Hardware-gated
    tests marked `@pytest.mark.gb10_hardware`; everything else runs on the dev box.

## Out of scope (deferred to v1.1)

- *USB audio devices as the data path* — architecture is source/sink-agnostic at the
  PipeWire-node level, but the discovery/CLI/UI for USB is a v1.1 follow-up.
- *Consecutive-interpretation mode* (live mic passthrough + queued TTS).
- *Delayed-apply-with-revert UI for hotspot mutations.*
- *Friendly hostname* (e.g. `meeting.demo`). v1.0 leaf cert covers `IP:10.42.0.1` only.
- *Multi-worker uvicorn.* Hardcoded `workers=1`; session/state model is single-process.
- *nftables migration for true atomicity.* Concurrent firewall writers handled
  best-effort via xtables wait lock + final under-lock revalidation + position-1
  invariant; documented as known limitation.

## Hardware-gated phases (cannot validate fully on this dev box)

- **Phase 13 (BT data plane)** — requires real BlueZ stack + paired audio device.
- **Phase 14 (lifespan auto-connect)** — same.
- **Phase 17 (concurrent STA + AP)** — requires MT7925 radio, second SSID, hotspot
  client to verify zero egress.
- **Phase 8 (firewall)** — code-side validation via subprocess mocks; final
  `iptables-save` parsing + position-1 invariant verification needs the GB10.

These phases land with full unit-test coverage on the dev box. End-to-end verification
is a hardware checklist captured in `docs/plans/sta-ap-unified-verification.md`
(written as part of Phase 19).

## Risk & open-questions register

The combined risk register is the union of Plan 1's R1–R66 and Plan 2's 1–38. They are
non-overlapping and tracked individually in their source plans. The position-1
invariant + canonical-generator pair (Plan 2 P0 / P1#1 / P1#2 / P2#1 / P2#2) supersedes
Plan 1's "exactly one jump" + RETURN-vs-ACCEPT story by being a strict superset.

## Progress (2026-05-04 working session)

Eight commits landed on `feature/sta-ap-unified-2026-05-04`:

| #  | Commit                                                                      | Phase  | Status |
|----|-----------------------------------------------------------------------------|--------|--------|
| 1  | docs: combined plan                                                         | docs   | ✅      |
| 2  | csp: extract inline scripts + AudioListener Protocol                        | 1      | ✅      |
| 3  | tls: per-device leaf-only cert + read-only SAN sanity check                 | 2      | ✅      |
| 4  | middleware: host canon + Origin allowlist + CSP injector + cache-headers    | 3      | ✅      |
| 5  | auth: per-boot subkey + session_id cookie + revocation hooks                | 4      | ✅      |
| 6  | hotspot: HTTP captive sub-app + 426 method guard + 308 catch-all            | 5      | ✅      |
| 7  | firewall: MS_* chain set + position-1 invariant + canonical generator       | 8      | ✅      |
| 8  | helper: privileged root-owned daemon + Unix-socket client                   | 10     | ✅      |

**Test counts**: 1453 unit tests pass (2 pre-existing skips, 0 regressions).
Each phase added focused unit tests:

* Phase 1: 73 audio/listener tests (existing) still pass.
* Phase 2: 9 new cert_check tests.
* Phase 3: 20 new security middleware tests.
* Phase 4: 11 new cookie session/revoke tests.
* Phase 5: 11 new captive HTTP sub-app tests.
* Phase 8: 25 new firewall position-1/canonical-generator/restore tests.
* Phase 10: 20 new helper daemon tests.

### Pending phases — deferred to follow-up sessions

| #   | Phase                                                                | Why deferred                                                                          |
|-----|----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 6   | Single-listener TLS bind + scope predicate flip                      | Deletes `_detect_management_ip*`/`_serve_dual` and rewires every CLI tool that probes via 127.0.0.1 (~30 tests). Needs hardware to verify the IP_FREEBIND bind on 10.42.0.1.        |
| 7   | Login/logout endpoints + bootstrap template                          | ~50 LOC + tests; depends on Phase 6 for the new cookie attrs to land in production.    |
| 9   | Admission control + per-IP/MAC rate limit                            | ~150 LOC + tests; no dependencies — could land independently.                          |
| 11  | Migrate sudo callsites to helper_client + remove sudoers grant       | Cross-cuts wifi.py + every CLI route + the systemd unit. Helper daemon is ready (Phase 10). |
| 12  | BT control plane (`bt.py` + `cli/bt.py`)                             | Pure code, unit-testable with subprocess mocks. ~600 LOC + tests.                      |
| 13  | BT data plane (`audio/bt_bridge.py` state machine)                   | Pure code skeleton possible; full validation needs BlueZ + paired device.              |
| 14  | Lifespan auto-connect + `/api/admin/diag/audio`                      | Depends on Phase 13.                                                                   |
| 15  | Admin UI BT card + WiFi STA card + endpoints + events WS             | Pure code — admin API + JS + HTML.                                                     |
| 16  | CLI ↔ UI parity matrix (`docs/cli-ui-parity.md` + CI lint)           | Pure docs + lint script.                                                               |
| 17  | Concurrent STA + AP `wifi.py` STA mode (Plan 2 mainline)             | Cannot validate without MT7925 radio on the GB10.                                      |
| 18  | Trust install CLI (`cli/trust.py`)                                   | Pure code — `meeting-scribe export-cert` / `trust-install --from-pem` / `--confirm-fingerprint` / `cert-fingerprint` / `trust-uninstall`. ~250 LOC + tests. |

### Hardware-gated verification list

End-to-end no-egress checklist + position-1-displacement scenario need a
real GB10 with the MT7925 radio + a hotspot client. Plan 1 §Verification +
Plan 2 §Verification both spell out the commands; consolidated as
`docs/plans/sta-ap-unified-verification.md` (to be written when the
hardware-gated work begins).

### Synthesis points (Plan 1 ↔ Plan 2)

The single non-trivial cross-plan merge is `wifi.py`'s firewall layer.
Resolution landed in `src/meeting_scribe/firewall.py`:

* Plan 2's `MS_INPUT` / `MS_FWD` / `MS_PRE` / `MS_POST` / `MS_INPUT6` chain
  naming + position-1 invariant + atomic combined-table restore.
* Plan 1's interface-scoped accept rules embedded inside Plan 2's
  `_expected_chain_rules` canonical generator.
* `sta_iface_present` toggle in the generator switches between AP-only
  and AP+STA isolation rules so hotspot clients have zero upstream egress
  in either mode.

## Memory pinning

This branch and plan respect the durable feedback recorded in
`/home/bradlay/.claude/projects/-home-bradlay-sddcinfo/memory/MEMORY.md`:

- *"No backwards-compat, keep codebase pristine"* — each phase's commit deletes the
  legacy code path it replaces (e.g. `_detect_management_ip*`, `:8080` strings, the
  `_apply_admin_firewall` dual-listener split).
- *"No re-ask after plan approval"* — once this combined plan is on the branch, phase
  execution does not pause to confirm intermediate steps.
- *"Strategic fixes not quick fixes"* — the privileged helper, position-1 invariant,
  and per-listener format binding are root-cause fixes, not symptom patches.
- *"Always use fastsafetensors"*, *"Never stop vLLM container"*, etc. — orthogonal to
  this branch but recorded for context.
