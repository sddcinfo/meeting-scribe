# WAN management — GB10 as travel-router

The GB10 can reach an upstream network via two paths:

| Path | Iface | When it wins | Configured via |
|------|-------|--------------|----------------|
| Wired | `enP7s7` | Always when cable is plugged in (NM metric=100) | OS / existing NM profile, claimed at first `wan up` |
| WiFi STA | `wlan_sta` | Fallback when wired is down (NM metric=600) | `meeting-scribe wifi wan ...` |

Both can be configured simultaneously. NM picks the lower-metric default
route, so cable-plug always wins — important for emergency admin
access (the admin UI is allowed inbound on `enP7s7` only; explicitly
denied on `wlan_sta`).

Replaces an external GL-MT3000 travel router for portable demos.

## First-time setup

1. Add the upstream PSK to the age store:

   ```
   ./scripts/encrypt-creds.sh   # opens an editor, add  YUNOMOTOCHO_PSK=<plaintext>
   ```

   PSK refs are SCREAMING_SNAKE_CASE keys (validated against
   `^[A-Z][A-Z0-9_]{0,63}$`). The plaintext PSK never lives in
   `settings.json` — only the ref.

2. Create a profile:

   ```
   meeting-scribe wifi wan profiles add --ssid Yunomotocho --psk-ref YUNOMOTOCHO_PSK
   ```

   The CLI prints a full uuid4 id. Copy it — that's how you address
   the profile from now on. IDs are never truncated in output.

3. Set it active and bring up the WAN:

   ```
   meeting-scribe wifi wan profiles set-active --id <uuid>
   meeting-scribe wifi wan up --id <uuid>
   ```

4. Confirm:

   ```
   meeting-scribe wifi wan status
   ```

   You should see wired + wifi reported independently, with their own
   connectivity badges. If wifi shows `connectivity: portal`, see
   [Captive portal auth](#captive-portal-auth) below.

## Captive portal auth

Many public-WiFi venues redirect the first HTTP request to a portal.
The GB10 detects this per-interface and surfaces the portal URL via
`wifi wan status` and the admin UI's WAN tab.

**v1 uses Tier A passthrough** — no portal-specific code on the GB10:

1. Connect a laptop to the GB10's hotspot AP (`wlP9s9`).
2. Open any HTTP URL in your browser. The upstream venue's portal
   intercepts you and presents its login page.
3. Authenticate. The upstream sees the GB10's NAT'd source MAC and
   whitelists it. From that moment, every device behind the GB10
   benefits — including the GB10's own outbound traffic.

This works for the majority of venues. Some portals tie auth to the
source IP+lease and you may have to re-authenticate after a DHCP renewal.

## Failover

NM watches default-route changes automatically.

- Unplug `enP7s7` → within ~5–10s, `wlan_sta` becomes the default
  route. Hotspot clients keep connectivity.
- Re-plug `enP7s7` → wired takes back over (lower metric wins).

There is **no boot-time autoconnect** — after a reboot, `wifi wan up`
must be re-run (deliberate v1 choice for predictable boot behavior).

## Security posture

When `wan_egress_mode=gateway`:

- `wlan_sta` inbound is **denied** on ports 22, 80, 443 (explicit DROP
  rules tagged `ms-fw`). The admin UI cannot be reached from upstream.
- `enP7s7` inbound is **allowed** on the same ports — emergency wired
  admin access is preserved.
- AP→WAN forwarding is allowed (new + established/related); WAN→AP
  only allows established/related (no upstream-initiated connections
  to hotspot clients).
- IPv6 is fully off on `wlan_sta` (`ipv6.method=disabled` +
  `net.ipv6.conf.wlan_sta.disable_ipv6=1`); `ip6tables -P FORWARD DROP`
  enforced steady-state.

When `wan_egress_mode=block` (default), AP→WAN forwarding is dropped
entirely — the legacy zero-egress hotspot posture for meeting privacy.

## PSK at rest

v1 explicitly accepts that the upstream PSK lives in
`/etc/NetworkManager/system-connections/meeting-scribe-sta-<id>.nmconnection`
(mode 0600, root-only) from `wan up` until an explicit `wan down`.
The PSK survives reboot in NM's keyfile — admin convenience over
exposure window. Settings JSON never contains the PSK, only `psk_ref`.

v2 candidate: NM secret-agent with `wifi-sec.psk-flags=0x2` for true
"only in age store" semantics.

## Recovery

If the CLI is broken or the firewall state looks wrong, restore the
safe AP-only posture:

```
# Option 1 — CLI works:
meeting-scribe wifi wan down
meeting-scribe wifi up --mode admin

# Option 2 — CLI broken:
#   1. edit ~/.config/meeting-scribe/settings.json
#   2. set "wan_egress_mode": "block"
#   3. systemctl restart meeting-scribe  (or `meeting-scribe restart`)
```

For the full hardware-cutover checklist, see
[`tests/manual/wifi_wan_cutover.md`](../tests/manual/wifi_wan_cutover.md).

## Files

| Concern | File |
|---------|------|
| Orchestration | `src/meeting_scribe/wifi_wan.py` |
| Firewall | `src/meeting_scribe/wifi.py` (`_ms_fw_rules`, `reconcile_network_state`) |
| Settings schema | `src/meeting_scribe/server_support/settings_store.py` |
| Secrets resolution | `src/meeting_scribe/server_support/secrets.py` |
| CLI | `src/meeting_scribe/cli/wifi.py` (`wifi wan` subgroup) |
| REST | `src/meeting_scribe/routes/admin_wan.py` |
| UI | `static/js/admin-wan-card.js` |
| Live tests | `tests/test_wifi_wan_live.py` (gated `@pytest.mark.real_gb10`) |
| Manual checklist | `tests/manual/wifi_wan_cutover.md` |
| Design doc | `docs/plans/wifi-wan-gateway.md` |
