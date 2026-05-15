# WAN cutover & recovery checklist

Hardware-only end-to-end test for the WAN feature. Runs on the GB10
(MT7925 radio, `wlP9s9` AP + `wlan_sta` STA, wired `enP7s7`). Walks
the operator through the supported lifecycle and the documented
recovery paths.

Owners: meeting-scribe maintainers. Re-run after any change to
`wifi.py`, `wifi_wan.py`, `wifi_sta.py`, `routes/admin_wan.py`, or the
firewall canonical generator.

## Pre-flight

- [ ] GL-MT3000 still upstream of the GB10 (wired). `nmcli dev status` shows
      `enP7s7  ethernet  connected  Wired connection 3`.
- [ ] Wired admin UI reachable from a laptop on the same LAN
      (`https://<gb10>/`).
- [ ] `meeting-scribe wifi status` reports the AP as desired/live = off
      (or AP up if you're cutting over mid-meeting — note which state).
- [ ] Yunomotocho PSK present in the age store:
      `bash scripts/decrypt-creds.sh | grep -c '^YUNOMOTOCHO_PSK='` → `1`.

## 1 — Add a profile

```
meeting-scribe wifi wan profiles add --ssid Yunomotocho --psk-ref YUNOMOTOCHO_PSK
```

- [ ] Command prints a full uuid4 id. Copy it for later. ID must NOT
      be truncated with an ellipsis anywhere in the output.
- [ ] `meeting-scribe wifi wan profiles ls` lists the profile and
      shows the active line (or "(no profiles)" if you haven't set
      active yet).

## 2 — Set active + bring up WAN

```
meeting-scribe wifi wan profiles set-active --id <uuid>
meeting-scribe wifi wan up --id <uuid>
```

- [ ] `nmcli con show` lists exactly one `meeting-scribe-sta-<uuid>`
      connection (no duplicates, no other `meeting-scribe-sta-*`).
- [ ] `ip -br link show wlan_sta` reports state `UP`.
- [ ] `ip -6 a show wlan_sta` lists **zero** v6 addresses (no global,
      no link-local). **Hard stop if any v6 address is present.**
- [ ] `meeting-scribe wifi wan status` shows wired `default-route` true
      AND wifi `up` true with metric=600.

## 3 — Dual-WAN priority + failover

- [ ] `ip route show default` lists two defaults, wired metric=100
      lower than wlan_sta metric=600. Wired wins.
- [ ] Unplug `enP7s7` cable. Within 10 seconds, `ip route show default`
      lists only the wlan_sta default. Hotspot client still has
      Internet (or `curl --interface wlan_sta -m 5 https://1.1.1.1`
      from the GB10 succeeds).
- [ ] Re-plug `enP7s7`. Within 10 seconds, the wired default is back.

## 4 — Per-iface captive portal isolation (P1 regression)

If the upstream WiFi requires captive-portal auth, this is the most
important regression test. If captive auth isn't available in the
test venue, skip this section and document.

- [ ] Wired connectivity reports `full`, wifi connectivity reports
      `portal`, both surfaced independently in
      `meeting-scribe wifi wan status`. Wired-full must NOT mask the
      wlan_sta portal state.
- [ ] On a laptop connected to the GB10's AP hotspot, opening
      `http://example.com` redirects to the upstream portal URL.
      After admin authenticates, traffic resumes.

## 5 — Firewall posture (gateway mode)

Flip `wan_egress_mode` to `gateway` (via `/api/admin/wan/up` or by
editing `~/.config/meeting-scribe/settings.json` and restarting):

- [ ] `sudo iptables -t nat -S POSTROUTING | grep ms-fw` shows two
      MASQUERADE rules — one `-o enP7s7`, one `-o wlan_sta`.
- [ ] `sudo iptables -S INPUT | grep -E 'ms-fw.*wlan_sta'` shows
      explicit DROP on ports 22/80/443 for wlan_sta.
- [ ] `sudo iptables -S INPUT | grep -E 'ms-fw.*enP7s7'` shows
      explicit ACCEPT on ports 22/80/443 for enP7s7 (preserves wired
      admin path).
- [ ] `sudo ip6tables -L FORWARD -n -v` shows policy DROP. No
      `ms-fw`-tagged ACCEPT rules in the v6 chain.
- [ ] `cat /etc/sysctl.d/99-meeting-scribe-gateway.conf` shows
      `net.ipv4.ip_forward = 1` and
      `net.ipv6.conf.all.forwarding = 0`.

## 6 — WAN-side INPUT isolation probe (security gate)

Requires a second host on the upstream WiFi (a phone is fine — join
the same upstream as `wlan_sta`).

- [ ] From that host: `curl -m 5 --connect-timeout 3 http://<wlan_sta_v4_ip>:443`
      times out or refuses. **Hard stop if the admin UI responds.**
- [ ] From the upstream host: `ping6 -I <upstream_iface> ff02::1` —
      the GB10 must NOT respond from wlan_sta (no v6 stack).
- [ ] From a host on `enP7s7`, the same v4 curl must succeed.
      Wired admin path stays open.

## 7 — Reconciliation idempotence

```
for i in 1 2 3 4 5; do
  meeting-scribe wifi wan status >/dev/null
  sudo iptables-save | grep ms-fw | sort > /tmp/ms-fw-snapshot-$i
done
diff /tmp/ms-fw-snapshot-1 /tmp/ms-fw-snapshot-5
```

- [ ] No drift between snapshots 1 and 5 (empty diff). Same rules,
      same counters position-wise. Sorted by line content.

## 8 — Recovery: kill mid-cycle

- [ ] `meeting-scribe stop` while WAN is active. Server clean stop.
- [ ] `meeting-scribe start`. After boot, `meeting-scribe wifi wan status`
      shows wifi `up=false` (no autoconnect by design).
- [ ] `nmcli con show | grep meeting-scribe-sta-` still lists the
      preserved active profile (PSK persists on disk per v1 decision —
      not a bug).
- [ ] `meeting-scribe wifi wan up --id <uuid>` succeeds. Exactly ONE
      `meeting-scribe-sta-<uuid>` profile in `nmcli con show` afterward
      (no duplicate, no add-collision).

## 9 — Recovery: CLI broken / safe fallback

- [ ] Edit `~/.config/meeting-scribe/settings.json` and set
      `wan_egress_mode: "block"`. Restart `meeting-scribe`.
- [ ] After restart, `sudo iptables -t nat -S POSTROUTING | grep ms-fw`
      shows **no** MASQUERADE rules (block mode wipes them).
- [ ] AP-only known-good posture restored: hotspot clients cannot
      reach the Internet, captive portal still serves on the AP iface.

## 10 — Profile teardown

- [ ] `meeting-scribe wifi wan down`. `nmcli con show` no longer lists
      the `meeting-scribe-sta-<uuid>` connection.
- [ ] `/etc/NetworkManager/system-connections/meeting-scribe-sta-<uuid>.nmconnection`
      no longer exists (PSK gone with the keyfile).
- [ ] `meeting-scribe wifi wan profiles rm --id <uuid>` removes the
      saved profile. `meeting-scribe wifi wan profiles ls` reports
      `(no profiles)`.

## Hard stops (file an issue, do NOT ship)

Any of these signals a regression in a load-bearing invariant:

1. v6 address present on `wlan_sta` after `wan up`
2. Admin UI reachable on `wlan_sta`'s IP from an upstream host
3. `ip6tables -L FORWARD -n -v` policy != DROP after gateway apply
4. More than one `meeting-scribe-sta-*` profile after `wan up`
5. `meeting-scribe-sta-<active-id>` deleted by orphan cleanup
6. Reconcile idempotence snapshots differ across runs
7. Block-mode firewall rule set changes vs the pre-WAN baseline
   (run `tests/test_iptables_rules.py` to guard the static layout)
