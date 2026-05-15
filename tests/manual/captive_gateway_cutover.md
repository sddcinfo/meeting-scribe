# Captive gateway — manual cutover checklist

Live hardware verification of Phase H (per-client captive gateway).
Requires two client devices (phone + laptop ideal) and a working
upstream WAN (wired or WiFi STA — see `docs/wifi-wan.md`).

## Pre-flight

- [ ] `sudo ipset --version` succeeds — userspace package present.
- [ ] `lsmod | grep ip_set` shows `ip_set` and `xt_set` loaded.
- [ ] `meeting-scribe status` reports `wifi: admin` and
      WAN reachable (`active_default = enP7s7` or `wlan_sta`).
- [ ] `meeting-scribe wifi wan mode get` returns `mode: captive,
      source: default` (post-migration) OR `source: operator` if you
      previously picked a different mode.

## 1 — Single-admin happy path

- [ ] Join `Dell Meeting <pin>` from device A (phone).
- [ ] OS captive-portal sheet pops within 5–10 s, lands on `/auth`.
- [ ] Sign in with `DellMeetingAdmin<pin>` (the deterministic admin
      password from `_mint_admin_password()`).
- [ ] CNA dismisses (Success body); device A can browse any external
      site, e.g. `https://example.com` resolves.
- [ ] `sudo ipset list ms-allowed-admins` shows device A's AP-side
      IP (`10.42.0.x`).

## 2 — Untrusted second device stays gated

- [ ] Join `Dell Meeting <pin>` from device B (laptop), do NOT sign in.
- [ ] Open a browser tab, try `http://example.com` → REDIRECT'd to
      `https://meeting-<pin>.local/auth`.
- [ ] Try `https://example.com` (HTTPS) → connection times out, no
      MITM warning. (Expected; we don't intercept HTTPS.)
- [ ] `curl -m 5 http://1.1.1.1` from device B → no response.
- [ ] `sudo ipset list ms-allowed-admins` still shows device A only.

## 3 — Guest tier

- [ ] On device B, enter the 4-digit guest PIN.
- [ ] CNA dismisses (no more captive nagging).
- [ ] Device B can reach meeting URLs on `10.42.0.1` (guest UI), but
      still cannot browse `example.com`.
- [ ] `sudo ipset list ms-allowed-guests` shows device B's IP.

## 4 — Admin logout drains WAN

- [ ] On device A, sign out of admin (admin menu → sign out, OR
      `meeting-scribe wifi wan mode set captive` from the host CLI to
      force a reconcile).
- [ ] `sudo ipset list ms-allowed-admins` no longer contains
      device A's IP.
- [ ] Device A's browser tries `https://example.com` → blocked.
- [ ] Device A's OS CNA pops again on next probe.

## 5 — Mode toggling

- [ ] `meeting-scribe wifi wan mode set gateway` from the host →
      device B (still unauthorized) can now reach `example.com`.
- [ ] `meeting-scribe wifi wan mode set captive` → device B blocked
      again, prompted to sign in.
- [ ] `meeting-scribe wifi wan mode set block` → ALL AP clients
      (including signed-in admins) lose WAN forwarding. Meeting
      privacy posture.
- [ ] `meeting-scribe wifi wan mode get` shows
      `source: operator` after each set.

## 6 — Migration regression

- [ ] On a fresh box with no persisted `wan_egress_mode`, restart
      meeting-scribe + bring up admin AP → `meeting-scribe wifi wan
      mode get` reports `mode: captive, source: default`.
- [ ] Explicit `meeting-scribe wifi wan mode set block` → source flips
      to `operator`. Restart the server. Verify `block` survives — the
      migration ladder does NOT re-flip to captive.

## 7 — GC tick

- [ ] After ~5 min, a client that fully disconnects from the AP gets
      pruned from its ipset by the next GC iteration. Verify with
      `sudo ipset list ms-allowed-admins` — old IPs gone.
- [ ] `journalctl -u meeting-scribe | grep "captive gc"` shows the
      "pruned N admin + M guest" log line.

## Hard stops

If any of these reproduce, file and revert before shipping:

- An unauthorized AP client succeeds at reaching an external IP
  (e.g. `curl -m 5 https://1.1.1.1` returns data). Per-IP gating is
  broken.
- Block mode iptables snapshot differs from a pre-Phase-H baseline
  for a real WAN bring-up (regression in `_ms_fw_rules`).
- An operator-set `block` reverts to `captive` after restart.
- Captive sub-app keeps returning 302 for an IP that's in
  `ms-allowed-admins` (CNA flicker).
- `sudo iptables-save` shows the same rule twice after a reconcile
  (rule duplication).
