# Captive gateway (Phase H)

Per-client authorization at the firewall layer. When the AP is in
`admin` mode and `wan_egress_mode=captive`, the FORWARD chain only
lets through traffic from clients whose IP is in the
`ms-allowed-admins` ipset. Everything else gets either DROP'd
(non-HTTP) or REDIRECT'd to the local captive sub-app (tcp/80).

Three trust tiers, gated at the kernel firewall:

| Tier        | Auth path                       | Reach 10.42.0.1 | Reach upstream WAN |
|-------------|---------------------------------|------------------|---------------------|
| Untrusted   | none (cookie + ipset both empty) | Yes — /auth     | **No**              |
| Guest       | guest PIN → `ms-allowed-guests` | Yes — guest UI  | **No**              |
| Admin       | admin password → `ms-allowed-admins` | Yes — admin UI | **Yes**             |

## Sign-in flow

1. Client joins `Dell Meeting <pin>` (or whatever the operator-set SSID is).
2. OS captive-portal probe hits `http://10.42.0.1/<probe-path>` (the
   AP is the gateway, so all unauthorized tcp/80 traffic gets
   REDIRECT'd to the local sub-app even when the OS dials a different
   hostname).
3. Probe returns a 302 to `https://meeting-<pin>.local/auth`.
4. Operating-system captive-portal sheet pops the admin login page.
5. Admin enters password → `POST /api/admin/authorize` validates +
   sets `scribe_admin` cookie **and** adds the caller's source IP to
   the `ms-allowed-admins` ipset via the captive-gateway hook.
6. Next probe response: now `_is_captive_acked` sees the IP in the
   ipset, returns the platform-specific "Success" body — CNA closes.
7. Admin can now browse externally through the AP normally; the
   FORWARD rule matches their src IP against the admins ipset and
   ACCEPTs.

Guests are identical except they hit `POST /api/auth/guest-pin` and
land in `ms-allowed-guests`. They get the CNA to close (so iOS stops
nagging) but never get FORWARD AP→WAN — only admins do.

## Egress mode posture

| Mode      | FORWARD AP→WAN     | tcp/80 to AP from unauthorized | Default for admin SSID |
|-----------|---------------------|--------------------------------|------------------------|
| `block`   | DROP all (zero-egress) | n/a                          | (historical default)   |
| `gateway` | ACCEPT all           | passes through                | Operator opt-in        |
| `captive` | ACCEPT iff src in ms-allowed-admins | REDIRECT to local :80 | Current default        |

A persisted `wan_egress_mode_source` field distinguishes "I haven't
touched it" (`default`) from "I explicitly picked this" (`operator`).
The AP-up auto-upgrade ladder promotes `block`→`captive` only when
source is `default`. Any explicit set via CLI / REST / UI stamps
source as `operator` and is sticky against future auto-upgrades.

## Operator surfaces

- **CLI**
  - `meeting-scribe wifi wan mode get` — print mode + source
  - `meeting-scribe wifi wan mode set {block|gateway|captive}` — set
    + reconcile firewall in one step
- **REST** (admin-cookie-gated)
  - `GET /api/admin/wan/mode` → `{"mode": ..., "source": ...}`
  - `PUT /api/admin/wan/mode` body `{"mode": ...}` — sets +
    reconciles, always stamps source `operator`
- **Admin UI**: "Egress mode" radio in the WAN settings card. The
  inline label under the radios shows whether the value is the
  shipped default or operator-set.

## What if `ipset` is missing?

The userspace `ipset` package isn't bundled in every Linux base image.
If it's missing on the GB10:

- `firewall_allowlist.is_available()` returns False
- Every CRUD function is a no-op + logs a single warning
- AP bring-up logs `captive: ensure_sets failed (non-fatal)`
- Captive firewall rules still try to `--match-set`; iptables will
  refuse them with a clear error in the server log

**Remediation**: `sudo apt install ipset && meeting-scribe restart`.
Bootstrap (`bootstrap.sh`) now includes `ipset` in the apt install
list, so fresh customer images self-heal.

If you can't install `ipset` for some reason, set
`wan_egress_mode=gateway` to fall back to pre-Phase-H behavior
(unconditional AP→WAN pass-through, no per-client gating).

## DHCP / IP identity

The captive gate is **IP-based**, not MAC-based. Lease changes that
hand a client a new IP require re-auth.

- NM's shared-mode dnsmasq is sticky-by-MAC, so a client that
  disconnects + reconnects usually gets the same IP back; their ipset
  entry stays valid.
- The current lease length is 60 minutes (NM default). When a client
  renews mid-session (typically at half-life), they keep the same IP
  — no re-auth.
- Worst case (IP exhaustion or NM dnsmasq restart): the client gets
  a new IP, the old IP stays in the ipset until the GC tick (every
  5 minutes) prunes it. The stale entry has no security impact
  because that IP is no longer assigned to a trusted client.

## Idle GC

`firewall_allowlist.gc_loop()` runs every 5 minutes from the server
lifespan. Each tick:

1. Read `/var/lib/NetworkManager/dnsmasq-wlP9s9.leases` (best-effort —
   a missing or unreadable file just skips the tick).
2. For each ipset entry, if the IP isn't in the current lease set,
   `ipset del` it.

This handles clients who disconnect without explicit logout. Explicit
logout already removes the entry synchronously.

## Edge cases

- **OS captive sheet vs full browser**: the OS captive sheet often
  doesn't share cookies with the regular browser. After admin signs
  in via the sheet, the caller IP is in the admins ipset — that's
  what gates WAN. The regular browser inherits WAN access **without
  re-authenticating** because the gate is per-IP, not per-cookie.
- **Multi-device admin**: phone + laptop on the same AP each need
  to authenticate separately. Per-IP gating means one device's
  authorization doesn't grant another's. Losing the laptop shouldn't
  compromise admin access from the phone.
- **Guest who later signs in as admin**: same device, same IP. Both
  ipsets contain the IP. WAN gate consults only the admins set.
  Admin logout removes from admins → WAN blocked → guest access still
  works.
- **Cookie expires but device stays connected**: GC removes IPs whose
  lease is gone, NOT IPs whose cookie has expired. An admin whose
  7-day cookie expires keeps WAN until they disconnect. If you want
  to drain on cookie expiry, the operator can `meeting-scribe wifi
  wan mode set captive` again — it triggers a reconcile that re-
  applies the rules but doesn't re-evaluate cookies. Drain manually
  with `sudo ipset flush ms-allowed-admins`.

## Hard-stop invariants

Tests guard each one (see `tests/test_firewall_gateway_mode.py`,
`tests/test_captive_authorize_hooks.py`,
`tests/test_captive_ack_tier_aware.py`,
`tests/test_wan_mode_control.py`):

- Block-mode firewall snapshot is byte-identical to the pre-captive
  baseline.
- Gateway mode does NOT carry the ipset clause (no `--match-set
  ms-allowed-admins`).
- Migration ladder NEVER overwrites a mode whose source is `operator`.
- Captive sub-app returns Success for an IP in either ipset.
- An external HTTP request from an unauthorized AP client REDIRECTs
  to the local port 80.
- An unauthorized AP client's TCP-to-1.1.1.1 attempt gets DROP'd on
  FORWARD (no MASQUERADE path for non-allowlisted source).
