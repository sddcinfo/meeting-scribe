"""wifi sub-group: up / down / status for the GB10 WiFi hotspot."""

from __future__ import annotations

import click

from meeting_scribe.cli import cli


@cli.group("wifi")
def wifi_group():
    """WiFi hotspot management (off / setup / meeting / admin mode).

    \b
    The GB10 MT7925 radio supports one AP at a time. Three modes:
      setup    — open SSID 'Dell Meeting Setup' for first-touch wizard
      meeting  — rotating SSID, captive portal, guest isolation
      admin    — fixed SSID, no portal, admin UI reachable over WiFi
    """


@wifi_group.command("up")
@click.option(
    "--mode",
    type=click.Choice(["setup", "meeting", "admin"]),
    default="meeting",
    show_default=True,
    help="setup = open AP for v1.0 first-touch wizard; "
    "meeting = rotating guest SSID + captive portal; "
    "admin = fixed SSID, no portal, admin UI over WiFi",
)
@click.option("--ssid", default=None, help="Override SSID (ignored in setup mode)")
@click.option("--password", default=None, help="Override password (ignored in setup mode)")
@click.option(
    "--band",
    default="a",
    show_default=True,
    type=click.Choice(["a", "bg"]),
    help="WiFi band (a=5GHz, bg=2.4GHz)",
)
@click.option("--channel", default=36, show_default=True, type=int, help="WiFi channel")
def wifi_up(mode: str, ssid: str | None, password: str | None, band: str, channel: int) -> None:
    """Bring up the WiFi AP in setup, meeting, or admin mode."""
    import asyncio as _asyncio

    from meeting_scribe.wifi import build_config, wifi_up_setup
    from meeting_scribe.wifi import wifi_up as _wifi_up

    if mode == "setup":
        click.echo(click.style("==> Bringing up open setup AP...", fg="cyan"))
        _asyncio.run(wifi_up_setup())
        click.echo()
        click.secho(
            "Setup-mode AP is live! (OWE — no password prompt on phones)",
            fg="green",
            bold=True,
        )
        click.echo("  SSID:     Dell Meeting Setup")
        click.echo("  Security: OWE (Opportunistic Wireless Encryption, RFC 8110)")
        click.echo("  Phones:   iOS 13+, Android 10+, macOS 11+, Win 11+")
        click.echo("  Wizard:   https://10.42.0.1/setup")
        return

    cfg = build_config(mode, ssid, password, band, channel)
    click.echo(click.style(f"==> Bringing up WiFi AP ({mode} mode)...", fg="cyan"))
    _asyncio.run(_wifi_up(cfg))
    click.echo()
    click.echo(click.style(f"WiFi AP is live! ({mode} mode)", fg="green", bold=True))
    click.echo(f"  Mode:      {mode}")
    click.echo(f"  SSID:      {cfg.ssid}")
    if cfg.security == "open":
        # OWE (Opportunistic Wireless Encryption): looks like an open
        # network on the join screen but the radio link is DH-encrypted.
        # Captive-gateway gates upstream access via the /auth flow.
        click.echo("  Security:  OWE (open join, no password)")
    else:
        click.echo(f"  Password:  {cfg.password}")
        click.echo("  Security:  WPA3-SAE (PMF required)")
    click.echo(f"  Band:      {'5 GHz' if band == 'a' else '2.4 GHz'} (channel {channel})")
    if mode == "admin":
        click.echo(f"  Admin UI:  https://{cfg.ap_ip}/")
        click.echo("  Portal:    captive — sign in at /auth (Phase H)")
    else:
        click.echo(f"  Web UI:    https://{cfg.ap_ip}/ (cookie picks admin vs guest)")
        click.echo(f"  Captive:   DNS wildcard → {cfg.ap_ip}")


@wifi_group.command("down")
def wifi_down_cmd() -> None:
    """Tear down the WiFi AP. Persists wifi_mode=off."""
    import asyncio as _asyncio

    from meeting_scribe.wifi import wifi_down as _wifi_down

    click.echo(click.style("==> Tearing down WiFi AP...", fg="cyan"))
    _asyncio.run(_wifi_down())
    click.secho("WiFi AP stopped. wifi_mode=off persisted.", fg="green")


@wifi_group.command("status")
def wifi_status_cmd() -> None:
    """Show live WiFi status (reads from nmcli/wpa_cli, not just state file)."""
    from meeting_scribe.wifi import wifi_status_sync

    info = wifi_status_sync()
    desired = info.get("desired_mode", "off")
    live = info.get("live_mode", "off")

    click.echo(click.style("WiFi Status", fg="cyan", bold=True))
    click.echo()

    dm_color = "green" if desired != "off" else "yellow"
    click.echo(f"  Desired:     {click.style(desired.upper(), fg=dm_color)}")

    lm_color = "green" if live != "off" else ("yellow" if live == "unknown" else "red")
    click.echo(f"  Live:        {click.style(live.upper(), fg=lm_color)}")

    if info.get("ssid"):
        click.echo(f"  SSID:        {info['ssid']}")
    if info.get("security"):
        sec = info["security"]
        km = sec.get("key_mgmt", "?")
        label = click.style("WPA3-SAE", fg="green") if km == "SAE" else click.style(km, fg="yellow")
        click.echo(f"  Security:    {label} pairwise={sec.get('pairwise_cipher', '?')}")
    if info.get("client_count") is not None:
        click.echo(f"  Clients:     {info['client_count']}")

    rd_configured = info.get("regdomain")
    rd_live = info.get("regdomain_live")
    if rd_live and rd_configured:
        if rd_live == rd_configured:
            click.echo(f"  Regdomain:   {click.style(rd_live, fg='green')}")
        else:
            click.echo(
                f"  Regdomain:   {click.style('DRIFT', fg='red')} "
                f"(configured={rd_configured} live={rd_live})"
            )

    cap = info.get("captive_active")
    if cap is True:
        click.echo(f"  Captive:     {click.style('active', fg='green')}")
    elif cap is False:
        click.echo(f"  Captive:     {click.style('disabled', fg='yellow')}")


# ─── wifi wan — upstream WAN management ────────────────────────


@wifi_group.group("wan")
def wan_group() -> None:
    """Upstream WAN (wired + WiFi STA) management.

    Replaces the GL-MT3000 travel router for portable demos. Wired
    (``enP7s7``) is always preferred when up; WiFi STA (``wlan_sta``) is
    the fallback. Profiles are stored by stable uuid4 id; PSK refs
    point at keys in the sddcinfo age store.
    """


def _format_uuid_table(rows: list[dict]) -> str:
    if not rows:
        return "(no profiles)"
    # IDs are printed FULL, never truncated (per feedback_no_id_truncation).
    lines = [f"{'ID':<36}  {'SSID':<24}  {'Band':<6}  {'BSSID':<17}  REF"]
    for r in rows:
        band = (r.get("band") or "auto").lower()
        band_display = {"a": "5", "bg": "2.4"}.get(band, "auto")
        ref_display = r.get("psk_ref") or "(OPEN)"
        lines.append(
            f"{r['id']:<36}  {r['ssid']:<24}  {band_display:<6}  "
            f"{(r.get('bssid') or '-'):<17}  {ref_display}"
        )
    return "\n".join(lines)


def _resolve_profile_or_die(*, id_arg: str | None, ssid: str | None) -> dict:
    """Return the WanProfile dict for ``id_arg`` or unique ``ssid``.

    Errors with usage hints if both are missing or ``ssid`` is ambiguous.
    Always called from CLI commands that take ``--id`` and/or ``--ssid``.
    """
    from meeting_scribe.server_support import settings_store as store

    if id_arg:
        prof = store._find_wan_profile_by_id(id_arg)
        if prof is None:
            raise click.ClickException(f"no wan profile with id {id_arg!r}")
        return prof
    if ssid:
        matches = store._find_wan_profiles_by_ssid(ssid)
        if not matches:
            raise click.ClickException(f"no wan profile with ssid {ssid!r}")
        if len(matches) > 1:
            ids = "\n  ".join(p["id"] for p in matches)
            raise click.ClickException(
                f"ssid {ssid!r} is ambiguous — multiple profiles match:\n  {ids}\n"
                f"re-run with --id <full-uuid> to disambiguate."
            )
        return matches[0]
    raise click.ClickException("specify --id or --ssid")


@wan_group.command("up")
@click.option("--id", "id_arg", default=None, help="Stable uuid4 profile id (preferred)")
@click.option("--ssid", default=None, help="SSID (errors if multiple profiles match)")
def wan_up_cmd(id_arg: str | None, ssid: str | None) -> None:
    """Bring up the WAN association for a saved profile."""
    import asyncio as _asyncio

    from meeting_scribe import wifi_wan

    prof = _resolve_profile_or_die(id_arg=id_arg, ssid=ssid)
    click.echo(click.style(f"==> Bringing up WAN: {prof['ssid']} (id={prof['id']})", fg="cyan"))
    _asyncio.run(wifi_wan.wan_up(prof["id"]))
    click.secho("WAN STA association up.", fg="green", bold=True)


@wan_group.command("down")
def wan_down_cmd() -> None:
    """Tear down the active WAN STA association and delete the NM profile."""
    import asyncio as _asyncio

    from meeting_scribe import wifi_wan

    click.echo(click.style("==> Tearing down WAN STA...", fg="cyan"))
    _asyncio.run(wifi_wan.wan_down())
    click.secho("WAN STA torn down.", fg="green")


@wan_group.command("status")
def wan_status_cmd() -> None:
    """Show per-interface WAN status (wired + WiFi STA + active default)."""
    import asyncio as _asyncio

    from meeting_scribe import wifi_wan

    status = _asyncio.run(wifi_wan.wan_status())
    click.echo(click.style("WAN Status", fg="cyan", bold=True))
    click.echo()

    def _pill(label: str, value: bool, *, true: str = "green", false: str = "yellow") -> str:
        return click.style(label, fg=true if value else false)

    wired = status["wired"]
    click.echo(f"  Wired ({wired['iface']}):")
    click.echo(
        f"    Link:        {_pill('UP', wired['up']) if wired['up'] else _pill('DOWN', False, false='red')}"
    )
    if wired["lease"]:
        click.echo(f"    Lease:       {wired['lease']}")
    if wired["profile_name"]:
        click.echo(f"    NM profile:  {wired['profile_name']}")
    if wired["route_metric"] is not None:
        click.echo(f"    Route metric:{wired['route_metric']}")
    click.echo(f"    Default:     {_pill('YES', wired['default_route'])}")

    wifi = status["wifi"]
    click.echo()
    click.echo(f"  WiFi STA ({wifi['iface']}):")
    click.echo(
        f"    Link:        {_pill('UP', wifi['up']) if wifi['up'] else _pill('DOWN', False, false='red')}"
    )
    if wifi["ssid"]:
        click.echo(f"    SSID:        {wifi['ssid']}")
    if wifi["bssid"]:
        click.echo(f"    BSSID:       {wifi['bssid']}")
    if wifi["signal_dbm"] is not None:
        click.echo(f"    Signal:      {wifi['signal_dbm']} dBm")
    if wifi["lease"]:
        click.echo(f"    Lease:       {wifi['lease']}")
    if wifi["profile_name"]:
        click.echo(f"    NM profile:  {wifi['profile_name']}")
    if wifi["route_metric"] is not None:
        click.echo(f"    Route metric:{wifi['route_metric']}")

    conn = wifi.get("connectivity", "unknown")
    conn_color = {"full": "green", "portal": "yellow", "limited": "yellow", "none": "red"}.get(
        conn, "white"
    )
    click.echo(f"    Connectivity:{click.style(conn, fg=conn_color)}")
    if wifi.get("portal_url"):
        click.echo(f"    Portal URL:  {click.style(wifi['portal_url'], fg='blue', underline=True)}")

    click.echo()
    click.echo(f"  Active default route: {status['active_default'] or '(none)'}")
    click.echo(f"  Egress mode:          {status['egress_mode']}")


@wan_group.command("scan")
@click.option(
    "--raw",
    is_flag=True,
    help="Show each BSS as its own row instead of consolidating by SSID",
)
def wan_scan_cmd(raw: bool) -> None:
    """Scan for upstream WiFi networks (consolidated by SSID by default)."""
    import asyncio as _asyncio

    from meeting_scribe import wifi_wan

    entries = _asyncio.run(wifi_wan.scan_upstream())
    if not entries:
        click.echo(
            "(no APs visible — radio may be busy with the AP or NM hasn't cached a scan yet)"
        )
        return
    if raw:
        entries = sorted(entries, key=lambda e: e.signal_dbm, reverse=True)
        click.echo(f"{'SSID':<32}  {'BSSID':<17}  {'Ch':<4} {'Signal':<10} Sec")
        for e in entries:
            sec = "WPA" if e.rsn_present else "OPEN"
            click.echo(f"{e.ssid:<32}  {e.bssid:<17}  {e.channel:<4} {e.signal_dbm:<10.1f} {sec}")
        return
    # Consolidated: one row per (SSID, security). Bands column shows
    # which of 2.4/5 GHz advertise this SSID — handy for picking a
    # band override in the Add form.
    groups = wifi_wan.consolidate_scan(entries)
    click.echo(f"{'SSID':<32}  {'Bands':<10}  {'APs':<5}  {'Signal':<10}  Sec")
    for g in groups:
        bands_str = "+".join("5" if b == "a" else "2.4" for b in g.bands) + " GHz"
        sec = "WPA" if g.rsn_present else "OPEN"
        click.echo(
            f"{g.ssid:<32}  {bands_str:<10}  {g.ap_count:<5}  {g.best_signal_dbm:<10.1f}  {sec}"
        )


# ─── wifi wan mode — egress posture ────────────────────────────


@wan_group.group("mode")
def wan_mode_group() -> None:
    """Read or set the WAN egress posture (block / gateway / captive)."""


@wan_mode_group.command("get")
def wan_mode_get_cmd() -> None:
    """Print the current egress mode + provenance (default vs operator)."""
    from meeting_scribe.server_support import settings_store

    mode = settings_store._effective_wan_egress_mode()
    source = settings_store._wan_egress_mode_source()
    click.echo(f"mode:   {mode}")
    click.echo(f"source: {source}")


@wan_mode_group.command("set")
@click.argument("mode", type=click.Choice(["block", "gateway", "captive"]))
def wan_mode_set_cmd(mode: str) -> None:
    """Set the WAN egress mode and trigger a single reconcile.

    Persists ``wan_egress_mode_source="operator"`` so the AP-up
    migration ladder never overwrites an explicit pick.
    """
    import asyncio as _asyncio

    from meeting_scribe.server_support import settings_store

    settings_store._set_wan_egress_mode(mode)  # default source="operator"
    click.echo(f"wan_egress_mode set to {mode!r} (source=operator)")
    # Trigger a single firewall reconcile so the new posture applies
    # without waiting for the next AP transition. Imported lazily —
    # CLI invocations from outside the server context should still
    # work (settings are persisted; next server start picks them up).
    try:
        from meeting_scribe.wifi import reconcile_network_state

        _asyncio.run(reconcile_network_state())
        click.secho("firewall reconciled.", fg="green")
    except Exception as exc:
        click.secho(
            f"settings saved, but live reconcile failed: {exc}",
            fg="yellow",
        )


# ─── wifi wan profiles — CRUD ─────────────────────────────────


@wan_group.group("profiles")
def wan_profiles_group() -> None:
    """Saved upstream-WiFi profile management (SSID + PSK ref)."""


@wan_profiles_group.command("ls")
def wan_profiles_ls_cmd() -> None:
    """List saved upstream profiles. IDs printed full."""
    from meeting_scribe.server_support import settings_store

    profiles = settings_store._load_wan_profiles()
    click.echo(_format_uuid_table(profiles))
    active = settings_store._effective_wan_active_profile_id()
    if active:
        click.echo()
        click.echo(f"Active: {active}")


@wan_profiles_group.command("add")
@click.option("--ssid", required=True, help="Upstream WiFi SSID")
@click.option(
    "--psk-ref",
    "psk_ref",
    default=None,
    help="Key in .credentials.env.age (SCREAMING_SNAKE_CASE, e.g. YUNOMOTOCHO_PSK). "
    "Omit when --open is set.",
)
@click.option(
    "--open",
    "open_network",
    is_flag=True,
    help="Mark this network as open (no PSK). Mutually exclusive with --psk-ref.",
)
@click.option(
    "--band",
    type=click.Choice(["auto", "2.4", "5"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Band preference: auto = NM picks, 2.4 = stay on 2.4 GHz, 5 = stay on 5 GHz",
)
@click.option(
    "--bssid",
    default=None,
    help="Advanced: pin a specific BSSID (disables roaming). Prefer --band for typical cases.",
)
@click.option(
    "--no-verify-psk-ref",
    is_flag=True,
    help="Skip dry-decrypt validation of psk-ref (useful for tests / first-touch setup)",
)
def wan_profiles_add_cmd(
    ssid: str,
    psk_ref: str | None,
    open_network: bool,
    band: str,
    bssid: str | None,
    no_verify_psk_ref: bool,
) -> None:
    """Add a new upstream-WiFi profile. PSK is referenced, never inlined."""
    import uuid as _uuid

    from meeting_scribe.server_support import secrets, settings_store

    if open_network and psk_ref:
        raise click.ClickException("--open and --psk-ref are mutually exclusive")
    if not open_network and not psk_ref:
        raise click.ClickException("either --psk-ref <REF> or --open is required (open = no auth)")
    if psk_ref and not no_verify_psk_ref and not secrets.psk_ref_exists(psk_ref):
        raise click.ClickException(
            f"psk-ref {psk_ref!r} not found in the age store — add it via "
            f"scripts/encrypt-creds.sh first, or re-run with --no-verify-psk-ref"
        )
    # Normalize "2.4" / "5" / "auto" → NM band code.
    band_code = {"2.4": "bg", "5": "a", "auto": "auto"}[band.lower()]
    profile = {
        "id": str(_uuid.uuid4()),
        "ssid": ssid,
        "bssid": bssid,
        "band": band_code,
        "psk_ref": psk_ref,  # None for open networks
        "regdomain": None,
        "last_seen": None,
    }
    settings_store._save_wan_profile(profile)
    click.echo("Added profile (full id printed for copy/paste):")
    click.echo(f"  id   {profile['id']}")
    click.echo(f"  ssid {profile['ssid']}")
    click.echo(f"  band {band_code} ({band.lower() if band_code != 'auto' else 'auto'})")
    if bssid:
        click.echo(f"  bssid {bssid} (pinned — roaming disabled)")
    if psk_ref:
        click.echo(f"  psk_ref {psk_ref}")
    else:
        click.echo("  security OPEN (no PSK)")


@wan_profiles_group.command("rm")
@click.option("--id", "id_arg", required=True, help="Profile id (full uuid4)")
def wan_profiles_rm_cmd(id_arg: str) -> None:
    """Remove a saved profile by id."""
    from meeting_scribe.server_support import settings_store

    if settings_store._delete_wan_profile(id_arg):
        click.echo(f"Removed profile {id_arg}")
    else:
        raise click.ClickException(f"no profile with id {id_arg!r}")


@wan_profiles_group.command("set-active")
@click.option("--id", "id_arg", required=True, help="Profile id (full uuid4)")
def wan_profiles_set_active_cmd(id_arg: str) -> None:
    """Mark a saved profile as the active one (does NOT bring it up)."""
    from meeting_scribe.server_support import settings_store

    settings_store._set_wan_active_profile_id(id_arg)
    click.echo(f"Active profile set to {id_arg}")
