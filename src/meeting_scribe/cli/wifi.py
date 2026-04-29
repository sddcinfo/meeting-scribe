"""wifi sub-group: up / down / status for the GB10 WiFi hotspot."""

from __future__ import annotations

import click

from meeting_scribe.cli import cli


@cli.group("wifi")
def wifi_group():
    """WiFi hotspot management (off / meeting / admin mode).

    \b
    The GB10 MT7925 radio supports one AP at a time. Two modes:
      meeting  — rotating SSID, captive portal, guest isolation
      admin    — fixed SSID, no portal, admin UI reachable over WiFi
    """


@wifi_group.command("up")
@click.option(
    "--mode",
    type=click.Choice(["meeting", "admin"]),
    default="meeting",
    show_default=True,
    help="meeting = rotating guest SSID + captive portal; "
    "admin = fixed SSID, no portal, admin UI over WiFi",
)
@click.option("--ssid", default=None, help="Override SSID")
@click.option("--password", default=None, help="Override password")
@click.option(
    "--band",
    default="a",
    show_default=True,
    type=click.Choice(["a", "bg"]),
    help="WiFi band (a=5GHz, bg=2.4GHz)",
)
@click.option("--channel", default=36, show_default=True, type=int, help="WiFi channel")
def wifi_up(mode: str, ssid: str | None, password: str | None, band: str, channel: int) -> None:
    """Bring up the WiFi AP in meeting or admin mode."""
    import asyncio as _asyncio

    from meeting_scribe.wifi import build_config
    from meeting_scribe.wifi import wifi_up as _wifi_up

    cfg = build_config(mode, ssid, password, band, channel)
    click.echo(click.style(f"==> Bringing up WiFi AP ({mode} mode)...", fg="cyan"))
    _asyncio.run(_wifi_up(cfg))
    click.echo()
    click.echo(click.style(f"WiFi AP is live! ({mode} mode)", fg="green", bold=True))
    click.echo(f"  Mode:      {mode}")
    click.echo(f"  SSID:      {cfg.ssid}")
    click.echo(f"  Password:  {cfg.password}")
    click.echo("  Security:  WPA3-SAE (PMF required)")
    click.echo(f"  Band:      {'5 GHz' if band == 'a' else '2.4 GHz'} (channel {channel})")
    if mode == "admin":
        click.echo(f"  Admin UI:  https://{cfg.ap_ip}:8080/")
        click.echo("  Portal:    disabled (admin mode)")
    else:
        click.echo(f"  Guest:     http://{cfg.ap_ip}/ (phones)")
        click.echo("  Admin:     https://<mgmt-ip>:8080/ (LAN only)")
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
