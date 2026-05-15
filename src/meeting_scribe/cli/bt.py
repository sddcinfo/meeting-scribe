"""``meeting-scribe bt`` subcommands — pair / connect / disconnect /
status / mic / forget / smoke.

Mirror of cli/wifi.py. The CLI is the source-of-truth for pairing flows
that need passkey confirmation; the admin web UI BT card uses the same
underlying primitives in :mod:`meeting_scribe.bt` for non-passkey
day-to-day ops.
"""

from __future__ import annotations

import asyncio
import json

import click

from meeting_scribe import bt as bt_mod
from meeting_scribe.cli import cli


@cli.group("bt")
def bt_group() -> None:
    """Bluetooth audio bridge controls."""


@bt_group.command("status")
def bt_status_cmd() -> None:
    """Print adapter + paired-device snapshot as JSON."""
    snap = asyncio.run(bt_mod.bt_status_sync())
    click.echo(json.dumps(snap, indent=2))


@bt_group.command("connect")
@click.argument("mac")
def bt_connect_cmd(mac: str) -> None:
    """Connect to a paired device by MAC."""
    try:
        asyncio.run(bt_mod.bt_connect(mac))
    except bt_mod.BluetoothError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"connected: {mac}")


@bt_group.command("disconnect")
@click.argument("mac", required=False)
def bt_disconnect_cmd(mac: str | None) -> None:
    """Disconnect the named device (or whatever's connected)."""
    try:
        asyncio.run(bt_mod.bt_disconnect(mac, user_initiated=True))
    except bt_mod.BluetoothError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("disconnected")


@bt_group.command("pair")
@click.argument("mac")
def bt_pair_cmd(mac: str) -> None:
    """Pair + trust + connect a device by MAC."""
    try:
        asyncio.run(bt_mod.bt_pair(mac))
    except bt_mod.BluetoothError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"paired: {mac}")


@bt_group.command("forget")
@click.argument("mac")
def bt_forget_cmd(mac: str) -> None:
    """Remove a paired device from BlueZ."""
    try:
        asyncio.run(bt_mod.bt_forget(mac))
    except bt_mod.BluetoothError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"forgotten: {mac}")


@bt_group.command("profile")
@click.argument("mac")
@click.argument("want", type=click.Choice(["a2dp", "hfp"]))
def bt_profile_cmd(mac: str, want: str) -> None:
    """Print the best-available profile name for the requested capability."""
    try:
        chosen = asyncio.run(bt_mod.bt_choose_profile(mac, want=want))  # type: ignore[arg-type]
    except bt_mod.BluetoothError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(chosen)


@bt_group.command("nodes")
@click.argument("mac")
def bt_nodes_cmd(mac: str) -> None:
    """Resolve the active source + sink node names for ``mac``."""
    src, sink = asyncio.run(bt_mod.bt_resolve_nodes(mac))
    click.echo(json.dumps({"source": src, "sink": sink}, indent=2))
