"""``meeting-scribe audio`` — authenticated audio routing controls.

The CLI talks to the same admin HTTP API as the UI. It does not edit the
settings file directly, so route validation, live server-mic reconciliation,
and local TTS listener retargeting stay identical across CLI and browser.
"""

from __future__ import annotations

import json
import os
from typing import Any

import click
import httpx

from meeting_scribe.cli import cli
from meeting_scribe.setup_state import _mint_admin_password

DEFAULT_BASE_URL = "https://127.0.0.1"


def _redact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "mic_node": selection.get("mic_node") or "",
        "admin_sink_node": selection.get("admin_sink_node") or "",
        "room_sink_node": selection.get("room_sink_node") or "",
        "mic_active": bool(selection.get("mic_active", False)),
        "server_mic_active_live": bool(selection.get("server_mic_active_live", False)),
    }


class AdminApi:
    def __init__(self, *, base_url: str, password: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.client = httpx.Client(verify=False, timeout=10.0, follow_redirects=False)

    def __enter__(self) -> AdminApi:
        password = self.password
        if password is None:
            password = os.environ.get("SCRIBE_ADMIN_PASSWORD")
        if password is None and self.base_url in {DEFAULT_BASE_URL, "https://localhost"}:
            password = _mint_admin_password()
        if password is None:
            password = click.prompt("Admin password", hide_input=True)
        resp = self.client.post(f"{self.base_url}/api/admin/authorize", data={"password": password})
        if resp.status_code not in {200, 303}:
            raise click.ClickException(f"admin authentication failed (HTTP {resp.status_code})")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.client.close()

    def get(self, path: str) -> dict[str, Any]:
        resp = self.client.get(f"{self.base_url}{path}")
        if resp.status_code in {401, 403}:
            raise click.ClickException("admin authentication required")
        if not resp.is_success:
            raise click.ClickException(f"GET {path} failed (HTTP {resp.status_code}): {resp.text}")
        return resp.json()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self.client.post(f"{self.base_url}{path}", json=body)
        if resp.status_code in {401, 403}:
            raise click.ClickException("admin authentication required")
        if not resp.is_success:
            detail = resp.text
            try:
                detail = resp.json().get("error", detail)
            except Exception:
                pass  # non-JSON error body — fall back to the raw response text
            raise click.ClickException(f"POST {path} failed (HTTP {resp.status_code}): {detail}")
        return resp.json()


def _api(base_url: str, password: str | None) -> AdminApi:
    return AdminApi(base_url=base_url, password=password)


@cli.group("audio")
def audio_group() -> None:
    """Authenticated audio routing and interpretation controls."""


def _common_options(fn):
    fn = click.option(
        "--password",
        envvar="SCRIBE_ADMIN_PASSWORD",
        help="Admin password. Defaults to SCRIBE_ADMIN_PASSWORD; local GB10 falls back to deterministic setup password.",
    )(fn)
    fn = click.option(
        "--base-url",
        default=DEFAULT_BASE_URL,
        show_default=True,
        envvar="SCRIBE_ADMIN_URL",
        help="Meeting Scribe admin API base URL.",
    )(fn)
    return fn


@audio_group.command("devices")
@_common_options
def audio_devices_cmd(base_url: str, password: str | None) -> None:
    """List audio devices and the persisted selection as JSON."""
    with _api(base_url, password) as api:
        click.echo(json.dumps(api.get("/api/admin/audio/devices"), indent=2))


@audio_group.command("status")
@_common_options
def audio_status_cmd(base_url: str, password: str | None) -> None:
    """Print current route + interpretation state."""
    with _api(base_url, password) as api:
        route = api.get("/api/admin/audio/route")
        interpretation = api.get("/api/admin/audio/interpretation")
    click.echo(
        json.dumps(
            {
                "route": _redact_selection(route),
                "interpretation": interpretation,
            },
            indent=2,
        )
    )


@audio_group.command("route")
@click.option("--mic-node", help="PipeWire source node for server-side ASR capture.")
@click.option("--admin-sink-node", help="PipeWire sink node for private/admin TTS.")
@click.option("--room-sink-node", help="PipeWire sink node for in-room TTS.")
@click.option(
    "--use-mic/--no-use-mic", default=None, help="Enable/disable server-side mic capture."
)
@_common_options
def audio_route_cmd(
    base_url: str,
    password: str | None,
    mic_node: str | None,
    admin_sink_node: str | None,
    room_sink_node: str | None,
    use_mic: bool | None,
) -> None:
    """Patch the audio route through the same validation as the UI."""
    body: dict[str, Any] = {}
    if mic_node is not None:
        body["mic_node"] = mic_node
    if admin_sink_node is not None:
        body["admin_sink_node"] = admin_sink_node
    if room_sink_node is not None:
        body["room_sink_node"] = room_sink_node
    if use_mic is not None:
        body["mic_active"] = use_mic
    if not body:
        raise click.UsageError("provide at least one route option")
    with _api(base_url, password) as api:
        click.echo(json.dumps(api.post("/api/admin/audio/route", body), indent=2))


@audio_group.command("interpretation")
@click.option("--enabled/--disabled", default=None, help="Enable/disable interpretation TTS.")
@click.option("--admin-language", help="Single target language for admin/private TTS.")
@click.option(
    "--room-language", help="Target language for room TTS; use 'all' for bidirectional room output."
)
@click.option(
    "--pause-flush-ms", type=int, help="Silence before flushing an interpretation phrase."
)
@click.option("--idle-drain-ms", type=int, help="Idle drain timeout for interpretation buffer.")
@click.option(
    "--mute",
    type=click.Choice(
        [
            "mute_room_speaker",
            "unmute_room_speaker",
            "mute_web",
            "unmute_web",
            "mute_bt_headsets",
            "unmute_bt_headsets",
        ]
    ),
    help="Apply the same targeted mute/unmute action exposed in the live admin UI.",
)
@_common_options
def audio_interpretation_cmd(
    base_url: str,
    password: str | None,
    enabled: bool | None,
    admin_language: str | None,
    room_language: str | None,
    pause_flush_ms: int | None,
    idle_drain_ms: int | None,
    mute: str | None,
) -> None:
    """Patch interpretation settings through the admin API."""
    body: dict[str, Any] = {}
    if enabled is not None:
        body["enabled"] = enabled
    if admin_language is not None:
        body["admin_tts_language"] = admin_language
    if room_language is not None:
        body["room_tts_language"] = room_language
    if pause_flush_ms is not None:
        body["pause_flush_ms"] = pause_flush_ms
    if idle_drain_ms is not None:
        body["idle_drain_ms"] = idle_drain_ms
    if mute is not None:
        body["mute"] = mute
    if not body:
        raise click.UsageError("provide at least one interpretation option")
    with _api(base_url, password) as api:
        click.echo(json.dumps(api.post("/api/admin/audio/interpretation", body), indent=2))
