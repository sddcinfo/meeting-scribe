"""config sub-group: get/set/unset/reload runtime-config knobs.

Hot-reloadable settings (translate_url, slide_translate_url,
slide_use_json_schema) persist to ``runtime-config.json`` and SIGHUP
the running server so it picks them up without a restart.
"""

from __future__ import annotations

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import _sighup_running_server


@cli.group("config")
def config_group() -> None:
    """Hot-reloadable runtime-config knobs (translate_url, slide_translate_url, ...).

    \b
    These three settings can be flipped without restarting the server:
      translate_url          — live-translation vLLM endpoint (Phase 7 rollback)
      slide_translate_url    — slide-pipeline vLLM endpoint
      slide_use_json_schema  — Phase 4b response_format flag

    `config set` persists to $XDG_DATA_HOME/meeting-scribe/runtime-config.json
    and sends SIGHUP to the running server so it re-reads on the next
    translation request.  Every other setting still lives on the static
    ServerConfig dataclass loaded from env.
    """


@config_group.command("get")
@click.argument("key", required=False)
def config_get(key: str | None) -> None:
    """Show one knob or all of them."""
    from meeting_scribe import runtime_config

    snapshot = runtime_config.instance().as_dict()
    if key is None:
        if not snapshot:
            click.echo("(no runtime overrides; all knobs fall back to ServerConfig)")
            return
        for k, v in snapshot.items():
            click.echo(f"{k} = {v!r}")
        return
    if key not in snapshot:
        click.echo(f"{key}: (unset)")
        return
    click.echo(f"{key} = {snapshot[key]!r}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.option(
    "--no-reload",
    is_flag=True,
    help="Persist to disk but do NOT send SIGHUP; the server will pick up on next restart.",
)
def config_set(key: str, value: str, no_reload: bool) -> None:
    """Persist KEY=VALUE and (by default) SIGHUP the running server."""
    from meeting_scribe import runtime_config

    # Coerce booleans for the known boolean knobs so CLI users don't
    # have to quote "true"/"false" and then worry about string vs bool.
    coerced: object = value
    if key == "slide_use_json_schema":
        if value.lower() in {"true", "1", "yes", "on"}:
            coerced = True
        elif value.lower() in {"false", "0", "no", "off"}:
            coerced = False
        else:
            raise click.BadParameter(
                f"slide_use_json_schema expects a boolean (true/false), got {value!r}"
            )

    try:
        runtime_config.instance().set(key, coerced)
    except KeyError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{key} = {coerced!r} (persisted to {runtime_config.instance().path})")

    if no_reload:
        return
    _sighup_running_server()


@config_group.command("unset")
@click.argument("key")
@click.option("--no-reload", is_flag=True, help="Don't SIGHUP.")
def config_unset(key: str, no_reload: bool) -> None:
    """Clear a knob so the next read falls back to ServerConfig."""
    from meeting_scribe import runtime_config

    try:
        runtime_config.instance().unset(key)
    except KeyError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{key} cleared")
    if not no_reload:
        _sighup_running_server()


@config_group.command("reload")
def config_reload() -> None:
    """SIGHUP the running server to re-read runtime-config.json."""
    _sighup_running_server()
