"""Meeting Scribe CLI — manage server lifecycle and development tools.

Usage:
    meeting-scribe start     # Start the server
    meeting-scribe stop      # Stop gracefully
    meeting-scribe restart   # Stop + start
    meeting-scribe status    # Check server + backend status
    meeting-scribe logs      # Tail server logs

The ``cli`` group is defined here; topic modules under
``meeting_scribe.cli.<topic>`` register their commands by importing
this module and decorating with ``@cli.command()`` or
``@<group>.command()``. The ``main`` entry point is what
``[project.scripts]`` in pyproject.toml resolves to.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import click


def _resolved_version() -> str:
    """Read the installed package version so ``--version`` reflects
    the actual deploy. Hardcoded literals drift across releases —
    the previous ``1.5.0`` literal stayed in place through the v2.0
    bump and showed the wrong number on every customer GB10."""
    try:
        return _pkg_version("meeting-scribe")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@click.group()
@click.version_option(version=_resolved_version())
def cli() -> None:
    """Meeting Scribe — real-time bilingual transcription."""


# Importing each topic module triggers its ``@cli.command()`` /
# ``@<group>.command()`` decorators, registering every command on the
# root ``cli`` group above. Order matters when one module's aliases
# reference another's command (e.g. ``up``/``down`` proxy the gb10
# group), so gb10 is imported before setup.
from meeting_scribe.cli import (
    audio,  # noqa: F401
    bench,  # noqa: F401
    benchmark,  # noqa: F401
    bt,  # noqa: F401
    config,  # noqa: F401
    demo_smoke,  # noqa: F401
    doctor,  # noqa: F401
    finalize,  # noqa: F401
    gb10,  # noqa: F401
    hf_probe,  # noqa: F401
    install_service,  # noqa: F401
    kiosk,  # noqa: F401
    library,  # noqa: F401
    lifecycle,  # noqa: F401
    meetings,  # noqa: F401
    precommit,  # noqa: F401
    queue,  # noqa: F401
    setup,  # noqa: F401
    speakerphone,  # noqa: F401
    terminal,  # noqa: F401
    trust,  # noqa: F401
    validate,  # noqa: F401
    versions,  # noqa: F401
    wifi,  # noqa: F401
)


def main() -> None:
    """Entry point — pyproject.toml resolves ``meeting-scribe`` to this."""
    cli()


if __name__ == "__main__":
    main()
