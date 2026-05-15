"""Locking test for the wrapper-free systemd unit (Phase 1.4).

The ``.venv/bin/meeting-scribe`` console-script wrapper has been
observed to land as a bare ``#!/bin/sh\\n`` after ``pip install -e .``
races. When the systemd unit invoked that wrapper, the truncated
shell script exited 0 immediately without sending ``READY=1``, so
``Type=notify`` startup failed with ``Result: protocol``.

The fix renders every ``Exec*`` directive using
``<venv>/bin/python3 -m meeting_scribe …`` instead of
``<venv>/bin/meeting-scribe …``. This test locks that contract.
"""

from __future__ import annotations

import re

from meeting_scribe.cli.install_service import _render_unit


def test_no_console_script_wrapper_in_any_exec_directive() -> None:
    """Every ``Exec*`` directive must use ``python3 -m meeting_scribe``,
    never the ``bin/meeting-scribe`` wrapper. A regression here puts
    the box back in the truncated-wrapper failure mode."""
    unit = _render_unit()
    exec_lines = [
        line
        for line in unit.splitlines()
        if re.match(
            r"^Exec(Start|StartPre|StartPost|Stop|StopPre|StopPost|Condition|Reload)\b", line
        )
    ]
    assert exec_lines, "rendered unit has no Exec* directives — template regression"
    for line in exec_lines:
        assert "/bin/meeting-scribe" not in line, (
            f"ExecStart-family directive invokes the truncatable wrapper:\n  {line}\n"
            "Use 'python3 -m meeting_scribe ...' instead."
        )
        assert "python3 -m meeting_scribe" in line, (
            f"ExecStart-family directive does not use 'python3 -m meeting_scribe':\n  {line}"
        )


def test_exec_directives_present() -> None:
    """The full preflight chain (precondition / boot / shutdown) plus
    ExecStart must all be rendered. A future template simplification
    that drops the gates would silently disable boot-time validation
    on the field devices."""
    unit = _render_unit()
    expected_directives = [
        "ExecCondition=",
        "ExecStartPre=",  # gb10 up --offline
        "ExecStart=",
        "ExecStopPost=",
    ]
    for needle in expected_directives:
        assert needle in unit, f"unit template missing {needle!r}: regression of preflight chain"
    # Two ExecStartPre lines specifically (gb10 + preflight boot).
    assert unit.count("ExecStartPre=") >= 2, "expected at least two ExecStartPre directives"


def test_type_notify_preserved() -> None:
    """``Type=notify`` + ``NotifyAccess=main`` are how systemd knows
    when the lifespan finished. Rendering a unit without them would
    let systemd mark a still-warming-up service as ``active`` and
    accept early traffic."""
    unit = _render_unit()
    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
