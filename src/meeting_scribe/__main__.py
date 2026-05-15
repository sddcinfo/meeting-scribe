"""Entry point for ``python -m meeting_scribe``.

The systemd unit invokes the venv interpreter directly via this module
instead of going through ``.venv/bin/meeting-scribe`` (the generated
console-script wrapper) because that wrapper has been observed to land
as a bare ``#!/bin/sh\\n`` after ``pip install -e .`` races, which
breaks ``Type=notify`` startup with ``Result: protocol``.

Going through ``-m meeting_scribe`` invokes Python directly on the
package, sidestepping the wrapper entirely. Operator shell use of
``meeting-scribe …`` still flows through the wrapper for convenience;
service start does not.

See ``docs/known-issues/wrapper-truncation.md`` for the wrapper-bug
write-up and current diagnostics state.
"""

from __future__ import annotations

from meeting_scribe.cli import main

if __name__ == "__main__":
    main()
