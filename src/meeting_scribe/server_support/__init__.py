"""Shared, non-state, non-route helpers extracted from ``server.py``.

This package collects helper functions and small dataclasses that were
historically defined at module scope inside ``server.py`` and are
needed by both the server entrypoint and the to-be-extracted route /
WebSocket / hotspot modules. Moving them here breaks the
"route module imports server.py" cycle that the upcoming route
extractions would otherwise create.

Each submodule is named after the cohesive concern it owns. Submodules
are imported lazily (per-callsite) so adding new helper modules here
doesn't snowball server.py's already-heavy import graph.
"""
