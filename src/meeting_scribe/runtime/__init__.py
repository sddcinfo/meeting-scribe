"""Runtime container for cross-module mutable state.

See :mod:`meeting_scribe.runtime.state` for the canonical home of
backends, queues, stores, and singletons that today live in
``server.py`` and need to be reachable from extracted route, WebSocket,
and hotspot modules without importing the server module itself.
"""
