"""HTTP route modules — APIRouter-based extraction of ``server.py``.

Each module declares a ``router`` (``APIRouter`` instance) plus the
handler functions, then ``server.py`` calls
``app.include_router(router)`` once per module during app
construction. Route modules import shared state from
``meeting_scribe.runtime.state`` and shared helpers from
``meeting_scribe.server_support``; they MUST NOT import from
``meeting_scribe.server`` (the parity test would catch the cycle, but
the convention keeps imports flat).
"""
