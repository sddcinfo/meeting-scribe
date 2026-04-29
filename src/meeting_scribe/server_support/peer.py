"""WebSocket peer-string helper.

One-liner used everywhere for log messages — pulled out so the
audio-output module can import it without dragging in the
WebSocket-related state from server.py.
"""

from __future__ import annotations

from typing import Any


def _peer_str(ws: Any) -> str:
    """Best-effort ``host:port`` string for a WebSocket client, for logs."""
    try:
        return f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    except Exception:
        return "?"
