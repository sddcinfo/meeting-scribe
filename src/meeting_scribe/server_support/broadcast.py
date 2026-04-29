"""WebSocket broadcast helpers — send to every connected client.

Two flavours, both fan out across ``state.ws_connections`` and prune
dropped sockets so dead references don't accumulate:

* ``_broadcast_json`` — arbitrary control messages (room layout
  edits, speaker rename, drift warnings, dev-reset) serialized via
  ``json.dumps``.
* ``_broadcast`` — transcript events serialized via Pydantic's
  ``model_dump_json``. Reaches admin, view-only popout, and hotspot
  guests since they all share the same connection set; frontends
  look up the block by ``segment_id`` and update in place.

Without the prune-on-fail loop, a tab that closed without
unregistering would stay in ``state.ws_connections`` as a dead
reference and slow every subsequent broadcast.
"""

from __future__ import annotations

import json as _json
import logging
from typing import TYPE_CHECKING

from meeting_scribe.runtime import state

if TYPE_CHECKING:
    from fastapi import WebSocket

    from meeting_scribe.models import TranscriptEvent

logger = logging.getLogger(__name__)


async def _broadcast_json(data: dict) -> None:
    """Send arbitrary JSON to all connected WebSocket clients.

    Dev-mode: validates ``data["type"]`` against ``WS_EVENT_TYPES`` and
    logs a warning if the value is unregistered. With env var
    ``WS_EVENT_TYPES_STRICT=1`` the warning becomes a ``ValueError``.

    The validation lives in ``ws/event_types.py``. Adding a new type?
    Update the enum AND ``tests/contracts/ws_event_samples/<type>.json``
    in the same change so the JS handler-coverage test can verify the
    client cascades have a named handler for it.
    """
    _validate_event_type(data)
    text = _json.dumps(data)
    dead: list[WebSocket] = []
    for ws in list(state.ws_connections):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_connections.discard(ws)


def _validate_event_type(data: dict) -> None:
    """Warn (or raise, in strict mode) when ``data["type"]`` is unknown.

    Inlined into ``_broadcast_json`` to enforce the registry at every
    server-side broadcast site without touching call sites. Pure
    string-comparison; no allocations on the happy path.
    """
    import os

    from meeting_scribe.ws.event_types import is_known_type

    type_value = data.get("type") if isinstance(data, dict) else None
    if type_value is None:
        return  # Untyped payloads (TranscriptEvent dict variants) are allowed.
    if is_known_type(type_value):
        return
    msg = (
        f"_broadcast_json: unregistered event type {type_value!r}. "
        f"Register in meeting_scribe/ws/event_types.py and add a sample "
        f"at tests/contracts/ws_event_samples/{type_value}.json."
    )
    if os.environ.get("WS_EVENT_TYPES_STRICT") == "1":
        raise ValueError(msg)
    logger.warning(msg)


async def _broadcast(event: TranscriptEvent) -> None:
    """Send transcript event to all connected WebSocket clients.

    Reaches admin audio WS, view-only WS (popout + guest), and hotspot
    guests since they all share the state.ws_connections set. Frontends look up
    the block by segment_id and update in place.
    """
    data = event.model_dump_json()
    dead: list[WebSocket] = []
    for ws in list(state.ws_connections):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_connections.discard(ws)
