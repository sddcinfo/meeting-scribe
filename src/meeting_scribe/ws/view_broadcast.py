"""``/api/ws/view`` — read-only transcript broadcast WebSocket.

Used by second browsers or guest devices that want to watch the
live transcript without uploading mic audio. On connect the
handler replays the current meeting's journal so a late-joining
client catches up to the same view as everyone else, then loops
on incoming text messages purely to receive language-preference
updates (``{"type": "set_language", "language": "en"}``).

Pings and other non-JSON text are silently ignored — they're a
common WebSocket library habit and shouldn't pollute the log.
"""

from __future__ import annotations

import json as _json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from meeting_scribe.runtime import state
from meeting_scribe.server_support.sessions import ClientSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/api/ws/view")
async def websocket_view(websocket: WebSocket) -> None:
    """View-only WebSocket — receives transcript events without sending audio.

    Used by second browsers or client devices that want to watch the
    live transcript without mic access.
    Accepts JSON text messages for language preference:
        {"type": "set_language", "language": "en"}
    """
    await websocket.accept()
    state.ws_connections.add(websocket)
    state._client_prefs[websocket] = ClientSession()
    logger.info("View-only WS connected (total=%d)", len(state.ws_connections))

    # Replay current meeting's journal so late-joining clients catch up
    if state.current_meeting:
        try:
            lines = state.storage.read_journal_raw(state.current_meeting.meeting_id, max_lines=500)
            if lines:
                total = len(lines)
                for line in lines:
                    if line.strip():
                        await websocket.send_text(line)
                logger.info("Replayed %d journal events to view WS", total)
        except Exception as e:
            logger.warning("Journal replay failed: %s", e)

    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = _json.loads(text)
                if isinstance(msg, dict) and msg.get("type") == "set_language":
                    lang = msg.get("language", "")
                    if lang:
                        state._client_prefs[websocket].preferred_language = lang
                        logger.info("View WS set language preference: %s", lang)
            except Exception:
                pass  # Ignore non-JSON (pings, etc.)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        state.ws_connections.discard(websocket)
        state._client_prefs.pop(websocket, None)
        logger.info("View-only WS disconnected (remaining=%d)", len(state.ws_connections))
