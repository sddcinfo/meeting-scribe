"""``/api/ws/view`` — read-only transcript broadcast WebSocket.

Used by second browsers or guest devices that want to watch the
live transcript without uploading mic audio. On connect the
handler replays the current meeting's journal so a late-joining
client catches up to the same view as everyone else. State-mutating
messages are rejected here; admin controls use admin-authenticated
routes/websockets.

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

_MUTATING_VIEW_MESSAGES = frozenset(
    {
        "set_language",
        "set_language_pair",
        "set_interpretation",
        "mute_room_speaker",
        "mute_web",
        "mute_bt_headsets",
    }
)


@router.websocket("/api/ws/view")
async def websocket_view(websocket: WebSocket) -> None:
    """View-only WebSocket - receives transcript events without sending audio.

    Used by:
      * second admin browsers / guest devices watching the live
        transcript without mic access (admin or guest cookies);
      * the GB10-local cage kiosk-runtime, which authenticates via
        the ``scribe_kiosk`` cookie. Kiosk connections are tracked in
        ``state.kiosk_ws_connections`` so the broadcast helper filters
        operator-only event types out of the kiosk fan-out.

    State-mutating JSON messages are rejected; the socket is read-only
    for every role. Inbound text frames from a kiosk-cookie socket are
    additionally dropped server-side (no language preference, no
    mutating message even attempted).
    """
    from meeting_scribe.terminal.auth import KIOSK_COOKIE_NAME

    is_kiosk = False
    kiosk_signer = getattr(state, "_kiosk_cookie_signer", None)
    kiosk_cookie = websocket.cookies.get(KIOSK_COOKIE_NAME)
    if kiosk_signer is not None and kiosk_signer.verify(kiosk_cookie):
        is_kiosk = True

    await websocket.accept()
    state.ws_connections.add(websocket)
    if is_kiosk:
        state.kiosk_ws_connections.add(websocket)
    state._client_prefs[websocket] = ClientSession()
    logger.info(
        "View-only WS connected (total=%d, role=%s)",
        len(state.ws_connections),
        "kiosk" if is_kiosk else "view",
    )

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
            # Read-only stream: kiosk sockets cannot send anything.
            if is_kiosk:
                continue
            try:
                msg = _json.loads(text)
                if isinstance(msg, dict) and msg.get("type") in _MUTATING_VIEW_MESSAGES:
                    await websocket.send_text(
                        _json.dumps(
                            {
                                "type": "error",
                                "code": 403,
                                "message": "admin websocket required for mutations",
                            }
                        )
                    )
                    await websocket.close(code=4403, reason="admin websocket required")
                    break
                if isinstance(msg, dict) and msg.get("type") == "set_language":
                    lang = msg.get("language", "")
                    if lang:
                        state._client_prefs[websocket].preferred_language = lang
                        logger.info("View WS set language preference: %s", lang)
            except Exception:
                pass  # Ignore non-JSON (pings, etc.)
    except WebSocketDisconnect:
        pass  # client closed the view-only socket — finally-block cleans up registrations
    except Exception:
        pass  # any other receive-loop error — drop this view; finally-block deregisters it
    finally:
        state.ws_connections.discard(websocket)
        state.kiosk_ws_connections.discard(websocket)
        state._client_prefs.pop(websocket, None)
        logger.info("View-only WS disconnected (remaining=%d)", len(state.ws_connections))
