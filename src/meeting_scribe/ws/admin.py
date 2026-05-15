"""``/api/ws/admin`` — admin-only state mutation WebSocket."""

from __future__ import annotations

import json as _json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from meeting_scribe.audio.audio_routing import get_routing_settings
from meeting_scribe.languages import is_supported
from meeting_scribe.runtime import state
from meeting_scribe.server_support.admin_guard import require_admin_ws
from meeting_scribe.server_support.broadcast import _broadcast_json
from meeting_scribe.server_support.settings_store import (
    _effective_interpretation_enabled,
    _load_settings_override,
    _save_settings_override,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def _broadcast_interpretation_status(payload: dict) -> None:
    await _broadcast_json({"type": "interpretation_status", **payload})


async def _set_language_pair(pair: list[str]) -> dict:
    parts = [str(p).strip().lower() for p in pair if str(p).strip()]
    if len(parts) not in (1, 2) or len(set(parts)) != len(parts):
        return {"ok": False, "error": "language_pair must contain 1 or 2 distinct codes"}
    if not all(is_supported(p) for p in parts):
        return {"ok": False, "error": "unsupported language in language_pair"}
    if state.current_meeting is None:
        return {"ok": False, "error": "no active meeting"}

    state.current_meeting.language_pair = parts
    try:
        state.storage._write_meta(state.current_meeting)
    except Exception:
        logger.exception("admin ws set_language_pair: meta write failed")
    if state.translation_queue:
        state.translation_queue.set_languages(parts)
    if state.asr_backend:
        state.asr_backend.set_languages(parts)
    if state.interpretation_buffer is not None:
        await state.interpretation_buffer.filter_for_language_pair(parts)
    return {"ok": True, "language_pair": parts}


@router.websocket("/api/ws/admin")
async def websocket_admin(websocket: WebSocket) -> None:
    if not await require_admin_ws(websocket):
        return
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            try:
                msg = _json.loads(text)
            except Exception:
                await websocket.send_text(
                    _json.dumps({"type": "error", "code": 400, "message": "invalid JSON"})
                )
                continue
            if not isinstance(msg, dict):
                await websocket.send_text(
                    _json.dumps({"type": "error", "code": 400, "message": "JSON object expected"})
                )
                continue

            msg_type = msg.get("type")
            if msg_type == "set_language_pair":
                pair = msg.get("language_pair", [])
                if isinstance(pair, str):
                    pair = [p.strip() for p in pair.split(",")]
                if not isinstance(pair, list):
                    result = {"ok": False, "error": "language_pair must be a list or comma string"}
                else:
                    result = await _set_language_pair(pair)
                await websocket.send_text(_json.dumps({"type": "language_pair_ack", **result}))
                continue

            if msg_type == "set_interpretation":
                enabled = msg.get("enabled")
                if not isinstance(enabled, bool):
                    await websocket.send_text(
                        _json.dumps(
                            {
                                "type": "error",
                                "code": 400,
                                "message": "enabled must be a boolean",
                            }
                        )
                    )
                    continue
                from meeting_scribe.routes.admin_audio import (
                    _apply_interpretation_live,
                    _interpretation_payload,
                )

                _save_settings_override({"interpretation_enabled": enabled})
                await _apply_interpretation_live(enabled)
                payload = _interpretation_payload()
                await websocket.send_text(_json.dumps({"type": "interpretation_ack", **payload}))
                await _broadcast_interpretation_status(payload)
                continue

            if msg_type in (
                "mute_room_speaker",
                "mute_web",
                "mute_bt_headsets",
                "unmute_room_speaker",
                "unmute_web",
                "unmute_bt_headsets",
            ):
                from meeting_scribe.routes.admin_audio import (
                    _apply_interpretation_mute,
                    _interpretation_payload,
                )

                await _apply_interpretation_mute(msg_type)
                payload = _interpretation_payload()
                await websocket.send_text(_json.dumps({"type": "interpretation_ack", **payload}))
                await _broadcast_interpretation_status(payload)
                continue

            if msg_type == "status":
                from meeting_scribe.routes.admin_audio import _interpretation_payload

                routing = get_routing_settings(_load_settings_override())
                await websocket.send_text(
                    _json.dumps(
                        {
                            "type": "admin_status",
                            "interpretation": _interpretation_payload(),
                            "audio_route": routing,
                            "interpretation_enabled": _effective_interpretation_enabled(),
                        }
                    )
                )
                continue

            await websocket.send_text(
                _json.dumps({"type": "error", "code": 400, "message": "unknown admin message"})
            )
    except WebSocketDisconnect:
        pass
