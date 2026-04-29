"""Session-scoped client preferences and draft room layouts.

``ClientSession`` carries the per-WebSocket preferences negotiated
when an interpretation listener connects (preferred language, voice
mode, audio wire format, etc.). The draft-layout helpers cache an
in-progress room layout per session so a user can edit and re-edit
without committing to disk.

Moved out of ``server.py`` so route modules and WebSocket handlers
can import ``ClientSession`` without dragging the full server graph
back in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import fastapi

from meeting_scribe.models import RoomLayout
from meeting_scribe.runtime import state

if TYPE_CHECKING:
    from meeting_scribe.backends.mse_encoder import Fmp4AacEncoder


_DEFAULT_SESSION = "default"


def _get_session_id(request: fastapi.Request) -> str:
    """Get or create a session ID from cookie."""
    return request.cookies.get("scribe_session", _DEFAULT_SESSION)


def _get_draft_layout(session_id: str) -> RoomLayout:
    """Get the draft layout for a session, creating if needed."""
    if session_id not in state._draft_layouts:
        state._draft_layouts[session_id] = RoomLayout()
    state._draft_layout_access[session_id] = time.monotonic()
    return state._draft_layouts[session_id]


def _set_draft_layout(session_id: str, layout: RoomLayout) -> None:
    """Set the draft layout for a session."""
    state._draft_layouts[session_id] = layout
    state._draft_layout_access[session_id] = time.monotonic()


# Per-client session preferences (language for interpretation, audio-out)
@dataclass
class ClientSession:
    preferred_language: str = ""
    send_audio: bool = False
    interpretation_mode: str = (
        "translation"  # "translation" (TTS only) or "full" (passthrough + TTS)
    )
    # "studio" — Qwen3-TTS named speaker per target language (default, fast)
    # "cloned" — clone each participant's voice from live meeting audio
    voice_mode: str = "studio"
    # Wire format negotiated via `set_format` on the audio-out WS. `None`
    # means the negotiation grace period is still running — audio for
    # this listener is buffered into `pending_audio` instead of being
    # sent immediately. After ``_AUDIO_FORMAT_GRACE_S`` the default is
    # "wav-pcm" for backward compatibility with cached legacy clients.
    audio_format: str | None = None
    # Lazy: stays None until the first audio delivery after a listener
    # negotiates "mse-fmp4-aac". Holds the per-connection PyAV encoder.
    mse_encoder: Fmp4AacEncoder | None = None
    # PCM delivery attempts that arrived while ``audio_format is None``.
    # Each item is ``(pcm: np.ndarray, source_sample_rate: int)``. Capped
    # to about 1 second of audio to bound memory for stuck handshakes.
    pending_audio: list = field(default_factory=list)
    # Monotonic deadline after which ``audio_format=None`` is promoted to
    # "wav-pcm" on the next delivery attempt. Set when the WS accepts.
    grace_deadline: float = 0.0
    # MSE stuck-health bookkeeping. Never drives encoder recreation —
    # only logs a diagnostic WARNING via the stuck-detection path.
    last_fragment_at: float = 0.0
    bytes_in_since_last_emit: int = 0
