"""WebSocket route modules.

Three WS endpoints, each in its own module so the audio-input
hot-path stays isolated from the view-broadcast and audio-output
state machines:

* ``audio_input`` (``/api/ws``) — admin-only audio upstream + JSON
  control messages. Receives binary 16 kHz PCM, forwards to ASR
  and diarization.
* ``view_broadcast`` (``/api/ws/view``) — read-only transcript
  stream. Replays the meeting journal to a late-joining client
  then receives only language-preference updates.
* ``audio_output`` (``/api/ws/audio-out``) — guest-scope listener
  endpoint. Negotiates an audio format on connect, then receives
  synthesized TTS audio plus optional pass-through original audio.

Helpers tightly bound to one endpoint live in that endpoint's
module; helpers shared across the audio delivery broadcast pipeline
(``_deliver_audio_to_listener``, ``_send_passthrough_audio`` etc.)
stay in ``server.py`` for now and are lazy-imported.
"""
