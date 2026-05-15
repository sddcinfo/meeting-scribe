"""In-browser terminal panel for the meeting-scribe popout.

Exposes a tmux-first PTY over a WebSocket with HMAC-ticket auth, byte-level
flow control, and a vendored xterm.js client.

Submodules are imported lazily — server.py pulls in what it needs by name,
and tests import leaf modules directly. Keeping this file empty of side
effects avoids circular-import hazards during ``server.py`` bootstrap.
"""

from __future__ import annotations
