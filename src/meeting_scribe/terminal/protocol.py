"""Wire protocol between the browser xterm.js client and the backend PTY.

Binary + JSON hybrid. Keystrokes and PTY output travel on the binary
channel (one-byte discriminator) to avoid JSON-wrapping every character;
everything else (attach/resize/ack/ping/status/error/bye) is JSON text.

Server → client binary:
    b'O' + <utf-8 output bytes>       PTY stdout/stderr

Client → server binary:
    b'I' + <raw stdin bytes>          keystrokes / paste chunks

Inbound binary frames are capped at :data:`INBOUND_FRAME_MAX` to bound
memory; violations raise :class:`ProtocolError` and the caller closes
the WebSocket with code 1009.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final

# ── Binary prefixes ───────────────────────────────────────────────
PREFIX_OUTPUT: Final[bytes] = b"O"
PREFIX_INPUT: Final[bytes] = b"I"

# ── Size limits (bytes) ───────────────────────────────────────────
INBOUND_FRAME_MAX: Final[int] = 256 * 1024
MAX_OUT_FRAME: Final[int] = 32 * 1024

# ── Flow-control watermarks (bytes) ───────────────────────────────
HIGH_WATER: Final[int] = 128 * 1024
LOW_WATER: Final[int] = 16 * 1024
OUT_BUFFER_HARD_CAP: Final[int] = HIGH_WATER * 4  # 512 KiB

# ── Inbound stdin admission budget ────────────────────────────────
STDIN_BUDGET: Final[int] = 1024 * 1024  # 1 MiB

# ── Geometry limits ───────────────────────────────────────────────
MIN_COLS: Final[int] = 20
MAX_COLS: Final[int] = 500
MIN_ROWS: Final[int] = 5
MAX_ROWS: Final[int] = 200

# ── Validation regexes ────────────────────────────────────────────
TMUX_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_\-]{1,32}$")
TICKET_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}\.[0-9a-f]{64}$")


class ProtocolError(ValueError):
    """Raised when a client frame violates the wire protocol."""


# ── Client → server ───────────────────────────────────────────────


@dataclass(frozen=True)
class AttachMessage:
    """First text frame the client sends after the WS handshake."""

    ticket: str
    tmux_session: str
    cols: int
    rows: int
    term: str = "xterm-256color"

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> AttachMessage:
        if payload.get("type") != "attach":
            raise ProtocolError(f"expected type=attach, got {payload.get('type')!r}")
        ticket = payload.get("ticket")
        if not isinstance(ticket, str) or not TICKET_RE.match(ticket):
            raise ProtocolError("invalid ticket format")
        session = payload.get("tmux_session", "scribe")
        if not isinstance(session, str) or not TMUX_NAME_RE.match(session):
            raise ProtocolError("invalid tmux_session name")
        cols = _coerce_int(payload.get("cols"), MIN_COLS, MAX_COLS, default=120)
        rows = _coerce_int(payload.get("rows"), MIN_ROWS, MAX_ROWS, default=40)
        term = payload.get("term", "xterm-256color")
        if not isinstance(term, str) or len(term) > 64:
            raise ProtocolError("invalid term")
        return cls(ticket=ticket, tmux_session=session, cols=cols, rows=rows, term=term)


@dataclass(frozen=True)
class ResizeMessage:
    cols: int
    rows: int

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> ResizeMessage:
        cols = _coerce_int(payload.get("cols"), MIN_COLS, MAX_COLS)
        rows = _coerce_int(payload.get("rows"), MIN_ROWS, MAX_ROWS)
        return cls(cols=cols, rows=rows)


@dataclass(frozen=True)
class AckMessage:
    """Monotonic cumulative-byte acknowledgement from the client."""

    bytes_total: int

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> AckMessage:
        raw = payload.get("bytes_total")
        # bool is a subclass of int — reject explicitly
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ProtocolError("bytes_total must be non-negative int")
        return cls(bytes_total=raw)


def parse_client_text(raw: str) -> AttachMessage | ResizeMessage | AckMessage | dict[str, Any]:
    """Parse a client text frame.

    Returns a typed message dataclass for known types, or the raw dict for
    lightweight messages like ``{"type": "ping"}``.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON: {e.msg}") from None
    if not isinstance(payload, dict):
        raise ProtocolError("text frame must be a JSON object")
    t = payload.get("type")
    if t == "attach":
        return AttachMessage.parse(payload)
    if t == "resize":
        return ResizeMessage.parse(payload)
    if t == "ack":
        return AckMessage.parse(payload)
    if t == "ping":
        return payload
    raise ProtocolError(f"unknown type: {t!r}")


# ── Server → client encoders ──────────────────────────────────────


def encode_output(chunk: bytes) -> bytes:
    """Prefix a PTY output chunk for binary WS send."""
    return PREFIX_OUTPUT + chunk


def encode_attached(*, cols: int, rows: int, tmux_session: str, pid: int) -> str:
    return json.dumps(
        {
            "type": "attached",
            "cols": cols,
            "rows": rows,
            "tmux_session": tmux_session,
            "pid": pid,
        },
        separators=(",", ":"),
    )


def encode_status(
    *,
    bytes_in: int,
    bytes_sent_total: int,
    bytes_acked_total: int,
    paused: bool,
    cols: int,
    rows: int,
) -> str:
    return json.dumps(
        {
            "type": "status",
            "bytes_in": bytes_in,
            "bytes_sent_total": bytes_sent_total,
            "bytes_acked_total": bytes_acked_total,
            "paused": paused,
            "cols": cols,
            "rows": rows,
        },
        separators=(",", ":"),
    )


def encode_error(code: str, message: str | None = None) -> str:
    obj: dict[str, str] = {"type": "error", "code": code}
    if message:
        obj["message"] = message
    return json.dumps(obj, separators=(",", ":"))


def encode_bye(reason: str, detail: str | None = None) -> str:
    obj: dict[str, str] = {"type": "bye", "reason": reason}
    if detail:
        obj["detail"] = detail
    return json.dumps(obj, separators=(",", ":"))


def encode_pong() -> str:
    return '{"type":"pong"}'


# ── Inbound binary ────────────────────────────────────────────────


def extract_stdin(frame: bytes) -> bytes:
    """Validate and strip the prefix from a client binary frame.

    Raises :class:`ProtocolError` on bad prefix or oversize.
    """
    if len(frame) > INBOUND_FRAME_MAX:
        raise ProtocolError(f"inbound frame too large: {len(frame)} > {INBOUND_FRAME_MAX}")
    if not frame:
        raise ProtocolError("empty binary frame")
    if frame[:1] != PREFIX_INPUT:
        raise ProtocolError(f"unknown binary prefix: {frame[:1]!r}")
    return bytes(frame[1:])


# ── Helpers ───────────────────────────────────────────────────────


def _coerce_int(value: Any, lo: int, hi: int, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"expected int in [{lo},{hi}], got {type(value).__name__}")
    return max(lo, min(hi, value))
