"""Parse/encode tests for the terminal WebSocket wire protocol."""

from __future__ import annotations

import json

import pytest

from meeting_scribe.terminal import protocol as p

_GOOD_TICKET = "a" * 64 + "." + "b" * 64


def test_attach_roundtrip() -> None:
    msg = p.parse_client_text(
        json.dumps(
            {
                "type": "attach",
                "ticket": _GOOD_TICKET,
                "tmux_session": "scribe-demo",
                "cols": 120,
                "rows": 40,
            }
        )
    )
    assert isinstance(msg, p.AttachMessage)
    assert msg.ticket == _GOOD_TICKET
    assert msg.tmux_session == "scribe-demo"
    assert msg.cols == 120 and msg.rows == 40
    assert msg.term == "xterm-256color"


def test_attach_clamps_geometry() -> None:
    msg = p.parse_client_text(
        json.dumps(
            {
                "type": "attach",
                "ticket": _GOOD_TICKET,
                "tmux_session": "scribe",
                "cols": 10,  # < MIN_COLS
                "rows": 9999,  # > MAX_ROWS
            }
        )
    )
    assert isinstance(msg, p.AttachMessage)
    assert msg.cols == p.MIN_COLS
    assert msg.rows == p.MAX_ROWS


@pytest.mark.parametrize(
    "bad_session",
    ["", "has spaces", "semi;colon", "back`tick", "$expand", "../traversal", "a" * 33],
)
def test_attach_rejects_bad_session(bad_session: str) -> None:
    with pytest.raises(p.ProtocolError):
        p.parse_client_text(
            json.dumps(
                {
                    "type": "attach",
                    "ticket": _GOOD_TICKET,
                    "tmux_session": bad_session,
                    "cols": 80,
                    "rows": 24,
                }
            )
        )


@pytest.mark.parametrize(
    "bad_ticket",
    [
        "",
        "short",
        "x" * 64 + "." + "y" * 63,  # wrong half-length
        "Z" * 64 + "." + "b" * 64,  # uppercase not allowed
        "not-a-ticket",
    ],
)
def test_attach_rejects_bad_ticket(bad_ticket: str) -> None:
    with pytest.raises(p.ProtocolError):
        p.parse_client_text(
            json.dumps(
                {
                    "type": "attach",
                    "ticket": bad_ticket,
                    "tmux_session": "scribe",
                    "cols": 80,
                    "rows": 24,
                }
            )
        )


def test_resize_roundtrip() -> None:
    msg = p.parse_client_text(json.dumps({"type": "resize", "cols": 100, "rows": 30}))
    assert isinstance(msg, p.ResizeMessage)
    assert msg.cols == 100 and msg.rows == 30


def test_ack_monotonic_shape() -> None:
    msg = p.parse_client_text(json.dumps({"type": "ack", "bytes_total": 12345}))
    assert isinstance(msg, p.AckMessage)
    assert msg.bytes_total == 12345


@pytest.mark.parametrize("bad", [-1, "12345", None, True])
def test_ack_rejects_non_int_or_negative(bad: object) -> None:
    with pytest.raises(p.ProtocolError):
        p.parse_client_text(json.dumps({"type": "ack", "bytes_total": bad}))


def test_ping_passthrough() -> None:
    msg = p.parse_client_text(json.dumps({"type": "ping"}))
    assert isinstance(msg, dict)
    assert msg["type"] == "ping"


def test_malformed_json_rejected() -> None:
    with pytest.raises(p.ProtocolError):
        p.parse_client_text("not { json")


def test_non_object_rejected() -> None:
    with pytest.raises(p.ProtocolError):
        p.parse_client_text(json.dumps(["a", "b"]))


def test_unknown_type_rejected() -> None:
    with pytest.raises(p.ProtocolError):
        p.parse_client_text(json.dumps({"type": "what"}))


# ── Binary frames ─────────────────────────────────────────────────


def test_extract_stdin_ok() -> None:
    out = p.extract_stdin(b"Ihello")
    assert out == b"hello"


def test_extract_stdin_empty_payload_ok() -> None:
    assert p.extract_stdin(b"I") == b""


def test_extract_stdin_rejects_empty() -> None:
    with pytest.raises(p.ProtocolError):
        p.extract_stdin(b"")


def test_extract_stdin_rejects_bad_prefix() -> None:
    with pytest.raises(p.ProtocolError):
        p.extract_stdin(b"Xpayload")


def test_extract_stdin_rejects_oversize() -> None:
    with pytest.raises(p.ProtocolError):
        p.extract_stdin(b"I" + b"\x00" * (p.INBOUND_FRAME_MAX + 1))


def test_extract_stdin_accepts_max_size() -> None:
    # Exactly at the cap is allowed (total frame length == cap).
    payload_size = p.INBOUND_FRAME_MAX - 1
    out = p.extract_stdin(b"I" + b"\x00" * payload_size)
    assert len(out) == payload_size


# ── Encoders ──────────────────────────────────────────────────────


def test_encode_output_prefixed() -> None:
    assert p.encode_output(b"hello") == b"Ohello"


def test_encode_attached_shape() -> None:
    raw = p.encode_attached(cols=80, rows=24, tmux_session="scribe", pid=12345)
    obj = json.loads(raw)
    assert obj == {
        "type": "attached",
        "cols": 80,
        "rows": 24,
        "tmux_session": "scribe",
        "pid": 12345,
    }


def test_encode_status_shape() -> None:
    raw = p.encode_status(
        bytes_in=10,
        bytes_sent_total=200,
        bytes_acked_total=150,
        paused=True,
        cols=80,
        rows=24,
    )
    obj = json.loads(raw)
    assert obj["type"] == "status"
    assert obj["paused"] is True
    assert obj["bytes_sent_total"] == 200
    assert obj["bytes_acked_total"] == 150


def test_encode_error_with_message() -> None:
    raw = p.encode_error("capacity", "no slots")
    assert json.loads(raw) == {"type": "error", "code": "capacity", "message": "no slots"}


def test_encode_error_without_message() -> None:
    assert json.loads(p.encode_error("auth")) == {"type": "error", "code": "auth"}


def test_encode_bye_with_detail() -> None:
    raw = p.encode_bye("pty_exited", "tmux: command not found")
    assert json.loads(raw) == {
        "type": "bye",
        "reason": "pty_exited",
        "detail": "tmux: command not found",
    }


def test_encode_pong_literal() -> None:
    assert p.encode_pong() == '{"type":"pong"}'
