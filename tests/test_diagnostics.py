"""Unit tests for meeting_scribe.diagnostics.

Covers the in-memory ring buffer (capture, filter, snapshot semantics)
and the rotating server-log tail/stream helpers. Endpoint-level auth
gating is shared with /api/meetings/{id}/terminal-log and is therefore
covered indirectly by the existing terminal_auth tests.
"""

from __future__ import annotations

import logging

import pytest

from meeting_scribe import diagnostics as diag


@pytest.fixture
def fresh_buffer():
    return diag.RecentLogRingBuffer(capacity=10, level=logging.WARNING)


def _emit(handler: diag.RecentLogRingBuffer, level: int, name: str, msg: str) -> None:
    rec = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    handler.emit(rec)


def test_ring_buffer_captures_warning_and_above(fresh_buffer):
    _emit(fresh_buffer, logging.INFO, "meeting_scribe.tts", "info should be ignored")
    _emit(fresh_buffer, logging.WARNING, "meeting_scribe.tts", "warn 1")
    _emit(fresh_buffer, logging.ERROR, "meeting_scribe.translation", "err 1")

    snap = fresh_buffer.snapshot()
    # Logging Handler.emit is unconditional — the level check happens in
    # logger.handle, NOT in the handler. So all three records land here.
    # The diag.snapshot() level filter is what users care about.
    assert len(snap) == 3
    snap = fresh_buffer.snapshot(level="WARNING")
    assert [e["level"] for e in snap] == ["WARNING", "ERROR"]


def test_ring_buffer_capacity_caps(fresh_buffer):
    for i in range(15):
        _emit(fresh_buffer, logging.WARNING, "meeting_scribe.x", f"msg {i}")
    snap = fresh_buffer.snapshot()
    assert len(snap) == 10
    # Newest 10 retained; ids are sequential.
    assert snap[0]["message"] == "msg 5"
    assert snap[-1]["message"] == "msg 14"


def test_ring_buffer_since_id_returns_only_new(fresh_buffer):
    for i in range(5):
        _emit(fresh_buffer, logging.WARNING, "meeting_scribe.x", f"msg {i}")
    first = fresh_buffer.snapshot()
    last_id = first[-1]["id"]
    _emit(fresh_buffer, logging.ERROR, "meeting_scribe.x", "after")
    delta = fresh_buffer.snapshot(since_id=last_id)
    assert len(delta) == 1
    assert delta[0]["message"] == "after"


def test_ring_buffer_component_filter(fresh_buffer):
    _emit(fresh_buffer, logging.WARNING, "meeting_scribe.translation.queue", "trans warn")
    _emit(fresh_buffer, logging.WARNING, "meeting_scribe.tts.worker", "tts warn")
    _emit(fresh_buffer, logging.WARNING, "meeting_scribe.slides.runner", "slides warn")
    snap = fresh_buffer.snapshot(component="translation")
    assert len(snap) == 1
    assert "trans" in snap[0]["message"]


def test_setup_diagnostics_logging_is_idempotent(tmp_path, monkeypatch):
    # Reset module state so this test doesn't depend on import order.
    monkeypatch.setattr(diag, "_setup_done", False)
    monkeypatch.setattr(diag, "_ring_buffer", None)
    monkeypatch.setattr(diag, "_log_file_path", None)

    target_logger = logging.getLogger("meeting_scribe_test_diag")
    # Detach any existing handlers from a previous test run.
    for h in list(target_logger.handlers):
        target_logger.removeHandler(h)

    diag.setup_diagnostics_logging(tmp_path, logger_name="meeting_scribe_test_diag")
    n1 = len(target_logger.handlers)
    diag.setup_diagnostics_logging(tmp_path, logger_name="meeting_scribe_test_diag")
    n2 = len(target_logger.handlers)
    assert n1 == n2, "setup_diagnostics_logging should be idempotent"

    log_path = diag.get_log_file_path()
    assert log_path is not None
    assert log_path.parent.name == "diagnostics"
    assert log_path.exists()


def test_tail_log_lines_returns_last_n(tmp_path, monkeypatch):
    log = tmp_path / "diagnostics" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(f"line-{i} INFO meeting_scribe: hello" for i in range(50)) + "\n")
    monkeypatch.setattr(diag, "_log_file_path", log)

    out = diag.tail_log_lines(max_lines=10)
    assert len(out) == 10
    assert out[-1].startswith("line-49")
    assert out[0].startswith("line-40")


def test_tail_log_lines_filters_by_level(tmp_path, monkeypatch):
    log = tmp_path / "diagnostics" / "server.log"
    log.parent.mkdir(parents=True)
    lines = []
    for i in range(30):
        lvl = "WARNING" if i % 5 == 0 else "INFO"
        lines.append(f"2026-04-22 12:00:00 {lvl} meeting_scribe: msg {i}")
    log.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(diag, "_log_file_path", log)

    only_warn = diag.tail_log_lines(max_lines=100, level="WARNING")
    assert all(" WARNING " in ln for ln in only_warn)
    assert len(only_warn) == 6  # 0,5,10,15,20,25


def test_tail_log_lines_search_substring(tmp_path, monkeypatch):
    log = tmp_path / "diagnostics" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "2026 INFO meeting_scribe: TTS fire segment 1\n"
        "2026 INFO meeting_scribe: ASR partial\n"
        "2026 INFO meeting_scribe: TTS fire segment 2\n"
    )
    monkeypatch.setattr(diag, "_log_file_path", log)

    out = diag.tail_log_lines(max_lines=100, search="tts fire")
    assert len(out) == 2
    assert all("TTS fire" in ln for ln in out)


def test_tail_log_lines_handles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "_log_file_path", tmp_path / "nope.log")
    assert diag.tail_log_lines(max_lines=100) == []
