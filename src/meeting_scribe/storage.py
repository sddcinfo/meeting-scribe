"""Meeting storage with crash durability.

Manages the on-disk layout for meetings including metadata, audio
chunks, and the append-only transcript journal. Uses atomic
write-rename for state transitions and fdatasync for journal
durability.

Directory structure per meeting::

    meetings/{id}/
        meta.json          # Atomic write-rename on each state transition
        audio/             # Float32 PCM chunks (AudioChunkWriter)
        journal.jsonl      # Append-only TranscriptEvents, fsync every 5s
        live.json          # Current live transcript state (for reconnects)
        polished.json      # Final polished transcript (after finalizing)

State machine:
    created -> recording -> finalizing -> complete | interrupted
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from meeting_scribe.config import ServerConfig
from meeting_scribe.models import MeetingMeta, MeetingState, RoomLayout, TranscriptEvent

logger = logging.getLogger(__name__)

# Valid state transitions.
_TRANSITIONS: dict[MeetingState, set[MeetingState]] = {
    MeetingState.CREATED: {MeetingState.RECORDING, MeetingState.INTERRUPTED},
    MeetingState.RECORDING: {MeetingState.FINALIZING, MeetingState.INTERRUPTED},
    MeetingState.FINALIZING: {MeetingState.COMPLETE, MeetingState.INTERRUPTED},
    MeetingState.COMPLETE: set(),
    MeetingState.INTERRUPTED: set(),
}


class AudioWriter:
    """Time-aligned PCM writer for meeting audio recording.

    Uses absolute seeking to handle disconnects gracefully.
    If a gap occurs (client disconnect), the gap is filled with silence
    so that byte_offset always equals (ms / 1000) * 32000.
    """

    def __init__(self, path: Path, sample_rate: int = 16000) -> None:
        self._path = path
        self._fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self._sample_rate = sample_rate
        self._bytes_per_sec = sample_rate * 2  # s16le = 2 bytes/sample
        self._total_written = 0
        self._closed = False

    def write_at(self, pcm: bytes, elapsed_ms: int) -> None:
        """Write PCM at absolute time offset. Fills gaps with silence."""
        if self._closed:
            return
        if len(pcm) % 2 != 0:
            return  # Invalid s16le data

        target_offset = int(elapsed_ms / 1000 * self._bytes_per_sec)
        current_size = os.fstat(self._fd).st_size

        if target_offset > current_size:
            # Fill gap with silence
            gap = target_offset - current_size
            os.lseek(self._fd, current_size, os.SEEK_SET)
            os.write(self._fd, b"\x00" * gap)

        os.lseek(self._fd, target_offset, os.SEEK_SET)
        os.write(self._fd, pcm)
        self._total_written = max(self._total_written, target_offset + len(pcm))

    def write_append(self, pcm: bytes) -> None:
        """Simple append (when time tracking is not needed)."""
        if self._closed:
            return
        os.lseek(self._fd, 0, os.SEEK_END)
        os.write(self._fd, pcm)
        self._total_written += len(pcm)

    @property
    def total_bytes(self) -> int:
        if self._closed:
            return self._total_written
        return os.fstat(self._fd).st_size

    @property
    def duration_ms(self) -> int:
        return int(self.total_bytes / self._bytes_per_sec * 1000)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.fdatasync(self._fd)
        except AttributeError:
            os.fsync(self._fd)
        os.close(self._fd)
        logger.info("Audio recording closed: %s (%.1fs)", self._path, self.duration_ms / 1000)


class MeetingStorage:
    """Manages durable on-disk storage for meetings.

    Provides crash recovery on startup (recording -> interrupted),
    append-only journaling with periodic fsync, and automatic
    retention cleanup.

    Args:
        config: Server configuration (uses ``meetings_dir``,
            ``journal_fsync_seconds``, and ``retention_days``).
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        self._config = config or ServerConfig()
        self._meetings_dir = self._config.meetings_dir
        self._journal_fsync_seconds = self._config.journal_fsync_seconds
        self._retention_days = self._config.retention_days

        # Per-meeting journal state: meeting_id -> (fd, last_fsync_time)
        self._journals: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

        # Retention cleanup timer
        self._cleanup_timer: threading.Timer | None = None
        self._cleanup_interval_hours = 6

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Initialize storage: recover interrupted meetings and run cleanup.

        Call once at server start before accepting any connections.
        """
        self._meetings_dir.mkdir(parents=True, exist_ok=True)
        recovered = self.recover_interrupted()
        if recovered:
            logger.info("Recovered %d interrupted meeting(s)", recovered)

        self.cleanup_retention()
        self._schedule_cleanup()

    def shutdown(self) -> None:
        """Flush all open journals and cancel the cleanup timer."""
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

        with self._lock:
            for meeting_id in list(self._journals):
                self._close_journal(meeting_id)

    # ------------------------------------------------------------------
    # Meeting operations
    # ------------------------------------------------------------------

    def create_meeting(self, meta: MeetingMeta | None = None) -> MeetingMeta:
        """Create a new meeting directory and write initial metadata.

        Args:
            meta: Optional pre-populated metadata. If None, a new
                MeetingMeta with a fresh UUID is created.

        Returns:
            The MeetingMeta for the new meeting.
        """
        if meta is None:
            meta = MeetingMeta()

        if not meta.created_at:
            meta.created_at = datetime.now(UTC).isoformat()

        meeting_dir = self._meeting_dir(meta.meeting_id)
        meeting_dir.mkdir(parents=True, exist_ok=True)
        (meeting_dir / "audio").mkdir(exist_ok=True)

        self._write_meta(meta)
        logger.info("Meeting created: %s", meta.meeting_id)
        return meta

    def transition_state(self, meeting_id: str, new_state: MeetingState) -> MeetingMeta:
        """Atomically transition a meeting to a new state.

        Args:
            meeting_id: The meeting to transition.
            new_state: Target state.

        Returns:
            Updated MeetingMeta.

        Raises:
            ValueError: If the transition is not valid.
            FileNotFoundError: If the meeting does not exist.
        """
        meta = self._read_meta(meeting_id)
        allowed = _TRANSITIONS.get(meta.state, set())

        if new_state not in allowed:
            raise ValueError(
                f"Invalid state transition: {meta.state.value} -> {new_state.value} "
                f"(allowed: {[s.value for s in allowed]})"
            )

        meta.state = new_state
        self._write_meta(meta)

        # Close journal when leaving recording/finalizing states.
        if new_state in (MeetingState.COMPLETE, MeetingState.INTERRUPTED):
            with self._lock:
                self._close_journal(meeting_id)

        logger.info("Meeting %s: state -> %s", meeting_id, new_state.value)
        return meta

    def append_event(self, meeting_id: str, event: TranscriptEvent) -> None:
        """Append a TranscriptEvent to the meeting's journal.

        The journal is an append-only JSONL file. Events are buffered in
        the OS write cache and fsynced every ``journal_fsync_seconds``.

        Args:
            meeting_id: Target meeting.
            event: The transcript event to persist.
        """
        line = event.model_dump_json() + "\n"
        data = line.encode("utf-8")

        with self._lock:
            fd, last_sync = self._get_journal_fd(meeting_id)
            os.write(fd, data)

            now = time.monotonic()
            if now - last_sync >= self._journal_fsync_seconds:
                self._fsync_fd(fd)
                self._journals[meeting_id] = (fd, now)

    def flush_journal(self, meeting_id: str) -> None:
        """Force-fsync the journal for a meeting.

        Call this during state transitions or before shutdown to
        guarantee all events are durable.
        """
        with self._lock:
            if meeting_id in self._journals:
                fd, _ = self._journals[meeting_id]
                self._fsync_fd(fd)
                self._journals[meeting_id] = (fd, time.monotonic())
                logger.debug("Journal flushed: %s", meeting_id)

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def recover_interrupted(self) -> int:
        """Find meetings in ``recording`` state and mark them ``interrupted``.

        Returns:
            Number of meetings recovered.
        """
        count = 0
        if not self._meetings_dir.exists():
            return count

        for meeting_dir in self._meetings_dir.iterdir():
            if not meeting_dir.is_dir():
                continue

            meta_path = meeting_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                meta = MeetingMeta.model_validate_json(meta_path.read_text())
            except Exception:
                logger.warning("Corrupt meta.json in %s — skipping", meeting_dir.name)
                continue

            if meta.state in (MeetingState.RECORDING, MeetingState.FINALIZING):
                meta.state = MeetingState.INTERRUPTED
                self._write_meta(meta)
                count += 1
                logger.info(
                    "Recovered meeting %s: marked interrupted",
                    meta.meeting_id,
                )

        return count

    # ------------------------------------------------------------------
    # Retention cleanup
    # ------------------------------------------------------------------

    def cleanup_retention(self) -> int:
        """Delete meetings older than ``retention_days``.

        Returns:
            Number of meetings deleted.
        """
        if not self._meetings_dir.exists():
            return 0

        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days)
        deleted = 0

        for meeting_dir in self._meetings_dir.iterdir():
            if not meeting_dir.is_dir():
                continue

            meta_path = meeting_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                meta = MeetingMeta.model_validate_json(meta_path.read_text())
            except Exception:
                continue

            if not meta.created_at:
                continue

            try:
                created = datetime.fromisoformat(meta.created_at)
            except ValueError:
                continue

            if created < cutoff and meta.state in (
                MeetingState.COMPLETE,
                MeetingState.INTERRUPTED,
            ):
                shutil.rmtree(meeting_dir)
                deleted += 1
                logger.info(
                    "Retention cleanup: deleted meeting %s (created %s)",
                    meta.meeting_id,
                    meta.created_at,
                )

        if deleted:
            logger.info("Retention cleanup: %d meeting(s) deleted", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_meta(self, meeting_id: str) -> MeetingMeta:
        """Read and return meeting metadata.

        Raises:
            FileNotFoundError: If the meeting does not exist.
        """
        return self._read_meta(meeting_id)

    def get_meeting_dir(self, meeting_id: str) -> Path:
        """Return the directory path for a meeting."""
        return self._meeting_dir(meeting_id)

    def get_audio_dir(self, meeting_id: str) -> Path:
        """Return the audio subdirectory path for a meeting."""
        return self._meeting_dir(meeting_id) / "audio"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meeting_dir(self, meeting_id: str) -> Path:
        return self._meetings_dir / meeting_id

    def _meta_path(self, meeting_id: str) -> Path:
        return self._meeting_dir(meeting_id) / "meta.json"

    def _journal_path(self, meeting_id: str) -> Path:
        return self._meeting_dir(meeting_id) / "journal.jsonl"

    def _read_meta(self, meeting_id: str) -> MeetingMeta:
        """Read meta.json for a meeting."""
        path = self._meta_path(meeting_id)
        if not path.exists():
            raise FileNotFoundError(f"Meeting not found: {meeting_id}")
        return MeetingMeta.model_validate_json(path.read_text())

    def _write_meta(self, meta: MeetingMeta) -> None:
        """Atomically write meta.json using write-rename.

        Writes to a temporary file first, fsyncs, then renames into
        place. This guarantees that meta.json is never partially written.
        """
        meeting_dir = self._meeting_dir(meta.meeting_id)
        meeting_dir.mkdir(parents=True, exist_ok=True)

        target = meeting_dir / "meta.json"
        tmp = meeting_dir / "meta.json.tmp"

        data = meta.model_dump_json(indent=2).encode("utf-8")

        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            self._fsync_fd(fd)
        finally:
            os.close(fd)

        tmp.rename(target)

    def _get_journal_fd(self, meeting_id: str) -> tuple[int, float]:
        """Get or open the journal file descriptor for a meeting.

        Must be called with ``self._lock`` held.
        """
        if meeting_id in self._journals:
            return self._journals[meeting_id]

        journal_path = self._journal_path(meeting_id)
        fd = os.open(
            str(journal_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        now = time.monotonic()
        self._journals[meeting_id] = (fd, now)
        return fd, now

    def _close_journal(self, meeting_id: str) -> None:
        """Close a journal file descriptor. Must be called with lock held."""
        if meeting_id not in self._journals:
            return

        fd, _ = self._journals.pop(meeting_id)
        try:
            self._fsync_fd(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_fd(fd: int) -> None:
        """Fsync a file descriptor, falling back from fdatasync to fsync."""
        try:
            os.fdatasync(fd)
        except AttributeError:
            # macOS does not have fdatasync; fall back to fsync.
            os.fsync(fd)

    # ── Audio Recording ─────────────────────────────────────

    def open_audio_writer(self, meeting_id: str) -> AudioWriter:
        """Open an audio writer for the meeting. Creates audio/ directory."""
        audio_dir = self._meeting_dir(meeting_id) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = audio_dir / "recording.pcm"
        return AudioWriter(path)

    def read_audio_segment(self, meeting_id: str, start_ms: int, end_ms: int) -> bytes:
        """Read a slice of audio from the recording as raw PCM."""
        path = self._meeting_dir(meeting_id) / "audio" / "recording.pcm"
        if not path.exists():
            return b""
        bytes_per_sec = 16000 * 2  # s16le 16kHz mono
        start_byte = int(start_ms / 1000 * bytes_per_sec)
        end_byte = int(end_ms / 1000 * bytes_per_sec)
        with open(path, "rb") as f:
            f.seek(start_byte)
            return f.read(end_byte - start_byte)

    def save_detected_speakers(self, meeting_id: str, speakers: list) -> None:
        """Save detected speakers to detected_speakers.json."""
        import json as _json

        path = self._meeting_dir(meeting_id) / "detected_speakers.json"
        path.write_text(
            _json.dumps([s if isinstance(s, dict) else s.model_dump() for s in speakers], indent=2)
        )

    def load_detected_speakers(self, meeting_id: str) -> list:
        """Load detected speakers from detected_speakers.json."""
        import json as _json

        path = self._meeting_dir(meeting_id) / "detected_speakers.json"
        if not path.exists():
            return []
        try:
            return _json.loads(path.read_text())
        except Exception:
            return []

    def audio_duration_ms(self, meeting_id: str) -> int:
        """Get the duration of the recording in milliseconds."""
        path = self._meeting_dir(meeting_id) / "audio" / "recording.pcm"
        if not path.exists():
            return 0
        size = path.stat().st_size
        return int(size / (16000 * 2) * 1000)

    # ── Room Layout Persistence ──────────────────────────────

    def save_room_layout(self, meeting_id: str, layout: RoomLayout) -> None:
        """Atomically write room.json for a meeting."""

        meeting_dir = self._meeting_dir(meeting_id)
        meeting_dir.mkdir(parents=True, exist_ok=True)

        target = meeting_dir / "room.json"
        tmp = meeting_dir / "room.json.tmp"
        data = layout.model_dump_json(indent=2).encode("utf-8")

        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            self._fsync_fd(fd)
        finally:
            os.close(fd)
        tmp.rename(target)
        logger.info("Saved room layout to %s", target)

    def load_room_layout(self, meeting_id: str) -> RoomLayout | None:
        """Load room.json for a meeting. Returns None if not present."""

        path = self._meeting_dir(meeting_id) / "room.json"
        if not path.exists():
            return None
        try:
            return RoomLayout.model_validate_json(path.read_text())
        except Exception:
            logger.warning("Failed to load room layout from %s", path)
            return None

    def _schedule_cleanup(self) -> None:
        """Schedule the next retention cleanup run."""
        interval_seconds = self._cleanup_interval_hours * 3600
        self._cleanup_timer = threading.Timer(interval_seconds, self._cleanup_tick)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _cleanup_tick(self) -> None:
        """Run retention cleanup and reschedule."""
        try:
            self.cleanup_retention()
        except Exception:
            logger.exception("Retention cleanup failed")
        finally:
            self._schedule_cleanup()
