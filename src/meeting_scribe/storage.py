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

import json
import logging
import multiprocessing
import multiprocessing.connection
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
    MeetingState.COMPLETE: {MeetingState.REPROCESSING},
    MeetingState.INTERRUPTED: {MeetingState.RECORDING},  # resume
    MeetingState.REPROCESSING: {MeetingState.COMPLETE},  # reprocess done / crash recovery
}


def _is_reprocess_active(meeting_dir: Path) -> bool:
    """True if ``meeting_dir/.reprocess.lock`` exists AND names a
    still-live process.

    Used by ``MeetingStorage.recover_interrupted`` to distinguish a
    crashed reprocess (recoverable) from one running right now in
    another process (must not be touched). The lock is written by
    ``reprocess_meeting`` at step 0 and removed at step 7.

    Returns False on any parse error or when the PID is dead —
    treating ambiguity as "not active" is safe because the worst
    case is a recovery that races a live reprocess, which is the
    status quo we are trying to fix. A false negative here is no
    worse than today; a false positive (thinking it's active when
    it isn't) would permanently strand the meeting in REPROCESSING.
    """
    lock = meeting_dir / ".reprocess.lock"
    if not lock.exists():
        return False
    try:
        payload = json.loads(lock.read_text())
        pid = int(payload.get("pid", -1))
    except ValueError, OSError, json.JSONDecodeError:
        return False
    if pid <= 0:
        return False
    try:
        # signal 0 is the POSIX "is this PID alive" probe — no
        # signal delivered, just an errno check.
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user's PID — treat as alive (conservative). On a
        # single-user dev box this branch shouldn't fire; on a multi-
        # tenant host, better to strand the meeting than race.
        return True
    return True


class AudioWriter:
    """Time-aligned PCM writer for meeting audio recording.

    Uses absolute seeking to handle disconnects gracefully.
    If a gap occurs (client disconnect), the gap is filled with silence
    so that byte_offset always equals (ms / 1000) * 32000.
    """

    def __init__(self, path: Path, sample_rate: int = 16000, append: bool = False) -> None:
        self._path = path
        flags = os.O_WRONLY | os.O_CREAT | (0 if append else os.O_TRUNC)
        self._fd = os.open(str(path), flags, 0o644)
        self._sample_rate = sample_rate
        self._bytes_per_sec = sample_rate * 2  # s16le = 2 bytes/sample
        self._total_written = os.path.getsize(str(path)) if append else 0
        if append and self._total_written > 0:
            os.lseek(self._fd, 0, os.SEEK_END)
        self._closed = False
        # Drift detection
        self._chunk_count = 0
        self._first_chunk_size = 0
        self._last_elapsed_ms = 0
        self._expected_offset = 0  # accumulated from chunk sizes
        self._drift_ms = 0.0
        self._drift_warnings = 0

    def write_at(self, pcm: bytes, elapsed_ms: int) -> None:
        """Append PCM bytes as they arrive — no zero-gap filling.

        Earlier versions computed `target_offset = elapsed_ms * bytes_per_sec`
        from wall clock and filled any gap with zeros. That was a disaster
        for audio quality: any WebSocket latency, browser jitter, or main-thread
        stall produced a large zero run in the recording, making the playback
        sound choppy and cutting in and out (12% of a real meeting was
        zero-filled in the test case).

        The fix: just append. Wall-clock drift is still tracked for logging,
        but the audio stream itself is cumulative chunks in order — the same
        bytes the ASR backend receives.

        Large reconnection gaps (>10 s between chunks) DO insert proportional
        silence, because losing 10+ seconds of chunks is a real dropout and
        we need to preserve timing for the transcript. The 10 s threshold
        is set well above the worst-case event-loop stall (~2.5 s) so
        routine hiccups don't trigger false silence padding — that was the
        root cause of the 20% zero-fill + 88 fake silence stretches seen
        in meeting 18984813.
        """
        if self._closed:
            return
        if len(pcm) % 2 != 0:
            return  # Invalid s16le data

        current_size = os.fstat(self._fd).st_size

        # Only fill on genuinely LONG reconnection gaps. 10 s is well above
        # any server-side stall (event loop + GC + container restart all
        # fit under it) and well below any human-perceptible outage.
        elapsed_jump = elapsed_ms - self._last_elapsed_ms if self._last_elapsed_ms > 0 else 0
        is_reconnection_gap = elapsed_jump > 10000 and self._chunk_count > 1
        if is_reconnection_gap:
            gap_ms = elapsed_jump - int(len(pcm) / self._bytes_per_sec * 1000)
            if gap_ms > 0:
                gap_bytes = int(gap_ms / 1000 * self._bytes_per_sec)
                gap_bytes -= gap_bytes % 2  # keep alignment
                if gap_bytes > 0:
                    os.lseek(self._fd, current_size, os.SEEK_SET)
                    os.write(self._fd, b"\x00" * gap_bytes)
                    current_size += gap_bytes
                    logger.info(
                        "Audio writer: reconnection gap filled with %.0fs silence",
                        gap_ms / 1000,
                    )

        # Append new chunk — no position calculation from wall clock
        os.lseek(self._fd, current_size, os.SEEK_SET)
        os.write(self._fd, pcm)
        self._total_written = current_size + len(pcm)

        # Drift tracking for logging only (does NOT affect where bytes land)
        if self._chunk_count == 0:
            self._first_chunk_size = len(pcm)
        self._chunk_count += 1
        self._last_elapsed_ms = elapsed_ms

        if not is_reconnection_gap and self._first_chunk_size > 0:
            self._expected_offset += len(pcm)
            # How far behind wall clock are we?
            wall_clock_offset = int(elapsed_ms / 1000 * self._bytes_per_sec)
            drift_bytes = wall_clock_offset - self._expected_offset
            self._drift_ms = drift_bytes / self._bytes_per_sec * 1000
            if self._drift_ms > 1000 and self._drift_warnings < 10:
                self._drift_warnings += 1
                logger.warning(
                    "Audio clock drift: %.0fms behind wall clock "
                    "(cumulative %d bytes vs wall %d bytes) — chunks arriving slower than real-time",
                    self._drift_ms,
                    self._expected_offset,
                    wall_clock_offset,
                )
        elif is_reconnection_gap:
            self._expected_offset = current_size + len(pcm)

    @property
    def drift_ms(self) -> float:
        """Current drift in milliseconds (gap-adjusted)."""
        return self._drift_ms

    @property
    def current_offset(self) -> int:
        """Current write position in recording.pcm — the byte offset
        that the next write will start from.

        W6a uses this to mark the boundary of a recovery window: when
        the ASR watchdog escalates, the supervisor reads from the
        earliest unresolved submission's offset up to the current
        offset (which keeps growing while live audio is suppressed)
        and replays that range through the recovered backend.

        recording.pcm is the canonical archive of meeting audio
        regardless of recovery state — the writer keeps appending
        every chunk while the ASR backend's live submissions are
        suppressed during RECOVERY_PENDING / REPLAYING."""
        return self._total_written

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


class AudioWriterProcess:
    """Crash-resilient audio writer that runs in a separate process.

    The writer process owns the file descriptor and survives crashes in the
    main server process. Audio chunks are sent over a multiprocessing Pipe.
    If the main process dies, the writer detects pipe EOF, fsyncs, and exits.

    Usage::

        writer = AudioWriterProcess(path)
        writer.start()
        writer.write_at(pcm_bytes, elapsed_ms)  # non-blocking send over pipe
        writer.close()                           # graceful shutdown
    """

    # Sentinel values for pipe protocol
    _CMD_WRITE_AT = 0
    _CMD_CLOSE = 1

    def __init__(self, path: Path, sample_rate: int = 16000, append: bool = False) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._append = append
        self._process: multiprocessing.Process | None = None
        self._pipe: multiprocessing.connection.Connection | None = None
        self._started = False
        # On append mode, mirror the existing file size so callers that
        # read ``total_bytes`` for sample-offset alignment get the TRUE
        # audio-file position, not a post-restart zero. Without this,
        # ASR timestamps after a meeting resume collide with the
        # original transcript at t=0.
        self._total_written = (
            os.path.getsize(str(path)) if (append and os.path.exists(str(path))) else 0
        )
        self._last_elapsed_ms = 0

    def start(self) -> None:
        """Spawn the writer process."""
        import multiprocessing

        parent_conn, child_conn = multiprocessing.Pipe()
        self._pipe = parent_conn
        self._process = multiprocessing.Process(
            target=self._writer_loop,
            args=(child_conn, self._path, self._sample_rate, self._append),
            daemon=False,  # survives parent crash
            name="audio-writer",
        )
        self._process.start()
        child_conn.close()  # parent doesn't need the child end
        self._started = True

    @staticmethod
    def _writer_loop(
        conn: multiprocessing.connection.Connection,
        path: Path,
        sample_rate: int,
        append: bool,
    ) -> None:
        """Run in child process — write audio chunks until pipe closes."""
        import signal

        # Ignore SIGTERM/SIGINT from parent — we want to finish writing
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        writer = AudioWriter(path, sample_rate=sample_rate, append=append)
        try:
            while True:
                try:
                    if not conn.poll(timeout=2.0):
                        # Check if parent is still alive
                        if os.getppid() == 1:
                            # Parent died — orphaned. Flush and exit.
                            break
                        continue
                    msg = conn.recv()
                except EOFError, BrokenPipeError, OSError:
                    # Pipe closed — parent died or called close
                    break

                if msg[0] == AudioWriterProcess._CMD_WRITE_AT:
                    writer.write_at(msg[1], msg[2])
                elif msg[0] == AudioWriterProcess._CMD_CLOSE:
                    break
        finally:
            writer.close()

    def write_at(self, pcm: bytes, elapsed_ms: int) -> None:
        """Send a write command to the writer process (non-blocking)."""
        if not self._started or self._pipe is None:
            return
        try:
            self._pipe.send((self._CMD_WRITE_AT, pcm, elapsed_ms))
            self._total_written += len(pcm)
            self._last_elapsed_ms = elapsed_ms
        except BrokenPipeError, OSError:
            logger.warning("Audio writer process pipe broken — audio may be lost")
            self._started = False

    @property
    def duration_ms(self) -> int:
        """Estimated duration based on bytes sent."""
        bytes_per_sec = self._sample_rate * 2
        return int(self._total_written / bytes_per_sec * 1000) if bytes_per_sec else 0

    @property
    def total_bytes(self) -> int:
        return self._total_written

    @property
    def current_offset(self) -> int:
        """Current write position — see AudioWriter.current_offset.
        AudioWriterProcess shares the same byte-counter contract."""
        return self._total_written

    def close(self) -> None:
        """Gracefully shut down the writer process."""
        if not self._started:
            return
        self._started = False
        if self._pipe:
            try:
                self._pipe.send((self._CMD_CLOSE,))
            except BrokenPipeError, OSError:
                pass
            self._pipe.close()
            self._pipe = None
        if self._process and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        self._process = None
        logger.info("Audio writer process closed: %s (%.1fs)", self._path, self.duration_ms / 1000)

    @property
    def is_alive(self) -> bool:
        """Check if the writer process is still running."""
        return bool(self._process and self._process.is_alive())


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

        Also garbage-collects empty meetings that contain no final
        transcript events — these are the leftovers from a Start →
        nothing-said → Stop / crash path and only clutter the sidebar.

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
            elif meta.state == MeetingState.REPROCESSING:
                # REPROCESSING on disk has two interpretations:
                #   (a) a prior reprocess was killed mid-flight (e.g.
                #       systemd stop timeout) — we want to recover it
                #       back to COMPLETE so the meeting is viewable.
                #   (b) a reprocess is ACTIVELY RUNNING right now in
                #       another process (the CLI ``full-reprocess``
                #       path, or a second server). Flipping state
                #       here would race the live reprocess and crash
                #       its step-7 COMPLETE transition.
                #
                # The lock file at ``.reprocess.lock`` distinguishes
                # them: reprocess_meeting writes it before step 0's
                # state flip and removes it after step 7's. A lock
                # whose PID is still alive means (b); a missing or
                # stale-PID lock means (a). Only (a) is recovered.
                if _is_reprocess_active(meeting_dir):
                    logger.info(
                        "Meeting %s: reprocess in progress (lock held) — "
                        "leaving state=reprocessing, not recovering",
                        meta.meeting_id,
                    )
                    continue
                self.transition_state(meta.meeting_id, MeetingState.COMPLETE)
                # Clean up the stale lock so a subsequent recovery pass
                # doesn't trip the liveness check against a dead PID.
                (meeting_dir / ".reprocess.lock").unlink(missing_ok=True)
                count += 1
                logger.info(
                    "Recovered meeting %s: reprocess was killed; state -> complete",
                    meta.meeting_id,
                )

            # Delete any meeting (interrupted, complete, or just-recovered)
            # that has no final transcript events. These are the
            # zero-event carcasses the user doesn't want to see.
            #
            # NOTE (see ../../../UPGRADE-NOTES-2026-04.md): this cleanup
            # is the reason 8 integration tests in tests/test_pipeline.py,
            # tests/test_security.py, tests/test_server_endpoints.py, and
            # tests/test_system.py fail with 404. They do
            # start -> stop -> GET /api/meetings/{id}/... without ever
            # recording audio, so the meeting hits this branch between
            # stop and the follow-up fetch. Fix options: make the tests
            # record a minimal audio blob, expose a preserve_empty knob
            # on the test server, or mark the affected tests xfail.
            if meta.state in (MeetingState.INTERRUPTED, MeetingState.COMPLETE):
                journal_path = meeting_dir / "journal.jsonl"
                if not self._journal_has_final_events(journal_path):
                    # SAFETY INVARIANT (added 2026-04-14): never delete
                    # a meeting that still has real audio on disk.
                    # Audio is unrecoverable — the journal, timeline,
                    # and speaker files can all be regenerated from
                    # the PCM via reprocess, but a deleted PCM is gone
                    # forever. On 2026-04-14 a journal wipe + server
                    # restart triggered this cleanup and destroyed an
                    # 88-minute meeting recording. Never again.
                    pcm_path = meeting_dir / "audio" / "recording.pcm"
                    if pcm_path.exists() and pcm_path.stat().st_size > 0:
                        logger.info(
                            "Preserving %s meeting %s — has %d bytes of audio "
                            "(journal is empty but audio is recoverable via reprocess)",
                            meta.state.value,
                            meta.meeting_id,
                            pcm_path.stat().st_size,
                        )
                        continue
                    try:
                        import shutil

                        shutil.rmtree(meeting_dir)
                        logger.info(
                            "Deleted empty %s meeting %s (no audio, no journal)",
                            meta.state.value,
                            meta.meeting_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to delete empty meeting %s: %s",
                            meta.meeting_id,
                            e,
                        )

        return count

    @staticmethod
    def _journal_has_final_events(journal_path) -> bool:
        """Return True if the journal has at least one final, non-empty event."""
        if not journal_path.exists():
            return False
        try:
            import json as _json

            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    e = _json.loads(line)
                except Exception:
                    continue
                if e.get("is_final") and (e.get("text") or "").strip():
                    return True
        except Exception:
            # On unexpected read error, be conservative and keep the meeting.
            return True
        return False

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

    def read_journal_raw(self, meeting_id: str, max_lines: int = 500) -> list[str]:
        """Read raw JSON lines from a meeting's journal, deduped for replay.

        Under multi-target fan-out a single segment_id can have multiple
        translation entries (one per target language) plus the original
        ASR entry. We keep ONE surviving entry per bucket, where the
        bucket key is:

          - ``segment_id`` alone for ASR/non-translated entries
          - ``(segment_id, translation.target_language)`` for any entry
            that carries a translation

        Within a bucket the later-written entry wins — successive ASR
        revisions, ``in_progress`` → ``done`` upgrades, and furigana
        annotations all end up as the last line for their bucket. For
        legacy journals with exactly one translation per segment this
        produces the same output as the pre-fan-out dedup.

        Returns the last *max_lines* DEDUPED entries (most recent).
        """
        import json as _json

        path = self._journal_path(meeting_id)
        if not path.exists():
            return []
        lines = path.read_text().splitlines()

        # Chronology is ordered by segment_id first-seen, then by bucket
        # first-seen within each segment. That keeps
        # (seg1, en), (seg2, en), (seg3, "") in seg1/seg2/seg3 order
        # even when seg2's translation entry shows up after seg3.
        seen: dict[tuple[str, str], str] = {}
        bucket_order: dict[tuple[str, str], int] = {}
        segment_order: dict[str, int] = {}
        bucket_pos = 0
        segment_pos = 0

        for line in lines:
            if not line.strip():
                continue
            try:
                e = _json.loads(line)
            except ValueError, _json.JSONDecodeError:
                continue
            sid = e.get("segment_id")
            if not sid:
                continue
            if sid not in segment_order:
                segment_order[sid] = segment_pos
                segment_pos += 1
            translation = e.get("translation") or {}
            target = translation.get("target_language") or ""
            bucket = (sid, target)
            if bucket not in seen:
                bucket_order[bucket] = bucket_pos
                bucket_pos += 1
            seen[bucket] = line  # later line wins within a bucket

        # Drop the ASR-only ("") bucket for any segment that also has at
        # least one translation bucket — the translation entries carry
        # the source text and are strictly more informative. Segments
        # without any translation keep their ASR-only entry.
        segments_with_translation = {sid for (sid, target) in seen if target}
        surviving = [b for b in seen if not (b[1] == "" and b[0] in segments_with_translation)]
        surviving.sort(key=lambda b: (segment_order[b[0]], bucket_order[b]))
        deduped = [seen[b] for b in surviving]
        if len(deduped) > max_lines:
            return deduped[-max_lines:]
        return deduped

    def audit_meetings(self) -> list[dict]:
        """Scan meetings dir and return audit info for each meeting.

        Returns a list of dicts with keys:
            meeting_id, state, audio_duration_s, journal_lines,
            has_summary, has_timeline, has_audio, age_hours, meeting_dir
        """
        import json as _json
        from datetime import UTC, datetime

        meetings_dir = self._meetings_dir
        results: list[dict] = []
        if not meetings_dir.exists():
            return results

        now = datetime.now(UTC)
        for d in sorted(meetings_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _json.loads(meta_path.read_text())
            except _json.JSONDecodeError, OSError:
                continue

            audio_path = d / "audio" / "recording.pcm"
            audio_bytes = audio_path.stat().st_size if audio_path.exists() else 0
            # s16le 16kHz mono = 32000 bytes/sec
            audio_duration_s = audio_bytes / 32000 if audio_bytes else 0

            journal_path = d / "journal.jsonl"
            journal_lines = 0
            if journal_path.exists():
                with journal_path.open() as f:
                    journal_lines = sum(1 for _ in f)

            created_at = meta.get("created_at", "")
            age_hours = 999999.0
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_hours = (now - ts).total_seconds() / 3600
                except ValueError, TypeError:
                    pass

            results.append(
                {
                    "meeting_id": meta.get("meeting_id", d.name),
                    "state": meta.get("state", "unknown"),
                    "audio_duration_s": round(audio_duration_s, 1),
                    "journal_lines": journal_lines,
                    "has_summary": (d / "summary.json").exists(),
                    "has_timeline": (d / "timeline.json").exists(),
                    "has_audio": audio_bytes > 1000,
                    "age_hours": round(age_hours, 1),
                    "meeting_dir": d,
                }
            )
        return results

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
            # Some platforms lack fdatasync; fall back to fsync.
            os.fsync(fd)

    # ── Audio Recording ─────────────────────────────────────

    def open_audio_writer(
        self, meeting_id: str, isolated: bool = True
    ) -> AudioWriter | AudioWriterProcess:
        """Open an audio writer for the meeting. Creates audio/ directory.

        Args:
            meeting_id: Meeting identifier.
            isolated: If True, run writer in a separate process for crash
                      resilience. The writer survives main server crashes.
        """
        audio_dir = self._meeting_dir(meeting_id) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        path = audio_dir / "recording.pcm"
        if isolated:
            writer = AudioWriterProcess(path)
            writer.start()
            return writer
        return AudioWriter(path)

    def audio_offset_ms(self, meeting_id: str) -> int:
        """Get the audio recording offset in milliseconds.

        The PCM file may start later than the meeting's time origin
        (e.g., first audio chunk arrives seconds after meeting creation).
        Journal timestamps are absolute from meeting start, but PCM byte 0
        corresponds to audio_offset_ms. All audio reads must subtract this.

        Stored in meta.json as 'audio_offset_ms'. Falls back to 0 if not set
        (live recordings where AudioWriter.write_at aligns to elapsed_ms).
        """
        import json as _json

        meta_path = self._meeting_dir(meeting_id) / "meta.json"
        if meta_path.exists():
            try:
                meta = _json.loads(meta_path.read_text())
                return int(meta.get("audio_offset_ms", 0))
            except Exception:
                pass
        return 0

    def read_audio_segment(self, meeting_id: str, start_ms: int, end_ms: int) -> bytes:
        """Read a slice of audio from the recording as raw PCM.

        Accounts for audio_offset_ms: journal timestamps are absolute,
        but the PCM file starts at audio_offset_ms.
        """
        path = self._meeting_dir(meeting_id) / "audio" / "recording.pcm"
        if not path.exists():
            return b""
        offset = self.audio_offset_ms(meeting_id)
        bytes_per_sec = 16000 * 2  # s16le 16kHz mono
        start_byte = int(max(0, start_ms - offset) / 1000 * bytes_per_sec)
        end_byte = int(max(0, end_ms - offset) / 1000 * bytes_per_sec)
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
        """Get the total meeting duration in milliseconds (including offset)."""
        path = self._meeting_dir(meeting_id) / "audio" / "recording.pcm"
        if not path.exists():
            return 0
        size = path.stat().st_size
        pcm_duration_ms = int(size / (16000 * 2) * 1000)
        return pcm_duration_ms + self.audio_offset_ms(meeting_id)

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

    def update_journal_speaker_identity(
        self,
        meeting_id: str,
        cluster_id: int,
        display_name: str,
    ) -> int:
        """Rewrite journal.jsonl: set identity=display_name for all events
        where speakers[0].cluster_id == cluster_id.

        Atomic via write-to-tmp + rename. Returns the number of events updated.
        Safe to call on a live meeting — the journal fd is held by the writer
        but this appends a corrected copy atomically.
        """
        import json as _json

        journal_path = self._meeting_dir(meeting_id) / "journal.jsonl"
        if not journal_path.exists():
            return 0

        tmp_path = journal_path.with_suffix(".jsonl.tmp")
        updated = 0
        with self._lock:
            with open(journal_path) as rf, open(tmp_path, "w") as wf:
                for line in rf:
                    if not line.strip():
                        continue
                    try:
                        event = _json.loads(line)
                    except _json.JSONDecodeError:
                        wf.write(line)
                        continue

                    speakers = event.get("speakers") or []
                    if speakers and speakers[0].get("cluster_id") == cluster_id:
                        speakers[0]["identity"] = display_name
                        speakers[0]["source"] = "enrolled"
                        speakers[0]["identity_confidence"] = 1.0
                        event["speakers"] = speakers
                        wf.write(_json.dumps(event) + "\n")
                        updated += 1
                    else:
                        wf.write(line)

            # Atomic rename
            tmp_path.replace(journal_path)

        logger.info(
            "Journal updated: meeting=%s cluster=%d name=%s events=%d",
            meeting_id,
            cluster_id,
            display_name,
            updated,
        )
        return updated

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
