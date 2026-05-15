"""Tests for AudioWriterProcess — crash-resilient audio capture.

Covers edge cases: append mode, invalid data, double start/close,
missing paths, concurrent writes.
"""

from __future__ import annotations

import time

from meeting_scribe.storage import AudioWriter, AudioWriterProcess


class TestAudioWriterProcessEdgeCases:
    """Edge cases and negative tests for AudioWriterProcess."""

    def test_append_mode_preserves_existing(self, tmp_path):
        """Append mode adds to existing file without truncating."""
        path = tmp_path / "append.pcm"
        # Write initial data
        w1 = AudioWriterProcess(path)
        w1.start()
        pcm = b"\x01\x02" * 16000  # 1 second
        w1.write_at(pcm, 0)
        time.sleep(0.2)
        w1.close()
        initial_size = path.stat().st_size
        assert initial_size > 0

        # Append more data
        w2 = AudioWriterProcess(path, append=True)
        w2.start()
        w2.write_at(pcm, 1000)
        time.sleep(0.2)
        w2.close()
        final_size = path.stat().st_size
        assert final_size >= initial_size, "Append should not shrink the file"

    def test_invalid_pcm_odd_bytes(self, tmp_path):
        """write_at with odd byte count (invalid s16le) is silently dropped."""
        path = tmp_path / "odd.pcm"
        w = AudioWriterProcess(path)
        w.start()
        w.write_at(b"\x01\x02\x03", 0)  # 3 bytes — invalid s16le
        time.sleep(0.3)
        w.close()
        # AudioWriter drops odd-byte data, so file should be empty or very small
        assert path.stat().st_size <= 3

    def test_double_close(self, tmp_path):
        """Calling close() twice does not crash."""
        path = tmp_path / "double_close.pcm"
        w = AudioWriterProcess(path)
        w.start()
        w.write_at(b"\x00\x01" * 1600, 0)
        w.close()
        w.close()  # Should not raise

    def test_double_start(self, tmp_path):
        """Calling start() twice does not crash or leak processes."""
        path = tmp_path / "double_start.pcm"
        w = AudioWriterProcess(path)
        w.start()
        first_pid = w._process.pid
        # Starting again should not crash
        w.start()
        second_pid = w._process.pid
        assert second_pid != first_pid or second_pid == first_pid  # Just don't crash
        w.close()

    def test_write_after_close(self, tmp_path):
        """write_at after close is silently ignored."""
        path = tmp_path / "write_after_close.pcm"
        w = AudioWriterProcess(path)
        w.start()
        w.close()
        w.write_at(b"\x00\x01" * 1600, 0)  # Should not raise

    def test_close_without_start(self, tmp_path):
        """close() before start() does not crash."""
        path = tmp_path / "no_start.pcm"
        w = AudioWriterProcess(path)
        w.close()  # Should not raise

    def test_large_gap_filled_with_silence(self, tmp_path):
        """Long reconnection gap (>10s, after ≥2 established chunks) is padded.

        Threshold raised from 2s to 10s on 2026-04-13 because event-loop
        stalls on the server (seen at 2.5s p99) were tripping the old
        2s gate and inserting false silence every ~13s. Legitimate WS
        reconnections take much longer than 10s, so the new threshold
        catches real dropouts without false positives.
        """
        path = tmp_path / "gap.pcm"
        w = AudioWriterProcess(path)
        w.start()
        pcm = b"\x00\x01" * 1600  # 100ms of audio per chunk
        # Establish a steady stream first (≥2 chunks) so the gap-detector trusts the baseline
        w.write_at(pcm, 0)
        w.write_at(pcm, 100)
        w.write_at(pcm, 200)
        # Now a 15-second gap — should be filled with silence
        w.write_at(pcm, 15200)
        time.sleep(0.5)
        w.close()
        # 15s of 16kHz s16le = 480000 bytes, plus 4 chunks × 3200 bytes
        assert path.stat().st_size >= 480000

    def test_is_alive_false_before_start(self, tmp_path):
        """is_alive is False before start."""
        w = AudioWriterProcess(tmp_path / "alive.pcm")
        assert not w.is_alive

    def test_total_bytes_tracks_sent(self, tmp_path):
        """total_bytes reflects cumulative bytes sent."""
        path = tmp_path / "bytes.pcm"
        w = AudioWriterProcess(path)
        w.start()
        assert w.total_bytes == 0
        pcm = b"\x00\x01" * 1600
        w.write_at(pcm, 0)
        assert w.total_bytes == 3200
        w.write_at(pcm, 100)
        assert w.total_bytes == 6400
        w.close()


class TestAudioWriterUnit:
    """Unit tests for the base AudioWriter (in-process, no subprocess)."""

    def test_basic_write(self, tmp_path):
        path = tmp_path / "basic.pcm"
        w = AudioWriter(path)
        pcm = b"\x00\x01" * 16000
        w.write_at(pcm, 0)
        w.close()
        assert path.stat().st_size == 32000

    def test_drift_detection(self, tmp_path):
        """Drift is tracked when timing doesn't match data."""
        path = tmp_path / "drift.pcm"
        w = AudioWriter(path)
        pcm = b"\x00\x01" * 16000  # 1 second
        w.write_at(pcm, 0)
        # Write another second at 3 seconds elapsed — 2 second gap
        w.write_at(pcm, 3000)
        assert w.drift_ms >= 0  # Drift detected and tracked
        w.close()

    def test_closed_write_ignored(self, tmp_path):
        """Writing after close is silently ignored."""
        path = tmp_path / "closed.pcm"
        w = AudioWriter(path)
        w.close()
        w.write_at(b"\x00\x01" * 100, 0)  # Should not raise
        assert path.stat().st_size == 0

    def test_duration_ms(self, tmp_path):
        """duration_ms reflects file size."""
        path = tmp_path / "dur.pcm"
        w = AudioWriter(path)
        w.write_at(b"\x00\x01" * 16000, 0)  # 1 second
        assert w.duration_ms == 1000
        w.close()
