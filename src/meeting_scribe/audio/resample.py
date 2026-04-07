"""Audio resampling and wire-format parsing.

Handles conversion from device sample rates (44.1 kHz / 48 kHz) to
the 16 kHz mono expected by ASR backends. Parses the binary wire
format sent by the browser AudioWorklet and detects stream epoch
changes and sample-offset gaps.

Wire format (per WebSocket message):
    [4B uint32 sample_rate]
    [8B uint64 sample_offset]
    [4B uint32 stream_epoch]
    [... Float32 PCM payload]
"""

from __future__ import annotations

import logging
import struct

import numpy as np
import torch
import torchaudio.transforms  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Wire format header: uint32 sample_rate, uint64 sample_offset, uint32 stream_epoch
_HEADER_FORMAT = "<IQI"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # 4 + 8 + 4 = 16 bytes


def parse_audio_message(data: bytes) -> tuple[int, int, int, np.ndarray]:
    """Parse a binary audio message from the browser AudioWorklet.

    Args:
        data: Raw bytes received via WebSocket.

    Returns:
        Tuple of (sample_rate, sample_offset, stream_epoch, audio) where
        audio is a Float32 numpy array of PCM samples.

    Raises:
        ValueError: If the message is too short to contain a valid header.
    """
    if len(data) < _HEADER_SIZE:
        raise ValueError(
            f"Audio message too short: {len(data)} bytes (minimum {_HEADER_SIZE} for header)"
        )

    sample_rate, sample_offset, stream_epoch = struct.unpack_from(_HEADER_FORMAT, data)
    payload = data[_HEADER_SIZE:]
    audio = np.frombuffer(payload, dtype=np.float32)

    return sample_rate, sample_offset, stream_epoch, audio


class Resampler:
    """Resamples audio from device rate to 16 kHz mono.

    Tracks stream epochs and sample offsets to detect AudioContext
    restarts and sample gaps. When a gap exceeds the tolerance, a
    silence marker is inserted so downstream consumers can handle
    discontinuities.

    Args:
        target_rate: Output sample rate (default 16000).
        gap_tolerance_ms: Maximum acceptable gap before inserting a
            silence marker, in milliseconds (default 100).
    """

    def __init__(
        self,
        target_rate: int = 16000,
        gap_tolerance_ms: float = 100.0,
    ) -> None:
        self._target_rate = target_rate
        self._gap_tolerance_ms = gap_tolerance_ms

        # Cache torchaudio resamplers keyed by source rate.
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

        # Stream tracking
        self._current_epoch: int | None = None
        self._expected_offset: int | None = None
        self._current_source_rate: int | None = None

    def resample(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int = 16000,
    ) -> np.ndarray:
        """Resample audio from source_rate to target_rate.

        Args:
            audio: Float32 mono PCM samples.
            source_rate: Input sample rate (e.g., 44100, 48000).
            target_rate: Output sample rate (default 16000).

        Returns:
            Float32 numpy array at the target rate.
        """
        if source_rate == target_rate:
            return audio

        resampler = self._get_resampler(source_rate, target_rate)
        tensor = torch.from_numpy(audio.copy()).unsqueeze(0)  # (1, samples)
        resampled = resampler(tensor).squeeze(0).numpy()
        return resampled

    def process_message(self, data: bytes) -> tuple[np.ndarray, bool, bool]:
        """Parse a wire-format message, resample, and detect discontinuities.

        Args:
            data: Raw WebSocket binary message.

        Returns:
            Tuple of (resampled_audio, epoch_changed, gap_detected) where:
            - resampled_audio: Float32 numpy array at target rate.
            - epoch_changed: True if the stream epoch changed (new AudioContext).
            - gap_detected: True if a sample offset gap exceeded tolerance.
        """
        sample_rate, sample_offset, stream_epoch, audio = parse_audio_message(data)

        epoch_changed = False
        gap_detected = False

        # Detect epoch change (new AudioContext).
        if self._current_epoch is None or stream_epoch != self._current_epoch:
            if self._current_epoch is not None:
                logger.info(
                    "Stream epoch changed: %d -> %d (AudioContext restart)",
                    self._current_epoch,
                    stream_epoch,
                )
            self._current_epoch = stream_epoch
            self._expected_offset = sample_offset + len(audio)
            self._current_source_rate = sample_rate
            epoch_changed = True
        else:
            # Detect sample offset gap.
            if self._expected_offset is not None:
                gap_samples = sample_offset - self._expected_offset
                if gap_samples > 0:
                    gap_ms = (gap_samples / sample_rate) * 1000.0
                    if gap_ms > self._gap_tolerance_ms:
                        logger.warning(
                            "Sample gap detected: %d samples (%.1f ms) at offset %d",
                            gap_samples,
                            gap_ms,
                            sample_offset,
                        )
                        gap_detected = True

            self._expected_offset = sample_offset + len(audio)
            self._current_source_rate = sample_rate

        resampled = self.resample(audio, sample_rate, self._target_rate)
        return resampled, epoch_changed, gap_detected

    def reset(self) -> None:
        """Reset stream tracking state (e.g., between meetings)."""
        self._current_epoch = None
        self._expected_offset = None
        self._current_source_rate = None
        logger.debug("Resampler state reset")

    def _get_resampler(self, source_rate: int, target_rate: int) -> torchaudio.transforms.Resample:
        """Get or create a cached torchaudio Resample transform."""
        # Use source_rate as dict key since target_rate is almost always fixed,
        # but include both in the actual cache key for correctness.
        cache_key = source_rate * 100_000 + target_rate
        if cache_key not in self._resamplers:
            self._resamplers[cache_key] = torchaudio.transforms.Resample(
                orig_freq=source_rate,
                new_freq=target_rate,
            )
            logger.debug("Created resampler: %d Hz -> %d Hz", source_rate, target_rate)
        return self._resamplers[cache_key]
