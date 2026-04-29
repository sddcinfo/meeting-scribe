"""PyAV-backed fragmented-MP4 / AAC-LC encoder for MSE streaming.

This is the encoder that drives the first-class audio path for hotspot
guest listeners. It takes raw float32 mono PCM at any reasonable sample
rate (the TTS backend produces 24 kHz; the ASR-based passthrough path
produces 16 kHz) and emits:

  1. A single init segment (ftyp + moov boxes) on construction, cached
     on ``self._init_bytes`` and retrievable via ``init_segment()``.
  2. A sequence of media fragments (moof + mdat boxes) via
     ``encode(pcm, source_rate)`` — one fragment per internal flush.

The constants ``WINNING_MUX_FLAGS`` and ``ACCUMULATION_THRESHOLD_MS`` were
empirically determined by the Phase 0 Spike B probe on 2026-04-15 against
PyAV 17.0.0 + bundled ffmpeg.

All cadence variability lives inside this class. Callers just call
``encode()`` and send whatever non-empty bytes come back. Empty returns
are a normal "still accumulating" signal, not a fault. Callers must NOT
interpret empty returns as an error and must NOT recreate the encoder in
response — the accumulation buffer is the entire point of the contract.
"""

from __future__ import annotations

import io
import logging
from typing import Final

import av
import numpy as np

logger = logging.getLogger(__name__)

# ── Empirically-determined constants (Phase 0 Spike B, 2026-04-15) ──

# MSE output: 48 kHz mono AAC-LC in fragmented MP4. 48 kHz is the native
# rate AAC is tuned for; we resample everything to 48 kHz on ingest.
SAMPLE_RATE_OUT: Final[int] = 48000
CHANNELS: Final[int] = 1
DEFAULT_BITRATE: Final[int] = 96000

# Muxer configuration that produces one fragment per AAC frame instead of
# coalescing the whole stream into a single blob on close. The
# ``frag_duration=20000`` (microseconds) hint is critical — without it,
# the default ``empty_moov+default_base_moof+frag_keyframe`` only emits
# two fragments total (init + final monolithic blob). See Spike B output.
WINNING_MUX_FLAGS: Final[dict[str, str]] = {
    "movflags": "empty_moov+default_base_moof+frag_keyframe+separate_moof",
    "frag_duration": "20000",
}

# Minimum PCM duration (in ms) that must accumulate before the encoder
# reliably emits any output. Spike B showed first emission lands at
# ~50 ms cumulative input (priming frame + 4 subsequent 10 ms frames).
# That's AAC's natural 1024-sample frame (21.33 ms at 48 kHz) plus one
# full encoder frame of internal buffering. We set the threshold to
# 60 ms for safety margin — anything above that reliably emits.
ACCUMULATION_THRESHOLD_MS: Final[int] = 60
_THRESHOLD_SAMPLES: Final[int] = (SAMPLE_RATE_OUT * ACCUMULATION_THRESHOLD_MS) // 1000

# Priming is fed in small chunks (10 ms each) up to a hard cap so the
# construction path mirrors the validated spike pattern. Feeding a
# single large frame doesn't produce output for some PyAV/ffmpeg builds
# because the encoder only re-inspects its internal accumulation at
# frame boundaries.
_PRIME_CHUNK_MS: Final[int] = 10
_PRIME_CHUNK_SAMPLES: Final[int] = (SAMPLE_RATE_OUT * _PRIME_CHUNK_MS) // 1000
_PRIME_MAX_MS: Final[int] = 200  # absolute cap; init MUST emerge within this window

# MIME type advertised to MSE clients via ``MediaSource.isTypeSupported``.
# AAC-LC profile = mp4a.40.2. Matches the codec we open on the stream below.
MEDIA_MIME: Final[str] = 'audio/mp4; codecs="mp4a.40.2"'


class Fmp4AacEncoder:
    """Stateful fragmented-MP4/AAC-LC encoder with internal PCM accumulation.

    Lifecycle:
        enc = Fmp4AacEncoder()
        init = enc.init_segment()           # bytes, cached
        # ... for every audio delivery:
        frag = enc.encode(pcm, source_rate)
        if frag:
            ws.send_bytes(frag)
        # ... on listener disconnect:
        tail = enc.flush()                  # flush pending
        if tail:
            ws.send_bytes(tail)
        enc.close()

    Thread-safety: none. One instance per WebSocket connection. Never
    share across listeners.
    """

    def __init__(self, bitrate: int = DEFAULT_BITRATE) -> None:
        self._bitrate = bitrate
        self._buf: io.BytesIO = io.BytesIO()
        self._container = av.open(self._buf, mode="w", format="mp4", options=WINNING_MUX_FLAGS)
        self._stream = self._container.add_stream("aac", rate=SAMPLE_RATE_OUT)
        self._stream.bit_rate = bitrate
        self._stream.layout = "mono"
        self._resampler: av.AudioResampler | None = None
        self._resampler_rate: int | None = None
        # Monotonic AAC-sample PTS across every encode call.
        self._pts_samples: int = 0
        # Internal accumulation buffer at SAMPLE_RATE_OUT mono float32.
        self._pcm_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._closed: bool = False
        # Prime the encoder and capture the init segment. This feeds
        # _THRESHOLD_SAMPLES of silence, drains the output, and splits
        # at the first moof box so the silent priming fragment is
        # discarded.
        self._init_bytes: bytes = self._prime_and_capture_init()

    # ── Public API ──

    def init_segment(self) -> bytes:
        """Return the cached ISO BMFF init segment (ftyp + moov)."""
        return self._init_bytes

    def encode(self, pcm: np.ndarray, source_sample_rate: int) -> bytes:
        """Accumulate PCM internally; emit a fragment when threshold is crossed.

        ``pcm`` is float32 mono audio at ``source_sample_rate``. Return
        value is the concatenated bytes of any fragments (moof + mdat)
        produced for this call, or ``b""`` if the accumulation buffer is
        still below the threshold (this is normal, not a fault).

        The caller MUST treat ``b""`` as "still accumulating" and MUST
        NOT recreate the encoder in response.
        """
        if self._closed:
            raise RuntimeError("encode() called on a closed Fmp4AacEncoder")
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if len(pcm) == 0:
            return b""

        # Resample to 48 kHz if needed.
        if source_sample_rate != SAMPLE_RATE_OUT:
            pcm_48k = self._resample_to_out(pcm, source_sample_rate)
        else:
            pcm_48k = pcm

        # Append to the accumulation buffer.
        self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm_48k])

        # Below threshold → normal accumulation, return empty.
        if len(self._pcm_buffer) < _THRESHOLD_SAMPLES:
            return b""

        # Flush the whole buffer through the encoder and drain the muxer.
        self._feed_encoder(self._pcm_buffer)
        self._pcm_buffer = np.zeros(0, dtype=np.float32)
        return self._drain_buf()

    def flush(self) -> bytes:
        """Force-encode pending PCM and drain any remaining packets.

        Called on listener disconnect so the tail of the last utterance
        is not lost. Also used in unit tests. Idempotent — a second
        ``flush()`` with nothing pending returns ``b""``.
        """
        if self._closed:
            return b""
        # Feed whatever's in the accumulation buffer, even below threshold.
        if len(self._pcm_buffer) > 0:
            self._feed_encoder(self._pcm_buffer)
            self._pcm_buffer = np.zeros(0, dtype=np.float32)
        # Flush the encoder's own internal state (packets held for future
        # frames). Flushing a stream is signalled by encoding a None frame.
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        except (av.error.EOFError, ValueError):
            # Some ffmpeg builds raise EOFError after the stream is
            # flushed once — treat as idempotent no-op.
            pass
        return self._drain_buf()

    def close(self) -> None:
        """Flush pending bytes and close the underlying container.

        Idempotent — a second ``close()`` is a no-op. After ``close()``,
        any subsequent ``encode()`` call raises RuntimeError.
        """
        if self._closed:
            return
        try:
            # Discard the return value — nothing is listening anymore by
            # the time close() fires.
            self.flush()
        except Exception as e:
            logger.debug("fmp4 encoder flush during close: %r", e)
        try:
            self._container.close()
        except Exception as e:
            logger.debug("fmp4 encoder container close: %r", e)
        self._closed = True

    # ── Internals ──

    def _prime_and_capture_init(self) -> bytes:
        """Feed silence in small chunks until the muxer emits ftyp + moov.

        The spike-validated pattern is feeding N × 10 ms AudioFrames. A
        single large frame doesn't reliably trigger emission for some
        PyAV/ffmpeg builds because the encoder only re-inspects its
        internal accumulation at frame boundaries. We loop feeding
        silence chunks until the muxer produces output or we hit the
        _PRIME_MAX_MS cap.
        """
        silence_chunk = np.zeros(_PRIME_CHUNK_SAMPLES, dtype=np.float32)
        out = b""
        fed_ms = 0
        while fed_ms < _PRIME_MAX_MS:
            self._feed_encoder(silence_chunk)
            fed_ms += _PRIME_CHUNK_MS
            out = self._drain_buf()
            if out:
                break
        if not out:
            raise RuntimeError(
                "fmp4 encoder prime: muxer emitted 0 bytes after feeding "
                f"{fed_ms} ms of silence in {_PRIME_CHUNK_MS} ms chunks — "
                "WINNING_MUX_FLAGS may be incorrect for this PyAV/ffmpeg build"
            )
        moof_idx = out.find(b"moof")
        if moof_idx < 0:
            # Muxer returned init without a trailing fragment — the whole
            # blob is ftyp+moov. Validate and return.
            if b"ftyp" not in out or b"moov" not in out:
                raise RuntimeError(
                    f"fmp4 encoder prime: no moof and missing ftyp/moov "
                    f"(first 32 bytes: {out[:32].hex()})"
                )
            return out
        # moof is preceded by a 4-byte size prefix — the box starts 4
        # bytes earlier. Everything before that prefix is init.
        box_start = moof_idx - 4
        if box_start <= 0:
            raise RuntimeError(
                f"fmp4 encoder prime: moof appears at offset {moof_idx} "
                "with no preceding init bytes"
            )
        init = bytes(out[:box_start])
        if b"ftyp" not in init or b"moov" not in init:
            raise RuntimeError(
                f"fmp4 encoder prime: init segment missing ftyp/moov "
                f"(first 32 bytes: {init[:32].hex()})"
            )
        return init

    def _resample_to_out(self, pcm: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample float32 mono PCM from source_rate to SAMPLE_RATE_OUT."""
        if self._resampler is None or self._resampler_rate != source_rate:
            self._resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=SAMPLE_RATE_OUT,
            )
            self._resampler_rate = source_rate
        in_frame = av.AudioFrame(format="fltp", layout="mono", samples=len(pcm))
        in_frame.sample_rate = source_rate
        in_frame.pts = None
        in_frame.planes[0].update(pcm.astype(np.float32).tobytes())
        out_samples: list[np.ndarray] = []
        for out_frame in self._resampler.resample(in_frame):
            arr = out_frame.to_ndarray()  # shape (channels, samples) for planar
            if arr.ndim == 2:
                out_samples.append(arr[0].astype(np.float32))
            else:
                out_samples.append(arr.astype(np.float32))
        if not out_samples:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out_samples)

    def _feed_encoder(self, pcm_48k_mono: np.ndarray) -> None:
        """Wrap 48 kHz mono float32 PCM in an AudioFrame and encode+mux."""
        if len(pcm_48k_mono) == 0:
            return
        frame = av.AudioFrame(format="fltp", layout="mono", samples=len(pcm_48k_mono))
        frame.sample_rate = SAMPLE_RATE_OUT
        frame.pts = self._pts_samples
        frame.planes[0].update(pcm_48k_mono.astype(np.float32).tobytes())
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._pts_samples += len(pcm_48k_mono)

    def _drain_buf(self) -> bytes:
        """Consume all bytes written to the BytesIO and reset it."""
        self._buf.seek(0)
        data = self._buf.read()
        self._buf.seek(0)
        self._buf.truncate()
        return data
