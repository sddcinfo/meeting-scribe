"""Tests for Fmp4AacEncoder — contract-based, no fragment-count assumptions.

Uses PyAV itself as the decoder (guaranteed to understand its own output),
not soundfile, which has no reliable fMP4/AAC support across libsndfile
builds. See plan iteration-2 P2-1 for the rationale.
"""

from __future__ import annotations

import io

import av
import numpy as np
import pytest

from meeting_scribe.backends.mse_encoder import (
    ACCUMULATION_THRESHOLD_MS,
    MEDIA_MIME,
    SAMPLE_RATE_OUT,
    Fmp4AacEncoder,
)


def _sine(duration_s: float, rate: int = SAMPLE_RATE_OUT, freq: float = 440.0) -> np.ndarray:
    """Float32 mono sine wave in [-0.8, +0.8]."""
    n = int(duration_s * rate)
    t = np.arange(n, dtype=np.float32) / rate
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _walk_iso_bmff(data: bytes) -> list[tuple[int, bytes]]:
    """Walk top-level ISO BMFF boxes. Return [(size, fourcc_type), ...].

    Standard layout: [4B BE size][4B type][(size-8)B body]. A size of 1
    means the box uses a 64-bit ``largesize`` field after the type, and a
    size of 0 means "box extends to end of file". We handle both.
    """
    out: list[tuple[int, bytes]] = []
    pos = 0
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "big")
        btype = data[pos + 4:pos + 8]
        if size == 1:
            if pos + 16 > len(data):
                break
            size = int.from_bytes(data[pos + 8:pos + 16], "big")
        elif size == 0:
            size = len(data) - pos
        if size < 8 or pos + size > len(data):
            break
        out.append((size, btype))
        pos += size
    return out


def test_media_mime_is_aac_lc() -> None:
    """Sanity: the codec string we advertise to MSE is mp4a.40.2 (AAC-LC)."""
    assert MEDIA_MIME == 'audio/mp4; codecs="mp4a.40.2"'


def test_init_structure_is_ftyp_then_moov() -> None:
    """init_segment() parses cleanly as ISO BMFF: ftyp then moov, nothing else."""
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        assert len(init) > 0, "empty init segment"
        boxes = _walk_iso_bmff(init)
        assert len(boxes) >= 2, f"expected at least 2 top-level boxes, got {boxes}"
        assert boxes[0][1] == b"ftyp", f"first box should be ftyp, got {boxes[0][1]!r}"
        assert boxes[1][1] == b"moov", f"second box should be moov, got {boxes[1][1]!r}"
        # Verify no moof leaked into the init segment.
        assert b"moof" not in init, "moof found inside init segment (split failed)"
    finally:
        enc.close()


def test_under_threshold_returns_empty() -> None:
    """A single 10 ms chunk (below ACCUMULATION_THRESHOLD_MS) returns b''."""
    enc = Fmp4AacEncoder()
    try:
        pcm = _sine(0.010, rate=SAMPLE_RATE_OUT)
        assert ACCUMULATION_THRESHOLD_MS > 10, (
            "test assumes threshold > 10 ms; revisit if Spike B changes"
        )
        result = enc.encode(pcm, SAMPLE_RATE_OUT)
        assert result == b"", (
            f"expected empty below threshold, got {len(result)} bytes"
        )
    finally:
        enc.close()


def test_over_threshold_emits_fragment() -> None:
    """A single 500 ms chunk (well above threshold) emits a moof+mdat fragment."""
    enc = Fmp4AacEncoder()
    try:
        pcm = _sine(0.500, rate=SAMPLE_RATE_OUT)
        result = enc.encode(pcm, SAMPLE_RATE_OUT)
        assert len(result) > 0, "expected non-empty fragment from 500 ms input"
        assert b"moof" in result, "fragment missing moof box"
        assert b"mdat" in result, "fragment missing mdat box"
    finally:
        enc.close()


def _decode_stream(init: bytes, fragments: list[bytes]) -> tuple[float, float]:
    """Concatenate init + fragments, decode via PyAV, return (duration_s, peak)."""
    blob = init + b"".join(fragments)
    if not blob:
        return 0.0, 0.0
    decode_in = av.open(io.BytesIO(blob))
    total_samples = 0
    peak = 0.0
    try:
        for frame in decode_in.decode(audio=0):
            arr = frame.to_ndarray()
            # arr can be (channels, samples) for planar or (1, samples) for mono.
            if arr.ndim == 2:
                total_samples += arr.shape[1]
                peak = max(peak, float(np.abs(arr).max()))
            else:
                total_samples += arr.shape[0]
                peak = max(peak, float(np.abs(arr).max()))
    finally:
        decode_in.close()
    return total_samples / SAMPLE_RATE_OUT, peak


def test_no_audio_lost_48k_many_small_chunks() -> None:
    """Feed 1 s of 48 kHz sine in 50 × 20 ms chunks, flush, assert duration invariant.

    The key invariant: total audio in == total audio out (± resampler/encoder
    tolerance). Does NOT assert fragment count — a valid implementation
    can emit any number of fragments depending on internal batching.
    """
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        fragments: list[bytes] = []
        for _ in range(50):
            pcm = _sine(0.020, rate=SAMPLE_RATE_OUT)
            frag = enc.encode(pcm, SAMPLE_RATE_OUT)
            if frag:
                fragments.append(frag)
        tail = enc.flush()
        if tail:
            fragments.append(tail)
        duration_s, peak = _decode_stream(init, fragments)
    finally:
        enc.close()

    # The encoder may lose a tiny amount of audio at the very tail due to
    # AAC frame-alignment; allow ±10% tolerance on duration.
    assert 0.9 <= duration_s <= 1.1, (
        f"expected ~1.0 s decoded, got {duration_s:.3f} s "
        f"from {len(fragments)} fragment(s)"
    )
    assert peak >= 0.3, f"peak amplitude {peak:.3f} < 0.3 (expected ~0.8 sine)"


def test_no_audio_lost_16k_resampled() -> None:
    """Repeat with 16 kHz source to exercise the resampler path."""
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        fragments: list[bytes] = []
        # 1 s of 16 kHz sine in 20 × 50 ms chunks
        for _ in range(20):
            pcm = _sine(0.050, rate=16000, freq=440.0)
            frag = enc.encode(pcm, 16000)
            if frag:
                fragments.append(frag)
        tail = enc.flush()
        if tail:
            fragments.append(tail)
        duration_s, peak = _decode_stream(init, fragments)
    finally:
        enc.close()

    assert 0.9 <= duration_s <= 1.1, (
        f"expected ~1.0 s decoded from 16 kHz source, got {duration_s:.3f} s"
    )
    assert peak >= 0.3, f"peak amplitude {peak:.3f} < 0.3"


def test_flush_returns_residual_below_threshold() -> None:
    """Feed a sub-threshold chunk, encode returns empty, flush returns residual."""
    enc = Fmp4AacEncoder()
    try:
        # 30 ms at 48 kHz — well below the 40 ms threshold
        pcm = _sine(0.030, rate=SAMPLE_RATE_OUT)
        assert enc.encode(pcm, SAMPLE_RATE_OUT) == b"", "expected empty below threshold"
        tail = enc.flush()
        assert len(tail) > 0, "flush should force-emit the residual"
        # Decode the init + tail and assert we got audible duration back.
        init = enc.init_segment()
        duration_s, _peak = _decode_stream(init, [tail])
    finally:
        enc.close()

    # 30 ms input can round to slightly less after AAC frame alignment; at
    # minimum we should see at least one AAC frame (~20 ms) of output.
    assert duration_s >= 0.015, (
        f"flush returned too little audio: {duration_s:.3f} s"
    )


def test_close_is_idempotent() -> None:
    """close() on a fresh encoder doesn't raise; a second close() is a no-op."""
    enc = Fmp4AacEncoder()
    enc.close()  # first close
    enc.close()  # second close, should be silent


def test_encode_after_close_raises() -> None:
    """encode() after close() raises RuntimeError (not a silent swallow)."""
    enc = Fmp4AacEncoder()
    enc.close()
    with pytest.raises(RuntimeError, match="closed"):
        enc.encode(_sine(0.100, rate=SAMPLE_RATE_OUT), SAMPLE_RATE_OUT)


def test_empty_pcm_returns_empty() -> None:
    """encode() with a zero-length PCM array returns b'' cleanly."""
    enc = Fmp4AacEncoder()
    try:
        result = enc.encode(np.zeros(0, dtype=np.float32), SAMPLE_RATE_OUT)
        assert result == b""
    finally:
        enc.close()


# ── Edge cases (Phase 2) ───────────────────────────────────────────


def test_boundary_exact_threshold_emits() -> None:
    """Exactly ACCUMULATION_THRESHOLD_MS worth of samples triggers emission."""
    enc = Fmp4AacEncoder()
    try:
        # Exactly 60ms = 2880 samples at 48kHz
        n = (SAMPLE_RATE_OUT * ACCUMULATION_THRESHOLD_MS) // 1000
        pcm = _sine(ACCUMULATION_THRESHOLD_MS / 1000.0, rate=SAMPLE_RATE_OUT)
        assert len(pcm) == n, f"expected {n} samples, got {len(pcm)}"
        result = enc.encode(pcm, SAMPLE_RATE_OUT)
        assert len(result) > 0, (
            f"exactly {ACCUMULATION_THRESHOLD_MS}ms should emit a fragment"
        )
        assert b"moof" in result
    finally:
        enc.close()


def test_boundary_one_below_threshold_empty() -> None:
    """One sample below threshold returns empty."""
    enc = Fmp4AacEncoder()
    try:
        n = (SAMPLE_RATE_OUT * ACCUMULATION_THRESHOLD_MS) // 1000 - 1
        pcm = np.zeros(n, dtype=np.float32)
        result = enc.encode(pcm, SAMPLE_RATE_OUT)
        assert result == b"", (
            f"expected empty for {n} samples (1 below threshold), "
            f"got {len(result)} bytes"
        )
    finally:
        enc.close()


def test_mixed_sample_rates_interleaved() -> None:
    """Alternating 16kHz and 24kHz inputs: resampler recreated, total duration correct."""
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        fragments: list[bytes] = []
        # 10 rounds: alternating 50ms at 16kHz and 50ms at 24kHz = 1s total
        for i in range(10):
            rate = 16000 if i % 2 == 0 else 24000
            pcm = _sine(0.100, rate=rate)
            frag = enc.encode(pcm, rate)
            if frag:
                fragments.append(frag)
        tail = enc.flush()
        if tail:
            fragments.append(tail)
        duration_s, peak = _decode_stream(init, fragments)
    finally:
        enc.close()

    assert 0.8 <= duration_s <= 1.2, (
        f"expected ~1.0s from mixed rates, got {duration_s:.3f}s"
    )
    assert peak >= 0.3


def test_flush_between_encodes_no_data_loss() -> None:
    """encode below threshold then flush — all input audio is captured.

    After flush() the AAC stream is finalized (EOF sent to the encoder),
    so no further encode() calls are possible on the same instance. This
    test verifies that sub-threshold audio accumulated in the buffer is
    correctly emitted via flush().
    """
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        fragments: list[bytes] = []

        # Feed 30ms + 30ms + 30ms = 90ms in sub-threshold chunks
        for _ in range(3):
            pcm = _sine(0.030, rate=SAMPLE_RATE_OUT)
            result = enc.encode(pcm, SAMPLE_RATE_OUT)
            if result:
                fragments.append(result)

        # The first 60ms should have emitted (at threshold); remaining
        # 30ms is in the accumulation buffer.
        tail = enc.flush()
        if tail:
            fragments.append(tail)

        assert len(fragments) >= 1, "expected at least one fragment from 90ms of input"
        duration_s, peak = _decode_stream(init, fragments)
    finally:
        enc.close()

    # 90ms input → at least 60ms decoded (AAC frame alignment may trim tail)
    assert 0.05 <= duration_s <= 0.15, (
        f"expected ~0.09s from 90ms input, got {duration_s:.3f}s"
    )
    assert peak >= 0.3


def test_large_single_chunk_10s() -> None:
    """10 seconds in one call produces fragments totalling ~10s decoded."""
    enc = Fmp4AacEncoder()
    try:
        init = enc.init_segment()
        fragments: list[bytes] = []
        pcm = _sine(10.0, rate=SAMPLE_RATE_OUT)
        frag = enc.encode(pcm, SAMPLE_RATE_OUT)
        if frag:
            fragments.append(frag)
        tail = enc.flush()
        if tail:
            fragments.append(tail)
        duration_s, peak = _decode_stream(init, fragments)
    finally:
        enc.close()

    assert 9.5 <= duration_s <= 10.5, (
        f"expected ~10.0s decoded, got {duration_s:.3f}s"
    )
    assert peak >= 0.3


def test_resampler_lazy_creation() -> None:
    """Resampler is None initially, created on first non-48kHz input, reused on same rate."""
    enc = Fmp4AacEncoder()
    try:
        assert enc._resampler is None
        assert enc._resampler_rate is None

        # 48kHz input should NOT create a resampler
        pcm_48 = _sine(0.020, rate=SAMPLE_RATE_OUT)
        enc.encode(pcm_48, SAMPLE_RATE_OUT)
        assert enc._resampler is None

        # 16kHz input should create one
        pcm_16 = _sine(0.020, rate=16000)
        enc.encode(pcm_16, 16000)
        assert enc._resampler is not None
        assert enc._resampler_rate == 16000
        resampler_id = id(enc._resampler)

        # Same rate reuses existing resampler
        enc.encode(pcm_16, 16000)
        assert id(enc._resampler) == resampler_id

        # Different rate creates a new one
        pcm_24 = _sine(0.020, rate=24000)
        enc.encode(pcm_24, 24000)
        assert enc._resampler_rate == 24000
        assert id(enc._resampler) != resampler_id
    finally:
        enc.close()
