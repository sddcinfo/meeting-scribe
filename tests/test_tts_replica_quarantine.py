"""Tests for per-replica failure tracking + quarantine in Qwen3TTSBackend.

Covers the 2026-04-15 bug: one replica goes CUDA-corrupt (HTTP 500 on
every synth request while /health still returns 200), interleaved
successes from the healthy replica kept the global failure counter
bouncing to 0, and the dead replica never got quarantined.

These tests drive the failure/success bookkeeping directly and verify
that the round-robin skips quarantined replicas, respects the re-probe
cooldown, and falls back to a best-effort attempt if every replica is
quarantined at once.
"""

from __future__ import annotations

import time

from meeting_scribe.backends.tts_qwen3 import Qwen3TTSBackend

URL_A = "http://localhost:8002"
URL_B = "http://localhost:8012"


def _backend() -> Qwen3TTSBackend:
    b = Qwen3TTSBackend(vllm_url=f"{URL_A},{URL_B}")
    # start() actually probes /health; skip it — we only want the
    # `_urls` list populated for pool-picking unit tests.
    b._mode = "vllm"
    return b


def test_next_url_alternates_when_all_healthy():
    """Two healthy replicas round-robin alternately."""
    b = _backend()
    picks = [b._next_url() for _ in range(4)]
    assert set(picks) == {URL_A, URL_B}
    # Exactly two of each over 4 picks.
    assert picks.count(URL_A) == 2
    assert picks.count(URL_B) == 2


def test_per_url_failure_counter_quarantines_after_threshold():
    """Three consecutive failures on the same URL quarantines it."""
    b = _backend()
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_B, "HTTP 500")
    assert URL_B in b._quarantined
    assert URL_A not in b._quarantined


def test_next_url_skips_quarantined_replica():
    """Quarantined URL is skipped by round-robin."""
    b = _backend()
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_B, "HTTP 500")
    picks = [b._next_url() for _ in range(6)]
    # All 6 picks should be URL_A; URL_B is quarantined and not yet
    # eligible for a reprobe (< _QUARANTINE_REPROBE_S).
    assert picks == [URL_A] * 6


def test_success_on_other_replica_does_NOT_clear_quarantine():
    """A success on URL_A must not unquarantine URL_B.

    This is the exact failure mode from the 2026-04-15 bug: the old
    global counter reset on every success regardless of origin. The new
    per-URL counter only clears when the successful URL is the one
    being tracked.
    """
    b = _backend()
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_B, "HTTP 500")
    assert URL_B in b._quarantined
    # Many successes on the OTHER URL
    for _ in range(10):
        b._mark_url_success(URL_A)
    # URL_B is still quarantined
    assert URL_B in b._quarantined
    # And still skipped
    assert b._next_url() == URL_A


def test_reprobe_after_cooldown_restores_url_on_success():
    """After _QUARANTINE_REPROBE_S, a probe is issued; success clears state."""
    b = _backend()
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_B, "HTTP 500")
    # Back-date the quarantine timestamp to simulate cooldown expiry.
    b._quarantined[URL_B] = time.monotonic() - (Qwen3TTSBackend._QUARANTINE_REPROBE_S + 1.0)
    # URL_B is now eligible again for a probe; the next round-robin
    # cycle must include it.
    picks = set()
    for _ in range(6):
        picks.add(b._next_url())
    assert URL_B in picks, "reprobe-eligible URL should be offered"
    # A successful response clears everything
    b._mark_url_success(URL_B)
    assert URL_B not in b._quarantined
    assert b._url_failures.get(URL_B, 0) == 0


def test_all_quarantined_still_returns_a_best_effort_url():
    """If EVERY replica is quarantined, return the oldest instead of None.

    A silent None would drop the whole audio segment; a best-effort
    attempt either finds the replica has recovered or re-confirms it's
    still dead (which is no worse than the status quo).
    """
    b = _backend()
    now = time.monotonic()
    b._quarantined[URL_A] = now - 1.0  # newer
    b._quarantined[URL_B] = now - 10.0  # older
    # Cooldown hasn't expired for either. _next_url should pick the
    # oldest-quarantined URL (URL_B) as the probe candidate.
    assert b._next_url() == URL_B


def test_empty_pool_returns_none():
    """A backend with no URLs returns None and does not raise."""
    b = Qwen3TTSBackend(vllm_url=None)
    b._urls = []
    assert b._next_url() is None


def test_single_replica_never_quarantined_indefinitely():
    """With only one URL configured, even hitting the threshold still
    lets _next_url return that URL — otherwise we'd break single-replica
    deployments."""
    b = Qwen3TTSBackend(vllm_url=URL_A)
    b._mode = "vllm"
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_A, "HTTP 500")
    # Everyone quarantined → fall-back best-effort returns the URL.
    assert b._next_url() == URL_A


# ── Edge cases (Phase 5) ────────────────────────────────────────────


def test_reprobe_failure_re_quarantines():
    """After reprobe window, pick quarantined URL, fail again → re-quarantined
    with a fresh timestamp."""
    b = _backend()
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_B, "HTTP 500")
    assert URL_B in b._quarantined
    old_ts = b._quarantined[URL_B]

    # Back-date to make reprobe eligible
    b._quarantined[URL_B] = time.monotonic() - (Qwen3TTSBackend._QUARANTINE_REPROBE_S + 1.0)
    # The reprobe-eligible URL should be offered
    picks = set()
    for _ in range(6):
        picks.add(b._next_url())
    assert URL_B in picks

    # Fail again → re-quarantined with fresh timestamp
    b._mark_url_failure(URL_B, "HTTP 500 again")
    assert URL_B in b._quarantined
    assert b._quarantined[URL_B] > old_ts


def test_failure_count_resets_on_success():
    """2 failures (below threshold) + 1 success on same URL → counter reset to 0."""
    b = _backend()
    b._mark_url_failure(URL_B, "HTTP 500")
    b._mark_url_failure(URL_B, "HTTP 500")
    assert b._url_failures[URL_B] == 2
    assert URL_B not in b._quarantined  # below threshold

    b._mark_url_success(URL_B)
    assert b._url_failures.get(URL_B, 0) == 0
    assert URL_B not in b._quarantined


def test_sequential_next_url_both_skip_quarantined():
    """Two sequential _next_url() calls with URL_B quarantined → both return URL_A.

    Validates that the round-robin cursor advances correctly when
    skipping quarantined URLs. No artificial threading — matches the
    single-threaded production access pattern.
    """
    b = _backend()
    for _ in range(Qwen3TTSBackend.MAX_CONSECUTIVE_FAILURES):
        b._mark_url_failure(URL_B, "HTTP 500")

    pick1 = b._next_url()
    pick2 = b._next_url()
    assert pick1 == URL_A
    assert pick2 == URL_A
