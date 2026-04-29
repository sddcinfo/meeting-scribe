"""Percentile helpers for the rolling-deque metrics windows.

Used by the ``Metrics`` class (still in ``server.py``) to render
``p50/p95/p99/sample_count`` summaries from rolling windows of latency
samples. The "nearest-rank" method is intentional — it's stable on
small windows where ``statistics.quantiles`` produces noise [P1-6-i1].

Pulled out of ``server.py`` so future status / diag route modules can
reuse the same percentile shape without dragging the server graph in.
"""

from __future__ import annotations

_MIN_SAMPLES_FOR_PCT = 10


def _percentile(samples: list[float], q: float, *, presorted: bool = False) -> float | None:
    """Return the qth percentile (q in [0, 1]) or None for tiny windows.

    Guards against ``statistics.quantiles`` edge cases on empty / 1-sample
    windows [P1-6-i1]. Under ``_MIN_SAMPLES_FOR_PCT`` samples → None.
    """
    n = len(samples)
    if n < _MIN_SAMPLES_FOR_PCT:
        return None
    srt = samples if presorted else sorted(samples)
    # Nearest-rank method — simple, stable, no interpolation noise on
    # small windows.
    idx = min(n - 1, max(0, round(q * (n - 1))))
    return round(srt[idx], 2)


def _percentile_dict(samples) -> dict:
    """Serialise a rolling deque into a p50/p95/p99/sample_count dict.

    Returns null percentiles + actual ``sample_count`` when the window
    has fewer than ``_MIN_SAMPLES_FOR_PCT`` items. [P1-6-i1]
    """
    arr = list(samples)
    sample_count = len(arr)
    if sample_count < _MIN_SAMPLES_FOR_PCT:
        return {"p50": None, "p95": None, "p99": None, "sample_count": sample_count}
    srt = sorted(arr)
    return {
        "p50": _percentile(srt, 0.50, presorted=True),
        "p95": _percentile(srt, 0.95, presorted=True),
        "p99": _percentile(srt, 0.99, presorted=True),
        "sample_count": sample_count,
    }
