"""Microphone liveness probe shared by meeting-start preflight + demo gate.

Codex review (rounds 2 and 3) caught two subtle holes that this module
plugs:

1. A timestamp set BEFORE the probe started (prior session, background
   room noise from a different meeting) must not count as proof the mic
   is alive right now. The probe-local epoch contract is: reset
   ``state.last_nonzero_audio_ts = 0.0`` at the top of the probe, then
   require strict ``last_nonzero_audio_ts > probe_epoch_ts`` to pass.

2. ``state.server_mic_active = True`` alone is NOT proof. That flag
   means "the pw-record reader task is running", which says nothing
   about the audio it's pulling. A flapping USB connection, a kernel
   ALSA buffer hang, a mic with hardware mute on — all of these leave
   the capture happy while the data is all-zero. Always require fresh
   evidence of non-zero samples.

The probe is reused by:
  * ``routes/meeting_lifecycle._meeting_start_preflight`` (Phase 5)
  * ``preflight.run(mode="demo")`` (Phase 6 — the comprehensive
    pre-demo gate)

Sharing one helper means the contract cannot drift between the two
surfaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

from meeting_scribe.runtime import state

logger = logging.getLogger(__name__)


# Floor matches ``ws.audio_input._AUDIO_LIVENESS_FLOOR`` so the
# liveness contract is symmetric: the inbound bumper and the probe
# read agree on what "non-zero sample" means.
_LIVENESS_FLOOR = 1e-4

# Time the probe will wait for the running capture to produce a
# non-zero frame. Codex review caught that 250 ms is too short for
# cold USB; 1.5 s covers the device-warmup variance on the Poly Sync
# 20-M without making the preflight feel sluggish.
_PROBE_WAIT_S = 1.5

# Polling cadence inside the wait loop. Short enough to early-exit on
# the first frame; not so short the loop is CPU-busy.
_PROBE_POLL_S = 0.05


_LivenessKind = Literal[
    "no_mic_configured",
    "unresolved_route",
    "node_missing",
    "samples_all_zero",
    "capture_probe_timed_out",
    "ok",
]


@dataclass
class LivenessResult:
    """Outcome of a single mic-readiness probe.

    ``ok=True`` means the probe observed a non-zero sample in the
    wait window. Anything else carries an actionable ``reason`` and a
    short ``detail`` the UI can render verbatim.
    """

    ok: bool
    reason: _LivenessKind
    detail: str
    probe_epoch_ts: float
    observed_at: float | None


def _fmt_node_for_human(node: str) -> str:
    """Trim long PipeWire node names to a recognizable tail for UI text."""
    if not node:
        return "(unconfigured)"
    if len(node) <= 72:
        return node
    return node[:32] + "…" + node[-32:]


async def probe_mic_liveness(*, timeout_s: float = _PROBE_WAIT_S) -> LivenessResult:
    """Run a probe-local epoch liveness check against the configured mic.

    Contract:

    1. ``mic_node`` must be configured. If not, return
       ``no_mic_configured``.
    2. ``state.audio_route_status`` must be ``"ok"``. The mic
       reconcile sets ``ambiguous`` / ``unresolved`` / ``capture_failed``
       on the failure paths; the meeting-start preflight refuses to
       start until reconcile clears them.
    3. Reset ``state.last_nonzero_audio_ts = 0.0`` and capture
       ``probe_epoch_ts = monotonic()``. Pre-reset values, no matter
       how recent, never count.
    4. If ``state.server_mic_active`` is True, wait up to ``timeout_s``
       for the running capture to bump ``last_nonzero_audio_ts`` to a
       value strictly greater than ``probe_epoch_ts``. The bumper in
       ``ws/audio_input._handle_audio`` only fires on frames with
       ``peak > _LIVENESS_FLOOR``.
    5. Server mic NOT active: there's no in-process source to wait on;
       return ``samples_all_zero`` immediately (the operator is expected
       to enable mic_active first; the preflight will re-run on retry).
       A standalone live capture via ``pw-record`` would compete with
       any reconciler/listener attempts to (re)bind the device, so we
       don't run one here.

    Returns :class:`LivenessResult`; never raises on the happy path.
    """
    mic_node = ""
    settings_route_status = "ok"
    try:
        # Lazy import — settings/state can be imported at module load
        # but reading them touches the file system.
        from meeting_scribe.audio.audio_routing import (
            SETTINGS_AUDIO_MEETING_MIC_NODE,
        )
        from meeting_scribe.server_support.settings_store import _load_settings_override

        settings = _load_settings_override()
        mic_node = str(settings.get(SETTINGS_AUDIO_MEETING_MIC_NODE) or "").strip()
    except Exception:
        logger.exception("probe_mic_liveness: failed to read mic settings")

    settings_route_status = getattr(state, "audio_route_status", "ok")
    probe_epoch_ts = time.monotonic()

    if not mic_node:
        return LivenessResult(
            ok=False,
            reason="no_mic_configured",
            detail=(
                "No microphone is selected. Open the audio routing admin panel and pick a source."
            ),
            probe_epoch_ts=probe_epoch_ts,
            observed_at=None,
        )

    if settings_route_status in {"ambiguous", "unresolved", "capture_failed"}:
        return LivenessResult(
            ok=False,
            reason="unresolved_route",
            detail=(
                f"Microphone routing status is '{settings_route_status}'. "
                "Resolve the audio-routing notification before starting a meeting."
            ),
            probe_epoch_ts=probe_epoch_ts,
            observed_at=None,
        )

    # Probe-local epoch — pre-reset values never count.
    state.last_nonzero_audio_ts = 0.0

    server_mic_active = bool(getattr(state, "server_mic_active", False))
    if not server_mic_active:
        return LivenessResult(
            ok=False,
            reason="samples_all_zero",
            detail=(
                f"Server-side mic capture is not running (target={_fmt_node_for_human(mic_node)}). "
                "Enable mic_active in the audio routing admin panel."
            ),
            probe_epoch_ts=probe_epoch_ts,
            observed_at=None,
        )

    deadline = probe_epoch_ts + timeout_s
    while True:
        observed = float(getattr(state, "last_nonzero_audio_ts", 0.0))
        if observed > probe_epoch_ts:
            return LivenessResult(
                ok=True,
                reason="ok",
                detail="mic delivered a non-zero frame inside the probe window",
                probe_epoch_ts=probe_epoch_ts,
                observed_at=observed,
            )
        if time.monotonic() >= deadline:
            return LivenessResult(
                ok=False,
                reason="samples_all_zero",
                detail=(
                    f"Mic capture is running on {_fmt_node_for_human(mic_node)} "
                    "but every sample in the last "
                    f"{timeout_s:.1f}s was zero. Check the device cable, "
                    "gain, and hardware mute."
                ),
                probe_epoch_ts=probe_epoch_ts,
                observed_at=None,
            )
        await asyncio.sleep(_PROBE_POLL_S)
