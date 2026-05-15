"""Detect a capture device's native (preferred) sample rate.

Pulling audio at a higher rate than the device natively supports forces
either firmware-side resampling (telephony speakerphones do this poorly
— their AEC/NR/AGC is tuned for 16 kHz) or PipeWire-side resampling on
top of our own downstream 48 kHz → 16 kHz step. Both hurt quality and
add latency. The Poly Sync 20-M survives because it natively exposes
48 kHz; the Dell SP325 advertises **both 16 kHz and 48 kHz** for
capture, but with on-device DSP tuned for 16 kHz — running at 48 kHz
audibly degrades the signal we hand ASR.

This module reads ``/proc/asound/cardN/streamM`` (world-readable, no
sudo needed) for the source ALSA device behind a PipeWire node and
picks the rate ≥ 16000 closest to 16000. That's the right answer for
ASR pipelines (Qwen3-ASR wants 16 kHz mono int16 and our resampler
target is 16000): if the device supports 16 kHz natively we capture
at 16 kHz and the resampler is a no-op; otherwise we capture at the
device's native rate and let the resampler convert.

The detector is best-effort. On parse failure it logs and returns the
caller's default — usually 48000 — so a misconfigured device cannot
break ``server_mic`` startup.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Target sample rate downstream ASR expects. We prefer to pull native if
# the device supports this rate or higher; the resampler in
# ``audio/resample.py`` defaults to 16000 too.
_ASR_TARGET_RATE = 16000

# Hard upper bound — if the device only supports rates wildly above
# 48 kHz we still cap to 48 kHz to keep the rest of the pipeline sane.
_MAX_CAPTURE_RATE = 48000


def _wpctl_alsa_card_dev(node_name: str) -> tuple[int, int] | None:
    """Resolve a PipeWire node name to its ALSA card+device numbers.

    Uses ``pw-cli info <node_name>`` to pull ``api.alsa.pcm.card`` +
    ``api.alsa.pcm.device``. ``wpctl inspect`` only accepts numeric IDs,
    but ``pw-cli info`` accepts the node-name form ``server_mic`` is
    configured with. Returns ``None`` if pw-cli is missing, the node
    doesn't exist, or it doesn't have ALSA props.
    """
    binary = shutil.which("pw-cli")
    if binary is None:
        logger.debug("pw-cli not on PATH; cannot resolve %s", node_name)
        return None
    try:
        proc = subprocess.run(
            [binary, "info", node_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("pw-cli info %s failed: %r", node_name, e)
        return None
    if proc.returncode != 0:
        return None
    card_m = re.search(r'api\.alsa\.pcm\.card\s*=\s*"?(\d+)', proc.stdout)
    dev_m = re.search(r'api\.alsa\.pcm\.device\s*=\s*"?(\d+)', proc.stdout)
    if not card_m:
        return None
    # pw-cli may omit device when it's the conventional 0; default safely.
    device = int(dev_m.group(1)) if dev_m else 0
    return int(card_m.group(1)), device


def _parse_rates_from_stream_file(text: str) -> list[int]:
    """Pull every ``Rates:`` line from a ``/proc/asound/.../streamN`` file.

    Returns the union of all rate sets encountered under any Capture
    section (USB devices often expose several altsettings, each with
    its own rate list).
    """
    out: set[int] = set()
    in_capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Capture:"):
            in_capture = True
            continue
        if stripped.startswith("Playback:"):
            in_capture = False
            continue
        if not in_capture:
            continue
        if stripped.startswith("Rates:"):
            for token in stripped[len("Rates:") :].split(","):
                token = token.strip()
                if token.isdigit():
                    out.add(int(token))
    return sorted(out)


def supported_capture_rates(card: int) -> list[int]:
    """Return every rate the given ALSA card advertises for capture.

    The file is at ``/proc/asound/cardN/stream0`` (USB audio devices
    almost always have a single PCM stream; pick stream0). World-readable,
    so no sudo needed.
    """
    candidates = [Path(f"/proc/asound/card{card}/stream{i}") for i in range(0, 4)]
    rates: set[int] = set()
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError as e:
            logger.debug("read %s failed: %r", path, e)
            continue
        rates.update(_parse_rates_from_stream_file(text))
    return sorted(rates)


def pick_rate(supported: list[int], default: int = 48000) -> int:
    """Pick the device's "best" rate given the supported set.

    Preference order:
      1. ``_ASR_TARGET_RATE`` (16000) — perfect, resampler becomes a no-op.
      2. The smallest supported rate ≥ ``_ASR_TARGET_RATE`` and ≤ ``_MAX_CAPTURE_RATE``.
      3. ``default`` if no supported rate qualifies (defensive fallback).

    Smaller-rate-first matters because the device's on-board DSP is
    typically tuned for the lowest rate it advertises (telephony 8/16 kHz).
    Going higher means more firmware resampling and worse quality.
    """
    if not supported:
        return default
    if _ASR_TARGET_RATE in supported:
        return _ASR_TARGET_RATE
    eligible = [r for r in supported if _ASR_TARGET_RATE <= r <= _MAX_CAPTURE_RATE]
    if eligible:
        return min(eligible)
    return default


def detect_capture_rate(node_name: str | None, *, default: int = 48000) -> int:
    """Best-effort: pick the right pw-record ``--rate`` for this source.

    Args:
        node_name: PipeWire node name (e.g.
            ``alsa_input.usb-Dell_Inc._Dell_SP325_..._pro-input-0``).
            Passing ``None`` or an unresolvable name returns ``default``.
        default: Fallback rate when detection fails (typically 48000 to
            preserve historical behavior).

    Returns:
        The picked sample rate in Hz.
    """
    if not node_name:
        return default
    resolved = _wpctl_alsa_card_dev(node_name)
    if resolved is None:
        logger.info(
            "native-rate: could not resolve %s to ALSA card; using default %d",
            node_name,
            default,
        )
        return default
    card, _device = resolved
    rates = supported_capture_rates(card)
    picked = pick_rate(rates, default=default)
    logger.info(
        "native-rate: %s → card%d rates=%s picked=%d",
        node_name,
        card,
        rates,
        picked,
    )
    return picked
