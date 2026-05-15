"""Persistent admin notifications surfaced via the ``.meeting-banner`` SPA component.

Producers (``audio_routing.reconcile_audio_routing`` is the first one) call
:func:`put_notification` whenever they encounter a condition the operator
should see — a stale-mic auto-rebind, an ambiguous device fingerprint, a
transient-disconnect that left the mic unresolved, a capture process that
refused to start. Entries are keyed by ``kind`` so a producer can be called
repeatedly without spamming the UI: the latest payload for each ``kind``
replaces the prior un-dismissed row.

The admin SPA polls ``/api/status`` for the ``admin_notifications`` field
and renders each entry into the existing ``.meeting-banner`` element. The
``POST /api/admin/notifications/{kind}/dismiss`` endpoint flips the row's
``dismissed_at`` so subsequent ``/api/status`` polls hide it (the row stays
in storage so the dismissed state survives a reload).

Persistence lives in the regular settings.json blob under
``SETTINGS_KEY``; ``state.pending_admin_notifications`` mirrors the
persisted view for in-process readers. The two are kept in sync via this
module — no direct mutation outside.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from meeting_scribe.runtime import state
from meeting_scribe.server_support.settings_store import (
    _load_settings_override,
    _save_settings_override,
)

logger = logging.getLogger(__name__)


SETTINGS_KEY = "admin_notifications"


def _read_persisted() -> dict[str, dict[str, Any]]:
    raw = _load_settings_override().get(SETTINGS_KEY) or {}
    if not isinstance(raw, dict):
        return {}
    # Filter to dict entries; tolerate hand-edits without crashing.
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _write_persisted(entries: dict[str, dict[str, Any]]) -> None:
    _save_settings_override({SETTINGS_KEY: entries})


def load_into_state() -> None:
    """Mirror persisted notifications onto ``state.pending_admin_notifications``.

    Called once during lifespan boot so the in-process view matches what's
    on disk before any producer runs.
    """
    state.pending_admin_notifications = _read_persisted()


def put_notification(kind: str, **fields: Any) -> None:
    """Upsert a notification keyed on ``kind``.

    Idempotent: replaces any prior un-dismissed entry for the same ``kind``.
    Always resets ``dismissed_at`` to ``None`` so a recurrence after dismiss
    re-surfaces the banner. Producers pass arbitrary keyword fields
    (``from`` / ``to`` / ``candidates`` / ``detail`` / etc.) — the
    front-end renderer reads them by ``kind`` and ignores unknown fields.
    """
    entries = _read_persisted()
    entries[kind] = {
        "kind": kind,
        "created_at": time.time(),
        "dismissed_at": None,
        **fields,
    }
    _write_persisted(entries)
    state.pending_admin_notifications = entries
    logger.info("admin_notifications: put kind=%s fields=%s", kind, list(fields))


def dismiss_if_present(kind: str) -> bool:
    """Mark the entry for ``kind`` dismissed if it exists and isn't already dismissed.

    Returns True when an entry transitioned to dismissed; False when there
    was nothing to dismiss. Called from producer success paths to clear
    stale failure-state notifications on recovery, and from the dismiss
    HTTP endpoint when the operator clicks the banner button.
    """
    entries = _read_persisted()
    entry = entries.get(kind)
    if entry is None or entry.get("dismissed_at") is not None:
        return False
    entry["dismissed_at"] = time.time()
    _write_persisted(entries)
    state.pending_admin_notifications = entries
    logger.info("admin_notifications: dismissed kind=%s", kind)
    return True


def active_notifications() -> list[dict[str, Any]]:
    """Return un-dismissed entries, newest first.

    Used by ``/api/status`` to populate the ``admin_notifications`` field.
    Reads from in-memory state (which load_into_state primes at boot and
    put/dismiss keeps current); a settings.json hand-edit between polls
    won't show up until the next put/dismiss, which is acceptable.
    """
    entries = state.pending_admin_notifications or {}
    active = [e for e in entries.values() if e.get("dismissed_at") is None]
    active.sort(key=lambda e: e.get("created_at", 0.0), reverse=True)
    return active
