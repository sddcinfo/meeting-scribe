"""Mutable runtime config holder — the rollback axis for the 3.6 migration.

The static ``ServerConfig`` dataclass in ``config.py`` is loaded once at
process start and threaded through constructors.  That's fine for most
settings, but the Qwen3.6 migration (Phase 7 of the plan) needs a
handful of knobs — translation endpoint URLs, slide JSON-schema toggle —
to flip without a process restart.  During cutover the operator flips
``translate_url`` from the current prod model to the new one, hits the
reload handler, and the next translation request picks up the new URL.

Design:
  * The holder is a process-local dict guarded by a lock.
  * Persistence lives at ``$XDG_DATA_HOME/meeting-scribe/runtime-config.json``
    so the CLI (out-of-process) can write it and the server (in-process)
    can reload it via SIGHUP.
  * ``reload_from_disk()`` replaces the live dict atomically.
  * Every consumer reads fresh per-request — no caching in constructors.

Only four knobs live here by design:
    translate_url          — live-translation vLLM endpoint
    slide_translate_url    — slide-pipeline vLLM endpoint (usually same)
    slide_use_json_schema  — Phase 4b response_format flag
    slide_stats_dir        — where slide-translation-stats.jsonl lands;
                             eval runs redirect into the privacy-gated
                             shadow root under ~/.local/share/sddc/...

Everything else stays on ``ServerConfig`` — adding settings here is a
conscious decision, not a convenience, because each knob here is an
additional path that must survive hot-reload testing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Allowlist of keys — writes to any other key are rejected so typos
# ("translation_url" vs "translate_url") fail loudly instead of
# silently creating a no-op knob.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "translate_url",
        "slide_translate_url",
        "slide_use_json_schema",
        "slide_stats_dir",
        # Integer — rolling context window size for the refinement
        # worker's translate calls (follow-shortly-after path).
        # Hot-reloadable so operators can run the sweep harness
        # against a single process without a restart between legs.
        # See ServerConfig.refinement_context_window_segments for the
        # full rationale.
        "refinement_context_window_segments",
    }
)


def _default_path() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "meeting-scribe" / "runtime-config.json"


class _RuntimeConfig:
    """Thread-safe singleton for hot-reloadable knobs."""

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._path = path or _default_path()
        self._values: dict[str, Any] = {}
        self.reload_from_disk()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        with self._lock:
            return self._path

    def set_path(self, path: Path) -> None:
        """Redirect the persistence file (used by tests)."""
        with self._lock:
            self._path = path
            self.reload_from_disk()

    def reload_from_disk(self) -> None:
        """Re-read the persistence file and replace the live dict.

        Missing file → empty dict (first boot / never-been-set).
        Malformed file → empty dict + WARNING (don't crash the server).
        """
        with self._lock:
            if not self._path.is_file():
                self._values = {}
                return
            try:
                raw = json.loads(self._path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "runtime-config: failed to read %s (%s); using empty defaults",
                    self._path,
                    exc,
                )
                self._values = {}
                return
            if not isinstance(raw, dict):
                logger.warning(
                    "runtime-config: %s is not a JSON object; using empty defaults",
                    self._path,
                )
                self._values = {}
                return
            # Only keep allowlisted keys so a stale file from an earlier
            # version of the server can't inject surprise settings.
            self._values = {k: v for k, v in raw.items() if k in _ALLOWED_KEYS}

    def _persist(self) -> None:
        """Atomic write — assumes caller holds _lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._values, indent=2) + "\n")
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the live value for *key*, falling back to *default*.

        Per-request callers pass the static config value as *default* so
        an unset knob transparently falls through to the ``ServerConfig``
        URL loaded at startup.
        """
        with self._lock:
            return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key not in _ALLOWED_KEYS:
            msg = f"runtime-config: unknown key {key!r}; allowed: {sorted(_ALLOWED_KEYS)}"
            raise KeyError(msg)
        with self._lock:
            self._values[key] = value
            self._persist()

    def unset(self, key: str) -> None:
        """Drop a key so the next read falls back to the static default."""
        if key not in _ALLOWED_KEYS:
            msg = f"runtime-config: unknown key {key!r}; allowed: {sorted(_ALLOWED_KEYS)}"
            raise KeyError(msg)
        with self._lock:
            self._values.pop(key, None)
            self._persist()

    def as_dict(self) -> dict[str, Any]:
        """Return a snapshot of the current values."""
        with self._lock:
            return dict(self._values)


# Module-level singleton.  Tests that need a clean state call
# ``install_singleton()`` with an isolated path.
_SINGLETON: _RuntimeConfig | None = None
_SINGLETON_LOCK = threading.Lock()


def instance() -> _RuntimeConfig:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = _RuntimeConfig()
        return _SINGLETON


def install_singleton(rc: _RuntimeConfig) -> None:
    """Replace the process-wide singleton (test-only hook)."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = rc


# Convenience re-exports for per-request callers (keeps import blocks tidy):
#   from meeting_scribe.runtime_config import get, reload_from_disk


def get(key: str, default: Any = None) -> Any:
    return instance().get(key, default)


def reload_from_disk() -> None:
    instance().reload_from_disk()
