"""Static HTML page cache loaded at import time.

The cache is a small in-memory dict populated from
``static/*.html`` files. View routes hand the bytes to
``HTMLResponse`` — keeping the read off the request path saves the
disk round-trip on every page load and makes captive-portal probes
O(1) instead of touching the filesystem.

Pulled out of ``server.py`` so the upcoming ``routes/views.py`` and
``hotspot/`` modules can both reach the cache without circling back
through the server module.

Hot-reload: the SIGHUP handler in ``runtime/net.py`` calls
:func:`reload_cached_html` so HTML edits take effect without a full
server restart. Pre-fix, every edit to ``static/index.html`` (or
sibling pages) silently kept the old bytes in memory until the
server was bounced — caused the 2026-05-07 BT-settings-not-showing
gaslight where the served file on disk had the new section but the
running server kept handing out the pre-edit cached version.
"""

from __future__ import annotations

from pathlib import Path

_HTML: dict[str, str] = {}
# Last-loaded directory, captured by ``cache_html`` so
# ``reload_cached_html`` can re-read the same files without needing
# the caller to thread STATIC_DIR through the SIGHUP path.
_LAST_STATIC_DIR: Path | None = None


def cache_html(static_dir: Path) -> None:
    """Populate the module-scope ``_HTML`` cache from static files.

    Called once at server startup with ``STATIC_DIR``. Missing files
    are silently skipped — the route handlers fall back to empty
    strings, which is the correct behavior in tests where the static
    tree may not exist.
    """
    global _LAST_STATIC_DIR
    _LAST_STATIC_DIR = static_dir
    pages = {
        "index": static_dir / "index.html",
        "guest": static_dir / "guest.html",
        "portal": static_dir / "portal.html",
        "reader": static_dir / "reader.html",
        "demo": static_dir / "demo" / "index.html",
        "voice-clone": static_dir / "demo" / "voice-clone.html",
    }
    for key, path in pages.items():
        if path.exists():
            _HTML[key] = path.read_text()


def reload_cached_html() -> bool:
    """Re-read the same static dir that was last passed to ``cache_html``.

    Returns ``True`` if a reload was attempted, ``False`` if the cache
    was never primed (server hasn't started yet — happens in some
    test setups). The SIGHUP handler in ``runtime/net.py`` calls this
    alongside ``runtime_config.reload_from_disk()``.
    """
    if _LAST_STATIC_DIR is None:
        return False
    cache_html(_LAST_STATIC_DIR)
    return True
