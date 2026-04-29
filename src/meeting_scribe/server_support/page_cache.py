"""Static HTML page cache loaded at import time.

The cache is a small in-memory dict populated from
``static/*.html`` files. View routes hand the bytes to
``HTMLResponse`` — keeping the read off the request path saves the
disk round-trip on every page load and makes captive-portal probes
O(1) instead of touching the filesystem.

Pulled out of ``server.py`` so the upcoming ``routes/views.py`` and
``hotspot/`` modules can both reach the cache without circling back
through the server module.
"""

from __future__ import annotations

from pathlib import Path

_HTML: dict[str, str] = {}


def cache_html(static_dir: Path) -> None:
    """Populate the module-scope ``_HTML`` cache from static files.

    Called once at server startup with ``STATIC_DIR``. Missing files
    are silently skipped — the route handlers fall back to empty
    strings, which is the correct behavior in tests where the static
    tree may not exist.
    """
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
