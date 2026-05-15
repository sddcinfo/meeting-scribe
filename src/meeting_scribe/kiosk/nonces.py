"""Single-use, in-process bootstrap nonces for the kiosk auth flow.

The kiosk-runtime is a ``bradlay``-owned process inside the cage
session. It calls ``meeting-scribe kiosk mint-nonce`` (which posts to
``/api/admin/kiosk/mint-nonce`` using the deterministic local admin
password) to receive a 32-byte hex nonce, then chromium (or its own
httpx client) hits ``GET http://127.0.0.1:8444/kiosk-bootstrap?nonce=X``.
The bootstrap endpoint calls :func:`consume_nonce`. On success, the
nonce is removed and a fresh ``scribe_kiosk`` cookie is issued; on
miss/expiry/re-use the endpoint 401s.

In-process state is correct here because the kiosk listener is
loopback-only and the meeting-scribe server is single-process. No
helper daemon round-trip is required, which keeps the cold-start path
free of inter-process socket dances.

Defence in depth: nonces are 32 random bytes (hex-encoded), 60 s TTL,
strictly single-use, and the store is opportunistically GC'd to keep
the working set bounded.
"""

from __future__ import annotations

import secrets
import time
from threading import Lock
from typing import Final

# 60 s gives the launcher generous slack between mint and consume
# without exposing a meaningful replay window on a process whose
# entire surface is loopback-only.
_NONCE_TTL_SECONDS: Final[float] = 60.0

# {nonce -> expires_at (monotonic seconds)}
_nonces: dict[str, float] = {}
_lock = Lock()


def _gc_locked(now: float) -> None:
    """Drop any nonces past their TTL. Caller holds ``_lock``."""
    expired = [n for n, exp in _nonces.items() if exp <= now]
    for n in expired:
        _nonces.pop(n, None)


def mint_nonce() -> str:
    """Mint a fresh single-use bootstrap nonce.

    Returns the hex-encoded nonce. The mint timestamp is recorded
    internally; :func:`consume_nonce` enforces the TTL.
    """
    now = time.monotonic()
    nonce = secrets.token_hex(32)
    with _lock:
        _gc_locked(now)
        _nonces[nonce] = now + _NONCE_TTL_SECONDS
    return nonce


def consume_nonce(nonce: str) -> bool:
    """Atomically validate + remove a nonce.

    Returns True on a single matching, unexpired nonce. Returns False
    for missing, expired, or already-consumed nonces. Callers that
    return True can safely issue the ``scribe_kiosk`` cookie.
    """
    if not nonce or not isinstance(nonce, str):
        return False
    now = time.monotonic()
    with _lock:
        _gc_locked(now)
        exp = _nonces.pop(nonce, None)
        if exp is None:
            return False
        return exp > now


def _debug_pool_size() -> int:
    """Test/debug helper: current live nonce count. Not for app code."""
    with _lock:
        return len(_nonces)
