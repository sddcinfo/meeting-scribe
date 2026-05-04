"""Admin secret, signed cookies, and single-use HMAC tickets for the terminal.

Three layers of credential:

* :class:`AdminSecretStore` — high-entropy shared secret on disk. Source of
  truth. Generated on first run with mode 0o600; refuses to start if the
  file is world-readable.
* :class:`CookieSigner` — issues/verifies an HMAC-signed ``scribe_admin``
  cookie after the user presents the secret. The cookie is what makes
  subsequent admin browser traffic authenticated.
* :class:`TicketStore` — mints short-lived, single-use nonces that the
  terminal WebSocket handshake requires *in addition* to the cookie.
  Defense against cookie replay from a compromised tab.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import os
import secrets
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final

logger = logging.getLogger(__name__)


# ── HKDF-SHA256 (RFC 5869) using stdlib hmac/sha256 ─────────────────


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """RFC 5869 HKDF-Extract — PRK = HMAC-SHA256(salt, IKM)."""
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-Expand — feed-forward HMAC chain producing ``length`` bytes."""
    okm = b""
    previous = b""
    counter = 1
    while len(okm) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        okm += previous
        counter += 1
    return okm[:length]


def derive_cookie_subkey(master_secret: bytes, boot_session_id: bytes) -> bytes:
    """Derive a per-boot 32-byte HMAC subkey for cookie signing.

    The master admin secret on disk does not change across reboots. To make a
    server restart invalidate every previously-issued admin cookie (the
    "logout-all-on-restart" guarantee), the actual signing key is derived from
    the master secret salted with a per-boot random ``boot_session_id``. When
    the server starts, ``state.boot_session_id`` is regenerated; the subkey
    rotates; the previous subkey's HMACs no longer verify.
    """
    prk = _hkdf_extract(salt=boot_session_id, ikm=master_secret)
    return _hkdf_expand(prk, info=b"scribe-cookie-v36", length=32)


# ── Admin secret on disk ──────────────────────────────────────────


DEFAULT_SECRET_PATH: Final[Path] = Path.home() / ".config" / "meeting-scribe" / "admin-secret"


@dataclass
class AdminSecretStore:
    """Load or create the shared admin secret on disk.

    The file is always mode 0o600. If we find it with looser perms we
    refuse to start — that's a strong signal something is wrong (the user
    committed it, copied it around, or a permissive umask leaked it).
    """

    path: Path
    secret: bytes

    @classmethod
    def load_or_create(cls, path: Path | None = None) -> AdminSecretStore:
        actual_path = path or _env_path() or DEFAULT_SECRET_PATH
        if actual_path.exists():
            cls._enforce_perms(actual_path)
            data = actual_path.read_bytes().strip()
            if not data:
                raise RuntimeError(f"admin secret file is empty: {actual_path}")
            return cls(path=actual_path, secret=data)
        actual_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw = secrets.token_hex(32).encode()
        # Write with mode 0o600 atomically. Per-PID tmp filename so
        # concurrent imports (pytest-xdist workers) don't race on the
        # same path: worker A finishes os.replace, worker B's tmp is
        # already gone, second os.replace dies with FileNotFoundError.
        tmp = actual_path.with_suffix(actual_path.suffix + f".tmp.{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)
        try:
            os.replace(tmp, actual_path)
        except FileNotFoundError:
            # A racing worker beat us to it. Drop our tmp and use theirs.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            if actual_path.exists():
                cls._enforce_perms(actual_path)
                data = actual_path.read_bytes().strip()
                if data:
                    return cls(path=actual_path, secret=data)
            raise
        # If a racing worker also won, our tmp may have been replaced into
        # actual_path by their os.replace before ours. Either way, the
        # value at actual_path is what's authoritative — re-read it.
        if actual_path.exists():
            data = actual_path.read_bytes().strip()
            if data:
                raw = data
        os.chmod(actual_path, 0o600)
        logger.info(
            "generated admin secret at %s (mode 600) — required for terminal bootstrap", actual_path
        )
        return cls(path=actual_path, secret=raw)

    @staticmethod
    def _enforce_perms(path: Path) -> None:
        mode = path.stat().st_mode & 0o777
        # Owner-only: 0o600 or 0o400 both acceptable (read-only hardening is fine)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise RuntimeError(
                f"admin secret file {path} has mode {oct(mode)}; "
                f"expected 0o600 (owner-only). Refusing to start. "
                f"Fix with: chmod 600 {path}"
            )

    def verify(self, candidate: str) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.strip().encode(), self.secret)


def _env_path() -> Path | None:
    raw = os.environ.get("SCRIBE_ADMIN_SECRET_FILE")
    return Path(raw) if raw else None


# ── Signed cookie ────────────────────────────────────────────────


COOKIE_NAME: Final[str] = "scribe_admin"


# Cookie format identifier — written into the HMAC info field so an attacker
# can't take a v1.0 cookie from one app and replay it against another that
# shares the same secret.
_COOKIE_HMAC_INFO: Final[bytes] = b"scribe-admin-cookie-v36"


@dataclass
class CookieSigner:
    """Per-boot-keyed HMAC-signed cookie: ``<issued_unix>.<session_id>.<hex-hmac>``.

    The HMAC is computed over ``f"{issued_unix}|{session_id}"`` using a
    per-boot subkey (see :func:`derive_cookie_subkey`). Two complementary
    invalidation primitives are layered:

    * Restart rotates ``boot_session_id`` → previous cookies fail HMAC verify.
    * Logout adds the cookie's ``session_id`` to ``state._revoked_sessions``;
      :meth:`verify` consults the optional ``is_revoked`` callable on every
      check.

    Old-format cookies (``<issued>.<sig>``) issued by the pre-cutover signer
    are treated as "no cookie" — :meth:`verify` returns ``False`` without
    raising, the user lands on guest UI, signs in again, and gets a fresh
    new-format cookie. Migration is self-healing.
    """

    secret: bytes
    max_age_seconds: int = 7 * 24 * 3600
    cookie_name: str = COOKIE_NAME
    is_revoked: Callable[[str], bool] | None = None

    def _hmac(self, payload: bytes) -> str:
        """Compute the per-boot-keyed HMAC over ``payload``."""
        return hmac.new(self.secret, _COOKIE_HMAC_INFO + payload, hashlib.sha256).hexdigest()

    def issue(self, now: float | None = None, *, session_id: str | None = None) -> str:
        """Mint a cookie. Generates a fresh ``session_id`` if not provided.

        Callers that need the ``session_id`` separately (to register active
        admin WebSockets in ``state._admin_ws_by_session``) get it via
        :func:`decode_verified_cookie` after issuing.
        """
        issued = str(int(now if now is not None else time.time()))
        sid = session_id or secrets.token_hex(16)
        payload = f"{issued}|{sid}".encode()
        sig = self._hmac(payload)
        return f"{issued}.{sid}.{sig}"

    def verify(self, value: str | None, now: float | None = None) -> bool:
        """Pure-boolean validity check — preserved signature for every caller.

        Returns ``False`` (never raises) for any of:
          * empty / malformed value
          * old-format two-part cookie (graceful pre-cutover migration)
          * HMAC mismatch (wrong secret or rotated boot subkey)
          * expired / future-dated
          * revoked ``session_id`` (per the ``is_revoked`` callback)
        """
        ok, _, _ = self._decode(value, now)
        return ok

    def _decode(
        self,
        value: str | None,
        now: float | None,
    ) -> tuple[bool, str | None, int | None]:
        """Internal implementation shared by :meth:`verify` and the
        :func:`decode_verified_cookie` helper.

        Returns ``(ok, session_id, issued_at)``. ``session_id`` and
        ``issued_at`` are ``None`` whenever ``ok`` is ``False``.
        """
        if not value:
            return False, None, None
        parts = value.split(".")
        if len(parts) != 3:
            # Old-format pre-cutover cookies have exactly 2 parts; treat as
            # "no cookie" so the user re-auths and migrates self-healingly.
            return False, None, None
        issued_s, sid, sig = parts
        if not issued_s or not sid or not sig:
            return False, None, None
        try:
            issued = int(issued_s)
        except ValueError:
            return False, None, None
        current = now if now is not None else time.time()
        if current - issued > self.max_age_seconds:
            return False, None, None
        if issued - current > 5:  # clock skew tolerance, no future-dated tokens
            return False, None, None
        expected = self._hmac(f"{issued_s}|{sid}".encode())
        if not hmac.compare_digest(sig, expected):
            return False, None, None
        if self.is_revoked is not None and self.is_revoked(sid):
            return False, None, None
        return True, sid, issued


def decode_verified_cookie(
    signer: CookieSigner,
    value: str | None,
    now: float | None = None,
) -> tuple[bool, str | None, int | None]:
    """Verify a cookie and return ``(ok, session_id, issued_at)`` for callers
    that need the metadata (logout endpoint, WS registration).

    Internally calls :meth:`CookieSigner._decode`. On any failure returns
    ``(False, None, None)`` — the same fail-closed contract as
    :meth:`CookieSigner.verify`.
    """
    return signer._decode(value, now)


# ── Session revocation + admin-WS bookkeeping ──────────────────────


def revoke_session(
    session_id: str,
    *,
    expiry_epoch: float,
    revoked_sessions: dict[str, float],
) -> None:
    """Mark ``session_id`` as revoked until ``expiry_epoch``.

    Called from logout + re-auth flows to invalidate the cookie's
    session_id. The TTL prevents the dict from growing unbounded — once
    the entry's expiry passes, the cookie's natural max_age has also
    passed and the entry can be pruned by ``prune_revoked``.
    """
    revoked_sessions[session_id] = expiry_epoch


def prune_revoked(
    revoked_sessions: dict[str, float],
    *,
    now: float | None = None,
) -> int:
    """Drop revocation entries whose expiry has passed.

    Returns the number of entries pruned. Called periodically from the
    cookie/ticket sweeper; idempotent and safe to call from any task.
    """
    current = now if now is not None else time.time()
    stale = [sid for sid, exp in revoked_sessions.items() if exp < current]
    for sid in stale:
        revoked_sessions.pop(sid, None)
    return len(stale)


@contextlib.asynccontextmanager
async def register_admin_ws(session_id: str, ws, by_session: dict[str, set]):
    """Track ``ws`` under ``session_id`` for the duration of the with-block.

    On entry the WS joins ``by_session[session_id]`` (set is created on
    first use). On exit — normal close, exception, network error, server
    shutdown — the WS is removed; if the resulting set is empty, the
    ``session_id`` key is popped so the dict stays sized to active
    sessions.

    Logout (from the admin web/UI) iterates every WS in the logging-out
    session's set and closes each one with WS code 1008 ``revoked``;
    re-auth on the same browser does the same for the prior session_id
    before minting the fresh cookie.

    Usage::

        async with register_admin_ws(sid, websocket,
                                     state._admin_ws_by_session):
            await main_handler(websocket)
    """
    bucket = by_session.setdefault(session_id, set())
    bucket.add(ws)
    try:
        yield
    finally:
        bucket = by_session.get(session_id)
        if bucket is not None:
            bucket.discard(ws)
            if not bucket:
                by_session.pop(session_id, None)


# ── Single-use HMAC tickets ───────────────────────────────────────


TICKET_TTL_DEFAULT: Final[float] = 60.0


@dataclass
class TicketStore:
    """Mint/consume one-shot HMAC tickets for the terminal WS handshake."""

    secret: bytes
    ttl_seconds: float = TICKET_TTL_DEFAULT
    _unused: dict[str, float] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _sweep_interval_seconds: float = 30.0

    async def mint(self) -> str:
        """Return a single-use ticket of the form ``<rand64>.<hmac64>``."""
        random_half = secrets.token_hex(32)
        mac = hmac.new(self.secret, random_half.encode(), hashlib.sha256).hexdigest()
        ticket = f"{random_half}.{mac}"
        async with self._lock:
            self._unused[ticket] = time.monotonic() + self.ttl_seconds
        return ticket

    async def consume(self, ticket: str) -> bool:
        """Validate + consume. Returns True exactly once per valid ticket."""
        if not ticket or "." not in ticket:
            return False
        # Constant-time HMAC re-check first — catches tampered/fabricated tickets
        random_half, _, sig = ticket.partition(".")
        if not random_half or not sig:
            return False
        expected = hmac.new(self.secret, random_half.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        async with self._lock:
            expiry = self._unused.pop(ticket, None)
            if expiry is None:
                return False  # unknown / already consumed
            if time.monotonic() > expiry:
                return False  # expired
        return True

    async def sweep(self) -> None:
        """Long-running background task: drop expired tickets every 30s."""
        while True:
            try:
                await asyncio.sleep(self._sweep_interval_seconds)
                now = time.monotonic()
                async with self._lock:
                    stale = [t for t, exp in self._unused.items() if exp < now]
                    for t in stale:
                        self._unused.pop(t, None)
                    if stale:
                        logger.debug("ticket sweep: dropped %d expired", len(stale))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ticket sweep iteration failed; continuing")

    # Synchronous introspection — safe to call from tests, not from a hot path.
    def _size(self) -> int:
        return len(self._unused)
