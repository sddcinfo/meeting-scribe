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
from typing import Final

logger = logging.getLogger(__name__)


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


@dataclass
class CookieSigner:
    """Stateless HMAC-signed cookie: ``<issued_unix>.<hex-hmac>``."""

    secret: bytes
    max_age_seconds: int = 7 * 24 * 3600
    cookie_name: str = COOKIE_NAME

    def issue(self, now: float | None = None) -> str:
        issued = str(int(now if now is not None else time.time()))
        sig = hmac.new(self.secret, issued.encode(), hashlib.sha256).hexdigest()
        return f"{issued}.{sig}"

    def verify(self, value: str | None, now: float | None = None) -> bool:
        if not value or "." not in value:
            return False
        issued_s, sig = value.split(".", 1)
        try:
            issued = int(issued_s)
        except ValueError:
            return False
        current = now if now is not None else time.time()
        if current - issued > self.max_age_seconds:
            return False
        if issued - current > 5:  # clock skew tolerance but no future-dated tokens
            return False
        expected = hmac.new(self.secret, issued_s.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)


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
