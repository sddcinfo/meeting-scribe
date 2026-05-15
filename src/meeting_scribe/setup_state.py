"""Setup-mode state machine for the v1.0 first-touch wizard.

Two-state persistent layout under ``/var/lib/meeting-scribe/`` (env
override ``SCRIBE_STATE_DIR`` for tests):

* ``setup-pending/`` — staging while the wizard is in flight.
  Contains ``created-at``, ``session-id``, ``claim-ip``, the two
  plaintext credentials (``admin-password.txt``,
  ``guest-pin.txt``), and the ``acked`` marker. Created on
  ``GET /setup`` (single-page flow); plaintext is held until the
  operator taps Finish, at which point HMACs are written to the
  LIVE store and the plaintext is deleted.
* ``setup-complete`` — final marker. Written when the wizard
  finishes. From that point the lifespan no longer routes
  through the wizard redirect.

Live HMAC files (NOT inside ``setup-pending/``):
  * ``admin-password-hmac``  — admin sign-in
  * ``guest-pin-hmac``       — 4-digit guest PIN
  * ``auth-version``         — int that participates in
    ``CookieSigner`` HKDF info; bumping invalidates every cookie

Every mutation acquires ``setup.lock`` via ``fcntl.LOCK_EX``;
reads acquire ``LOCK_SH``.

The v1.0 simplification removed the open→locked AP rotation and
the printed-card / fingerprint-pin / bootstrap-secret apparatus.
The recovery-code flow was also dropped — admin password is
deterministically derived from the appliance pin and the operator
can recover it any time via ``meeting-scribe factory-reset``.
The AP stays OWE forever; auth happens at the application layer
via the guest PIN or admin password (see ``routes/guest_auth.py``).
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hmac as _hmac
import logging
import os
import secrets
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────


SETUP_EXPIRY_SECONDS: int = 1800  # 30 min
LOCK_TIMEOUT_S: float = 30.0


def _state_dir() -> Path:
    override = os.environ.get("SCRIBE_STATE_DIR", "").strip()
    return Path(override) if override else Path("/var/lib/meeting-scribe")


def _setup_pending_dir() -> Path:
    return _state_dir() / "setup-pending"


def _setup_complete_marker() -> Path:
    return _state_dir() / "setup-complete"


def _setup_lock_path() -> Path:
    return _state_dir() / "setup.lock"


def _auth_version_path() -> Path:
    return _state_dir() / "auth-version"


def _admin_password_hmac_path() -> Path:
    return _state_dir() / "admin-password-hmac"


def _guest_pin_hmac_path() -> Path:
    return _state_dir() / "guest-pin-hmac"


# ── Locking ──────────────────────────────────────────────────────


@contextlib.contextmanager
def _ex_lock(timeout: float = LOCK_TIMEOUT_S):
    """``LOCK_EX`` on ``setup.lock``; mutations wrap in this. Lock
    file is kernel-cleaned on crash so a stale wizard server can
    never strand it."""
    lock_path = _setup_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"setup.lock LOCK_EX not acquired within {timeout:.0f}s",
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextlib.contextmanager
def _sh_lock(timeout: float = LOCK_TIMEOUT_S):
    """``LOCK_SH`` on ``setup.lock``; reads wrap in this."""
    lock_path = _setup_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"setup.lock LOCK_SH not acquired within {timeout:.0f}s",
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ── Atomic helpers ───────────────────────────────────────────────


# Every state file under ``~/.config/meeting-scribe`` is owner-private. The
# marker, version counter, password HMACs, and pending-credential cache all
# share the same mode — no caller has a use case for world-read.


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via tempfile + ``os.replace``.

    Mode is hardcoded to 0o600; do not parameterize. CodeQL flags any taint
    flow into ``os.open``/``os.chmod`` mode arguments — keep this constant.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


# ── State predicates ─────────────────────────────────────────────


def is_setup_complete() -> bool:
    """``True`` when the appliance has finished first-touch setup."""
    return _setup_complete_marker().exists()


def mark_setup_complete() -> None:
    """Write the ``setup-complete`` marker. Called after
    ``ack_credentials_saved`` succeeds — there's no separate AP
    rotation phase in v1.0."""
    _atomic_write(_setup_complete_marker(), b"1\n")


def auth_version() -> int:
    """Integer ``auth-version``; defaults to 1. Participates in
    ``CookieSigner`` HKDF info so a bump invalidates every existing
    cookie. Recovery + factory reset bump."""
    raw = _read_text(_auth_version_path())
    if not raw:
        return 1
    try:
        return int(raw.strip())
    except ValueError:
        return 1


def bump_auth_version() -> int:
    new = auth_version() + 1
    _atomic_write(_auth_version_path(), f"{new}\n".encode())
    return new


def bump_auth_version_and_rotate() -> int:
    """Bump ``auth-version`` AND rotate the live admin cookie signer.

    Called only from ``factory_reset`` — the explicit "wipe
    everything" UX action. Bumping the version on disk is enough for
    fresh sign-ins (every new cookie picks up the new version), but
    in-flight admin WebSockets keep their pre-bump cookie until the
    connection drops naturally. Re-deriving the global signer here
    rejects those pre-bump cookies on the very next verify call,
    bouncing both browser tabs the wizard's hardware verification
    step expects.
    """
    new = bump_auth_version()
    # Local import: ``runtime.state`` imports setup_state at module
    # load to read the version, so a top-level import here would
    # cycle.
    from meeting_scribe.runtime import state as _state
    from meeting_scribe.terminal.auth import derive_cookie_subkey

    admin_secret = getattr(_state, "_terminal_admin_secret", None)
    signer = getattr(_state, "_terminal_cookie_signer", None)
    boot_id = getattr(_state, "boot_session_id", b"")
    if admin_secret is not None and signer is not None and boot_id:
        new_subkey = derive_cookie_subkey(
            admin_secret.secret,
            boot_id,
            new,
        )
        signer.rotate_secret(new_subkey)
    return new


# ── Result types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ClaimResult:
    """Outcome of a ``claim_setup`` call.

    ``status``:
        * ``"created"`` — fresh setup-pending/ minted; cookie should
          be set on the response.
        * ``"resumed"`` — the operator's existing session refreshed
          the wizard; reuse the existing cookie (no Set-Cookie
          override required).
        * ``"in_progress"`` — another client holds the wizard;
          caller returns HTTP 409 to the requester.
    """

    status: Literal["created", "resumed", "in_progress"]
    session_id: str


@dataclass(frozen=True)
class SetupCredentials:
    """The two plaintext credentials minted at claim time."""

    admin_password: str
    guest_pin: str


# ── Credential minting ───────────────────────────────────────────


def _mint_admin_password() -> str:
    """Memorable, deterministic admin password tied to the SSID
    suffix. Format: ``DellMeetingAdmin<NNNN>`` where ``<NNNN>`` is
    the same 4-digit appliance pin baked into the SSID name.

    The admin password is NOT a security secret in the traditional
    sense — anyone in Wi-Fi range can guess it from the SSID. Its
    job is to be memorable for the demo operator (one phrase, plus
    "the same number"). Real protection comes from the appliance's
    AP-only network surface and the cookie-signed admin session.
    """
    from meeting_scribe.cli._common import appliance_pin

    return f"DellMeetingAdmin{appliance_pin()}"


def _mint_guest_pin() -> str:
    """4 decimal digits derived deterministically from the
    appliance ID. Same as the SSID's 4-digit suffix — attendees
    see ``Dell Meeting 4239`` and type ``4239`` as the PIN, no
    second thing to remember.

    Not random per-setup, but: the PIN is NOT a security boundary
    on its own (it's derivable from the public SSID); application-
    layer auth (admin password, plus PIN gating of the live view)
    is what protects sensitive operations.
    """
    from meeting_scribe.cli._common import appliance_pin

    return appliance_pin()


# ── claim_setup ──────────────────────────────────────────────────


def _read_pending_session_id() -> bytes | None:
    raw = _read_text(_setup_pending_dir() / "session-id")
    if not raw:
        return None
    return raw.strip().encode("utf-8")


def _pending_age_seconds() -> float | None:
    raw = _read_text(_setup_pending_dir() / "created-at")
    if not raw:
        return None
    try:
        return time.time() - float(raw.strip())
    except ValueError:
        return None


def claim_setup(client_ip: str, request_cookie_sid: bytes | None) -> ClaimResult:
    """Acquire (or resume) wizard ownership.

    Cookie-bound: an AP client without a matching ``ms_setup_sid``
    cookie cannot reclaim a pending wizard, even on the same IP.

    Caller contract: ``GET /setup`` invokes this on the simplified
    v1.0 flow — first AP client to load the wizard mints credentials.
    The AP-iface origin gate + cookie-bound session are the sole
    anti-takeover layers.
    """
    with _ex_lock():
        pending = _setup_pending_dir()
        age = _pending_age_seconds()
        if pending.exists() and age is not None and age < SETUP_EXPIRY_SECONDS:
            on_disk_sid = _read_pending_session_id()
            if (
                request_cookie_sid is not None
                and on_disk_sid is not None
                and _hmac.compare_digest(on_disk_sid, request_cookie_sid)
            ):
                return ClaimResult(
                    status="resumed",
                    session_id=on_disk_sid.decode("utf-8"),
                )
            return ClaimResult(status="in_progress", session_id="")

        if pending.exists():
            shutil.rmtree(pending, ignore_errors=True)

        session_id = secrets.token_urlsafe(32)
        admin_password = _mint_admin_password()
        guest_pin = _mint_guest_pin()

        pending.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_write(pending / "created-at", f"{time.time()}\n".encode())
        _atomic_write(pending / "session-id", session_id.encode("utf-8"))
        _atomic_write(pending / "claim-ip", client_ip.encode("utf-8"))
        _atomic_write(pending / "admin-password.txt", admin_password.encode("utf-8"))
        _atomic_write(pending / "guest-pin.txt", guest_pin.encode("utf-8"))
        return ClaimResult(status="created", session_id=session_id)


# ── read_credentials / ack ──────────────────────────────────────


class CredentialsAlreadyAcked(RuntimeError):
    """Raised by ``read_credentials`` after ``ack_credentials_saved``
    has run — caller returns HTTP 410 Gone."""


class NoPendingSetup(RuntimeError):
    """Raised when no ``setup-pending/`` dir exists (or it expired
    while the wizard tab sat open)."""


def _is_acked() -> bool:
    return (_setup_pending_dir() / "acked").exists()


def read_credentials() -> SetupCredentials:
    """Read the two plaintext credentials. Idempotent until acked."""
    with _sh_lock():
        pending = _setup_pending_dir()
        if not pending.exists():
            raise NoPendingSetup("no setup-pending/ directory")
        if _is_acked():
            raise CredentialsAlreadyAcked("ack-saved already fired")
        admin = _read_text(pending / "admin-password.txt") or ""
        guest_pin = _read_text(pending / "guest-pin.txt") or ""
        if not admin or not guest_pin:
            raise NoPendingSetup("setup-pending/ is missing one or more credential files")
        return SetupCredentials(
            admin_password=admin,
            guest_pin=guest_pin,
        )


def ack_credentials_saved(*, admin_secret: bytes) -> None:
    """Persist LIVE HMACs for every credential, then delete the
    plaintext. HMAC-write-first / plaintext-delete-second so a
    crash between the two leaves a recoverable state.

    Callers also typically follow up with ``mark_setup_complete()``
    once the wizard is truly done — keeping the two as separate
    operations means a crash between ack and mark just re-asks
    the operator to confirm rather than wiping state.
    """
    with _ex_lock():
        pending = _setup_pending_dir()
        if not pending.exists():
            raise NoPendingSetup("no setup-pending/ directory")
        admin_text = _read_text(pending / "admin-password.txt")
        pin_text = _read_text(pending / "guest-pin.txt")
        if not admin_text or not pin_text:
            raise NoPendingSetup("admin-password/guest-pin missing")
        admin_hmac = _hmac.new(admin_secret, admin_text.encode("utf-8"), "sha256").hexdigest()
        pin_hmac = _hmac.new(admin_secret, pin_text.encode("utf-8"), "sha256").hexdigest()
        # HMAC writes FIRST.
        _atomic_write(_admin_password_hmac_path(), admin_hmac.encode("utf-8"))
        _atomic_write(_guest_pin_hmac_path(), pin_hmac.encode("utf-8"))
        # Plaintext deletions SECOND.
        for name in ("admin-password.txt", "guest-pin.txt"):
            with contextlib.suppress(FileNotFoundError):
                (pending / name).unlink()
        _atomic_write(pending / "acked", b"1\n")


# ── Credential verification ─────────────────────────────────────


def verify_admin_password(candidate: str, *, admin_secret: bytes) -> bool:
    """Constant-time-compare ``candidate`` against the persisted
    ``admin-password-hmac``. Returns ``False`` when the file is
    missing so callers can return a generic "password does not
    match" without leaking which condition fired (pre-setup vs
    bad password).

    HMAC-SHA256 (not Argon2/scrypt/bcrypt) is INTENTIONAL here. The
    v1.0 demo-appliance threat model deliberately exposes the admin
    password to anyone in WiFi range — it's deterministically derived
    from the SSID's 4-digit suffix as ``DellMeetingAdmin<NNNN>``, and
    that derivation is documented on the wizard page. The HMAC on
    disk is keyed by the master ``admin-secret`` and protects against
    a stolen-disk-image scenario where someone reads the file
    contents but doesn't have the master key. A password-hashing KDF
    would add no security here — the keyspace is 10000 candidates,
    trivially brute-forceable regardless of hash speed, and the
    password is already public to anyone who can read the SSID. See
    the user-direction header in the sunny-seeking-pebble plan.
    """
    candidate = (candidate or "").strip()
    if not candidate:
        return False
    stored = _read_text(_admin_password_hmac_path())
    if stored is None:
        return False
    expected = _hmac.new(admin_secret, candidate.encode("utf-8"), "sha256").hexdigest()
    return _hmac.compare_digest(stored.strip(), expected)


def verify_guest_pin(candidate: str, *, admin_secret: bytes) -> bool:
    """Constant-time-compare ``candidate`` against the persisted
    ``guest-pin-hmac``. Returns ``False`` when the file is missing
    so callers can return a generic "code does not match" without
    leaking which condition fired."""
    candidate = (candidate or "").strip()
    if not candidate or not candidate.isdigit() or len(candidate) != 4:
        return False
    stored = _read_text(_guest_pin_hmac_path())
    if stored is None:
        return False
    expected = _hmac.new(admin_secret, candidate.encode("utf-8"), "sha256").hexdigest()
    return _hmac.compare_digest(stored.strip(), expected)


# ── cancel ──────────────────────────────────────────────────────


def cancel_setup() -> None:
    """Operator-initiated cancel from the wizard. Removes
    ``setup-pending/``."""
    with _ex_lock():
        shutil.rmtree(_setup_pending_dir(), ignore_errors=True)


# ── reconcile (startup) ─────────────────────────────────────────


def _reconcile_setup_at_startup() -> None:
    """Run at lifespan startup. Drops any expired ``setup-pending/``
    directory so a fresh wizard claim isn't blocked by yesterday's
    abandoned attempt."""
    with _ex_lock():
        pending = _setup_pending_dir()
        if pending.exists():
            age = _pending_age_seconds()
            if age is not None and age >= SETUP_EXPIRY_SECONDS:
                logger.info("reconcile: removing expired setup-pending/")
                shutil.rmtree(pending, ignore_errors=True)


# ── factory_reset ───────────────────────────────────────────────


def factory_reset(*, regenerate_secrets: Callable[[], None] | None = None) -> None:
    """Atomic reset for re-deployment.

      1. Bump ``auth-version`` (invalidates every existing cookie).
      2. Remove ``setup-complete``, ``setup-pending/``.
      3. Remove the LIVE HMACs (admin-password, guest-pin) so the
         next operator must complete the wizard before any auth
         works.
      4. Also clear any legacy ``recovery-code-hmac`` left behind
         by an older deploy (the recovery flow was retired in v1.0;
         the file is otherwise unreachable but lingering on disk).
      5. Optional ``regenerate_secrets`` callback for future use.

    Some on-disk state may be owned by root (e.g. when an earlier
    setup-pending was created by a sudo'd test/CLI invocation).
    The shutil.rmtree falls back to a sudo-rm via the existing
    sudoers fragment if the user-mode unlink hits PermissionError.
    """
    with _ex_lock():
        bump_auth_version_and_rotate()
        with contextlib.suppress(FileNotFoundError):
            _setup_complete_marker().unlink()
        _force_rmtree(_setup_pending_dir())
        for f in (
            _admin_password_hmac_path(),
            _guest_pin_hmac_path(),
            _state_dir() / "recovery-code-hmac",
        ):
            _force_unlink(f)
        if regenerate_secrets is not None:
            try:
                regenerate_secrets()
            except Exception as exc:
                logger.exception(
                    "factory_reset: regenerate_secrets raised: %s",
                    exc,
                )


def _force_unlink(path: Path) -> None:
    """Unlink ``path``; fall back to ``sudo -n rm -f`` if the
    user-mode unlink can't (PermissionError because the file was
    written by a sudo'd process). No-op if the file doesn't
    exist."""
    if not path.exists():
        return
    try:
        path.unlink()
        return
    except PermissionError:
        pass
    import subprocess

    subprocess.run(
        ["sudo", "-n", "rm", "-f", str(path)],
        capture_output=True,
        timeout=5,
        check=False,
    )


def _force_rmtree(path: Path) -> None:
    """Recursively remove ``path``; fall back to ``sudo -n rm -rf``
    when files inside are root-owned."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        return
    except PermissionError:
        pass
    import subprocess

    subprocess.run(
        ["sudo", "-n", "rm", "-rf", str(path)],
        capture_output=True,
        timeout=10,
        check=False,
    )
