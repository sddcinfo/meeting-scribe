"""Tests for the simplified v1.0 setup-state machine.

The AP-rotation machinery (begin_commit, finalize_phase3,
await_proof_of_reconnect, rollback_to_pending,
resume_pending_proof) was deleted. The wizard mints credentials,
operator saves them, ack writes HMACs + setup-complete marker, AP
stays OWE forever.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from meeting_scribe import setup_state


@pytest.fixture(autouse=True)
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def admin_secret() -> bytes:
    return b"x" * 64


# ── Predicates ───────────────────────────────────────────────────


def test_predicates_clean_state(_state_dir: Path) -> None:
    assert setup_state.is_setup_complete() is False
    assert setup_state.auth_version() == 1


def test_bump_auth_version_persists(_state_dir: Path) -> None:
    assert setup_state.bump_auth_version() == 2
    assert setup_state.bump_auth_version() == 3
    assert setup_state.auth_version() == 3


def test_bump_and_rotate_invalidates_admin_cookie(
    _state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bump_auth_version_and_rotate`` rotates the live admin signer
    so a cookie minted before the bump fails verify after the bump.
    This is the in-process invalidation guarantee factory_reset
    relies on (vs. waiting for the next restart)."""
    from meeting_scribe.runtime import state
    from meeting_scribe.terminal.auth import (
        AdminSecretStore,
        CookieSigner,
        derive_cookie_subkey,
    )

    secret_path = _state_dir / "admin-secret"
    admin_secret = AdminSecretStore.load_or_create(secret_path)
    boot_id = b"\x07" * 32
    monkeypatch.setattr(state, "_terminal_admin_secret", admin_secret)
    monkeypatch.setattr(state, "boot_session_id", boot_id)
    initial_subkey = derive_cookie_subkey(admin_secret.secret, boot_id, setup_state.auth_version())
    signer = CookieSigner(secret=initial_subkey, max_age_seconds=60)
    monkeypatch.setattr(state, "_terminal_cookie_signer", signer)

    # Mint a cookie under the current version.
    cookie = signer.issue()
    assert signer.verify(cookie) is True

    # Bump + rotate. The same cookie no longer verifies.
    setup_state.bump_auth_version_and_rotate()
    assert signer.verify(cookie) is False, (
        "rotate_secret must replace the signer's HMAC key in-place so "
        "pre-bump cookies stop verifying immediately"
    )


def test_mark_setup_complete(_state_dir: Path) -> None:
    assert setup_state.is_setup_complete() is False
    setup_state.mark_setup_complete()
    assert setup_state.is_setup_complete() is True


# ── Claim ────────────────────────────────────────────────────────


def test_claim_creates_pending_with_session_cookie(_state_dir: Path) -> None:
    result = setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    assert result.status == "created"
    assert result.session_id

    pending = _state_dir / "setup-pending"
    assert pending.exists()
    assert (pending / "session-id").read_text() == result.session_id
    assert (pending / "claim-ip").read_text() == "10.42.0.50"
    for name in ("admin-password.txt", "guest-pin.txt"):
        assert (pending / name).stat().st_mode & 0o777 == 0o600
    assert not (pending / "recovery-code.txt").exists()


def test_claim_resumes_with_matching_cookie(_state_dir: Path) -> None:
    first = setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    sid_bytes = first.session_id.encode("utf-8")
    second = setup_state.claim_setup("10.42.0.50", request_cookie_sid=sid_bytes)
    assert second.status == "resumed"
    assert second.session_id == first.session_id


def test_claim_rejects_competitor_without_cookie(_state_dir: Path) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    blocked = setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    assert blocked.status == "in_progress"


def test_claim_rejects_wrong_cookie_same_ip(_state_dir: Path) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    blocked = setup_state.claim_setup("10.42.0.50", request_cookie_sid=b"forged-session-id")
    assert blocked.status == "in_progress"


def test_claim_remints_after_expiry(_state_dir: Path) -> None:
    import time

    first = setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    pending = _state_dir / "setup-pending"
    (pending / "created-at").write_text(f"{time.time() - 7200}\n")
    second = setup_state.claim_setup("10.42.0.51", request_cookie_sid=None)
    assert second.status == "created"
    assert second.session_id != first.session_id


# ── Credentials read / ack ───────────────────────────────────────


def test_read_credentials_idempotent(_state_dir: Path) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    a = setup_state.read_credentials()
    b = setup_state.read_credentials()
    assert a == b
    assert a.admin_password.startswith("DellMeetingAdmin")
    assert a.admin_password.endswith(a.guest_pin)
    assert len(a.guest_pin) == 4
    assert a.guest_pin.isdigit()


def test_read_credentials_after_ack_raises(_state_dir: Path, admin_secret: bytes) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=admin_secret)
    with pytest.raises(setup_state.CredentialsAlreadyAcked):
        setup_state.read_credentials()


def test_ack_writes_hmacs_and_deletes_plaintext(_state_dir: Path, admin_secret: bytes) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    creds = setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=admin_secret)

    import hmac as _hmac

    admin_hmac = (_state_dir / "admin-password-hmac").read_text()
    pin_hmac = (_state_dir / "guest-pin-hmac").read_text()

    assert (
        admin_hmac == _hmac.new(admin_secret, creds.admin_password.encode(), "sha256").hexdigest()
    )
    assert pin_hmac == _hmac.new(admin_secret, creds.guest_pin.encode(), "sha256").hexdigest()

    pending = _state_dir / "setup-pending"
    for name in ("admin-password.txt", "guest-pin.txt"):
        assert not (pending / name).exists()
    assert not (_state_dir / "recovery-code-hmac").exists()
    assert (pending / "acked").exists()


def test_ack_without_pending_raises(_state_dir: Path, admin_secret: bytes) -> None:
    with pytest.raises(setup_state.NoPendingSetup):
        setup_state.ack_credentials_saved(admin_secret=admin_secret)


# ── Guest PIN verification ─────────────────────────────────────


def test_verify_admin_password_pre_finish_returns_false(
    _state_dir: Path, admin_secret: bytes
) -> None:
    """Pre-``finish``, no ``admin-password-hmac`` file exists. The
    verifier returns False so a half-completed setup can't mint
    cookies."""
    assert setup_state.verify_admin_password("anything", admin_secret=admin_secret) is False


def test_verify_admin_password_post_finish_round_trip(
    _state_dir: Path, admin_secret: bytes
) -> None:
    """Wizard-style write → ``verify_admin_password`` round trip. The
    HMAC stored on disk under ``admin-password-hmac`` accepts the
    same password and rejects others."""
    import hmac

    # Don't use the production-shaped ``DellMeetingAdmin<NNNN>`` form
    # in test fixtures — secret scanners (GitGuardian) flag any
    # commit that looks like a deployed appliance's password.
    candidate = "test-fixture-credential-XYZ"
    expected = hmac.new(admin_secret, candidate.encode("utf-8"), "sha256").hexdigest()
    (_state_dir / "admin-password-hmac").write_text(expected)
    assert setup_state.verify_admin_password(candidate, admin_secret=admin_secret) is True


def test_verify_admin_password_wrong_secret(_state_dir: Path, admin_secret: bytes) -> None:
    """Stored HMAC is for one password; verifying any other returns
    False (constant-time mismatch)."""
    import hmac

    expected = hmac.new(admin_secret, b"the-real-one", "sha256").hexdigest()
    (_state_dir / "admin-password-hmac").write_text(expected)
    assert setup_state.verify_admin_password("not-it", admin_secret=admin_secret) is False
    assert setup_state.verify_admin_password("", admin_secret=admin_secret) is False


def test_verify_guest_pin_match(_state_dir: Path, admin_secret: bytes) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    creds = setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=admin_secret)
    assert setup_state.verify_guest_pin(creds.guest_pin, admin_secret=admin_secret)


def test_verify_guest_pin_mismatch(_state_dir: Path, admin_secret: bytes) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=admin_secret)
    assert not setup_state.verify_guest_pin("0000", admin_secret=admin_secret)


def test_verify_guest_pin_no_hmac_file(_state_dir: Path, admin_secret: bytes) -> None:
    """Pre-setup → constant-time false (no info leak)."""
    assert not setup_state.verify_guest_pin("1234", admin_secret=admin_secret)


def test_verify_guest_pin_rejects_non_4digit(_state_dir: Path, admin_secret: bytes) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=admin_secret)
    for bad in ("", "abc", "12345", "abcd", "12 34"):
        assert not setup_state.verify_guest_pin(bad, admin_secret=admin_secret)


# ── Cancel ──────────────────────────────────────────────────────


def test_cancel_clears_pending(_state_dir: Path) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    setup_state.cancel_setup()
    assert not (_state_dir / "setup-pending").exists()


# ── Reconcile ──────────────────────────────────────────────────


def test_reconcile_no_pending_is_noop(_state_dir: Path) -> None:
    setup_state._reconcile_setup_at_startup()
    assert not (_state_dir / "setup-pending").exists()


def test_reconcile_drops_expired_pending(_state_dir: Path) -> None:
    import time

    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    pending = _state_dir / "setup-pending"
    (pending / "created-at").write_text(f"{time.time() - 7200}\n")
    setup_state._reconcile_setup_at_startup()
    assert not pending.exists()


# ── factory_reset ────────────────────────────────────────────────


def test_factory_reset_clears_state_and_bumps_auth_version(
    _state_dir: Path, admin_secret: bytes
) -> None:
    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    setup_state.read_credentials()
    setup_state.ack_credentials_saved(admin_secret=admin_secret)
    setup_state.mark_setup_complete()
    starting_version = setup_state.auth_version()

    setup_state.factory_reset()

    assert setup_state.auth_version() == starting_version + 1
    assert not setup_state.is_setup_complete()
    assert not (_state_dir / "setup-pending").exists()
    assert not (_state_dir / "admin-password-hmac").exists()
    assert not (_state_dir / "recovery-code-hmac").exists()
    assert not (_state_dir / "guest-pin-hmac").exists()


def test_factory_reset_calls_optional_regen(_state_dir: Path) -> None:
    fired: list[bool] = []

    def regen() -> None:
        fired.append(True)

    setup_state.factory_reset(regenerate_secrets=regen)
    assert fired == [True]


# ── Concurrency lock ────────────────────────────────────────────


def test_lock_excludes_concurrent_writers(_state_dir: Path) -> None:
    import errno
    import fcntl

    setup_state.claim_setup("10.42.0.50", request_cookie_sid=None)
    lock_path = _state_dir / "setup.lock"
    fd_a = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd_a, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fd_b = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(OSError) as exc_info:
            fcntl.flock(fd_b, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert exc_info.value.errno in (errno.EAGAIN, errno.EACCES)
    finally:
        fcntl.flock(fd_a, fcntl.LOCK_UN)
        os.close(fd_a)
        os.close(fd_b)
