"""Auth-version + boot-session-id participation in cookie HKDF.

The two-cookie model splits restart vs factory-reset semantics:

* Admin cookie — signed with ``derive_cookie_subkey(master, boot_id,
  auth_version)``. A regular service restart regenerates
  ``boot_id`` → admin cookies invalidate. A factory_reset bumps
  ``auth_version`` → admin AND guest cookies invalidate together.

* Guest cookie — signed with ``derive_cookie_subkey(master, None,
  auth_version, info=b"scribe-guest-cookie-v1")``. ``boot_id=None``
  uses a constant salt, so a regular restart leaves the signer
  intact (guest sessions persist). Only ``auth_version`` rotates
  the guest signer.
"""

from __future__ import annotations

from meeting_scribe.terminal.auth import derive_cookie_subkey

_MASTER = b"x" * 64
_BOOT_A = b"\x01" * 32
_BOOT_B = b"\x02" * 32


def test_admin_subkey_changes_with_boot_id() -> None:
    """Restart simulation: same master, different boot_id → different
    subkey. This is the "logout-all-on-restart" guarantee."""
    a = derive_cookie_subkey(_MASTER, _BOOT_A, auth_version=1)
    b = derive_cookie_subkey(_MASTER, _BOOT_B, auth_version=1)
    assert a != b


def test_admin_subkey_changes_with_auth_version() -> None:
    """factory_reset simulation: same boot_id, different
    auth_version → different subkey. Bumping the version invalidates
    every previously-issued admin cookie even mid-boot."""
    a = derive_cookie_subkey(_MASTER, _BOOT_A, auth_version=1)
    b = derive_cookie_subkey(_MASTER, _BOOT_A, auth_version=2)
    assert a != b


def test_guest_subkey_survives_restart() -> None:
    """boot_id=None means the salt is constant — two derivations at
    the same auth_version produce the SAME subkey, so a guest cookie
    issued before a restart still verifies after."""
    a = derive_cookie_subkey(_MASTER, None, auth_version=1, info=b"scribe-guest-cookie-v1")
    b = derive_cookie_subkey(_MASTER, None, auth_version=1, info=b"scribe-guest-cookie-v1")
    assert a == b


def test_guest_subkey_rotates_on_auth_version_bump() -> None:
    """factory_reset: even with boot_id=None, bumping auth_version
    rotates the guest signer. Both cookies stop verifying together."""
    a = derive_cookie_subkey(_MASTER, None, auth_version=1, info=b"scribe-guest-cookie-v1")
    b = derive_cookie_subkey(_MASTER, None, auth_version=2, info=b"scribe-guest-cookie-v1")
    assert a != b


def test_admin_and_guest_subkeys_diverge_at_same_inputs() -> None:
    """Different ``info`` separates the two namespaces: a cookie
    signed in the admin namespace cannot be replayed as a guest
    cookie even at identical version + boot inputs (boot_id matters
    only on the admin side; the guest side ignores it via None)."""
    admin = derive_cookie_subkey(_MASTER, _BOOT_A, auth_version=1, info=b"scribe-cookie-v36")
    guest = derive_cookie_subkey(_MASTER, None, auth_version=1, info=b"scribe-guest-cookie-v1")
    assert admin != guest
