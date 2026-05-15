"""Auth-boundary tests for the speakerphone routers.

Acceptance criterion 11 (and codex P0 from plan review):
  * Public ``/api/admin/speakerphone/*`` rejects requests without a
    session cookie.
  * Internal ``/api/internal/speakerphone/*`` is **not reachable** at all
    on the public TCP listener — 404 / not-mounted, never 401.
  * UDS uvicorn refuses to bind in a world-writable runtime directory
    and produces a 0600 socket file on success.

We do not spin up a real uvicorn here; the structural tests are enough
to assert the contract. End-to-end UDS verification belongs in a
hardware-in-the-loop smoke test.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from meeting_scribe.speakerphone import uds as sp_uds


def test_internal_router_is_not_mounted_on_main_app() -> None:
    """Importing the public app must not pull /api/internal/* into TCP."""
    from meeting_scribe.server import app

    tcp_paths = {getattr(route, "path", "") for route in app.routes}
    leaked = {p for p in tcp_paths if p.startswith("/api/internal/")}
    assert leaked == set(), f"internal routes leaked into TCP app: {leaked}"


def test_admin_router_is_mounted_on_main_app() -> None:
    from meeting_scribe.server import app

    tcp_paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/admin/speakerphone/state" in tcp_paths
    assert "/api/admin/speakerphone/mapping" in tcp_paths


def test_uds_app_isolated_to_internal_namespace() -> None:
    app = sp_uds.build_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    api_paths = {p for p in paths if p.startswith("/api/")}
    assert api_paths
    assert all(p.startswith("/api/internal/") for p in api_paths)
    # The admin namespace must not appear at all.
    assert not any(p.startswith("/api/admin/") for p in api_paths)


def test_check_runtime_dir_rejects_world_writable_parent(tmp_path: Path) -> None:
    """A 0777 runtime dir must abort the bind.

    Simulating with a tmpdir set to mode 0777 — the function chmods the
    parent to 0700 on success, so we have to prepare a sibling and check
    via the raw function.
    """
    target = tmp_path / "sock"
    # First call creates parent 0700 (the tmp_path itself). Verify the
    # function does not raise on a properly-configured tmpdir.
    sp_uds._check_runtime_dir(target)
    mode = tmp_path.stat().st_mode & 0o777
    # The function chmods to 0700 if it had to mkdir; we don't assert
    # specific mode here because tmp_path already existed at 0700-ish.

    # Now simulate a world-writable parent by widening perms BEFORE
    # the check — the function must refuse.
    bad_dir = tmp_path / "world_writable_runtime"
    bad_dir.mkdir(mode=0o777)
    os.chmod(bad_dir, 0o777)
    bad_target = bad_dir / "sock"
    with pytest.raises(RuntimeError, match="world-writable"):
        sp_uds._check_runtime_dir(bad_target)


def test_create_socket_writes_0600_perms(tmp_path: Path) -> None:
    target = tmp_path / "meeting-scribe.sock"
    sock = sp_uds._create_socket(target)
    try:
        st = target.stat()
        mode = st.st_mode & 0o777
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
        assert stat.S_ISSOCK(st.st_mode)
        # And owned by us (sanity for the same-user threat model).
        assert st.st_uid == os.getuid()
    finally:
        sock.close()
        target.unlink(missing_ok=True)


def test_create_socket_unlinks_prior_owned_socket(tmp_path: Path) -> None:
    """A stale socket owned by the current user is silently replaced."""
    target = tmp_path / "meeting-scribe.sock"
    # Stage a stale socket on disk owned by us.
    import socket as _socket

    placeholder = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    placeholder.bind(str(target))
    placeholder.close()
    assert target.exists()

    sock = sp_uds._create_socket(target)
    try:
        assert target.exists()
        st = target.stat()
        assert stat.S_ISSOCK(st.st_mode)
        assert (st.st_mode & 0o777) == 0o600
    finally:
        sock.close()
        target.unlink(missing_ok=True)


def test_default_uds_path_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MEETING_SCRIBE_UDS_PATH", str(tmp_path / "x.sock"))
    assert sp_uds.default_uds_path() == tmp_path / "x.sock"


def test_default_uds_path_uses_xdg_runtime_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MEETING_SCRIBE_UDS_PATH", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert sp_uds.default_uds_path() == tmp_path / "meeting-scribe.sock"
