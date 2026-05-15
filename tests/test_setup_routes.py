"""Tests for the simplified Phase F wizard routes (no rotation).

The v1.0 simplification reduced the wizard to a single page +
single finish action:

  * GET /setup            →  claim, mint, render
  * POST /api/setup/finish →  ack + mark setup-complete
  * POST /api/setup/cancel →  drop pending state

The fingerprint check, bootstrap-secret entry, AP rotation,
reconnect-proof flow, and recovery-code apparatus were dropped —
see routes/setup.py docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from meeting_scribe import setup_state
from meeting_scribe.routes.setup import router as setup_router


@pytest.fixture
def _state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SCRIBE_STATE_DIR", str(tmp_path))
    (tmp_path / "setup-complete").unlink(missing_ok=True)
    return tmp_path


@pytest.fixture
def app(_state_dir: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(setup_router)
    return app


def _client(app: FastAPI) -> TestClient:
    return TestClient(app, base_url="https://10.42.0.1")


# ── GET /setup ─────────────────────────────────────────────────


def test_get_setup_claims_and_renders(app: FastAPI, _state_dir: Path) -> None:
    """First hit mints credentials + sets the cookie + renders page."""
    client = _client(app)
    resp = client.get("/setup")
    assert resp.status_code == 200
    body = resp.text
    assert "Guest PIN" in body
    assert "Admin password" in body
    assert "DellMeetingAdmin" in body
    assert "Recovery code" not in body
    assert "saved them" in body
    assert "no-store" in resp.headers["cache-control"]
    assert "ms_setup_sid" in resp.headers.get("set-cookie", "")
    assert (_state_dir / "setup-pending").exists()


def test_get_setup_idempotent_for_same_session(app: FastAPI, _state_dir: Path) -> None:
    """Reload by the same browser (cookie matches) re-renders with
    the same credentials, doesn't mint new ones."""
    client = _client(app)
    first = client.get("/setup")
    assert first.status_code == 200
    creds_a = setup_state.read_credentials()
    second = client.get("/setup")
    assert second.status_code == 200
    creds_b = setup_state.read_credentials()
    assert creds_a == creds_b
    assert creds_a.guest_pin in first.text
    assert creds_a.guest_pin in second.text


def test_get_setup_competing_client_409(app: FastAPI, _state_dir: Path) -> None:
    a = _client(app)
    a.get("/setup")
    b = _client(app)
    resp = b.get("/setup")
    assert resp.status_code == 409
    assert "already in progress" in resp.text


def test_get_setup_redirects_to_auth_when_complete(app: FastAPI, _state_dir: Path) -> None:
    """Setup is one-shot. Once the complete marker exists, ``/setup``
    must 302 the operator straight to the sign-in page rather than
    render an interstitial that requires another click."""
    (_state_dir / "setup-complete").write_text("1\n")
    client = _client(app)
    resp = client.get("/setup", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/auth"


# ── POST /api/setup/finish ─────────────────────────────────────


def test_finish_requires_session(app: FastAPI, _state_dir: Path) -> None:
    client = _client(app)
    resp = client.post("/api/setup/finish")
    assert resp.status_code == 403


def test_finish_acks_and_marks_complete(app: FastAPI, _state_dir: Path) -> None:
    client = _client(app)
    client.get("/setup")
    resp = client.post("/api/setup/finish")
    assert resp.status_code == 200
    assert resp.json()["state"] == "complete"
    # Both HMACs persisted; recovery-code-hmac is gone with v1.0.
    assert (_state_dir / "admin-password-hmac").exists()
    assert (_state_dir / "guest-pin-hmac").exists()
    assert not (_state_dir / "recovery-code-hmac").exists()
    # setup-complete marker written.
    assert (_state_dir / "setup-complete").exists()


# ── POST /api/setup/cancel ─────────────────────────────────────


def test_cancel_clears_pending(app: FastAPI, _state_dir: Path) -> None:
    client = _client(app)
    client.get("/setup")
    assert (_state_dir / "setup-pending").exists()
    resp = client.post("/api/setup/cancel")
    assert resp.status_code == 200
    assert not (_state_dir / "setup-pending").exists()


def test_pages_have_no_remote_references(app: FastAPI, _state_dir: Path) -> None:
    client = _client(app)
    body = client.get("/setup").text
    assert "https://fonts.googleapis.com" not in body
    assert "https://cdn." not in body
    assert "@import url(http" not in body
    assert '<script src="http' not in body
    assert '<link rel="stylesheet" href="http' not in body
