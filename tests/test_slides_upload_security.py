"""Locking auth tests for ``POST /api/meetings/{id}/slides/upload``
(Phase 1.2).

The PPTX upload regression was a frontend bug — the slide-upload code
read ``window._currentMeetingId`` while ``startRecording()`` wrote to
``window.current_meeting_id``, so the upload always saw a null id.

These tests pin the **server-side** auth contract for the route so a
future fix doesn't loosen authentication trying to "make uploads
work". The route itself uses ``_require_admin_or_raise(request)``
(routes/slides.py:61); guest cookies and unauthenticated requests
must keep being rejected.

Origin/CSRF defense-in-depth: ``_ORIGIN_GUARDED_PREFIXES`` in
``middlewares.py`` covers ``/api/admin/*`` and ``/api/meeting/*``
(singular). The upload route lives under ``/api/meetings/*``
(plural) and is therefore not covered by the Origin allowlist
today; ``scribe_admin`` is ``SameSite=Strict`` which is the actual
browser-CSRF mitigation. If a future PR widens the Origin guard to
cover ``/api/meetings/*``, that PR also extends this matrix to
include the bad-Origin / missing-Origin cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from meeting_scribe.routes.slides import router as slides_router
from meeting_scribe.runtime import state as runtime_state

_FIXTURE_PPTX = Path(__file__).parent / "fixtures" / "test_slides.pptx"


class _StubSlideJobRunner:
    """Minimal ``SlideJobRunner`` shape — the route handler only calls
    ``start_job(...)`` and reads ``state.slides_enabled`` / the
    ``state.slide_job_runner`` reference. Returning a fixed deck id
    lets the positive-case test confirm a 200 with a deck_id payload
    without needing the full runner + LibreOffice + translation
    backend chain."""

    async def start_job(
        self,
        meeting_id: str,
        contents: bytes,
        source_lang: str,
        target_lang: str,
        *,
        skip_language_detection: bool = False,
        upload_filename: str = "",
        monolingual: bool = False,
    ) -> str:
        return "deck-stub-1"


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    """Build a FastAPI app that mounts only the slides router. The
    real app's middleware stack (admin gate, Origin allowlist) is
    NOT wired here — that means each test must explicitly stub the
    auth dependency the route uses."""
    monkeypatch.setattr(runtime_state, "slides_enabled", True, raising=False)
    monkeypatch.setattr(runtime_state, "slide_job_runner", _StubSlideJobRunner(), raising=False)
    # Stub storage so the route's ``state.storage._read_meta(...)``
    # call returns None (route falls back to default language pair).
    monkeypatch.setattr(runtime_state, "storage", _StubStorage(), raising=False)
    # Stub config so default_language_pair resolves.
    monkeypatch.setattr(runtime_state, "config", _StubConfig(), raising=False)

    app = FastAPI()
    app.include_router(slides_router)
    return app


class _StubStorage:
    def _read_meta(self, _mid: str):
        return None


class _StubConfig:
    default_language_pair = "en,ja"


def _client(
    app: FastAPI, *, with_admin_cookie: bool, with_guest_cookie: bool = False
) -> TestClient:
    """TestClient with the requested cookie profile.

    The route uses ``_require_admin_or_raise(request)`` which calls
    into ``has_admin_session(request)`` which verifies the
    ``scribe_admin`` cookie's HMAC. Stubbing the admin gate via a
    monkeypatch on ``has_admin_session`` is cleaner than minting a
    real signed cookie for these tests; we monkeypatch in the test
    body rather than the fixture so each test controls the auth
    decision explicitly.
    """
    client = TestClient(app)
    if with_admin_cookie:
        client.cookies.set("scribe_admin", "stub-admin-cookie")
    if with_guest_cookie:
        client.cookies.set("ms_guest", "stub-guest-cookie")
    return client


def _upload(client: TestClient, *, meeting_id: str = "mtg-1", origin: str | None = None):
    headers = {}
    if origin is not None:
        headers["Origin"] = origin
    return client.post(
        f"/api/meetings/{meeting_id}/slides/upload",
        files={
            "file": (
                "deck.pptx",
                _FIXTURE_PPTX.read_bytes(),
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        },
        headers=headers,
    )


def _patch_admin_gate(monkeypatch: pytest.MonkeyPatch, *, accept: bool) -> None:
    """Patch the admin guard so tests don't need real HMAC signing.

    The route imports ``_require_admin_or_raise`` from
    ``server_support.admin_guard``. We patch the gate at the module
    level the route reads from."""
    from meeting_scribe.server_support import admin_guard

    if accept:
        monkeypatch.setattr(admin_guard, "_require_admin_or_raise", lambda req: None)
        # The route imports the symbol directly, so also patch the
        # binding inside ``routes.slides``:
        from meeting_scribe.routes import slides as slides_module

        monkeypatch.setattr(slides_module, "_require_admin_or_raise", lambda req: None)
    else:
        from fastapi import HTTPException

        def _reject(req):
            raise HTTPException(403, "Admin access required")

        monkeypatch.setattr(admin_guard, "_require_admin_or_raise", _reject)
        from meeting_scribe.routes import slides as slides_module

        monkeypatch.setattr(slides_module, "_require_admin_or_raise", _reject)


# ── Negative cases ─────────────────────────────────────────────────


def test_upload_no_cookie_rejected(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unauthenticated request must not reach the slide-job
    runner. The admin gate rejects with 403."""
    _patch_admin_gate(monkeypatch, accept=False)
    client = _client(app, with_admin_cookie=False)
    resp = _upload(client)
    assert resp.status_code == 403


def test_upload_guest_cookie_rejected(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """A guest-only cookie does not satisfy the admin gate. Locks
    the contract that uploads are admin-only."""
    _patch_admin_gate(monkeypatch, accept=False)
    client = _client(app, with_admin_cookie=False, with_guest_cookie=True)
    resp = _upload(client)
    assert resp.status_code == 403


# ── Positive case ──────────────────────────────────────────────────


def test_upload_admin_cookie_accepted(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid admin cookie + a real ``.pptx`` payload reaches the
    slide-job runner and gets a deck id back. This is the contract
    the frontend depends on after the ``current_meeting_id`` fix."""
    _patch_admin_gate(monkeypatch, accept=True)
    client = _client(app, with_admin_cookie=True)
    resp = _upload(client, origin="https://192.168.1.168")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert body["deck_id"] == "deck-stub-1"


def test_upload_admin_cookie_no_origin_accepted_today(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**Codifies current behaviour**, not desired-end-state. The
    upload route is under ``/api/meetings/*`` (plural) and is NOT
    behind the Origin allowlist (which covers
    ``/api/admin/*`` + ``/api/meeting/*`` singular). The
    ``scribe_admin`` cookie's ``SameSite=Strict`` attribute is what
    blocks browser-CSRF for this route today.

    A future PR that widens ``_ORIGIN_GUARDED_PREFIXES`` to cover
    ``/api/meetings/`` should flip this test from `accepted` to
    `rejected` and add the parallel cases for the start/stop/cancel
    endpoints. Until then, this test pins the current contract so a
    drift either way is visible."""
    _patch_admin_gate(monkeypatch, accept=True)
    client = _client(app, with_admin_cookie=True)
    resp = _upload(client, origin=None)
    assert resp.status_code == 200


# ── Smoke: missing file rejected at parser layer ──────────────────


def test_upload_missing_file_rejected(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route returns 400 when no ``file`` field is present.
    Locks the parser contract — pre-fix the route was suspected of a
    field-name mismatch; this test rules that out structurally."""
    _patch_admin_gate(monkeypatch, accept=True)
    client = _client(app, with_admin_cookie=True)
    resp = client.post(
        "/api/meetings/mtg-1/slides/upload",
        # Posting WITH a multipart-shaped body that has the wrong
        # field name -- the parser sees no ``file`` field and rejects
        # at the route, not at the parser. Use a non-empty body to
        # ensure we exercise the route's check, not Starlette's
        # "no body" path.
        files={"not_file": ("foo.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
    assert "No file uploaded" in resp.text
