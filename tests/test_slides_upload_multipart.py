"""Regression test for the slides-upload PPTX multipart path.

The bug this guards against: ``python-multipart`` was missing from
the runtime dependency list, so a real multipart POST to
``/api/meetings/{id}/slides/upload`` would crash with
``AssertionError: The python-multipart library must be installed to
use form parsing.``  None of the existing unit tests exercised the
HTTP form-parsing path, so the regression made it all the way to a
customer GB10 install.

This test mounts the slides router on a minimal FastAPI app, stubs
the admin guard and runtime state, and POSTs a real multipart body.
If the multipart dependency drifts again (or fastapi changes how it
loads the parser), this test will fail loudly in CI instead of
silently shipping.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_scribe.routes import slides as slides_route
from meeting_scribe.runtime import state as runtime_state

PPTX_FIXTURE = Path(__file__).parent / "fixtures" / "test_slides.pptx"


def test_python_multipart_importable():
    """If this fails, runtime form parsing will crash. Catch that here
    rather than at the customer's HTTP boundary."""
    import python_multipart  # noqa: F401


@pytest.fixture
def slides_app(monkeypatch):
    """Minimal FastAPI app with just the slides router + stubbed deps."""
    # The route imports admin-guard helpers as module-local names, so
    # we patch the *names the route looks up*, not the source module.
    monkeypatch.setattr(slides_route, "_require_admin_or_raise", lambda _r: None)
    monkeypatch.setattr(slides_route, "_require_admin_response", lambda _r: None)

    # Make the route reach form parsing — but bail with 503 immediately
    # after, so we don't have to stand up a real SlideJobRunner / disk
    # layout. The point of this test is to validate multipart, not the
    # downstream pipeline.
    monkeypatch.setattr(runtime_state, "slides_enabled", True)

    class _StubRunner:
        async def submit(self, *_args, **_kwargs):
            return {"deck_id": "test-deck", "status": "queued"}

    monkeypatch.setattr(runtime_state, "slide_job_runner", _StubRunner())

    app = FastAPI()
    app.include_router(slides_route.router)
    return app


def test_pptx_upload_multipart_parses_without_error(slides_app):
    """Real multipart POST. If python-multipart is missing, FastAPI's
    ``request.form()`` raises AssertionError before any route logic
    runs — the response would be a 500 with ``Internal Server
    Error`` and the assertion in the body. With the dep declared,
    we expect the route to read the file and dispatch to the
    SlideJobRunner stub.
    """
    assert PPTX_FIXTURE.exists(), f"missing fixture {PPTX_FIXTURE}"

    # raise_server_exceptions=False so we get a real 500 response with
    # the assertion message in the body if multipart parsing fails,
    # rather than the exception bubbling up out of the test client.
    client = TestClient(slides_app, raise_server_exceptions=False)
    with PPTX_FIXTURE.open("rb") as f:
        resp = client.post(
            "/api/meetings/test-meeting/slides/upload",
            files={
                "file": (
                    "test_slides.pptx",
                    f,
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            },
        )

    # The point of this test is to validate the multipart code path —
    # everything past ``request.form()`` is downstream pipeline that
    # we don't bother stubbing here. We just want a hard fail if the
    # python-multipart assertion comes back.
    assert "python-multipart" not in resp.text and "must be installed" not in resp.text, (
        f"slides upload still hit the python-multipart assertion: "
        f"{resp.status_code} {resp.text[:200]}"
    )


def test_pptx_upload_missing_file_returns_400(slides_app):
    """Multipart with no ``file`` part should reach the route's own
    handling and return 400 — not crash on form parsing.
    """
    client = TestClient(slides_app)
    resp = client.post(
        "/api/meetings/test-meeting/slides/upload",
        # Send a multipart body with a different field name so
        # request.form() succeeds but form.get("file") is None.
        files={"not_file": ("x.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
    assert "No file uploaded" in resp.text
