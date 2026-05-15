"""Import-smoke tests for the speakerphone routers.

Catches the trivial "I broke an import" failure mode before integration
tests run.
"""

from __future__ import annotations


def test_admin_router_registers_expected_routes() -> None:
    from meeting_scribe.routes.admin_speakerphone import router

    paths = {route.path for route in router.routes}
    assert "/api/admin/speakerphone/state" in paths
    assert "/api/admin/speakerphone/mapping" in paths
    assert "/api/admin/speakerphone/led-test" in paths
    assert "/api/admin/speakerphone/reset-defaults" in paths


def test_internal_router_registers_expected_routes() -> None:
    from meeting_scribe.routes.internal_speakerphone import router

    paths = {route.path for route in router.routes}
    assert "/api/internal/speakerphone/state" in paths
    assert "/api/internal/speakerphone/interpretation" in paths
    assert "/api/internal/speakerphone/mic-mute" in paths
    assert "/api/internal/speakerphone/meeting-toggle" in paths
    assert "/api/internal/speakerphone/press" in paths


def test_internal_namespace_disjoint_from_admin_namespace() -> None:
    """Belt-and-braces: no internal path leaks into the admin router.

    Acceptance criterion 11 requires the internal namespace to be
    unreachable on the public TCP listener. The two routers are
    separate; this asserts no accidental overlap.
    """
    from meeting_scribe.routes.admin_speakerphone import router as admin_router
    from meeting_scribe.routes.internal_speakerphone import router as internal_router

    admin_paths = {route.path for route in admin_router.routes}
    internal_paths = {route.path for route in internal_router.routes}
    assert admin_paths.isdisjoint(internal_paths)
    assert all(p.startswith("/api/admin/") for p in admin_paths)
    assert all(p.startswith("/api/internal/") for p in internal_paths)


def test_uds_app_only_includes_internal_routes() -> None:
    from meeting_scribe.speakerphone.uds import build_app

    app = build_app()
    paths = {route.path for route in app.routes}
    # Every /api/* path on the UDS app must be in the internal namespace.
    api_paths = {p for p in paths if p.startswith("/api/")}
    assert api_paths
    assert all(p.startswith("/api/internal/") for p in api_paths)


def test_meeting_client_resolves_uds_path() -> None:
    from meeting_scribe.speakerphone.meeting_client import UdsMeetingClient

    client = UdsMeetingClient()
    assert str(client.path).endswith(".sock")
