"""Route-parity regression gate.

This test guards the FastAPI app's externally visible surface during
``server.py`` decomposition.
Every entry in ``app.routes`` is captured into a stable JSON inventory
covering path, methods, route class, endpoint identity, response class,
status code, and dependency callable identity. The test compares the
live inventory against ``tests/fixtures/route_inventory_baseline.json``
on every run.

Updating the baseline:
    Set ``UPDATE_ROUTE_BASELINE=1`` in the environment and run the
    test once to regenerate the fixture. Commit the change in the
    same commit that intentionally renames/adds/removes a route, with
    a one-line rationale in the commit message.

Why dependency callable identity matters:
    Recording only ``[type(d).__name__ for d in route.dependencies]``
    collapses every guard to ``Depends`` or ``Security``. A handler
    silently swapping ``_require_admin`` for a less restrictive guard
    would slip through. We record the resolved callable's
    ``__module__.__qualname__`` plus security scopes, sourced from
    ``route.dependant.dependencies`` (FastAPI's normalized tree).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

_BASELINE = Path(__file__).parent / "fixtures" / "route_inventory_baseline.json"


def _qualname(fn: Any) -> str:
    mod = getattr(fn, "__module__", "?") or "?"
    qn = getattr(fn, "__qualname__", repr(fn))
    return f"{mod}.{qn}"


def _capture_dependencies(route: Any) -> list[dict]:
    """Return ordered list of dependency-callable identities for ``route``.

    Uses FastAPI's resolved ``route.dependant.dependencies`` tree so the
    underlying callable (``_require_admin`` etc.) is visible — not just
    the wrapper type (``Depends``/``Security``).
    """
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return []
    deps = getattr(dependant, "dependencies", None) or []
    out: list[dict] = []
    for d in deps:
        call = getattr(d, "call", None)
        if call is None:
            continue
        scopes = list(getattr(d, "security_scopes", []) or [])
        out.append({"call": _qualname(call), "security_scopes": scopes})
    return out


def _route_record(route: Any) -> dict:
    rec: dict[str, Any] = {
        "type": type(route).__name__,
        "path": getattr(route, "path", None),
        "name": getattr(route, "name", None),
    }
    methods = getattr(route, "methods", None)
    if methods is not None:
        rec["methods"] = sorted(methods)
    endpoint = getattr(route, "endpoint", None)
    if endpoint is not None:
        rec["endpoint"] = _qualname(endpoint)
    response_class = getattr(route, "response_class", None)
    if response_class is not None:
        # Wrapped via DefaultPlaceholder in some FastAPI versions; unwrap if possible.
        inner = getattr(response_class, "value", response_class)
        rec["response_class"] = _qualname(inner) if callable(inner) else type(inner).__name__
    status_code = getattr(route, "status_code", None)
    if status_code is not None:
        rec["status_code"] = status_code
    rec["dependencies"] = _capture_dependencies(route)
    sub_app = getattr(route, "app", None)
    if sub_app is not None and rec["type"] in {"Mount", "Host"}:
        rec["mounted_app"] = type(sub_app).__name__
    return rec


def _capture_inventory() -> list[dict]:
    # Importing `app` runs server.py module-level init; that's fine —
    # the parity test is exactly meant to anchor that public surface.
    from meeting_scribe.server import app

    records = [_route_record(r) for r in app.routes]
    # Sort by (path, methods or "", type) so commit-order doesn't leak in.
    return sorted(
        records,
        key=lambda r: (r.get("path") or "", "|".join(r.get("methods", [])), r.get("type", "")),
    )


def _maybe_update_baseline(inventory: list[dict]) -> bool:
    if os.environ.get("UPDATE_ROUTE_BASELINE") != "1":
        return False
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return True


def test_route_inventory_matches_baseline() -> None:
    inventory = _capture_inventory()
    if _maybe_update_baseline(inventory):
        pytest.skip(
            "UPDATE_ROUTE_BASELINE=1 — wrote new baseline to "
            f"{_BASELINE.relative_to(Path.cwd())} (re-run without the env var to verify)"
        )
    if not _BASELINE.exists():
        pytest.fail(
            f"baseline missing at {_BASELINE}. "
            "Run with UPDATE_ROUTE_BASELINE=1 to capture the initial inventory."
        )
    expected = json.loads(_BASELINE.read_text())
    if inventory != expected:
        # Generate a compact, reviewable diff in the failure message.
        # The full inventories can be hundreds of records — show only
        # the keys that differ.
        from difflib import unified_diff

        live = json.dumps(inventory, indent=2, sort_keys=True).splitlines()
        base = json.dumps(expected, indent=2, sort_keys=True).splitlines()
        diff = "\n".join(unified_diff(base, live, fromfile="baseline", tofile="live", lineterm=""))
        pytest.fail(
            "route inventory drifted from baseline. If the change is "
            "intentional, set UPDATE_ROUTE_BASELINE=1 and re-run, then "
            "commit the new fixture with a one-line rationale.\n\n" + diff
        )
