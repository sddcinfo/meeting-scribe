"""Unit tests for the popout layout tree / presets / storage.

These modules are pure-JS and live at `window.PopoutLayout*` once loaded.
We exercise them from an in-browser page context so the same runtime
asserts the same behavior that ships to users — no Node shim required.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

pytestmark = pytest.mark.browser


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

_HOST = """<!doctype html>
<html><head><meta charset="utf-8"><title>popout layout unit</title></head>
<body>
<script src="/static/js/popout-layout.js"></script>
<script src="/static/js/popout-layout-presets.js"></script>
<script src="/static/js/popout-layout-storage.js"></script>
</body></html>
"""


@pytest.fixture
def unit_server() -> Generator[dict[str, Any]]:
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(_HOST)

    @app.get("/static/js/{fname}")
    async def js(fname: str):
        path = STATIC_DIR / "js" / fname
        if not path.exists():
            return HTMLResponse(status_code=404, content="")
        return FileResponse(path, media_type="application/javascript")

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time

    deadline = time.monotonic() + 5.0
    while (not server.started or not server.servers) and time.monotonic() < deadline:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield {"base_url": f"http://127.0.0.1:{port}"}
    server.should_exit = True
    thread.join(timeout=3.0)


def _load(page, url: str) -> None:
    page.goto(url, wait_until="networkidle")
    page.wait_for_function(
        "() => window.PopoutLayout && window.PopoutLayoutPresets && window.PopoutLayoutStorage",
        timeout=3000,
    )


def test_clampratio(page, unit_server):
    _load(page, unit_server["base_url"])
    assert page.evaluate("() => window.PopoutLayout.clampRatio(0.5)") == 0.5
    assert page.evaluate("() => window.PopoutLayout.clampRatio(0)") == 0.15
    assert page.evaluate("() => window.PopoutLayout.clampRatio(1.5)") == 0.85
    assert page.evaluate("() => window.PopoutLayout.clampRatio(NaN)") == 0.5


def test_collect_panels(page, unit_server):
    _load(page, unit_server["base_url"])
    panels = page.evaluate(
        """() => {
            const { PRESETS } = window.PopoutLayoutPresets;
            const out = {};
            for (const slug of Object.keys(PRESETS)) {
                out[slug] = window.PopoutLayout.collectPanels(PRESETS[slug].tree).sort();
            }
            return out;
        }"""
    )
    assert panels["translator"] == ["slides", "transcript"]
    assert panels["developer"] == ["terminal", "transcript"]
    assert panels["fullstack"] == ["slides", "terminal", "transcript"]
    assert panels["triple"] == ["slides", "terminal", "transcript"]
    assert panels["sidebyside"] == ["slides", "terminal", "transcript"]
    assert panels["demo"] == ["slides", "terminal"]


def test_split_ids_unique_and_namespaced(page, unit_server):
    _load(page, unit_server["base_url"])
    # Every preset validator must return [] (no errors).
    errs_by_slug = page.evaluate(
        """() => {
            const { PRESETS } = window.PopoutLayoutPresets;
            const out = {};
            for (const slug of Object.keys(PRESETS)) {
                out[slug] = window.PopoutLayout.validatePreset(PRESETS[slug].tree, slug);
            }
            return out;
        }"""
    )
    for slug, errs in errs_by_slug.items():
        assert errs == [], f"preset {slug} validator errors: {errs}"

    # And ids do not collide across presets (critical for per-preset ratio persistence).
    all_ids = page.evaluate(
        """() => {
            const { PRESETS } = window.PopoutLayoutPresets;
            const all = [];
            for (const slug of Object.keys(PRESETS)) {
                all.push(...window.PopoutLayout.collectSplitIds(PRESETS[slug].tree));
            }
            return all;
        }"""
    )
    assert len(all_ids) == len(set(all_ids)), f"split ids collide: {all_ids}"


def test_set_get_ratio_is_non_destructive(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const t = window.PopoutLayoutPresets.PRESETS.fullstack.tree;
            const t2 = L.setRatio(t, 'fullstack:bottom', 0.3);
            return {
                originalBottom: L.getRatio(t,  'fullstack:bottom'),
                updatedBottom:  L.getRatio(t2, 'fullstack:bottom'),
                originalMain:   L.getRatio(t,  'fullstack:main'),
                updatedMain:    L.getRatio(t2, 'fullstack:main'),
            };
        }"""
    )
    assert result["originalBottom"] == 0.65
    assert result["updatedBottom"] == 0.3
    assert result["originalMain"] == result["updatedMain"]


def test_set_ratio_clamps(page, unit_server):
    _load(page, unit_server["base_url"])
    assert (
        page.evaluate(
            """() => {
            const L = window.PopoutLayout;
            const t = window.PopoutLayoutPresets.PRESETS.fullstack.tree;
            return L.getRatio(L.setRatio(t, 'fullstack:bottom', 0.99), 'fullstack:bottom');
        }"""
        )
        == 0.85
    )
    assert (
        page.evaluate(
            """() => {
            const L = window.PopoutLayout;
            const t = window.PopoutLayoutPresets.PRESETS.fullstack.tree;
            return L.getRatio(L.setRatio(t, 'fullstack:bottom', -0.1), 'fullstack:bottom');
        }"""
        )
        == 0.15
    )


def test_resolve_renderable_tree_prunes(page, unit_server):
    _load(page, unit_server["base_url"])
    # fullstack with no terminal → transcript + slides, no empty rectangle.
    kept = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.fullstack.tree,
                { transcript: true, slides: true, terminal: false },
            );
            return L.collectPanels(resolved).sort();
        }"""
    )
    assert kept == ["slides", "transcript"]

    # fullstack with no slides → transcript + terminal (terminal promoted from inner split).
    kept = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.fullstack.tree,
                { transcript: true, slides: false, terminal: true },
            );
            return L.collectPanels(resolved).sort();
        }"""
    )
    assert kept == ["terminal", "transcript"]

    # no deck + no terminal → single-leaf transcript.
    shape = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.fullstack.tree,
                { transcript: true, slides: false, terminal: false },
            );
            return { kind: resolved.kind, panel: resolved.panel };
        }"""
    )
    assert shape == {"kind": "leaf", "panel": "transcript"}


def test_resolve_demo_with_no_deck_collapses_to_terminal(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.demo.tree,
                { transcript: true, slides: false, terminal: true },
            );
            return { kind: resolved.kind, panel: resolved.panel };
        }"""
    )
    # Demo without slides loses half — terminal leaf is promoted.
    assert result == {"kind": "leaf", "panel": "terminal"}


def test_storage_load_migrates_from_terminal_visible(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            localStorage.setItem('terminal_visible', '1');
            const st = window.PopoutLayoutStorage.load();
            return { preset: st.preset, lastTermPreset: st.lastTermPreset, lastNoTermPreset: st.lastNoTermPreset };
        }"""
    )
    assert result["preset"] == "fullstack"
    assert result["lastTermPreset"] == "fullstack"


def test_storage_default_on_blank_state(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const st = window.PopoutLayoutStorage.load();
            return { preset: st.preset, version: st.version };
        }"""
    )
    assert result["preset"] == "translator"
    assert result["version"] == 2


def test_storage_setRatio_is_namespaced_per_preset(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const S = window.PopoutLayoutStorage;
            let st = S.load();
            st = S.setPreset(st, 'fullstack');
            st = S.setRatio(st, 'fullstack:bottom', 0.3);
            st = S.setPreset(st, 'translator');
            st = S.setRatio(st, 'translator:main', 0.7);
            const raw = JSON.parse(localStorage.getItem('popout_layout_v2'));
            return raw.ratiosByPreset;
        }"""
    )
    assert result == {
        "fullstack": {"fullstack:bottom": 0.3},
        "translator": {"translator:main": 0.7},
    }


def test_storage_resolvedTree_applies_overrides(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const S = window.PopoutLayoutStorage;
            const L = window.PopoutLayout;
            let st = S.load();
            st = S.setPreset(st, 'fullstack');
            st = S.setRatio(st, 'fullstack:bottom', 0.25);
            const tree = S.resolvedTree(st);
            return {
                bottomRatio: L.getRatio(tree, 'fullstack:bottom'),
                mainRatio:   L.getRatio(tree, 'fullstack:main'),
            };
        }"""
    )
    assert result == {"bottomRatio": 0.25, "mainRatio": 0.5}


def test_term_toggle_memory_persists(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const S = window.PopoutLayoutStorage;
            let st = S.load();
            st = S.setPreset(st, 'sidebyside');
            st = S.setPreset(st, 'developer');
            st = S.setPreset(st, 'translator');
            return {
                lastTerm: st.lastTermPreset,
                lastNoTerm: st.lastNoTermPreset,
            };
        }"""
    )
    # sidebyside + developer both had terminal; last-term should be developer.
    assert result == {"lastTerm": "developer", "lastNoTerm": "translator"}


def test_validate_rejects_duplicate_split_id(page, unit_server):
    _load(page, unit_server["base_url"])
    errs = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const bad = L.split(
                'v', 0.5,
                L.split('h', 0.5, L.leaf('slides'), L.leaf('terminal'), 'x:dup'),
                L.split('h', 0.5, L.leaf('slides'), L.leaf('terminal'), 'x:dup'),
                'x:main',
            );
            return L.validatePreset(bad, 'x');
        }"""
    )
    assert any("duplicate" in e for e in errs)


def test_validate_rejects_cross_preset_slug(page, unit_server):
    _load(page, unit_server["base_url"])
    errs = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const bad = L.split('v', 0.5, L.leaf('slides'), L.leaf('terminal'), 'other:main');
            return L.validatePreset(bad, 'my-preset');
        }"""
    )
    assert any("namespaced" in e for e in errs)


# ── v2 edit operations ───────────────────────────────────────────


def test_split_at_wraps_leaf(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const tree = L.leaf('transcript');
            // Split the lone leaf at path [] right-side.
            const out = L.splitAt(tree, [], 'terminal', 'h', 'b', 'custom');
            return {
                kind: out.kind,
                dir: out.dir,
                a: out.a.panel,
                b: out.b.panel,
            };
        }"""
    )
    assert result == {"kind": "split", "dir": "h", "a": "transcript", "b": "terminal"}


def test_split_at_deep_target(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const tree = L.split('v', 0.5, L.leaf('transcript'), L.leaf('terminal'), 'custom:main');
            // Split terminal (at path ['b']) downward with slides.
            const out = L.splitAt(tree, ['b'], 'slides', 'v', 'b', 'custom');
            return {
                rootDir: out.dir,
                rightKind: out.b.kind,
                rightDir: out.b.dir,
                rightA: out.b.a.panel,
                rightB: out.b.b.panel,
            };
        }"""
    )
    assert result == {
        "rootDir": "v",
        "rightKind": "split",
        "rightDir": "v",
        "rightA": "terminal",
        "rightB": "slides",
    }


def test_remove_at_promotes_sibling(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const tree = L.split('v', 0.5, L.leaf('transcript'), L.leaf('terminal'), 'custom:main');
            const out = L.removeAt(tree, ['b']);
            return { kind: out.kind, panel: out.panel };
        }"""
    )
    assert result == {"kind": "leaf", "panel": "transcript"}


def test_remove_at_root_returns_null(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const tree = L.leaf('transcript');
            return L.removeAt(tree, []);
        }"""
    )
    assert result is None


def test_change_panel_at_updates_leaf(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const tree = L.split('v', 0.5, L.leaf('transcript'), L.leaf('terminal'), 'custom:main');
            const out = L.changePanelAt(tree, ['b'], 'slides');
            return { a: out.a.panel, b: out.b.panel };
        }"""
    )
    assert result == {"a": "transcript", "b": "slides"}


def test_find_panel_path(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const tree = L.split('v', 0.5,
                L.leaf('transcript'),
                L.split('h', 0.5, L.leaf('slides'), L.leaf('terminal'), 'c:bottom'),
                'c:main');
            return {
                transcript: L.findPanelPath(tree, 'transcript'),
                slides: L.findPanelPath(tree, 'slides'),
                terminal: L.findPanelPath(tree, 'terminal'),
                absent: L.findPanelPath(tree, 'nope'),
            };
        }"""
    )
    assert result == {
        "transcript": ["a"],
        "slides": ["b", "a"],
        "terminal": ["b", "b"],
        "absent": None,
    }


def test_storage_setCustomTree_transitions(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const S = window.PopoutLayoutStorage;
            const L = window.PopoutLayout;
            let st = S.load();
            // Simulate user dragging slides next to transcript via splitAt.
            const tree = L.split('v', 0.5, L.leaf('transcript'), L.leaf('terminal'), 'custom:1');
            st = S.setCustomTree(st, tree);
            return {
                preset: st.preset,
                storedPreset: JSON.parse(localStorage.getItem('popout_layout_v2')).preset,
                hasCustomTree: !!st.customTree,
                resolvedPanels: L.collectPanels(S.resolvedTree(st)).sort(),
                lastTermPreset: st.lastTermPreset,   // 'custom' since tree has terminal
            };
        }"""
    )
    assert result == {
        "preset": "custom",
        "storedPreset": "custom",
        "hasCustomTree": True,
        "resolvedPanels": ["terminal", "transcript"],
        "lastTermPreset": "custom",
    }
