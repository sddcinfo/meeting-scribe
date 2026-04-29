"""Unit tests for the popout layout tree / presets / storage.

These modules are pure-JS and live at ``window.PopoutLayout*`` once
loaded. We exercise them from an in-browser page context so the same
runtime asserts the same behavior that ships to users — no Node shim
required.

The current registry has three presets (see
``static/js/popout-layout-presets.js``):

  * ``translate``  — single ``transcript`` leaf, no terminal.
  * ``translator`` — ``transcript`` over ``slides`` (Presentation), no terminal.
  * ``triple``     — ``transcript`` over (``slides`` over ``terminal``); the
                     only preset that includes a terminal pane.
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


# ── Preset registry ───────────────────────────────────────────────


def test_preset_registry_lists_three_presets_in_order(page, unit_server):
    _load(page, unit_server["base_url"])
    order = page.evaluate("() => window.PopoutLayoutPresets.PRESET_ORDER")
    assert order == ["translate", "translator", "triple"]


def test_preset_panel_membership(page, unit_server):
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
    assert panels == {
        "translate": ["transcript"],
        "translator": ["slides", "transcript"],
        "triple": ["slides", "terminal", "transcript"],
    }


def test_only_triple_carries_terminal(page, unit_server):
    """``hasTerminal`` is the predicate the keyboard shortcut + storage
    use to decide last-term vs last-no-term memory. Triple is the only
    current preset with a terminal pane.
    """
    _load(page, unit_server["base_url"])
    flags = page.evaluate(
        """() => {
            const P = window.PopoutLayoutPresets;
            return {
                translate: P.hasTerminal('translate'),
                translator: P.hasTerminal('translator'),
                triple: P.hasTerminal('triple'),
            };
        }"""
    )
    assert flags == {"translate": False, "translator": False, "triple": True}


def test_unknown_slug_falls_back_to_translator(page, unit_server):
    _load(page, unit_server["base_url"])
    slug = page.evaluate("() => window.PopoutLayoutPresets.get('not-a-preset').slug")
    assert slug == "translator"


def test_split_ids_unique_and_namespaced(page, unit_server):
    _load(page, unit_server["base_url"])
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


# ── Tree utilities ────────────────────────────────────────────────


def test_clampratio(page, unit_server):
    _load(page, unit_server["base_url"])
    assert page.evaluate("() => window.PopoutLayout.clampRatio(0.5)") == 0.5
    assert page.evaluate("() => window.PopoutLayout.clampRatio(0)") == 0.15
    assert page.evaluate("() => window.PopoutLayout.clampRatio(1.5)") == 0.85
    assert page.evaluate("() => window.PopoutLayout.clampRatio(NaN)") == 0.5


def test_set_get_ratio_is_non_destructive(page, unit_server):
    """Editing ``triple:bottom`` must leave ``triple:main`` untouched."""
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const t = window.PopoutLayoutPresets.PRESETS.triple.tree;
            const t2 = L.setRatio(t, 'triple:bottom', 0.3);
            return {
                originalBottom: L.getRatio(t,  'triple:bottom'),
                updatedBottom:  L.getRatio(t2, 'triple:bottom'),
                originalMain:   L.getRatio(t,  'triple:main'),
                updatedMain:    L.getRatio(t2, 'triple:main'),
            };
        }"""
    )
    # Triple ships with main=0.45, bottom=0.55.
    assert result["originalBottom"] == 0.55
    assert result["updatedBottom"] == 0.3
    assert result["originalMain"] == result["updatedMain"]


def test_set_ratio_clamps(page, unit_server):
    _load(page, unit_server["base_url"])
    assert (
        page.evaluate(
            """() => {
            const L = window.PopoutLayout;
            const t = window.PopoutLayoutPresets.PRESETS.triple.tree;
            return L.getRatio(L.setRatio(t, 'triple:bottom', 0.99), 'triple:bottom');
        }"""
        )
        == 0.85
    )
    assert (
        page.evaluate(
            """() => {
            const L = window.PopoutLayout;
            const t = window.PopoutLayoutPresets.PRESETS.triple.tree;
            return L.getRatio(L.setRatio(t, 'triple:bottom', -0.1), 'triple:bottom');
        }"""
        )
        == 0.15
    )


def test_resolve_renderable_tree_prunes(page, unit_server):
    """``resolveRenderableTree`` collapses unavailable panels and
    promotes siblings up the tree."""
    _load(page, unit_server["base_url"])
    # Triple with no terminal → transcript + slides.
    kept = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.triple.tree,
                { transcript: true, slides: true, terminal: false },
            );
            return L.collectPanels(resolved).sort();
        }"""
    )
    assert kept == ["slides", "transcript"]

    # Triple with no slides → transcript + terminal.
    kept = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.triple.tree,
                { transcript: true, slides: false, terminal: true },
            );
            return L.collectPanels(resolved).sort();
        }"""
    )
    assert kept == ["terminal", "transcript"]

    # No deck + no terminal → single-leaf transcript.
    shape = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.triple.tree,
                { transcript: true, slides: false, terminal: false },
            );
            return { kind: resolved.kind, panel: resolved.panel };
        }"""
    )
    assert shape == {"kind": "leaf", "panel": "transcript"}


def test_resolve_translator_with_no_deck_collapses_to_transcript(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            const L = window.PopoutLayout;
            const resolved = L.resolveRenderableTree(
                window.PopoutLayoutPresets.PRESETS.translator.tree,
                { transcript: true, slides: false, terminal: false },
            );
            return { kind: resolved.kind, panel: resolved.panel };
        }"""
    )
    # Translator without slides loses half — transcript leaf is promoted.
    assert result == {"kind": "leaf", "panel": "transcript"}


# ── Storage ───────────────────────────────────────────────────────


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


def test_storage_load_migrates_terminal_visible_to_triple(page, unit_server):
    """Legacy: ``terminal_visible=1`` was the old per-key boolean. The
    6→3 collapse remapped the only terminal-bearing preset to ``triple``.
    """
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            localStorage.setItem('terminal_visible', '1');
            const st = window.PopoutLayoutStorage.load();
            return {
                preset: st.preset,
                lastTermPreset: st.lastTermPreset,
                lastNoTermPreset: st.lastNoTermPreset,
            };
        }"""
    )
    assert result["preset"] == "triple"
    assert result["lastTermPreset"] == "triple"


def test_storage_migrates_removed_slug(page, unit_server):
    """A stale localStorage from before the 6→3 collapse must be
    rewritten on load — ``developer/fullstack/sidebyside/demo`` were
    removed and have an explicit replacement map.
    """
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            // Seed a stale v2 record with a removed preset.
            localStorage.setItem('popout_layout_v2', JSON.stringify({
                version: 2,
                preset: 'fullstack',
                lastTermPreset: 'sidebyside',
                lastNoTermPreset: 'developer',
                ratiosByPreset: {},
            }));
            const st = window.PopoutLayoutStorage.load();
            return {
                preset: st.preset,
                lastTermPreset: st.lastTermPreset,
                lastNoTermPreset: st.lastNoTermPreset,
            };
        }"""
    )
    # Mapping per popout-layout-storage.js#_migrateRemovedSlug.
    assert result == {
        "preset": "triple",            # fullstack → triple
        "lastTermPreset": "triple",    # sidebyside → triple
        "lastNoTermPreset": "translate",  # developer → translate
    }


def test_storage_setRatio_is_namespaced_per_preset(page, unit_server):
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const S = window.PopoutLayoutStorage;
            let st = S.load();
            st = S.setPreset(st, 'triple');
            st = S.setRatio(st, 'triple:bottom', 0.3);
            st = S.setPreset(st, 'translator');
            st = S.setRatio(st, 'translator:main', 0.7);
            const raw = JSON.parse(localStorage.getItem('popout_layout_v2'));
            return raw.ratiosByPreset;
        }"""
    )
    assert result == {
        "triple": {"triple:bottom": 0.3},
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
            st = S.setPreset(st, 'triple');
            st = S.setRatio(st, 'triple:bottom', 0.25);
            const tree = S.resolvedTree(st);
            return {
                bottomRatio: L.getRatio(tree, 'triple:bottom'),
                mainRatio:   L.getRatio(tree, 'triple:main'),
            };
        }"""
    )
    # Override applies to bottom; main keeps its preset default of 0.45.
    assert result == {"bottomRatio": 0.25, "mainRatio": 0.45}


def test_term_toggle_memory_persists(page, unit_server):
    """``setPreset`` records the last-with-terminal and last-without
    presets so the Ctrl+Shift+T toggle has a target to swap to.
    """
    _load(page, unit_server["base_url"])
    result = page.evaluate(
        """() => {
            localStorage.clear();
            const S = window.PopoutLayoutStorage;
            let st = S.load();
            st = S.setPreset(st, 'translate');
            st = S.setPreset(st, 'triple');
            st = S.setPreset(st, 'translator');
            return {
                lastTerm: st.lastTermPreset,
                lastNoTerm: st.lastNoTermPreset,
            };
        }"""
    )
    # triple is the terminal-bearing preset; translator was last without.
    assert result == {"lastTerm": "triple", "lastNoTerm": "translator"}


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
            const tree = L.split('v', 0.5, L.leaf('transcript'), L.leaf('terminal'), 'custom:1');
            st = S.setCustomTree(st, tree);
            return {
                preset: st.preset,
                storedPreset: JSON.parse(localStorage.getItem('popout_layout_v2')).preset,
                hasCustomTree: !!st.customTree,
                resolvedPanels: L.collectPanels(S.resolvedTree(st)).sort(),
                lastTermPreset: st.lastTermPreset,
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
