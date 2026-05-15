/* Popout layout persistence.
 *
 * Stores { version, preset, lastTermPreset, lastNoTermPreset, ratiosByPreset }
 * under a single localStorage key. Ratios are namespaced per preset so
 * editing `fullstack:main` never touches `translator:main`.
 *
 * First-load migration: if no layout key but any of the old ad-hoc keys
 * (terminal_visible, popout_slides_lang) are present, infer a preset
 * from the legacy state — never regress the user from "terminal was
 * open" to "no terminal".
 */

(function () {
  'use strict';

  const KEY = 'popout_layout_v2';
  const VERSION = 2;

  const DEFAULTS = {
    version: VERSION,
    preset: 'translator',
    lastTermPreset: 'triple',
    lastNoTermPreset: 'translator',
    ratiosByPreset: {},
    // v2: when the user runs any edit op (split/swap/remove) the active
    // preset becomes 'custom' and the resulting tree is stored here.
    customTree: null,
  };

  function _validSlug(slug) {
    return slug && (
      slug === 'custom' ||
      window.PopoutLayoutPresets.PRESET_ORDER.includes(slug)
    );
  }

  function loadRaw() {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj || obj.version !== VERSION) return null;
      return obj;
    } catch {
      return null;
    }
  }

  function saveRaw(state) {
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
    } catch (e) {
      console.warn('[popout-layout] save failed', e);
    }
  }

  /** First-load migration from the old scalar keys. Idempotent. */
  function inferLegacyPreset() {
    const termVisible = localStorage.getItem('terminal_visible') === '1';
    // The 6-preset history collapsed to 3 (translate / translator /
    // triple). Map "terminal-visible" to triple (the only remaining
    // preset that includes a terminal pane); otherwise default to
    // translator (Presentation: transcript top, slides below).
    if (termVisible) return 'triple';
    return 'translator';
  }

  /** Map a removed preset slug to its closest current equivalent.
   *  Called from `load()` so a stale localStorage from before the
   *  6→3 collapse falls back to a valid view instead of breaking. */
  function _migrateRemovedSlug(slug) {
    const removed = {
      developer: 'translate',     // transcript-only is the closest no-slides view
      fullstack: 'triple',        // had transcript + slides + terminal
      sidebyside: 'triple',       // had transcript + slides + terminal
      demo: 'translator',         // slides+terminal collapses to slides-only here
    };
    return removed[slug];
  }

  function load() {
    const existing = loadRaw();
    if (existing) {
      const merged = {
        ...DEFAULTS,
        ...existing,
        ratiosByPreset: { ...(existing.ratiosByPreset || {}) },
      };
      // Migrate any removed slug (developer / fullstack / sidebyside /
      // demo) to its closest current equivalent so a stale localStorage
      // doesn't break the picker.
      const replacement = _migrateRemovedSlug(merged.preset);
      if (replacement) {
        merged.preset = replacement;
        saveRaw(merged);
      }
      const ltReplacement = _migrateRemovedSlug(merged.lastTermPreset);
      if (ltReplacement) merged.lastTermPreset = ltReplacement;
      const lntReplacement = _migrateRemovedSlug(merged.lastNoTermPreset);
      if (lntReplacement) merged.lastNoTermPreset = lntReplacement;
      return merged;
    }
    // Migrate on first load.
    const preset = inferLegacyPreset();
    const migrated = {
      ...DEFAULTS,
      preset,
      lastTermPreset:   window.PopoutLayoutPresets.hasTerminal(preset) ? preset : 'triple',
      lastNoTermPreset: window.PopoutLayoutPresets.hasTerminal(preset) ? 'translator' : preset,
    };
    saveRaw(migrated);
    return migrated;
  }

  function save(state) { saveRaw(state); return state; }

  function setPreset(state, slug) {
    if (!_validSlug(slug)) return state;
    const next = { ...state, preset: slug };
    if (window.PopoutLayoutPresets.hasTerminal(slug)) next.lastTermPreset = slug;
    else next.lastNoTermPreset = slug;
    save(next);
    return next;
  }

  function setRatio(state, splitId, ratio) {
    const preset = state.preset;
    const current = state.ratiosByPreset[preset] || {};
    const next = {
      ...state,
      ratiosByPreset: {
        ...state.ratiosByPreset,
        [preset]: { ...current, [splitId]: window.PopoutLayout.clampRatio(ratio) },
      },
    };
    save(next);
    return next;
  }

  /** Resolve the tree + applied ratio overrides for the currently active preset.
   *  When preset === 'custom', the stored customTree is authoritative. */
  function resolvedTree(state) {
    if (state.preset === 'custom' && state.customTree) {
      return window.PopoutLayout.deepClone(state.customTree);
    }
    const preset = window.PopoutLayoutPresets.get(state.preset);
    const overrides = (state.ratiosByPreset && state.ratiosByPreset[preset.slug]) || {};
    return window.PopoutLayout.applyRatios(
      window.PopoutLayout.deepClone(preset.tree),
      overrides,
    );
  }

  /** Transition the active layout to a custom tree (triggered by any
   *  edit-menu or drag-drop mutation). */
  function setCustomTree(state, tree) {
    const next = { ...state, preset: 'custom', customTree: tree };
    // Custom layouts can include or exclude terminal too — keep last-*
    // memory accurate so Ctrl+Shift+T still works.
    const hasTerm = window.PopoutLayout.collectPanels(tree).includes('terminal');
    if (hasTerm) next.lastTermPreset = 'custom';
    else next.lastNoTermPreset = 'custom';
    save(next);
    return next;
  }

  // ── Server-side sync ───────────────────────────────────────
  //
  // The laptop admin's popout layout mirrors to the HDMI kiosk
  // browser via PUT /api/admin/popout-layout + WS event
  // 'popout_layout_changed'. We:
  //
  //  * fetch the server layout once on boot; if present, apply
  //    authoritatively over local cache and seed localVersion;
  //  * debounce-PUT on every local save() so a quick drag of a
  //    pane ratio batches into one POST;
  //  * subscribe to the popout_layout_changed event via the global
  //    ``scribeWs`` published by the admin SPA boot; on a remote
  //    version > local, apply silently (no echo);
  //  * the kiosk role never PUTs (the server's role gate would
  //    403 anyway), the role check is just a UX optimization.

  let _localVersion = 0;
  let _tabId = null;
  function _ensureTabId() {
    if (_tabId) return _tabId;
    try {
      const url = new URL(window.location.href);
      const fromHash = (url.hash || "").match(/tabId=([0-9a-f-]+)/i);
      if (fromHash) {
        _tabId = fromHash[1];
        return _tabId;
      }
    } catch {}
    _tabId = (crypto && crypto.randomUUID ? crypto.randomUUID() : String(Math.random()));
    return _tabId;
  }
  function _isKioskRole() {
    try { return document.body && document.body.dataset.role === 'kiosk'; } catch { return false; }
  }
  let _putTimer = null;
  function _scheduleServerPut(state) {
    if (_isKioskRole()) return;  // kiosk never writes
    if (_putTimer) clearTimeout(_putTimer);
    _putTimer = setTimeout(async () => {
      _putTimer = null;
      try {
        const resp = await fetch('/api/admin/popout-layout', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ layout: state, source_tab_id: _ensureTabId() }),
          credentials: 'same-origin',
        });
        if (resp.ok) {
          const data = await resp.json();
          if (typeof data.version === 'number' && data.version > _localVersion) {
            _localVersion = data.version;
          }
        }
      } catch (err) {
        console.warn('[popout-layout] server PUT failed', err);
      }
    }, 200);
  }

  async function _seedFromServer() {
    // Kiosk: read-only narrow projection at /api/kiosk/popout-layout.
    // Admin: full endpoint at /api/admin/popout-layout. Try kiosk
    // first so a kiosk-cookied browser doesn't 403 on the admin path.
    const tryPaths = ['/api/kiosk/popout-layout', '/api/admin/popout-layout'];
    for (const path of tryPaths) {
      try {
        const resp = await fetch(path, { credentials: 'same-origin' });
        if (!resp.ok) continue;
        const body = await resp.json();
        const layout = body && body.layout;
        const version = (body && typeof body.version === 'number') ? body.version : 0;
        if (layout && typeof layout === 'object') {
          saveRaw(layout);
          _localVersion = version;
          return layout;
        }
        if (typeof version === 'number') _localVersion = version;
        return null;
      } catch {
        // try next
      }
    }
    return null;
  }

  function _handleRemoteLayoutEvent(msg) {
    if (!msg || msg.type !== 'popout_layout_changed') return;
    if (typeof msg.version === 'number' && msg.version <= _localVersion) return;
    if (msg.source_tab_id && msg.source_tab_id === _ensureTabId()) {
      // Our own echo - update version but skip the re-render.
      if (typeof msg.version === 'number') _localVersion = msg.version;
      return;
    }
    if (!msg.layout || typeof msg.layout !== 'object') return;
    saveRaw(msg.layout);
    if (typeof msg.version === 'number') _localVersion = msg.version;
    // Fire a CustomEvent so the popout render module can pick the
    // change up without coupling to this storage module.
    try {
      window.dispatchEvent(new CustomEvent('popout-layout:remote-apply', {
        detail: { layout: msg.layout, version: _localVersion },
      }));
    } catch {}
  }

  // Public hook for the WS dispatcher (in scribe/features/view-only-ws.js
  // and the popout SPA) to call once it has a message. Returns true if
  // the event was recognized + handled by this module.
  function handleWsMessage(msg) {
    if (!msg || typeof msg !== 'object') return false;
    if (msg.type !== 'popout_layout_changed') return false;
    _handleRemoteLayoutEvent(msg);
    return true;
  }

  // Wrap save() so every local save() schedules a server PUT.
  const _origSave = save;
  function saveWithServerSync(state) {
    _origSave(state);
    _scheduleServerPut(state);
    return state;
  }

  window.PopoutLayoutStorage = {
    KEY, VERSION, DEFAULTS,
    load,
    save: saveWithServerSync,
    setPreset, setRatio,
    resolvedTree,
    setCustomTree,
    handleWsMessage,
    seedFromServer: _seedFromServer,
    _inferLegacyPreset: inferLegacyPreset,  // exposed for tests
  };

  // Best-effort initial seed: don't await — the legacy code path
  // calls load() synchronously on boot and the server seed will
  // refresh into place asynchronously. The remote-apply event will
  // re-render even if the first load was from localStorage.
  _seedFromServer().catch(() => {});
})();
