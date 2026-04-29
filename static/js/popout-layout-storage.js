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

  window.PopoutLayoutStorage = {
    KEY, VERSION, DEFAULTS,
    load, save,
    setPreset, setRatio,
    resolvedTree,
    setCustomTree,
    _inferLegacyPreset: inferLegacyPreset,  // exposed for tests
  };
})();
