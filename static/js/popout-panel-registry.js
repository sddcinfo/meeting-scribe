/* Popout panel registry — cached root elements + lifecycle hooks.
 *
 * The registry owns *cached DOM references* for the three singleton
 * panels. `renderLayout()` re-parents these nodes as the user switches
 * presets; because the cache holds the HTMLElement directly, detachment
 * during a re-layout never produces null.
 *
 * Panel implementations register themselves early in popout bootstrap.
 * Each `ensure()` returns the root element the renderer will move.
 * Each `notifyResize()` tells the panel to recompute internal layout
 * (xterm fit, slide viewer dims) after a move/resize.
 */

(function () {
  'use strict';

  const _roots = new Map();       // panelId → HTMLElement
  const _impls = new Map();       // panelId → { ensure, notifyResize? }

  function register(panelId, impl) {
    _impls.set(panelId, impl);
  }

  async function ensureRoot(panelId) {
    if (_roots.has(panelId)) return _roots.get(panelId);
    const impl = _impls.get(panelId);
    if (!impl) throw new Error(`popout-panel-registry: no impl for '${panelId}'`);
    const root = await impl.ensure();
    if (!(root instanceof HTMLElement)) {
      throw new Error(`popout-panel-registry: '${panelId}'.ensure() did not return an element`);
    }
    _roots.set(panelId, root);
    return root;
  }

  function notifyResize(panelId) {
    const impl = _impls.get(panelId);
    if (impl && typeof impl.notifyResize === 'function') {
      try { impl.notifyResize(); } catch (e) { console.warn('[panel-registry] notifyResize', panelId, e); }
    }
  }

  function notifyAll() {
    for (const id of _roots.keys()) notifyResize(id);
  }

  function cachedRoots() {
    return Array.from(_roots.entries());
  }

  window.PopoutPanelRegistry = { register, ensureRoot, notifyResize, notifyAll, cachedRoots };
})();
