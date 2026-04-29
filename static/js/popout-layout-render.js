/* Popout layout renderer.
 *
 *  · renderLayout(root, state, availability)
 *      Clears `root`, builds nested `.lyt-split` divs for the active
 *      preset's tree (after pruning unavailable panels), re-parents
 *      each cached panel root into its slot, and wires one resize
 *      gutter per split.
 *
 *  · Below 768px the tree auto-flattens to a vertical stack via
 *    buildMobileTree(). matchMedia watches the breakpoint and
 *    re-renders on cross only (never on same-mode resizes).
 *
 *  · Subscribes to 'popout-availability:change' — emitted by the app
 *    when slides/terminal availability flips at runtime — and re-
 *    renders the current preset with the updated availability.
 *
 *  · Emits 'popout-layout:resize' after every successful render so
 *    panels (xterm, slide viewer) can refit.
 *
 *  · v2: per-leaf-slot menu (split-right / split-down / swap / change /
 *    remove) + drag-and-drop rearrangement. Any of these mutations
 *    transitions the layout to `preset:'custom'` + customTree storage.
 */

(function () {
  'use strict';

  const BREAKPOINT = 768;
  const MOBILE_WEIGHTS = { transcript: 0.55, slides: 0.25, terminal: 0.2 };
  const PANEL_META = {
    transcript: {
      label: 'Transcript', emoji: '💬',
      // Two distinct empty states — the app flips between them via
      // _hasActiveMeeting() so we don't tell the user to "start a
      // meeting" while a meeting is already recording.
      emptyNoMeeting: 'Waiting for meeting audio.\nStart a meeting to begin live transcription + translation.',
      emptyActive:   'Listening for speech…\nTranscript + translation will appear here as people talk.',
    },
    slides:     { label: 'Slides',     emoji: '📊',
      empty: 'No slide deck loaded.\nUpload a .pptx via the “Slides” button in the header.' },
    terminal:   { label: 'Terminal',   emoji: '⌁',
      empty: 'Terminal connecting…' },
  };

  // Popout opens with no meeting visibility; the app pings /api/status
  // during init and sets this via setMeetingActive(). Default false keeps
  // the "Start a meeting" copy correct for cold-open popouts.
  let _meetingActive = false;

  let _state = null;
  let _availability = { transcript: true, slides: false, terminal: true };
  let _rootEl = null;
  let _mqList = null;
  let _onStateChange = null;    // callback set by install()
  let _renderGen = 0;           // incremented per renderLayout call
  let _pendingRender = null;    // promise of the in-flight render (for sequencing)

  function _computeAvailability(tree) {
    const panels = window.PopoutLayout.collectPanels(tree);
    // Slides used to be pruned from the tree when no deck was loaded,
    // which hid the "where slides go" slot entirely on multi-panel
    // presets (sidebyside / triple / fullstack). We now keep the slot
    // rendered so the empty-state hint ("Upload a .pptx…") stays
    // visible — the user is then aware that slides belong here.
    // `_availability.slides` still drives the empty/populated flag
    // via _isPanelEmpty().
    return {
      transcript: panels.includes('transcript'),
      slides:     panels.includes('slides'),
      terminal:   panels.includes('terminal'),
    };
  }

  function setAvailability(next) {
    _availability = { ..._availability, ...next };
    if (_rootEl && _state) renderLayout(_rootEl, _state);
  }

  function _isNarrow() {
    return window.matchMedia && window.matchMedia(`(max-width: ${BREAKPOINT - 1}px)`).matches;
  }

  function buildMobileTree(desktopTree, weights = MOBILE_WEIGHTS) {
    const L = window.PopoutLayout;
    const panels = L.collectPanels(desktopTree);
    const ordered = ['transcript', 'slides', 'terminal'].filter(p => panels.includes(p));
    if (ordered.length === 0) return desktopTree;
    if (ordered.length === 1) return L.leaf(ordered[0]);
    const picked = ordered.map(p => (p in weights ? weights[p] : 1 / ordered.length));
    const total = picked.reduce((s, w) => s + w, 0);
    const w = picked.map(x => x / total);
    function nest(i) {
      if (i === ordered.length - 1) return L.leaf(ordered[i]);
      const remaining = w.slice(i + 1).reduce((s, x) => s + x, 0);
      const ratio = w[i] / (w[i] + remaining);
      return L.split('v', ratio, L.leaf(ordered[i]), nest(i + 1), `mobile:${ordered[i]}→rest`);
    }
    return nest(0);
  }

  // ── Empty-state decoration ─────────────────────────────────────

  function _paintEmptyStates() {
    if (!_rootEl) return;
    const slots = _rootEl.querySelectorAll('.lyt-slot-leaf');
    slots.forEach((slot) => {
      const panel = slot.dataset.panel;
      const empty = _isPanelEmpty(panel);
      slot.dataset.empty = empty ? 'true' : 'false';
      if (empty) {
        slot.dataset.emptyText = _emptyTextFor(panel);
      } else {
        slot.removeAttribute('data-empty-text');
      }
    });
  }

  function _emptyTextFor(panel) {
    const meta = PANEL_META[panel];
    if (!meta) return '';
    if (panel === 'transcript') {
      return _meetingActive ? meta.emptyActive : meta.emptyNoMeeting;
    }
    return meta.empty || '';
  }

  function _isPanelEmpty(panel) {
    if (panel === 'slides') return !_availability.slides;
    if (panel === 'transcript') {
      const grid = document.getElementById('transcript-grid');
      // Heuristic: transcript is "empty" if it has zero utterance children.
      return !grid || grid.children.length === 0;
    }
    if (panel === 'terminal') {
      const dot = document.querySelector('.term-status');
      const state = dot?.dataset?.state;
      return !state || state === 'idle' || state === 'connecting';
    }
    return false;
  }

  // ── Per-leaf action menu ───────────────────────────────────────

  const ACTIONS = [
    { id: 'split-right', label: 'Split right',   title: 'Split this panel horizontally.',
      run: (path) => _mutateTree((t) => window.PopoutLayout.splitAt(t, path, _suggestPanel(path, 'a'), 'h', 'b', 'custom')) },
    { id: 'split-down',  label: 'Split down',    title: 'Split this panel vertically.',
      run: (path) => _mutateTree((t) => window.PopoutLayout.splitAt(t, path, _suggestPanel(path, 'a'), 'v', 'b', 'custom')) },
    { id: 'change',      label: 'Change panel…', title: 'Swap this slot to a different panel type.',
      run: (path, panel, el) => _openChangeSubmenu(path, panel, el) },
    { id: 'remove',      label: 'Remove',        title: 'Take this panel out of the layout.',
      run: (path) => _mutateTree((t) => window.PopoutLayout.removeAt(t, path)) },
  ];

  /** Pick a default panel for a newly split pane — prefer one not already in the tree. */
  function _suggestPanel(pathOfSplit, side) {
    if (!_state) return 'terminal';
    const tree = window.PopoutLayoutStorage.resolvedTree(_state);
    const existing = new Set(window.PopoutLayout.collectPanels(tree));
    for (const p of ['slides', 'terminal', 'transcript']) {
      if (!existing.has(p)) return p;
    }
    return 'terminal';
  }

  function _openMenu(anchor, path, panel) {
    _closeMenus();
    const menu = document.createElement('div');
    menu.className = 'lyt-menu';
    menu.setAttribute('role', 'menu');
    menu.innerHTML = `
      <div class="lyt-menu-header">${PANEL_META[panel]?.label || panel}</div>
      ${ACTIONS.map((a) => `
        <button type="button" class="lyt-menu-item" data-action="${a.id}"
                title="${a.title}">${a.label}</button>
      `).join('')}
    `;
    menu.addEventListener('click', (ev) => {
      const btn = ev.target.closest('[data-action]');
      if (!btn) return;
      const action = ACTIONS.find((a) => a.id === btn.dataset.action);
      if (action) action.run(path, panel, btn);
      _closeMenus();
    });
    document.body.appendChild(menu);
    const rect = anchor.getBoundingClientRect();
    menu.style.top  = `${rect.bottom + 4}px`;
    menu.style.left = `${rect.right - menu.offsetWidth}px`;
    // Clamp to viewport.
    const mRect = menu.getBoundingClientRect();
    if (mRect.right > window.innerWidth - 8) {
      menu.style.left = `${window.innerWidth - mRect.width - 8}px`;
    }
    if (mRect.bottom > window.innerHeight - 8) {
      menu.style.top = `${rect.top - mRect.height - 4}px`;
    }
    setTimeout(() => {
      document.addEventListener('click', _closeMenus, { once: true });
      document.addEventListener('keydown', _onMenuKey, { once: true });
    }, 0);
  }

  function _openChangeSubmenu(path, currentPanel, anchor) {
    _closeMenus();
    const menu = document.createElement('div');
    menu.className = 'lyt-menu lyt-submenu';
    menu.innerHTML = `
      <div class="lyt-menu-header">Change to…</div>
      ${window.PopoutLayout.PANELS.map((p) => `
        <button type="button" class="lyt-menu-item" data-panel="${p}"
                ${p === currentPanel ? 'disabled' : ''}>${PANEL_META[p]?.label || p}</button>
      `).join('')}
    `;
    menu.addEventListener('click', (ev) => {
      const btn = ev.target.closest('[data-panel]');
      if (!btn || btn.disabled) return;
      _mutateTree((t) => window.PopoutLayout.changePanelAt(t, path, btn.dataset.panel));
      _closeMenus();
    });
    document.body.appendChild(menu);
    const rect = anchor.getBoundingClientRect();
    menu.style.top  = `${rect.top}px`;
    menu.style.left = `${rect.right + 4}px`;
    setTimeout(() => {
      document.addEventListener('click', _closeMenus, { once: true });
    }, 0);
  }

  function _closeMenus() {
    document.querySelectorAll('.lyt-menu').forEach((m) => m.remove());
  }

  function _onMenuKey(ev) {
    if (ev.key === 'Escape') _closeMenus();
  }

  // ── Drag-and-drop (VS Code-style edge-drop zones) ──────────────

  let _drag = null;

  function _onHandleDragStart(ev, path, panel) {
    ev.dataTransfer.effectAllowed = 'move';
    ev.dataTransfer.setData('text/lyt-path', JSON.stringify(path));
    ev.dataTransfer.setData('text/lyt-panel', panel);
    _drag = { path, panel };
    document.body.classList.add('lyt-dragging');
  }
  function _onHandleDragEnd() {
    _drag = null;
    document.body.classList.remove('lyt-dragging');
    document.querySelectorAll('.lyt-slot-leaf').forEach((s) => {
      s.classList.remove('lyt-drop-top','lyt-drop-bottom','lyt-drop-left','lyt-drop-right','lyt-drop-center');
    });
  }

  function _dropZone(ev, slot) {
    const r = slot.getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width;
    const y = (ev.clientY - r.top) / r.height;
    const margin = 0.3;          // 30% edge bands
    if (x < margin) return 'left';
    if (x > 1 - margin) return 'right';
    if (y < margin) return 'top';
    if (y > 1 - margin) return 'bottom';
    return 'center';
  }

  function _onSlotDragOver(ev, slot) {
    if (!_drag) return;
    ev.preventDefault();
    ev.dataTransfer.dropEffect = 'move';
    const zone = _dropZone(ev, slot);
    ['top','bottom','left','right','center'].forEach((z) => {
      slot.classList.toggle(`lyt-drop-${z}`, z === zone);
    });
  }
  function _onSlotDragLeave(slot) {
    ['top','bottom','left','right','center'].forEach((z) => slot.classList.remove(`lyt-drop-${z}`));
  }
  function _onSlotDrop(ev, slot, targetPath, targetPanel) {
    if (!_drag) return;
    ev.preventDefault();
    const zone = _dropZone(ev, slot);
    _onSlotDragLeave(slot);
    const srcPath = _drag.path;
    const srcPanel = _drag.panel;
    if (JSON.stringify(srcPath) === JSON.stringify(targetPath)) return;   // no-op

    const L = window.PopoutLayout;
    _mutateTree((tree) => {
      // 1. Remove from source. This may collapse a parent split.
      let t = L.removeAt(tree, srcPath);
      if (!t) return tree;   // removing source would empty the tree; abort

      // 2. Locate the drop target AFTER removal (its path may have changed).
      const targetLeaf = (() => {
        // Heuristic: walk tree looking for the leaf with targetPanel.
        // Won't collide because the source has been removed.
        return L.findPanelPath(t, targetPanel);
      })();
      if (!targetLeaf) return tree;

      if (zone === 'center') {
        return L.changePanelAt(t, targetLeaf, srcPanel);    // swap roles
      }
      const dir  = (zone === 'left' || zone === 'right') ? 'h' : 'v';
      const side = (zone === 'right' || zone === 'bottom') ? 'b' : 'a';
      return L.splitAt(t, targetLeaf, srcPanel, dir, side, 'custom');
    });
    _drag = null;
  }

  // ── Tree mutations (persist as custom preset) ──────────────────

  function _mutateTree(fn) {
    if (!_state || !_rootEl) return;
    const S = window.PopoutLayoutStorage;
    const L = window.PopoutLayout;
    // Paths stored on leaf-chrome elements are paths in the tree that
    // was actually rendered, i.e. the pruned + (maybe mobile) tree. We
    // must mutate the same tree the user sees, or the paths won't line
    // up when the preset's stored tree differs from what's on screen.
    const baseTree = S.resolvedTree(_state);
    const viewTree = _isNarrow() ? buildMobileTree(baseTree) : baseTree;
    const availability = _computeAvailability(viewTree);
    const current = L.resolveRenderableTree(viewTree, availability) || viewTree;
    const next = fn(current);
    if (!next) return;                       // mutation refused (e.g. removing last leaf)
    _state = S.setCustomTree(_state, next);
    if (_onStateChange) _onStateChange(_state);
    renderLayout(_rootEl, _state);
  }

  // ── DOM construction ───────────────────────────────────────────

  function _buildLeafChrome(slot, node, path, panel) {
    // Small handle bar in the top-right of each leaf: a ⋮ menu button
    // and a drag handle. Positioned absolute so it floats over panel
    // content without reserving layout space. The ensureRoot content
    // is added AFTER this so it stacks behind.
    const chrome = document.createElement('div');
    chrome.className = 'lyt-leaf-chrome';
    chrome.innerHTML = `
      <span class="lyt-leaf-handle"
            draggable="true"
            role="button"
            tabindex="0"
            title="Drag to rearrange · ${PANEL_META[panel]?.label || panel}">⇕</span>
      <button type="button" class="lyt-leaf-menu-btn" aria-label="Panel options">⋮</button>
    `;
    const handle = chrome.querySelector('.lyt-leaf-handle');
    const menuBtn = chrome.querySelector('.lyt-leaf-menu-btn');
    handle.addEventListener('dragstart', (ev) => _onHandleDragStart(ev, path, panel));
    handle.addEventListener('dragend', _onHandleDragEnd);
    menuBtn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      _openMenu(menuBtn, path, panel);
    });
    slot.appendChild(chrome);

    // Drop zone wiring on the slot itself (not the chrome) so the
    // entire pane is a valid drop target.
    slot.addEventListener('dragover', (ev) => _onSlotDragOver(ev, slot));
    slot.addEventListener('dragleave', () => _onSlotDragLeave(slot));
    slot.addEventListener('drop', (ev) => _onSlotDrop(ev, slot, path, panel));
  }

  async function _buildSlot(node, ratio, onRatioCommit, path) {
    const slot = document.createElement('div');
    slot.className = 'lyt-slot';
    if (ratio !== null) slot.style.setProperty('--slot-flex', String(ratio));
    if (window.PopoutLayout.isLeaf(node)) {
      slot.classList.add('lyt-slot-leaf');
      slot.dataset.panel = node.panel;
      _buildLeafChrome(slot, node, path, node.panel);
      const root = await window.PopoutPanelRegistry.ensureRoot(node.panel);
      root.hidden = false;
      slot.appendChild(root);
    } else {
      const nested = await _buildSplit(node, onRatioCommit, path);
      slot.appendChild(nested);
    }
    return slot;
  }

  async function _buildSplit(node, onRatioCommit, path) {
    const splitEl = document.createElement('div');
    splitEl.className = 'lyt-split';
    splitEl.dataset.dir = node.dir;
    splitEl.dataset.id = node.id;
    splitEl.dataset.ratio = String(node.ratio);

    const slotA = await _buildSlot(node.a, node.ratio, onRatioCommit, [...path, 'a']);
    const gutter = document.createElement('div');
    gutter.className = 'lyt-gutter';
    gutter.setAttribute('role', 'separator');
    gutter.setAttribute('aria-orientation', node.dir === 'h' ? 'vertical' : 'horizontal');
    const slotB = await _buildSlot(node.b, 1 - node.ratio, onRatioCommit, [...path, 'b']);

    splitEl.appendChild(slotA);
    splitEl.appendChild(gutter);
    splitEl.appendChild(slotB);

    window.PopoutLayoutResizer.createResizer(gutter, splitEl, {
      dir: node.dir,
      onCommit: (r) => onRatioCommit(node.id, r),
    });

    return splitEl;
  }

  async function _buildTop(node, onRatioCommit) {
    if (window.PopoutLayout.isLeaf(node)) {
      const wrap = document.createElement('div');
      wrap.className = 'lyt-slot lyt-slot-leaf lyt-slot-toplevel';
      wrap.dataset.panel = node.panel;
      _buildLeafChrome(wrap, node, [], node.panel);
      const root = await window.PopoutPanelRegistry.ensureRoot(node.panel);
      root.hidden = false;
      wrap.appendChild(root);
      return wrap;
    }
    return _buildSplit(node, onRatioCommit, []);
  }

  async function renderLayout(rootEl, state) {
    // Serialize renders: each call bumps the generation counter and
    // waits for any in-flight render to finish before starting. Every
    // await point checks that the generation hasn't moved on — if a
    // newer render started, we abort and let the newer one commit. This
    // prevents two renders racing on ensureRoot + appendChild and
    // ending up with panels in slots from the wrong preset (the bug
    // that made the transcript vanish after a rapid preset swap).
    const myGen = ++_renderGen;
    const prev = _pendingRender;
    let resolve;
    _pendingRender = new Promise((r) => { resolve = r; });
    if (prev) { try { await prev; } catch {} }
    if (myGen !== _renderGen) { resolve(); return; }

    try {
      await _doRender(rootEl, state, myGen);
    } finally {
      if (myGen === _renderGen) _pendingRender = null;
      resolve();
    }
  }

  async function _doRender(rootEl, state, myGen) {
    _rootEl = rootEl;
    _state = state;
    _closeMenus();

    const S = window.PopoutLayoutStorage;
    const L = window.PopoutLayout;
    const baseTree = S.resolvedTree(state);

    const tree = _isNarrow() ? buildMobileTree(baseTree) : baseTree;
    const presetAvailability = _computeAvailability(tree);
    const renderable = L.resolveRenderableTree(tree, presetAvailability);

    if (!renderable) {
      const root = await window.PopoutPanelRegistry.ensureRoot('transcript');
      if (myGen !== _renderGen) return;
      rootEl.textContent = '';
      const wrap = document.createElement('div');
      wrap.className = 'lyt-slot lyt-slot-leaf lyt-slot-toplevel';
      wrap.dataset.panel = 'transcript';
      root.hidden = false;
      wrap.appendChild(root);
      rootEl.appendChild(wrap);
      _paintEmptyStates();
      return;
    }

    function onRatioCommit(id, ratio) {
      _state = S.setRatio(_state, id, ratio);
      if (_onStateChange) _onStateChange(_state);
    }
    const newSubtree = await _buildTop(renderable, onRatioCommit);
    if (myGen !== _renderGen) return;   // a newer render arrived — drop ours
    rootEl.textContent = '';
    rootEl.appendChild(newSubtree);

    requestAnimationFrame(() => {
      if (myGen !== _renderGen) return;
      _paintEmptyStates();
      window.PopoutPanelRegistry.notifyAll();
      rootEl.dispatchEvent(new CustomEvent('popout-layout:resize', { bubbles: true }));
      window.dispatchEvent(new CustomEvent('popout-layout:resize'));
    });
  }

  function install(rootEl, state, availability = {}, opts = {}) {
    _availability = { transcript: true, slides: false, terminal: true, ...availability };
    _onStateChange = opts.onStateChange || null;
    renderLayout(rootEl, state);

    if (_mqList) _mqList.removeEventListener('change', _onBreakpointCross);
    _mqList = window.matchMedia(`(max-width: ${BREAKPOINT - 1}px)`);
    _mqList.addEventListener('change', _onBreakpointCross);

    window.addEventListener('popout-availability:change', _onAvailabilityChange);
    // Transcript grows rows via CompactGridRenderer → watch its child list
    // so empty-state placeholder flips off as utterances arrive.
    const grid = document.getElementById('transcript-grid');
    if (grid) {
      const obs = new MutationObserver(() => _paintEmptyStates());
      obs.observe(grid, { childList: true });
    }
  }

  function _onBreakpointCross() {
    if (_rootEl && _state) renderLayout(_rootEl, _state);
  }

  function _onAvailabilityChange(e) {
    if (e && e.detail) setAvailability(e.detail);
  }

  function setMeetingActive(active) {
    const next = !!active;
    if (next === _meetingActive) return;
    _meetingActive = next;
    _paintEmptyStates();
  }

  window.PopoutLayoutRender = {
    install,
    renderLayout,
    buildMobileTree,
    setAvailability,
    setMeetingActive,
    _state: () => _state,   // test hook
  };
})();
