// Meeting Scribe — Pop-out window SPA.
//
// The pop-out (`?popout=view`) is the customer-facing surface: a clean
// translate/slides/terminal layout opened in its own browser window
// (often projected to a TV, floated in PiP, or mirrored to the HDMI
// kiosk). It runs in the SAME admin SPA bundle as the operator view
// but takes a completely different code path — different header
// chrome, its own WebSocket lifecycle (`/api/ws/view` with auto-
// reconnect + reset-state-on-meeting-rollover), its own slide
// viewer with pane-by-language alignment, its own keyboard nav, its
// own layout-preset picker (translate / translator / triple / etc.).
//
// The admin SPA boot orchestrator stamps `popout-view` + `view-only`
// onto `<body>` before any DOM measurements run so the CSS cascade is
// right at first paint. `features/popout-spa.bootstrap.js` then gates
// on `?popout=` and calls `bootPopoutSpa()` exactly once.
//
// Dependency surface (all via clean named imports):
//   state.js          — `state`, `store` singletons
//   lib/escape.js     — `esc` (HTML-escape helper used in innerHTML strings)
//   lib/meeting-url.js— `_enc` (encodeURIComponent shorthand)
//   lib/time-format.js— `formatTime` (transcript clock formatter)
//   lib/lang-helpers.js— `getLangA` / `getLangB` / `isMonolingual`
//   compact-grid.js   — `CompactGridRenderer` (transcript surface)
//   column-headers.js — `updateColumnHeaders` (called via _updateColumnHeaders wrapper)
//   speaker-registry.js— `_speakerRegistry`, `refreshTranscriptSpeakerLabels`, `renameSpeaker`
//   live-session.js   — `ingestFromLiveWs` (SegmentStore.ingest + test hook)
//   bg-finalize-toast.js— `_renderBackgroundFinalizeToast`
//   modal-system.js   — `showModal` (inline `onclick="closeModal()"` resolves via window.*)
//
// Window globals consumed: `PopoutLayoutPresets`, `PopoutLayoutStorage`,
// `PopoutLayoutRender`, `PopoutPanelRegistry`, `TerminalPanel` (lazy-loaded),
// plus the test hooks (`__test_msg_log`, `__test_unhandled_count`,
// `__test_unhandled_types`) the cross-window-sync browser test depends on.

import { state, store } from "../state.js";
import { esc } from "../lib/escape.js";
import { _enc } from "../lib/meeting-url.js";
import { CompactGridRenderer } from "./compact-grid.js";
import { formatTime } from "../lib/time-format.js";
import {
  getLangA as _getLangA,
  getLangB as _getLangB,
  isMonolingual as _isMonolingual,
} from "../lib/lang-helpers.js";
import { updateColumnHeaders as _updateColumnHeadersRaw } from "./column-headers.js";
import {
  _speakerRegistry,
  refreshTranscriptSpeakerLabels as _refreshTranscriptSpeakerLabels,
  renameSpeaker,
} from "./speaker-registry.js";
import { ingestFromLiveWs } from "./live-session.js";
import { _renderBackgroundFinalizeToast } from "./bg-finalize-toast.js";
import { showModal } from "./modal-system.js";

const API = '';
const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';

// Thin wrapper so the popout block can call `_updateColumnHeaders()`
// without threading `state` through every call site.
function _updateColumnHeaders() {
  return _updateColumnHeadersRaw(state);
}

export function bootPopoutSpa() {
  // Show meeting mode, hide everything else
  document.getElementById('meeting-mode').style.display = '';

  // Insert minimal pop-out header before <main>
  const popHeader = document.createElement('div');
  popHeader.className = 'popout-header';
  popHeader.innerHTML = `
    <div class="popout-header-left">
      <div class="popout-dot"></div>
      <span class="popout-title">Meeting Scribe</span>
      <span class="popout-lang" id="popout-lang-label"></span>
    </div>
    <div class="popout-header-right">
      <div class="popout-lang-toggle" role="tablist" aria-label="Language display">
        <button class="popout-btn popout-lang-btn" id="popout-lang-both" data-mode="both" title="Show both languages">Both</button>
        <button class="popout-btn popout-lang-btn" id="popout-lang-a" data-mode="a" title="Show language A only"></button>
        <button class="popout-btn popout-lang-btn" id="popout-lang-b" data-mode="b" title="Show language B only"></button>
      </div>
      <button class="popout-btn" id="popout-direction" title="Toggle scroll direction">↑ Newest first</button>
      <!-- Options are rebuilt by _alignSlideLanguageToMeeting() from the
           actual meeting language_pair so any pair the user picks shows
           up here (not just the original ja/en/zh/ko hardcoded set). -->
      <select class="popout-slides-lang" id="popout-slides-lang" title="Slide language pair (Auto = follow speech direction)">
        <option value="auto">Auto</option>
      </select>
      <label class="popout-btn" id="popout-slides-btn" title="Upload PPTX slides for translation">
        Slides <input type="file" accept=".pptx,.potx,.pptm,.ppsx,.ppsm" id="popout-slides-input" style="display:none">
      </label>
      <button class="popout-btn popout-text-size" id="popout-text-smaller" title="Decrease text size">A-</button>
      <button class="popout-btn popout-text-size" id="popout-text-larger" title="Increase text size">A+</button>
      <button class="popout-btn" id="popout-pip-btn" title="Float above other windows (always on top)" style="display:none">Float</button>
      <select class="popout-layout-picker" id="popout-layout-picker" title="Layout (Ctrl+Shift+L cycles)">
        <option value="translate">Translate</option>
        <option value="translator">Presentation</option>
        <option value="triple">Triple stack</option>
        <option value="terminal">Translate + terminal</option>
      </select>
      <button class="popout-btn" id="popout-qr-btn" title="Show WiFi QR code">QR</button>
      <div class="popout-qr" id="popout-qr" style="display:none"></div>
    </div>
  `;
  document.body.insertBefore(popHeader, document.querySelector('main'));

  // Initialize compact renderer for pop-out (groups by speaker turn)
  window._gridRenderer = new CompactGridRenderer(document.getElementById("transcript-grid"), null, formatTime);

  // Direction toggle — persisted
  const savedDir = localStorage.getItem('popout_direction') || 'newest';
  const dirBtn = document.getElementById('popout-direction');
  if (savedDir === 'oldest') {
    dirBtn.textContent = '↓ Oldest first';
    setTimeout(() => {
      if (window._gridRenderer && window._gridRenderer._newestFirst) {
        window._gridRenderer.toggleDirection();
      }
    }, 100);
  }
  dirBtn.addEventListener('click', () => {
    const newestFirst = window._gridRenderer.toggleDirection();
    dirBtn.textContent = newestFirst ? '↑ Newest first' : '↓ Oldest first';
    localStorage.setItem('popout_direction', newestFirst ? 'newest' : 'oldest');
  });

  // Text size controls — A+/A- with localStorage persistence
  let _textScale = parseFloat(localStorage.getItem('scribe_text_scale') || '1');
  function _applyTextScale() {
    document.documentElement.style.setProperty('--text-scale', _textScale.toFixed(2));
    localStorage.setItem('scribe_text_scale', _textScale.toFixed(2));
  }
  _applyTextScale();
  document.getElementById('popout-text-larger')?.addEventListener('click', () => {
    _textScale = Math.min(2.5, _textScale + 0.15);
    _applyTextScale();
  });
  document.getElementById('popout-text-smaller')?.addEventListener('click', () => {
    _textScale = Math.max(0.5, _textScale - 0.15);
    _applyTextScale();
  });

  // Language mode toggle — both | a | b
  const setLangMode = (mode) => {
    document.body.classList.remove('lang-mode-both', 'lang-mode-a-only', 'lang-mode-b-only');
    if (mode === 'a') document.body.classList.add('lang-mode-a-only');
    else if (mode === 'b') document.body.classList.add('lang-mode-b-only');
    else document.body.classList.add('lang-mode-both');
    document.querySelectorAll('.popout-lang-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    localStorage.setItem('popout_lang_mode', mode);
  };
  document.querySelectorAll('.popout-lang-btn').forEach(btn => {
    btn.addEventListener('click', () => setLangMode(btn.dataset.mode));
  });
  // Restore saved mode
  setLangMode(localStorage.getItem('popout_lang_mode') || 'both');

  // Load meeting data + connect WS
  (async () => {
    try {
      // Fetch language registry
      try {
        const langResp = await fetch(`${API}/api/languages`);
        const langData = await langResp.json();
        (langData.languages || []).forEach(l => { state.languageNames[l.code] = l; });
      } catch {}

      // Check for historical meeting hash
      const hashMatch = location.hash.match(/^#meeting\/(.+)/);
      let mid = hashMatch?.[1];

      if (!mid) {
        // Live mode — find active meeting
        const statusResp = await fetch(`${API}/api/status`);
        const statusData = await statusResp.json();
        mid = statusData.meeting?.id;
      }
      // Tell the layout renderer whether a meeting is live so the
      // transcript empty-state text reads "Listening for speech…"
      // instead of "Start a meeting…" when audio just hasn't arrived
      // yet (common: popout opened mid-meeting, or ASR is warming up).
      try { window.PopoutLayoutRender?.setMeetingActive(!!mid); } catch {}

      if (mid) {
        try {
          const meetResp = await fetch(`${API}/api/meetings/${_enc(mid)}`);
          const meetData = await meetResp.json();
          if (meetData.meta?.language_pair?.length >= 1) {
            state.currentLanguagePair = meetData.meta.language_pair.join(',');
            _updateColumnHeaders();
          }
          if (meetData.events) {
            for (const event of meetData.events) store.ingest(event);
          }
        } catch {}
      }

      // Reflect monolingual mode on <body> so CSS hides the A/Both/B
      // toggle, the slide-language dropdown, and the translated-slides
      // sub-pane — freeing up real estate for the original deck.
      function _applyMonolingualBody() {
        document.body.classList.toggle('monolingual', _isMonolingual());
      }
      _applyMonolingualBody();

      // Reusable: update the popout language label + toggle button text
      // from `state.currentLanguagePair`. Called on init AND on every refresh
      // poll, so the popout picks up a meeting-language change (e.g.
      // user picked Dutch but the popout was opened pre-meeting and was
      // showing the default ja↔en pair).
      function _updatePopoutLangDisplay() {
        const langLabel = document.getElementById('popout-lang-label');
        const a = _getLangA();
        const b = _getLangB();
        const nameA = state.languageNames[a]?.name || a.toUpperCase();
        if (_isMonolingual()) {
          // Single language — no arrow, no B button label.
          if (langLabel) langLabel.textContent = nameA;
          return;
        }
        const nameB = state.languageNames[b]?.name || b.toUpperCase();
        if (langLabel) langLabel.textContent = `${nameA} ↔ ${nameB}`;
        const langBtnA = document.getElementById('popout-lang-a');
        const langBtnB = document.getElementById('popout-lang-b');
        if (langBtnA) langBtnA.textContent = nameA;
        if (langBtnB) langBtnB.textContent = nameB;
      }
      _updatePopoutLangDisplay();

      // Periodic refresh: pick up a meeting language change made after
      // the popout was opened (or when the popout was opened before the
      // user even started the meeting). 5s cadence is fast enough that
      // the user notices within a beat but won't hammer the API.
      setInterval(async () => {
        try {
          const r = await fetch(`${API}/api/status`);
          if (!r.ok) return;
          const sd = await r.json();
          const liveMid = sd?.meeting?.id;
          // Update the empty-state copy as meetings start/stop so the
          // popout stops saying "Start a meeting" the moment one starts.
          try { window.PopoutLayoutRender?.setMeetingActive(!!liveMid); } catch {}
          if (!liveMid) return;
          // If the live meeting changed (or we never had a meeting on init),
          // refresh language pair from its meta.
          if (liveMid !== mid) mid = liveMid;
          const m = await fetch(`${API}/api/meetings/${_enc(liveMid)}`);
          if (!m.ok) return;
          const md = await m.json();
          const lp = md?.meta?.language_pair;
          if (Array.isArray(lp) && lp.length >= 1 && lp.length <= 2) {
            const next = lp.join(',');
            if (next !== state.currentLanguagePair) {
              state.currentLanguagePair = next;
              _updateColumnHeaders();
              _updatePopoutLangDisplay();
              _applyMonolingualBody();
              _alignSlideLanguageToMeeting();
            }
          }
        } catch {}
      }, 5000);

      // Sync the slide-upload language dropdown to the meeting's pair when
      // it's still on "Auto". If the user has explicitly pinned a slide
      // Rebuild the slide-language dropdown to mirror the meeting's
      // language_pair AND its direction (speech goes a→b, default slides
      // go a→b too — that's the alignment the user wants). Keeps the
      // user's explicit pin if they set one in localStorage.
      function _alignSlideLanguageToMeeting() {
        const sel = document.getElementById('popout-slides-lang');
        if (!sel) return;
        const a = _getLangA();
        const b = _getLangB();
        if (!a || !b) return;

        const fwd = `${a}:${b}`;
        const rev = `${b}:${a}`;
        const nameA = (state.languageNames[a]?.name || a.toUpperCase());
        const nameB = (state.languageNames[b]?.name || b.toUpperCase());
        const desired = [
          { value: 'auto', label: `Auto (${nameA} → ${nameB})` },
          { value: fwd, label: `${nameA} → ${nameB}` },
          { value: rev, label: `${nameB} → ${nameA}` },
        ];
        // Only rebuild when the option set changes — avoids fighting the
        // user's open dropdown while they're choosing.
        const currentValues = Array.from(sel.options).map((o) => o.value);
        const desiredValues = desired.map((o) => o.value);
        if (
          currentValues.length !== desiredValues.length
          || currentValues.some((v, i) => v !== desiredValues[i])
        ) {
          sel.innerHTML = desired.map((o) =>
            `<option value="${o.value}">${o.label}</option>`
          ).join('');
        }

        // Restore the user's pin if it's still valid for this pair, else
        // fall back to "Auto" (which now means: use speech direction).
        const stored = (() => { try { return localStorage.getItem('popout_slides_lang'); } catch { return null; } })();
        if (stored && desiredValues.includes(stored)) {
          sel.value = stored;
        } else {
          sel.value = 'auto';
          // Drop a stale pin from a different language pair so it doesn't
          // resurrect itself if the user navigates back to that pair.
          if (stored && stored !== 'auto') {
            try { localStorage.removeItem('popout_slides_lang'); } catch {}
          }
        }
      }
      _alignSlideLanguageToMeeting();

      // Persistent meeting PIN in the popout header. QR codes are
      // gone and the SSID is open, so the 4-digit appliance PIN
      // ("1618") is the only join-hint the operator needs and it
      // is shown continuously, in-meeting and idle.
      let _popoutLastPin = null;
      async function refreshPopoutPin(retries = 15, delayMs = 2000) {
        try {
          const resp = await fetch(`${API}/api/admin/settings`);
          if (!resp.ok) {
            if (retries > 0) setTimeout(() => refreshPopoutPin(retries - 1, delayMs), delayMs);
            return;
          }
          const data = await resp.json();
          const pin = data.appliance_pin || '';
          if (!pin || pin === _popoutLastPin) return;
          _popoutLastPin = pin;
          const target = document.getElementById('popout-qr');
          if (target) {
            target.classList.add('popout-meeting-pin');
            target.innerHTML = `
              <span class="popout-meeting-pin-label">Meeting ID</span>
              <span class="popout-meeting-pin-value">${esc(pin)}</span>
            `;
          }
        } catch (pinErr) {
          console.warn('Popout PIN load failed:', pinErr);
          if (retries > 0) setTimeout(() => refreshPopoutPin(retries - 1, delayMs), delayMs);
        }
      }
      refreshPopoutPin();
      // Periodic consistency poll: factory_reset rotates the
      // appliance_id, so the PIN can change without a tab reload.
      setInterval(() => refreshPopoutPin(0, 0), 10000);

      // ── Terminal panel ───────────────────────────────────────
      // Lazy-load the ~750KB xterm.js bundle only on first toggle so the
      // popout's initial paint isn't paying for a feature nobody's using yet.
      let _terminalPanel = null;
      let _terminalPanelLoading = null;
      async function _ensureTerminalPanel() {
        if (_terminalPanel) return _terminalPanel;
        if (_terminalPanelLoading) return _terminalPanelLoading;
        _terminalPanelLoading = (async () => {
          if (!window.TerminalPanel) {
            await new Promise((resolve, reject) => {
              const s = document.createElement('script');
              s.src = '/static/js/terminal-panel.js?v=2';
              s.async = false;
              s.onload = resolve;
              s.onerror = () => reject(new Error('terminal-panel.js failed to load'));
              document.head.appendChild(s);
            });
          }
          _terminalPanel = new window.TerminalPanel({
            apiBase: API,
            wsBase: API.replace(/^http/, 'ws'),
          });
          await _terminalPanel.mount();
          window._terminalPanel = _terminalPanel;   // pin for console inspection
          return _terminalPanel;
        })();
        try {
          return await _terminalPanelLoading;
        } finally {
          _terminalPanelLoading = null;
        }
      }

      // ── Popout layout wiring ──────────────────────────────────
      // Replaces the previous Term-button-only flow. The layout picker is
      // now the authoritative control: presets decide which of transcript
      // / slides / terminal are visible. Each panel's cached root element
      // gets re-parented (never rebuilt) as the user swaps presets.
      async function _ensureTerminalPanelRoot() {
        const p = await _ensureTerminalPanel();
        // The panel builds its own `.terminal-panel` root inside mount();
        // we just surface it to the registry. The panel's initial
        // insertBefore in mount() also drops it into the DOM — the
        // layout renderer will take over on first render.
        if (p._root && !p._root.isConnected) {
          // Panel was mounted but detached (e.g., mid-render). Nothing to do.
        }
        p.show();                         // ensures visible state is tracked
        return p._root;
      }

      function _ensureTranscriptRoot() {
        return document.getElementById('transcript-grid');
      }

      function _ensureSlideViewerRoot() {
        _ensureSlideViewer();
        return _slideViewerEl;
      }

      if (window.PopoutPanelRegistry && window.PopoutLayoutRender) {
        const PPR = window.PopoutPanelRegistry;
        PPR.register('transcript', {
          ensure: _ensureTranscriptRoot,
          notifyResize: () => { /* CompactGridRenderer reflows via its own ResizeObserver */ },
        });
        PPR.register('slides', {
          ensure: _ensureSlideViewerRoot,
          notifyResize: () => {
            // Let any sizing listener (_autoSizeSlidePane) recompute.
            try { window.dispatchEvent(new Event('resize')); } catch {}
          },
        });
        PPR.register('terminal', {
          ensure: _ensureTerminalPanelRoot,
          notifyResize: () => {
            if (window._terminalPanel && typeof window._terminalPanel._scheduleRefit === 'function') {
              window._terminalPanel._scheduleRefit();
            }
          },
        });

        const main = document.querySelector('main');
        const S = window.PopoutLayoutStorage;
        const P = window.PopoutLayoutPresets;
        let _layoutState = S.load();

        // Sync the picker selectbox with the loaded/current preset.
        const picker = document.getElementById('popout-layout-picker');
        function _syncPicker() {
          if (!picker) return;
          // Ensure a 'custom' option exists when the tree is user-edited.
          const hasCustom = [...picker.options].some(o => o.value === 'custom');
          if (_layoutState.preset === 'custom' && !hasCustom) {
            const opt = document.createElement('option');
            opt.value = 'custom';
            opt.textContent = 'Custom';
            picker.appendChild(opt);
          }
          picker.value = _layoutState.preset;
        }
        _syncPicker();

        async function _applyPreset(slug, { skipRender = false } = {}) {
          if (!P.PRESET_ORDER.includes(slug)) return;
          _layoutState = S.setPreset(_layoutState, slug);
          _syncPicker();
          if (!skipRender) {
            try {
              await window.PopoutLayoutRender.renderLayout(main, _layoutState);
            } catch (err) {
              console.warn('[popout-layout] render failed', err);
            }
          }
        }

        if (picker) {
          picker.addEventListener('change', () => _applyPreset(picker.value));
        }

        // Keyboard: Ctrl+Shift+L cycles; Ctrl+Shift+1..6 jumps; Ctrl+Shift+T toggles terminal.
        document.addEventListener('keydown', async (e) => {
          if (!e.ctrlKey || !e.shiftKey) return;
          if (e.key === 'L' || e.key === 'l') {
            e.preventDefault();
            const order = P.PRESET_ORDER;
            const next = order[(order.indexOf(_layoutState.preset) + (e.altKey ? -1 : 1) + order.length) % order.length];
            await _applyPreset(next);
            return;
          }
          if (/^[1-6]$/.test(e.key)) {
            e.preventDefault();
            const slug = P.PRESET_ORDER[parseInt(e.key, 10) - 1];
            if (slug) await _applyPreset(slug);
            return;
          }
          if (e.key === 'T' || e.key === 't') {
            e.preventDefault();
            const hasTerm = P.hasTerminal(_layoutState.preset);
            const next = hasTerm ? _layoutState.lastNoTermPreset : _layoutState.lastTermPreset;
            await _applyPreset(next || (hasTerm ? 'translator' : 'triple'));
          }
        });

        // Live availability: the app flips slide availability based on
        // meeting/deck lifecycle. Seed with current state and update on
        // every slide-state change our broadcast dispatcher already handles.
        const availNow = {
          transcript: true,
          slides:     !!(window._slideState && window._slideState.deckId),
          terminal:   true,
        };
        window.PopoutLayoutRender.install(main, _layoutState, availNow, {
          // Edit-menu + drag-drop commit mutations → renderer calls this
          // so the picker + local state closure stay in sync.
          onStateChange: (next) => {
            _layoutState = next;
            _syncPicker();
          },
        });

        // Re-export current state so we can update ratios on commit.
        window._popoutLayoutState = () => _layoutState;
      }

      // ── Slide upload handler ─────────────────────────────────
      const slidesInput = document.getElementById('popout-slides-input');
      const slidesBtn = document.getElementById('popout-slides-btn');
      let _slideViewerEl = null;
      let _slideState = { deckId: null, total: 0, current: 0 };

      function _ensureSlideViewer() {
        if (_slideViewerEl) return;
        _slideViewerEl = document.createElement('div');
        _slideViewerEl.className = 'popout-slides';
        // Monolingual meetings: only the original-slide pane is shown.
        // Class is applied once here (on the element that owns the
        // panes); CSS collapses the translated sub-pane.
        if (_isMonolingual()) _slideViewerEl.classList.add('monolingual-slides');
        _slideViewerEl.style.display = 'none';
        _slideViewerEl.innerHTML = `
          <div class="sv-resize-handle" style="height:6px;cursor:ns-resize;background:linear-gradient(to bottom,#e8e6e1,#d0cec8);display:flex;align-items:center;justify-content:center;user-select:none" title="Drag to resize">
            <div style="width:40px;height:2px;background:#aaa;border-radius:1px"></div>
          </div>
          <div class="sv-slides" style="display:flex;gap:0;height:40vh;overflow:hidden">
            <div class="sv-pane" id="sv-orig-pane" style="flex:var(--sv-pane-a-flex,1);min-width:80px;align-items:stretch;justify-content:stretch;background:#fff;min-height:100px;overflow:hidden">
              <div id="sv-orig-container" style="flex:1;display:flex;align-items:center;justify-content:center;width:100%;overflow:hidden"></div>
            </div>
            <!-- Draggable divider between original and translated panes.
                 Drag horizontally to favour one side over the other (e.g.
                 enlarge the English pane while shrinking the Japanese
                 source). Ratio persists in localStorage as
                 popout_slide_pane_ratio. Double-click resets to 50/50. -->
            <div id="sv-pane-divider" class="sv-pane-divider"
                 style="width:6px;cursor:ew-resize;background:linear-gradient(to right,#e8e6e1,#d0cec8 50%,#e8e6e1);flex:0 0 6px;display:flex;align-items:center;justify-content:center;user-select:none"
                 title="Drag to resize · double-click to reset">
              <div style="width:2px;height:32px;background:#9a9aa2;border-radius:1px"></div>
            </div>
            <div class="sv-pane" id="sv-trans-pane" style="flex:var(--sv-pane-b-flex,1);min-width:80px;display:flex;align-items:center;justify-content:center;background:#1a1a1e;min-height:100px;position:relative">
              <img class="sv-img" id="sv-trans" style="max-width:100%;max-height:100%;object-fit:contain" alt="Translated">
              <!-- Initially hidden — flipped on by _renderSlide when a
                   translated slide load is in flight. With no deck the
                   layout's empty-state placeholder ("No slide deck
                   loaded") shows through cleanly instead. -->
              <div id="sv-trans-status" style="position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#9a9aa2;font-size:0.8rem;font-style:italic;background:#1a1a1e">Translating…</div>
            </div>
          </div>
          <div class="sv-nav" style="display:flex;align-items:center;justify-content:center;gap:1rem;padding:0.3rem;background:#fff;border-top:1px solid #e8e6e1">
            <!-- Deck switcher: hidden when there's only one deck. Populated
                 by _refreshDeckSwitcher() from /api/meetings/{id}/decks. -->
            <select id="sv-deck-picker" title="Switch slide deck" style="display:none;font-size:0.7rem;padding:0.15rem 0.3rem;border:1px solid #e8e6e1;border-radius:4px;background:#fff;cursor:pointer;max-width:18rem"></select>
            <button id="sv-prev" style="font-size:0.8rem;padding:0.2rem 0.5rem;border:1px solid #e8e6e1;border-radius:4px;background:#fff;cursor:pointer">\u25C0</button>
            <span id="sv-label" style="font-size:0.7rem;color:#6a6a72;font-weight:600;min-width:4rem;text-align:center"></span>
            <button id="sv-next" style="font-size:0.8rem;padding:0.2rem 0.5rem;border:1px solid #e8e6e1;border-radius:4px;background:#fff;cursor:pointer">\u25B6</button>
            <span id="sv-progress" style="font-size:0.65rem;color:#9a9aa2;font-style:italic"></span>
          </div>
        `;
        // If the popout layout registry is loaded, it will own placement.
        // Legacy fallback (main view, non-popout flows) keeps the old
        // after-main insertion.
        if (!window.PopoutPanelRegistry) {
          const main = document.querySelector('main') || document.querySelector('.transcript-grid') || document.body;
          if (main.nextSibling) {
            main.parentNode.insertBefore(_slideViewerEl, main.nextSibling);
          } else {
            main.parentNode.appendChild(_slideViewerEl);
          }
        }

        // Query INSIDE _slideViewerEl, not the document — the popout
        // panel registry caches this element detached and only mounts
        // it later via _buildSlot. Document-scope lookups would return
        // null and crash with "Cannot read properties of null
        // (reading 'addEventListener')", which broke EVERY layout that
        // includes a slides panel (translator, fullstack, triple,
        // sidebyside) the moment the registry tried to ensure() it.
        _slideViewerEl.querySelector('#sv-prev').addEventListener('click', () => _slideNav(-1));
        _slideViewerEl.querySelector('#sv-next').addEventListener('click', () => _slideNav(1));

        // The right-pane image is created statically in the markup above
        // (sv-trans), so we attach the auto-size load listener here once.
        // The left-pane image is created lazily by _ensureOrigPng and gets
        // its listener attached at injection time. Whichever side loads
        // first triggers the resize.
        const transStaticImg = _slideViewerEl.querySelector('#sv-trans');
        if (transStaticImg) transStaticImg.addEventListener('load', _autoSizeSlidePane);

        // Deck switcher — POST the chosen deck_id to switch the active
        // deck server-side. The server broadcasts slide_deck_changed
        // back to us (and any other connected viewer) so the popout
        // re-renders against the new deck.
        const deckPicker = _slideViewerEl.querySelector('#sv-deck-picker');
        if (deckPicker) {
          deckPicker.addEventListener('change', async () => {
            const targetDeck = deckPicker.value;
            const smid = _slideState.meetingId || mid;
            if (!targetDeck || !smid || targetDeck === _slideState.deckId) return;
            try {
              await fetch(`${API}/api/meetings/${_enc(smid)}/decks/active`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ deck_id: targetDeck }),
              });
              // Server will broadcast slide_deck_changed → _showSlides
              // picks it up. No further work needed here.
            } catch {}
          });
        }

        // Pane divider — drag horizontally to favour one side over the
        // other (e.g. give the English translation more room than the
        // Japanese source). Persists to localStorage so the layout
        // survives a page reload. Double-click resets to 50/50.
        const paneDivider = _slideViewerEl.querySelector('#sv-pane-divider');
        const slidesRow = _slideViewerEl.querySelector('.sv-slides');
        if (paneDivider && slidesRow) {
          // Restore saved ratio (a-flex / total)
          const savedRatio = parseFloat(localStorage.getItem('popout_slide_pane_ratio'));
          if (Number.isFinite(savedRatio) && savedRatio > 0.05 && savedRatio < 0.95) {
            slidesRow.style.setProperty('--sv-pane-a-flex', String(savedRatio));
            slidesRow.style.setProperty('--sv-pane-b-flex', String(1 - savedRatio));
          }
          let dragStartX = 0, dragStartTotal = 0, dragStartA = 0;
          paneDivider.addEventListener('mousedown', (e) => {
            e.preventDefault();
            dragStartX = e.clientX;
            const rect = slidesRow.getBoundingClientRect();
            dragStartTotal = rect.width - paneDivider.offsetWidth;
            const aPane = _slideViewerEl.querySelector('#sv-orig-pane');
            dragStartA = aPane ? aPane.getBoundingClientRect().width : dragStartTotal / 2;
            const onMove = (e2) => {
              const delta = e2.clientX - dragStartX;
              const newAWidth = Math.max(80, Math.min(dragStartTotal - 80, dragStartA + delta));
              const ratio = Math.max(0.05, Math.min(0.95, newAWidth / dragStartTotal));
              slidesRow.style.setProperty('--sv-pane-a-flex', String(ratio));
              slidesRow.style.setProperty('--sv-pane-b-flex', String(1 - ratio));
            };
            const onUp = () => {
              document.removeEventListener('mousemove', onMove);
              document.removeEventListener('mouseup', onUp);
              // Persist final ratio
              const aPane = _slideViewerEl.querySelector('#sv-orig-pane');
              if (aPane) {
                const total = slidesRow.getBoundingClientRect().width - paneDivider.offsetWidth;
                if (total > 0) {
                  const finalRatio = aPane.getBoundingClientRect().width / total;
                  localStorage.setItem('popout_slide_pane_ratio', finalRatio.toFixed(3));
                }
              }
              // Re-fit slides to new pane widths.
              if (typeof _autoSizeSlidePane === 'function') _autoSizeSlidePane();
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
          });
          paneDivider.addEventListener('dblclick', () => {
            slidesRow.style.removeProperty('--sv-pane-a-flex');
            slidesRow.style.removeProperty('--sv-pane-b-flex');
            localStorage.removeItem('popout_slide_pane_ratio');
            if (typeof _autoSizeSlidePane === 'function') _autoSizeSlidePane();
          });
        }

        // Resize handle — drag to adjust slide viewer height
        const resizeHandle = _slideViewerEl.querySelector('.sv-resize-handle');
        const slidesContainer = _slideViewerEl.querySelector('.sv-slides');
        if (resizeHandle && slidesContainer) {
          let startY = 0, startH = 0;
          resizeHandle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            startY = e.clientY;
            startH = slidesContainer.offsetHeight;
            // Mark user-resized so the auto-size pass stops fighting the
            // user's preferred height. Stays set until they reload.
            slidesContainer.dataset.userResized = '1';
            const onMove = (e2) => {
              // Dragging UP = making slides taller (since slides are below transcript)
              const delta = startY - e2.clientY;
              const newH = Math.max(80, Math.min(window.innerHeight * 0.7, startH + delta));
              slidesContainer.style.maxHeight = newH + 'px';
              slidesContainer.style.height = newH + 'px';
            };
            const onUp = () => {
              document.removeEventListener('mousemove', onMove);
              document.removeEventListener('mouseup', onUp);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
          });
        }

        document.addEventListener('keydown', (e) => {
          if (!_slideViewerEl || _slideViewerEl.style.display === 'none') return;
          if (e.key === 'ArrowLeft') _slideNav(-1);
          else if (e.key === 'ArrowRight') _slideNav(1);
        });

        // Re-auto-size when the window changes (popout floating to PiP,
        // user resizes window, etc.). Debounced via rAF.
        let _resizeRaf = null;
        window.addEventListener('resize', () => {
          if (_resizeRaf) return;
          _resizeRaf = requestAnimationFrame(() => {
            _resizeRaf = null;
            _autoSizeSlidePane();
          });
        });
      }

      function _slideNav(delta) {
        const newIdx = _slideState.current + delta;
        if (newIdx < 0 || newIdx >= _slideState.total) return;
        _slideState.current = newIdx;
        _renderSlide();
        // Persist locally so a reload / restart drops the user back on the
        // same slide (per deck — keyed by deck_id so a new upload doesn't
        // inherit a stale index).
        try {
          if (_slideState.deckId) {
            localStorage.setItem(`popout_slide_idx_${_slideState.deckId}`, String(newIdx));
          }
        } catch {}
        // Broadcast to other viewers
        const navMid = _slideState.meetingId || mid;
        fetch(`${API}/api/meetings/${_enc(navMid)}/slides/current`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ index: newIdx }),
        }).catch(() => {});
      }

      // The original side just renders the same PNG path as the translated
      // side — backend already renders 300 DPI from LibreOffice, the popout
      // owns the prev/next/keyboard nav, and there's no second toolbar to
      // fight with. OnlyOffice was previously layered on top as a "lossless
      // vector view" upgrade but it brought its own loading state, its own
      // chrome, and a "Download failed" overlay when the editor couldn't
      // reach source.pptx. Removed entirely — PNG-only is simpler + faster.

      function _renderSlide() {
        const bust = _slideState.deckId ? `?d=${_slideState.deckId}` : '';
        const smid = _slideState.meetingId || mid;
        if (!smid) return;
        const transImg = document.getElementById('sv-trans');
        const transStatus = document.getElementById('sv-trans-status');
        const label = document.getElementById('sv-label');

        // Pane-by-language alignment: the LEFT pane is always lang A
        // (e.g. JA in a ja↔en meeting) and the RIGHT pane is always
        // lang B (e.g. EN). Whichever side the deck is in, that side
        // gets ``/original``; the OTHER side gets ``/translated``.
        // Falls back to "left=original, right=translated" when we
        // don't yet know the deck's source language.
        const langA = _getLangA();
        const langB = _getLangB();
        const deckSrc = (_slideState.sourceLang || '').toLowerCase();
        const leftIsTranslated = !!(deckSrc && deckSrc === langB);
        const leftEndpoint = leftIsTranslated ? 'translated' : 'original';
        const rightEndpoint = leftIsTranslated ? 'original' : 'translated';

        _ensureOrigPng();
        const origImg = document.getElementById('sv-orig-img');
        if (origImg) {
          origImg.src = `${API}/api/meetings/${_enc(smid)}/slides/${_slideState.current}/${leftEndpoint}${bust}`;
          origImg.alt = `Slide in ${langA?.toUpperCase() || 'A'}`;
        }
        if (transImg) {
          transImg.src = `${API}/api/meetings/${_enc(smid)}/slides/${_slideState.current}/${rightEndpoint}${bust}`;
          transImg.alt = `Slide in ${langB?.toUpperCase() || 'B'}`;
          transImg.onerror = () => { transImg.style.display = 'none'; if (transStatus) transStatus.style.display = ''; };
          transImg.onload = () => { transImg.style.display = ''; if (transStatus) transStatus.style.display = 'none'; };
        }
        if (label) label.textContent = `${_slideState.current + 1} / ${_slideState.total}`;
        document.getElementById('sv-prev').disabled = _slideState.current <= 0;
        document.getElementById('sv-next').disabled = _slideState.current >= _slideState.total - 1;
      }

      function _ensureOrigPng() {
        // Idempotent — injects the PNG <img> into the original-side
        // container if it isn't already there. The img fills the container
        // via width/height 100% + object-fit contain.
        const container = document.getElementById('sv-orig-container');
        if (!container) return;
        if (container.querySelector('#sv-orig-img')) return; // already wired
        container.innerHTML = `<img id="sv-orig-img" style="width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain;display:block" alt="Original slide">`;
        // Once the slide image loads, size the entire slide pane to match
        // the slide aspect ratio so the slide fills the available width
        // exactly (no letterbox). Caps at 70vh so the transcript still
        // gets meaningful screen real estate.
        const img = container.querySelector('#sv-orig-img');
        if (img) img.addEventListener('load', _autoSizeSlidePane);
      }

      function _autoSizeSlidePane() {
        if (!_slideViewerEl) return;
        const slidesEl = _slideViewerEl.querySelector('.sv-slides');
        if (!slidesEl) return;
        // Use whichever pane has loaded an image first. Both render the
        // same slides so they share an aspect ratio. Originally we read
        // sv-orig-img exclusively, but with pane-by-language alignment
        // the LEFT pane sometimes shows the translated PNG (which arrives
        // ~5-15s later) — waiting for it left the pane stuck at the
        // initial 40vh until the second pane finally loaded.
        const left = document.getElementById('sv-orig-img');
        const right = document.getElementById('sv-trans');
        const refImg =
          (left && left.naturalWidth && left.naturalHeight) ? left :
          (right && right.naturalWidth && right.naturalHeight) ? right :
          null;
        if (!refImg) return;
        // Monolingual: a single pane claims the FULL container width,
        // and we let the slide grow taller (no translated pane is
        // competing for vertical space). Bilingual: each pane is
        // flex:1 so each gets ~half the width, capped tighter.
        const mono = _slideViewerEl.classList.contains('monolingual-slides');
        const containerWidth = slidesEl.clientWidth || _slideViewerEl.clientWidth;
        const paneWidth = mono ? containerWidth : (containerWidth - 2) / 2;
        if (paneWidth <= 0) return;
        const aspect = refImg.naturalWidth / refImg.naturalHeight;
        const targetH = Math.round(paneWidth / aspect);
        // Reserve vertical space for the popout header (~36px), the
        // slide nav bar below the image (~40px), and a transcript
        // strip at the top (~160px — two or three lines of compact
        // blocks, enough to read without squinting). Everything else
        // goes to the slide so it's as big as possible. Bilingual
        // stays capped at 70vh so the second transcript column still
        // has air.
        const RESERVE_MONO = 36 + 40 + 160;
        const maxH = mono
          ? Math.max(240, window.innerHeight - RESERVE_MONO)
          : Math.round(window.innerHeight * 0.70);
        const minH = 120;
        const finalH = Math.max(minH, Math.min(maxH, targetH));
        // Don't fight the user if they've manually resized via the drag handle.
        if (slidesEl.dataset.userResized === '1') return;
        slidesEl.style.height = finalH + 'px';
        slidesEl.style.maxHeight = finalH + 'px';
      }

      async function _refreshDeckSwitcher() {
        const picker = document.getElementById('sv-deck-picker');
        if (!picker) return;
        const smid = _slideState.meetingId || mid;
        if (!smid) return;
        let decks = [];
        try {
          const r = await fetch(`${API}/api/meetings/${_enc(smid)}/decks`);
          if (!r.ok) return;
          const data = await r.json();
          decks = (data && data.decks) || [];
        } catch { return; }
        // Hide picker entirely when there's only one (or zero) decks —
        // no need to clutter the nav bar.
        if (decks.length < 2) {
          picker.style.display = 'none';
          picker.innerHTML = '';
          return;
        }
        picker.style.display = '';
        // Build options. Prefer a friendly label from saved meta filename
        // when present; fall back to "Deck N" + short hash.
        picker.innerHTML = decks.map((d, i) => {
          const label = d.upload_filename || `Deck ${decks.length - i}`;
          const slidesN = d.total_slides || '?';
          const cached = d.from_cache ? ' (cached)' : '';
          const stage = d.stage && d.stage !== 'complete' ? ` [${d.stage}]` : '';
          return `<option value="${d.deck_id}">${esc(label)} · ${slidesN} slides${stage}${cached}</option>`;
        }).join('');
        picker.value = _slideState.deckId || decks[0].deck_id;
      }

      async function _showSlides(total, deckId, opts) {
        _ensureSlideViewer();
        // On a new upload, clear the container so the old deck's PNG
        // doesn't briefly flash before the new one's first paint.
        if (_slideState.deckId !== deckId) {
          const container = document.getElementById('sv-orig-container');
          if (container) container.innerHTML = '';
        }
        const carriedMid = _slideState.meetingId || mid;
        _slideState = {
          deckId,
          total,
          current: (opts && opts.startIndex) || 0,
          meetingId: carriedMid,
          sourceLang: (opts && opts.sourceLang) || null,
          targetLang: (opts && opts.targetLang) || null,
        };
        _slideViewerEl.style.display = '';
        if (slidesBtn) slidesBtn.classList.add('active-toggle');
        _refreshDeckSwitcher();
        // Notify the popout layout renderer so a preset that includes
        // slides can swap from "transcript fills" to "transcript + slides"
        // without a page reload. The renderer prunes unavailable panels
        // on every availability change (see popout-layout-render.js).
        try {
          window.dispatchEvent(new CustomEvent('popout-availability:change', {
            detail: { slides: true },
          }));
        } catch {}

        // Resolve deck source/target language BEFORE the first render.
        // Pane-by-language alignment (left=langA, right=langB) depends
        // on sourceLang to decide which side gets /original vs
        // /translated. Previously we rendered immediately with a null
        // source-lang fallback (left=original), then re-rendered on
        // backfill — causing a visible flip when the deck language
        // didn't match langA (e.g. English deck in a JA↔EN meeting:
        // user saw the EN slide on the "translated" side first, then
        // the JA slide "instantly" replaced it, while the EN slide
        // appeared on the "original" side only after the re-render —
        // perceived as "translated instant, original slow").
        if (!_slideState.sourceLang) {
          const smid = _slideState.meetingId || mid;
          if (smid) {
            try {
              const r = await fetch(`${API}/api/meetings/${_enc(smid)}/slides`);
              if (r.ok) {
                const meta = await r.json();
                if (meta && meta.deck_id === deckId) {
                  _slideState.sourceLang = (meta.source_lang || '').toLowerCase() || null;
                  _slideState.targetLang = (meta.target_lang || '').toLowerCase() || null;
                }
              }
            } catch {}
          }
        }
        _renderSlide();
      }

      function _handleSlideWsEvent(data) {
        if (data.type === 'slide_deck_changed') {
          _showSlides(data.total_slides, data.deck_id);
        } else if (data.type === 'slide_change') {
          _slideState.current = data.slide_index;
          _renderSlide();
        } else if (data.type === 'slide_partial_ready') {
          // A single slide PNG just landed on disk. Pick the pane based
          // on the deck's source language (pane-by-language alignment),
          // not the kind blindly. The "original" PNG goes into whichever
          // pane corresponds to the deck's language; "translated" goes
          // into the other pane.
          if (data.deck_id !== _slideState.deckId) return;
          if (data.index !== _slideState.current) return;
          const smid = _slideState.meetingId || mid;
          if (!smid) return;
          const cb = `?d=${_slideState.deckId}&t=${Date.now()}`;
          const deckSrc = (_slideState.sourceLang || '').toLowerCase();
          const langB = _getLangB();
          const leftIsTranslated = !!(deckSrc && deckSrc === langB);
          const targetIsLeft =
            (data.kind === 'original' && !leftIsTranslated) ||
            (data.kind === 'translated' && leftIsTranslated);

          if (targetIsLeft) {
            const origImg = document.getElementById('sv-orig-img');
            if (origImg) {
              origImg.src = `${API}/api/meetings/${_enc(smid)}/slides/${data.index}/${data.kind}${cb}`;
            }
          } else {
            const transImg = document.getElementById('sv-trans');
            const transStatus = document.getElementById('sv-trans-status');
            if (transImg) {
              transImg.src = `${API}/api/meetings/${_enc(smid)}/slides/${data.index}/${data.kind}${cb}`;
              transImg.onload = () => { transImg.style.display = ''; if (transStatus) transStatus.style.display = 'none'; };
            }
          }
        } else if (data.type === 'slide_job_progress') {
          // Track pipeline state so the in-pane spinner can show what's
          // actually happening to THIS slide rather than a static
          // "Translating...". The footer progress strip still shows the
          // overall stage in the original raw form.
          const progress = document.getElementById('sv-progress');
          if (progress) {
            if (data.stage === 'complete') {
              progress.textContent = '';
              _renderSlide(); // reload to pick up translated version
            } else {
              progress.textContent = `${data.stage.replace(/_/g, ' ')}${data.progress ? ' (' + data.progress + ')' : ''}`;
            }
          }
          // Tag the deck-level pipeline state on _slideState so the spinner
          // copy can reflect "translating slide X of N" → "rendering slide
          // X of N" → ready, instead of indefinite "Translating...".
          _slideState.pipelineStage = data.stage;
          _slideState.pipelineProgress = data.progress || null;
          _updateTransSpinnerText();
        }
      }

      function _updateTransSpinnerText() {
        // Monolingual meetings don't have a translated pane — the whole
        // right side of the slide viewer is collapsed via CSS. Skip the
        // spinner copy entirely so we don't narrate a pipeline that
        // isn't running.
        if (_isMonolingual()) return;
        const status = document.getElementById('sv-trans-status');
        if (!status || status.style.display === 'none') return;
        const stage = _slideState.pipelineStage || 'translating';
        const prog = _slideState.pipelineProgress;
        let label = 'Translating…';
        if (stage === 'translating') {
          label = prog ? `Translating (${prog})…` : 'Translating…';
        } else if (stage === 'reinserting') {
          label = 'Inserting translations into slides…';
        } else if (stage === 'rendering_translated') {
          label = prog ? `Rendering translated slides (${prog})…` : 'Rendering translated slides…';
        } else if (stage === 'rendering_original') {
          label = prog ? `Preparing slides (${prog})…` : 'Preparing slides…';
        } else if (stage === 'complete') {
          label = 'Translated slide unavailable';
        }
        status.textContent = label;
      }

      // Slide-language override: persist selection so it sticks across reloads
      const slidesLangSel = document.getElementById('popout-slides-lang');
      if (slidesLangSel) {
        const savedLang = localStorage.getItem('popout_slides_lang');
        if (savedLang) slidesLangSel.value = savedLang;
        slidesLangSel.addEventListener('change', () => {
          localStorage.setItem('popout_slides_lang', slidesLangSel.value);
        });
      }

      if (slidesInput) {
        slidesInput.addEventListener('change', async (e) => {
          const file = e.target.files[0];
          if (!file) return;
          // Accept the full PowerPoint OOXML family — python-pptx + LibreOffice
          // both handle every variant the same way (template/slideshow/macro
          // bits don't change the underlying zip structure we extract from).
          const _validExt = /\.(pptx|potx|pptm|ppsx|ppsm)$/i;
          if (!_validExt.test(file.name)) { showModal(`<div class="modal-confirm-title">Invalid file</div><div class="modal-confirm-message">Supported PowerPoint formats: .pptx, .potx, .pptm, .ppsx, .ppsm</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm'); return; }
          if (file.size > 50 * 1024 * 1024) { showModal(`<div class="modal-confirm-title">File too large</div><div class="modal-confirm-message">Maximum file size is 50 MB.</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm'); return; }
          const formData = new FormData();
          formData.append('file', file);
          // If the user pinned a slide language pair, send it as an explicit
          // override so the backend skips auto-detection.
          const langChoice = slidesLangSel ? slidesLangSel.value : 'auto';
          if (langChoice && langChoice !== 'auto' && langChoice.includes(':')) {
            const [src, tgt] = langChoice.split(':');
            formData.append('source_lang', src);
            formData.append('target_lang', tgt);
          }
          const progressEl = document.getElementById('sv-progress');
          try {
            // Fetch current meeting ID at upload time (not from init-time mid)
            let uploadMid = mid;
            if (!uploadMid) {
              const sr = await fetch(`${API}/api/status`);
              const sd = await sr.json();
              uploadMid = sd.meeting?.id;
              if (uploadMid) mid = uploadMid; // update for future use
            }
            if (!uploadMid) { showModal(`<div class="modal-confirm-title">No active meeting</div><div class="modal-confirm-message">Start a meeting before uploading slides.</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm'); return; }
            _ensureSlideViewer();
            _slideViewerEl.style.display = '';
            // Mount the slides panel into the layout NOW — presets that
            // only include slides when available (translator, fullstack,
            // triple, sidebyside, demo) otherwise won't render a slot
            // until slide_deck_changed arrives via WS, so the user sees
            // nothing during upload + processing.
            try {
              window.dispatchEvent(new CustomEvent('popout-availability:change', {
                detail: { slides: true },
              }));
            } catch {}
            // Scope the progress lookup to the viewer element — `sv-progress`
            // lives inside _slideViewerEl and is only reachable via
            // document.getElementById AFTER the layout renderer has
            // inserted the viewer into the DOM. Query by scope so we
            // can always update the progress text.
            const getProgressEl = () => _slideViewerEl?.querySelector('#sv-progress') || progressEl;
            const prog0 = getProgressEl();
            if (prog0) prog0.textContent = 'Uploading...';
            const resp = await fetch(`${API}/api/meetings/${_enc(uploadMid)}/slides/upload`, { method: 'POST', body: formData });
            if (!resp.ok) {
              const err = await resp.json().catch(() => ({}));
              throw new Error(err.error || `Upload failed: ${resp.status}`);
            }
            const data = await resp.json();
            _slideState.deckId = data.deck_id;
            _slideState.meetingId = uploadMid;
            const prog1 = getProgressEl();
            if (prog1) prog1.textContent = 'Processing...';
            // Resolve the real total_slides + deck source/target by
            // polling /api/meetings/{mid}/slides (same logic used on
            // popout boot). The earlier fix called _showSlides(0, ...)
            // optimistically — but total=0 means the prev/next
            // bounds-check (`newIdx >= _slideState.total`) refuses
            // every navigation until slide_deck_changed arrives. Route
            // through _restoreSlideState so we never land in state with
            // a mounted deck we can't navigate.
            try { _restoreSlideState(); } catch {}
          } catch (err) {
            showModal(`<div class="modal-confirm-title">Upload failed</div><div class="modal-confirm-message">${esc(err.message)}</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm');
            const progErr = _slideViewerEl?.querySelector('#sv-progress') || progressEl;
            if (progErr) progErr.textContent = '';
          }
          slidesInput.value = '';
        });
      }

      // ── Slide state restore on popout (re)load ───────────────
      // Survive a server restart, page reload, or popout window reopen by:
      //  1. Looking up the active deck for the current meeting from the API.
      //  2. Restoring the user's last-viewed slide index from localStorage
      //     (per deck_id, so deck-switches don't carry stale state).
      //  3. Polling briefly if the meeting id isn't known yet (e.g. /api/status
      //     hasn't responded). Cap retries so a truly slide-less meeting
      //     doesn't keep hammering the API.
      const _LAST_SLIDE_KEY = (deckId) => `popout_slide_idx_${deckId}`;
      async function _restoreSlideState() {
        const startedAt = Date.now();
        let attempts = 0;
        while (Date.now() - startedAt < 8000) {
          attempts += 1;
          const checkMid = mid;
          if (checkMid) {
            try {
              const r = await fetch(`${API}/api/meetings/${_enc(checkMid)}/slides`);
              if (r.ok) {
                const meta = await r.json();
                if (meta && meta.total_slides > 0 && meta.deck_id) {
                  // If the deck has changed since the user's last visit,
                  // toss the stale localStorage entry so we don't jump
                  // them to a slide index from a different deck.
                  let restored = 0;
                  try {
                    const v = localStorage.getItem(_LAST_SLIDE_KEY(meta.deck_id));
                    if (v != null) restored = Math.max(0, Math.min(meta.total_slides - 1, parseInt(v, 10) || 0));
                  } catch {}
                  // Server-side current_slide_index wins if larger than 0
                  // (presenter is broadcasting), else use restored.
                  const idx = (meta.current_slide_index && meta.current_slide_index > 0)
                    ? meta.current_slide_index
                    : restored;
                  // Pass source_lang/target_lang so pane-by-language
                  // alignment kicks in immediately (no extra fetch).
                  _showSlides(meta.total_slides, meta.deck_id, {
                    startIndex: idx,
                    sourceLang: (meta.source_lang || '').toLowerCase() || null,
                    targetLang: (meta.target_lang || '').toLowerCase() || null,
                  });
                  return;
                }
                // No active deck — nothing to restore.
                return;
              }
              // 404/503 are expected when slides haven't been started or
              // the worker isn't ready. Don't keep polling on those.
              if (r.status === 404 || r.status === 503) return;
            } catch {}
          }
          // Wait then retry — meeting id may still be loading from /api/status
          await new Promise((res) => setTimeout(res, 800));
          if (attempts > 10) return;
        }
      }
      _restoreSlideState();

      // ── Document PiP (Float button) ──────────────────────────
      // Uses an iframe inside the PiP window loading the reader.html page,
      // which has its own WS connection and self-contained rendering.
      const pipBtn = document.getElementById('popout-pip-btn');
      if (pipBtn && 'documentPictureInPicture' in window) {
        pipBtn.style.display = '';
        let pipWin = null;
        pipBtn.addEventListener('click', async () => {
          if (pipWin) { pipWin.close(); pipWin = null; pipBtn.textContent = 'Float'; return; }
          try {
            // Request max allowed size — full width, top 20% of screen.
            // Chrome caps PiP at ~80% of screen area; user can resize freely.
            pipWin = await documentPictureInPicture.requestWindow({
              width: Math.round(screen.width * 0.8),
              height: Math.round(screen.height * 0.2),
            });
            // Load reader.html in an iframe — it has its own WS + rendering
            const iframe = pipWin.document.createElement('iframe');
            iframe.src = `${location.origin}/reader`;
            iframe.style.cssText = 'width:100%;height:100%;border:none;margin:0;padding:0';
            pipWin.document.body.style.cssText = 'margin:0;padding:0;overflow:hidden';
            pipWin.document.body.appendChild(iframe);
            pipBtn.textContent = 'Unfloat';
            pipWin.addEventListener('pagehide', () => {
              pipBtn.textContent = 'Float';
              pipWin = null;
            });
          } catch (err) { console.warn('PiP failed:', err); }
        });
      }

      // Connect view WS for live updates. We subscribe when either:
      //  - No hash (normal live-mode popout from the active meeting), OR
      //  - The hash-pinned meeting IS the currently-recording one
      //    (user opened the popout from a review-mode "Live" button on
      //    the live meeting — same meeting, just a different entry
      //    point). The previous "if (!hashMatch)" guard left that case
      //    replay-only, so the popout appeared frozen after the initial
      //    fetch. The WS broadcasts events for whichever meeting is
      //    currently `current_meeting` server-side, so connecting when
      //    mid matches that is always safe.
      let _shouldConnectWs = !hashMatch;
      if (hashMatch) {
        try {
          const s = await fetch(`${API}/api/status`);
          if (s.ok) {
            const sd = await s.json();
            if (sd?.meeting?.id && sd.meeting.id === mid) _shouldConnectWs = true;
          }
        } catch {}
      }
      if (_shouldConnectWs) {
        // Track which meeting the popout is currently rendering. When a
        // new meeting starts (the live current_meeting becomes a
        // different id than what we've been ingesting for), wipe the
        // store + grid so we don't show prior meeting's transcript
        // mixed with the new one. This is the popout-side counterpart
        // of the AudioPipeline's cancel handling — popout doesn't have
        // an audio WS so it must do the cleanup itself.
        let _popoutCurrentMeetingId = mid || null;
        function _resetPopoutMeetingState(nextMid) {
          try { store.clear(); } catch {}
          try {
            if (window._gridRenderer) window._gridRenderer._clear(true);
          } catch {}
          _popoutCurrentMeetingId = nextMid || null;
        }

        // Connection-status pill in the popout header so the user can
        // see when the WS drops/reconnects rather than silently going
        // stale. Tinted dot next to the meeting title.
        const _popoutDot = document.querySelector('.popout-dot');
        function _setConnState(state) {
          if (!_popoutDot) return;
          _popoutDot.dataset.connState = state;  // open | connecting | down
          _popoutDot.title = (
            state === 'open' ? 'Live — connected to server' :
            state === 'connecting' ? 'Reconnecting…' :
            'Disconnected — retrying'
          );
        }

        // Auto-reconnect with exponential backoff. Fires after every
        // close/error event. Without this, a server restart left the
        // popout silently stale forever — user had to manually refresh.
        // Backoff caps at 30s; resets to 1s on each successful open.
        let _viewWs = null;
        let _viewKeepAlive = null;
        let _viewReconnectDelay = 1000;
        let _viewReconnectTimer = null;
        let _viewClosed = false;  // user-intentional close (e.g. tab unload)

        function _scheduleViewReconnect() {
          if (_viewReconnectTimer || _viewClosed) return;
          const delay = _viewReconnectDelay;
          _viewReconnectDelay = Math.min(_viewReconnectDelay * 2, 30000);
          _setConnState('connecting');
          _viewReconnectTimer = setTimeout(() => {
            _viewReconnectTimer = null;
            _connectViewWs();
          }, delay);
        }

        function _connectViewWs() {
          if (_viewClosed) return;
          if (_viewWs && (
            _viewWs.readyState === WebSocket.OPEN ||
            _viewWs.readyState === WebSocket.CONNECTING
          )) return;

          _setConnState('connecting');
          let ws;
          try {
            ws = new WebSocket(`${WS_PROTO}//${location.host}/api/ws/view`);
          } catch (e) {
            console.warn('[popout] WS construct failed, will retry:', e);
            _scheduleViewReconnect();
            return;
          }
          _viewWs = ws;

          ws.addEventListener('open', () => {
            _viewReconnectDelay = 1000;
            _setConnState('open');
            // After a reconnect, replay missed segments from the
            // current meeting (if any) so the user doesn't sit
            // looking at a frozen transcript until the next utterance
            // arrives. Best-effort — failure here is non-fatal.
            (async () => {
              try {
                const sresp = await fetch(`${API}/api/status`);
                const sd = await sresp.json();
                const liveMid = sd?.meeting?.id;
                if (!liveMid) return;
                if (_popoutCurrentMeetingId && _popoutCurrentMeetingId !== liveMid) {
                  _resetPopoutMeetingState(liveMid);
                }
                if (!_popoutCurrentMeetingId) _popoutCurrentMeetingId = liveMid;
                const r = await fetch(`${API}/api/meetings/${_enc(liveMid)}`);
                if (!r.ok) return;
                const md = await r.json();
                if (md.events) {
                  for (const ev of md.events) {
                    try { store.ingest(ev); } catch {}
                  }
                }
              } catch {}
            })();
          });

          ws.addEventListener('close', () => {
            if (_viewKeepAlive) { clearInterval(_viewKeepAlive); _viewKeepAlive = null; }
            _setConnState('down');
            _scheduleViewReconnect();
          });

          ws.addEventListener('error', () => {
            // Browser will fire close after error — handler above takes
            // care of reconnect scheduling.
          });

          ws.addEventListener('message', (evt) => {
            try {
              const msg = JSON.parse(evt.data);
              // Meeting-lifecycle resets — clear stale transcript/state
              // before any new events flow in.
              if (
                msg.type === 'meeting_stopped' ||
                msg.type === 'meeting_cancelled'
              ) {
                _resetPopoutMeetingState(null);
                return;
              }
              // meeting_started fires when the operator hits Start; the
              // popout (and HDMI kiosk mirror) must clear any content
              // left over from the previous meeting AND reset the
              // layout back to the transcript-only 'translate' preset
              // so the next meeting begins with a clean view.
              if (msg.type === 'meeting_started') {
                _resetPopoutMeetingState(msg.meeting_id || null);
                try {
                  if (window.PopoutLayoutStorage) {
                    // Force the transcript-only 'translate' preset;
                    // 'translator' (Presentation, transcript + slides)
                    // is the storage default but the user spec is
                    // "just translation" on a fresh meeting.
                    const fresh = {
                      ...window.PopoutLayoutStorage.DEFAULTS,
                      preset: 'translate',
                      customTree: null,
                      ratiosByPreset: {},
                    };
                    window.PopoutLayoutStorage.save(fresh);
                    window.dispatchEvent(new CustomEvent('popout-layout:remote-apply', {
                      detail: { layout: fresh, version: 0 },
                    }));
                  }
                } catch (err) {
                  console.warn('[popout] meeting_started layout reset failed:', err);
                }
                // Kiosk idle splash fade: hide via the body cascade by
                // toggling data-recording. The CSS in _state.css already
                // hides the splash when body[data-role="kiosk"] also has
                // data-recording set.
                try {
                  document.body.setAttribute('data-recording', '');
                  document.body.classList.add('recording');
                } catch {}
                return;
              }
              // Detect meeting_id rollover on any event that carries
              // one (the server stamps most events with meeting_id).
              const evMid = msg.meeting_id || msg.meetingId || null;
              if (evMid && _popoutCurrentMeetingId && evMid !== _popoutCurrentMeetingId) {
                _resetPopoutMeetingState(evMid);
              } else if (evMid && !_popoutCurrentMeetingId) {
                _popoutCurrentMeetingId = evMid;
              }

              // ── Per-type cascade ────────────────────────────────
              // Every server-emitted control type MUST have a named branch
              // here. The catch-all `default` at the bottom funnels into
              // a counter + dev-mode warning so drift surfaces in the JS
              // handler-coverage test (tests/js/ws-event-handler-coverage.test.mjs).
              // Explicit no-ops are encoded as `/* noop */ break;` so the
              // intent is visible to readers.
              switch (msg.type) {
                case 'speaker_rename':
                  if (msg.cluster_id != null) {
                    renameSpeaker(msg.cluster_id, msg.display_name);
                    _refreshTranscriptSpeakerLabels();
                  }
                  break;
                case 'speaker_remap':
                  // Server collapsed cluster_ids when diarize centroids
                  // merged. Update the speaker registry so retired ids
                  // don't linger as ghost names in the popout transcript.
                  if (msg.renames && typeof msg.renames === 'object') {
                    for (const [retiredStr, survivor] of Object.entries(msg.renames)) {
                      const retired = parseInt(retiredStr, 10);
                      if (!Number.isFinite(retired)) continue;
                      const survivorEntry = _speakerRegistry.clusters.get(survivor);
                      const retiredEntry = _speakerRegistry.clusters.get(retired);
                      if (retiredEntry && survivorEntry && retiredEntry.displayName && !survivorEntry.displayName) {
                        survivorEntry.displayName = retiredEntry.displayName;
                      }
                      _speakerRegistry.clusters.delete(retired);
                    }
                    _refreshTranscriptSpeakerLabels();
                  }
                  break;
                case 'speaker_pulse':
                  /* noop in popout — no seat strip to pulse */
                  break;
                case 'seat_update':
                case 'speaker_assignment':
                case 'speaker_correction':
                case 'room_layout_update':
                case 'summary_regenerated':
                case 'transcript_revision':
                case 'interpretation_status':
                case 'bt_status':
                  /* noop in popout — these surface in the admin view */
                  break;
                case 'meeting_warning':
                case 'meeting_warning_cleared':
                case 'audio_drift':
                case 'finalize_progress':
                  // Intentionally not surfaced in popout UI today (admin
                  // owns these pills); listed so the cascade is exhaustive
                  // and the handler-coverage gate stays green.
                  /* noop */
                  break;
                case 'background_finalize_progress':
                  // Phase B's progress channel — mounts a corner toast
                  // for the meeting whose finalize is running in the
                  // background after Stop. Inert in popout view today;
                  // the admin tab handler below renders the visible toast.
                  _renderBackgroundFinalizeToast(msg);
                  break;
                case 'slide_change':
                case 'slide_deck_changed':
                case 'slide_partial_ready':
                case 'slide_job_progress':
                  _handleSlideWsEvent(msg);
                  break;
                case undefined:
                  // No `type` field → it's a TranscriptEvent (segment carrier).
                  ingestFromLiveWs(msg);
                  break;
                default:
                  // Unknown control type — surface in dev-mode counter
                  // so the JS handler-coverage test detects drift, AND
                  // log so a maintainer notices in the browser console.
                  if (window.__test_msg_log) {
                    window.__test_unhandled_count = (window.__test_unhandled_count || 0) + 1;
                    window.__test_unhandled_types = window.__test_unhandled_types || new Set();
                    window.__test_unhandled_types.add(msg.type);
                  }
                  console.warn('popout WS: unhandled event type', msg.type, msg);
                  // DELIBERATELY do NOT call ingestFromLiveWs(msg) — that
                  // was the bug class: control events being funnelled into
                  // SegmentStore.ingest with segment_id=undefined.
                  break;
              }
            } catch {}
          });

          _viewKeepAlive = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              try { ws.send('ping'); } catch {}
            } else {
              clearInterval(_viewKeepAlive);
              _viewKeepAlive = null;
            }
          }, 30000);
        }

        // Close cleanly on tab unload so the server doesn't keep a
        // dead client around for the silence-watchdog grace window.
        window.addEventListener('beforeunload', () => {
          _viewClosed = true;
          if (_viewWs && _viewWs.readyState === WebSocket.OPEN) {
            try { _viewWs.close(1000, 'page-unload'); } catch {}
          }
        });

        _connectViewWs();
      }
    } catch (e) { console.error('Pop-out init error:', e); }
  })();
}
