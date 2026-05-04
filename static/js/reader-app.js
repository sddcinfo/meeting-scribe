const API = location.origin;
const _wsO = window.__SCRIBE_WS_OVERRIDE || null;
const WSP = _wsO ? 'wss:' : (location.protocol === 'https:' ? 'wss:' : 'ws:');
const WSH = _wsO ? _wsO.replace(/^wss?:\/\//, '') : location.host;
// Dynamic cap: fill available pane height. Recalculated on first render.
let MAX_VISIBLE = 20;

let languages = {};
let langA = null;
let langB = null;
const segments = new Map();
const pending = new Map();

const scrollA = document.getElementById('scroll-a');
const scrollB = document.getElementById('scroll-b');
const liveA = document.getElementById('live-a');
const liveB = document.getElementById('live-b');
const labelA = document.getElementById('label-a');
const labelB = document.getElementById('label-b');
const waitEl = document.getElementById('r-waiting');
const splitEl = document.getElementById('r-split');
const dotEl = document.getElementById('r-dot');
const langLabel = document.getElementById('r-lang-label');
const dirBtn = document.getElementById('r-dir-btn');

// Direction: newest-first matches the guest view default. Persisted so
// the toggle survives reloads on the same kiosk/iPad.
let newestFirst = (localStorage.getItem('reader_newestFirst') ?? 'true') !== 'false';

function _syncDirBtn() {
  dirBtn.textContent = newestFirst ? '↑ Newest' : '↓ Oldest';
}

function _applyReaderDirection() {
  for (const scroll of [scrollA, scrollB]) {
    const live = scroll.querySelector('.live');
    const segs = [...scroll.querySelectorAll('.seg')];
    segs.forEach(s => s.remove());
    if (newestFirst) {
      scroll.prepend(live);
      // Oldest was first in DOM; reversing + inserting after .live one
      // by one lands the newest directly under .live.
      segs.reverse().forEach(s => live.after(s));
    } else {
      scroll.appendChild(live);
      segs.reverse().forEach(s => scroll.insertBefore(s, live));
    }
    requestAnimationFrame(() => {
      scroll.scrollTop = newestFirst ? 0 : scroll.scrollHeight;
    });
  }
}

_syncDirBtn();
_applyReaderDirection();

dirBtn.addEventListener('click', () => {
  newestFirst = !newestFirst;
  localStorage.setItem('reader_newestFirst', String(newestFirst));
  _syncDirBtn();
  _applyReaderDirection();
});

// Text size controls (A+/A-)
let _rTextScale = parseFloat(localStorage.getItem('scribe_text_scale') || '1');
function _rApplyTextScale() {
  document.documentElement.style.setProperty('--text-scale', _rTextScale.toFixed(2));
  localStorage.setItem('scribe_text_scale', _rTextScale.toFixed(2));
}
_rApplyTextScale();
document.getElementById('r-text-larger').addEventListener('click', () => {
  _rTextScale = Math.min(2.5, _rTextScale + 0.15);
  _rApplyTextScale();
});
document.getElementById('r-text-smaller').addEventListener('click', () => {
  _rTextScale = Math.max(0.5, _rTextScale - 0.15);
  _rApplyTextScale();
});

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function langName(code) {
  const l = languages[code];
  return l ? (l.native_name || l.name) : code?.toUpperCase() || '?';
}

// `html` is a pre-rendered HTML body (already esc()'d plain text OR
// `furigana_html` ruby markup from the server). The caller decides which.
function addSeg(scroll, id, html, langClass) {
  let el = scroll.querySelector(`[data-seg="${id}"]`);
  if (!el) {
    el = document.createElement('div');
    el.className = 'seg';
    el.dataset.seg = id;
    const live = scroll.querySelector('.live');
    if (newestFirst) {
      live.after(el);
    } else {
      scroll.insertBefore(el, live);
    }
    // Drop the oldest once we exceed the cap. Oldest lives at the far
    // end of the scroll from .live in both directions.
    const segs = scroll.querySelectorAll('.seg');
    if (segs.length > MAX_VISIBLE) {
      (newestFirst ? segs[segs.length - 1] : segs[0]).remove();
    }
    requestAnimationFrame(() => {
      scroll.scrollTop = newestFirst ? 0 : scroll.scrollHeight;
    });
  }
  el.innerHTML = `<div class="seg-text ${langClass}">${html}</div>`;
}

function renderSeg(id) {
  const evt = segments.get(id);
  if (!evt?.text) return;

  // Monolingual meetings (detected from meeting metadata in _initViewer):
  // everything flows into pane A, pane B is hidden via CSS. No
  // translation-target inference below.
  if (document.body.classList.contains('monolingual')) {
    if (!langA && evt.language) {
      langA = evt.language;
      labelA.textContent = langName(langA);
      langLabel.textContent = langName(langA);
    }
    const body = evt.furigana_html || esc(evt.text);
    const cls = evt.language === 'ja' ? 'ja' : '';
    addSeg(scrollA, id, body, cls);
    return;
  }

  // Detect languages from first segment with translation (bilingual only).
  if (!langA && evt.language) {
    langA = evt.language;
    labelA.textContent = langName(langA);
  }
  if (!langB && evt.translation?.target_language) {
    langB = evt.translation.target_language;
    labelB.textContent = langName(langB);
    langLabel.textContent = `${langName(langA)} / ${langName(langB)}`;
  }

  // Route by language, not by original/translation. Each segment
  // contributes to both panes: the original text goes to the pane matching
  // its language, the translation to the other pane.
  //
  // For each pane, prefer the server-rendered ruby HTML (`furigana_html` /
  // `translation.furigana_html`) when present so kanji shows pronunciation.
  // Fall back to esc()'d plain text for non-JA languages or before the
  // furigana revision lands.
  const origLang = evt.language;
  const trLang = evt.translation?.target_language;

  // Pane A body: original if it matches langA, otherwise translation
  let bodyA = null, classA = '';
  if (origLang === langA) {
    bodyA = evt.furigana_html || esc(evt.text);
    classA = langA === 'ja' ? 'ja' : '';
  } else if (trLang === langA && evt.translation?.text) {
    bodyA = evt.translation.furigana_html || esc(evt.translation.text);
    classA = langA === 'ja' ? 'ja' : '';
  }

  // Pane B body: original if it matches langB, otherwise translation
  let bodyB = null, classB = '';
  if (origLang === langB) {
    bodyB = evt.furigana_html || esc(evt.text);
    classB = langB === 'ja' ? 'ja' : '';
  } else if (trLang === langB && evt.translation?.text) {
    bodyB = evt.translation.furigana_html || esc(evt.translation.text);
    classB = langB === 'ja' ? 'ja' : '';
  }

  if (bodyA) addSeg(scrollA, id, bodyA, classA);
  if (bodyB) {
    addSeg(scrollB, id, bodyB, classB);
    liveB.textContent = '';
  }
}

// Merge an incoming event onto an existing record, taking the union of
// additive dimensions (translation, furigana_html, speakers). The server
// broadcasts each dimension independently and the translation rebroadcast
// does NOT carry furigana, so naive overwrite drops it. Mirror of the
// admin SegmentStore merge, simplified for this view.
function _mergeReader(existing, incoming) {
  if (!existing) return { ...incoming };
  const merged = { ...existing, ...incoming };
  if (existing.text && (!incoming.text || incoming.text.length < existing.text.length)) {
    merged.text = existing.text;
  }
  merged.revision = Math.max(existing.revision || 0, incoming.revision || 0);
  if (!merged.furigana_html && existing.furigana_html) {
    merged.furigana_html = existing.furigana_html;
  }
  const exTr = existing.translation;
  const inTr = incoming.translation;
  if (exTr?.text && !inTr?.text) {
    merged.translation = exTr;
  } else if (inTr?.text) {
    merged.translation = { ...inTr };
  } else if (exTr || inTr) {
    merged.translation = inTr || exTr;
  }
  if (exTr?.furigana_html && !merged.translation?.furigana_html) {
    merged.translation = { ...(merged.translation || {}), furigana_html: exTr.furigana_html };
  }
  return merged;
}

function ingest(evt) {
  const id = evt.segment_id;
  if (!id || !evt.text) return;

  if (!evt.is_final) {
    liveA.textContent = evt.text;
    return;
  }

  // Locate any prior copy (displayed OR pending) and merge into it. This
  // is what makes furigana survive: a furigana revision arriving while
  // the segment is still in `pending` (waiting for translation) gets
  // folded into the pending record, and the eventual promotion carries
  // the furigana along instead of overwriting it with the translation
  // event (which doesn't carry furigana_html).
  const wasInSegments = segments.has(id);
  const wasInPending = pending.has(id);
  const prior = wasInSegments ? segments.get(id) : (wasInPending ? pending.get(id) : null);
  const merged = _mergeReader(prior, evt);

  const hasTr = !!(merged.translation?.text && merged.translation.status === 'done');

  // Already displayed → update in place.
  if (wasInSegments) {
    segments.set(id, merged);
    renderSeg(id);
    return;
  }

  // Ready to display.
  if (hasTr) {
    if (wasInPending) pending.delete(id);
    segments.set(id, merged);
    liveA.textContent = '';
    liveB.textContent = '';
    renderSeg(id);
    return;
  }

  // Hold pending until translation arrives. 4 s safety net so a segment
  // that never gets a translation still surfaces eventually.
  pending.set(id, merged);
  liveA.textContent = merged.text;
  if (!wasInPending) {
    setTimeout(() => {
      if (pending.has(id) && !segments.has(id)) {
        segments.set(id, pending.get(id));
        pending.delete(id);
        liveA.textContent = '';
        liveB.textContent = '';
        renderSeg(id);
      }
    }, 4000);
  }
}

async function loadLangs() {
  try {
    const r = await fetch(`${API}/api/languages`);
    const d = await r.json();
    if (d.languages) {
      for (const l of d.languages) languages[l.code] = l;
    }
  } catch {}
}

let _wsc = false;
let _kaInterval = null;
function connectWs() {
  if (_wsc) return;
  _wsc = true;
  const ws = new WebSocket(`${WSP}//${WSH}/api/ws/view`);
  ws.onmessage = (e) => { try { ingest(JSON.parse(e.data)); } catch {} };
  ws.onclose = () => {
    _wsc = false;
    if (_kaInterval) { clearInterval(_kaInterval); _kaInterval = null; }
    dotEl.classList.add('idle');
    setTimeout(poll, 3000);
  };
  ws.onopen = () => { dotEl.classList.remove('idle'); };
  _kaInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping');
  }, 30000);
}

async function poll() {
  try {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) throw 0;
    const d = await r.json();
    if (d.meeting?.state === 'recording') {
      if (!Object.keys(languages).length) await loadLangs();
      waitEl.style.display = 'none';
      splitEl.style.display = '';
      dotEl.classList.remove('idle');
      if (!_wsc) connectWs();
      return;
    }
  } catch {}
  waitEl.style.display = '';
  splitEl.style.display = 'none';
  dotEl.classList.add('idle');
  setTimeout(poll, 3000);
}

// ── Slide viewer ────────────────────────────────────────────
// Dynamically load slide-viewer.js and initialize
(function() {
  const script = document.createElement('script');
  script.src = '/static/js/slide-viewer.js';
  script.onload = function() {
    const container = document.getElementById('slide-viewer-container');
    // Determine meeting ID from status API when meeting starts
    let slideViewer = null;

    // Patch the WS message handler to forward slide events
    const origOnMessage = null;
    const _origIngest = window.ingest || function(){};

    // Override connectWs to intercept slide events
    const _origConnectWs = connectWs;
    connectWs = function() {
      if (_wsc) return;
      _wsc = true;
      const ws = new WebSocket(WSP + '//' + WSH + '/api/ws/view');
      ws.onmessage = function(e) {
        try {
          const data = JSON.parse(e.data);
          // Forward slide events to viewer
          if (slideViewer && (
            data.type === 'slide_deck_changed' ||
            data.type === 'slide_change' ||
            data.type === 'slide_job_progress'
          )) {
            slideViewer.handleWsEvent(data);
          }
          // Regular transcript events
          if (data.segment_id) {
            ingest(data);
          }
        } catch {}
      };
      ws.onclose = function() {
        _wsc = false;
        if (_kaInterval) { clearInterval(_kaInterval); _kaInterval = null; }
        dotEl.classList.add('idle');
        setTimeout(poll, 3000);
      };
      ws.onopen = function() { dotEl.classList.remove('idle'); };
      _kaInterval = setInterval(function() {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping');
      }, 30000);
    };

    // Initialize viewer when meeting becomes active
    const _origPoll = poll;
    const _initViewer = async function(meetingId) {
      // Drive monolingual layout from meeting metadata, not from segment
      // content. A bilingual meeting can legitimately have early
      // untranslated segments while translation catches up — inferring
      // the layout from segment sparsity would misclassify it. If the
      // metadata fetch fails (offline / stale) we leave the default
      // two-pane layout intact.
      try {
        const mr = await fetch(API + '/api/meetings/' + encodeURIComponent(meetingId));
        if (mr.ok) {
          const md = await mr.json();
          const lp = md?.meta?.language_pair;
          if (Array.isArray(lp) && lp.length === 1) {
            document.body.classList.add('monolingual');
            langA = lp[0];
            langB = null;
            labelA.textContent = langName(langA);
            langLabel.textContent = langName(langA);
          } else if (Array.isArray(lp) && lp.length === 2) {
            document.body.classList.remove('monolingual');
          }
        }
      } catch { /* best-effort; keep default two-pane layout */ }

      if (!slideViewer) {
        slideViewer = new SlideViewer(container, meetingId, {
          isAdmin: location.protocol === 'https:',
        });
        await slideViewer.checkExisting();
      }
    };

    // Patch poll to capture meeting ID
    const __origPoll = poll;
    poll = async function() {
      try {
        const r = await fetch(API + '/api/status');
        if (!r.ok) throw 0;
        const d = await r.json();
        if (d.meeting && d.meeting.state === 'recording') {
          if (!Object.keys(languages).length) await loadLangs();
          waitEl.style.display = 'none';
          splitEl.style.display = '';
          dotEl.classList.remove('idle');
          await _initViewer(d.meeting.meeting_id);
          if (!_wsc) connectWs();
          return;
        }
      } catch {}
      waitEl.style.display = '';
      splitEl.style.display = 'none';
      dotEl.classList.add('idle');
      if (slideViewer) { slideViewer.hide(); slideViewer = null; }
      setTimeout(poll, 3000);
    };
  };
  document.head.appendChild(script);
})();

// ── Document Picture-in-Picture overlay ─────────────────────
// Chrome 116+: float the transcript above fullscreen apps.
const pipBtn = document.getElementById('pip-btn');
if ('documentPictureInPicture' in window) {
  pipBtn.style.display = '';
  let pipWin = null;

  pipBtn.addEventListener('click', async () => {
    if (pipWin) {
      pipWin.close();
      pipWin = null;
      return;
    }
    try {
      pipWin = await documentPictureInPicture.requestWindow({
        width: Math.min(800, Math.round(screen.width * 0.6)),
        height: Math.min(400, Math.round(screen.height * 0.35)),
      });
      // Copy all stylesheets into the PiP window
      for (const sheet of document.styleSheets) {
        try {
          const style = pipWin.document.createElement('style');
          for (const rule of sheet.cssRules) style.textContent += rule.cssText + '\n';
          pipWin.document.head.appendChild(style);
        } catch { /* cross-origin sheet, skip */ }
      }
      // Move the root element into the PiP window
      const root = document.querySelector('.root');
      pipWin.document.body.appendChild(root);
      pipBtn.textContent = 'Unfloat';

      // When PiP window closes, move content back
      pipWin.addEventListener('pagehide', () => {
        document.body.appendChild(pipWin.document.querySelector('.root') || root);
        pipBtn.textContent = 'Float';
        pipWin = null;
      });
    } catch (err) {
      console.warn('Document PiP failed:', err);
    }
  });
}

poll();
