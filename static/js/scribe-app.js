/**
 * Meeting Scribe — main application controller.
 *
 * Event-driven architecture:
 *   AudioWorklet → WebSocket (binary) → Server ASR/Translate → WebSocket (JSON) → SegmentStore → DOM
 *
 * SegmentStore: keyed by segment_id, tracks highest revision per segment.
 * DOM updates are batched via requestAnimationFrame for smooth rendering.
 */

const API = '';
const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTO}//${location.host}/api/ws`;
// View-only mode: determined by server (only the recording session is admin)
let VIEW_ONLY = false;

// Pop-out mode: ?popout=view opens a clean 2-column translation window
const POPOUT_MODE = new URLSearchParams(location.search).get('popout');
if (POPOUT_MODE) {
  document.body.classList.add('popout-view');
  VIEW_ONLY = true;
}

// Dynamic language pair — set from /api/languages or meeting meta
let currentLanguagePair = 'en,ja';

// Server-configured default pair. Populated by wireUp() once /api/languages
// responds. Mono-language meetings are NOT exposed via this default — they
// live behind the landing page's "Quick start English" button. This variable
// is read by showSetup() to reset the setup-screen dropdowns back to a
// bilingual pair every time the user returns to setup, so a prior mono
// meeting can't leak into the next one.
let _defaultLanguagePairCache = ['en', 'ja'];
function _defaultLanguagePair() {
  // Guarantee two distinct codes. If the server ever returned a mono default
  // (legacy config), we still promote it to bilingual here — the setup
  // screen never shows mono.
  const pair = _defaultLanguagePairCache.slice(0, 2);
  if (pair.length < 2) pair.push(pair[0] === 'en' ? 'ja' : 'en');
  if (pair[0] === pair[1]) pair[1] = pair[0] === 'en' ? 'ja' : 'en';
  return pair;
}
let _languageNames = {};  // code → {name, native_name}

// ─── Mic warm-up singleton ────────────────────────────────────────────────
//
// `getUserMedia` + `new AudioContext()` cost ~500-1500 ms the first time the
// page touches the microphone (permission prompt, device init, audio graph
// boot, sample-rate negotiation). Doing that work the moment the user taps a
// chair leaves a noticeable dead zone before capture actually starts. Instead
// we prime the mic the instant the setup panel becomes visible: the stream
// is acquired once, kept alive for the whole setup session, and reused by
// every `_enrollSeat` call. The chair tap then just attaches a fresh
// ScriptProcessor — instant.
//
// Released either when the meeting starts (mic is taken over by the recorder)
// or when the page leaves the setup view.
const micWarmup = {
  ac: null,
  stream: null,
  source: null,
  primed: false,
  primingPromise: null,

  async prime() {
    if (this.primed) return true;
    if (this.primingPromise) return this.primingPromise;
    this.primingPromise = (async () => {
      try {
        // Use the same constraints as the meeting-recording pipeline so the
        // warmed-up stream can be handed over to AudioPipeline without
        // re-prompting / re-negotiating the device.
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
        });
        this.ac = new AudioContext();
        if (this.ac.state === 'suspended') {
          // Some browsers won't resume an AudioContext until the next user
          // gesture. That's fine — _enrollSeat re-resumes inside the click
          // handler. Either way the device + permission are already warm.
          try { await this.ac.resume(); } catch {}
        }
        this.source = this.ac.createMediaStreamSource(this.stream);
        this.primed = true;
        return true;
      } catch (e) {
        console.warn('Mic warm-up failed (will retry on first chair tap):', e);
        this.release();
        return false;
      } finally {
        this.primingPromise = null;
      }
    })();
    return this.primingPromise;
  },

  release() {
    try { this.source?.disconnect(); } catch {}
    if (this.stream) {
      try { this.stream.getTracks().forEach(t => t.stop()); } catch {}
    }
    if (this.ac && this.ac.state !== 'closed') {
      try { this.ac.close(); } catch {}
    }
    this.ac = null;
    this.stream = null;
    this.source = null;
    this.primed = false;
  },

  // Hand the warmed-up stream + AudioContext to a long-lived owner (e.g. the
  // meeting recorder). The caller becomes responsible for tearing them down;
  // the warm-up forgets them so it doesn't double-close.
  consume() {
    if (!this.primed) return null;
    const handover = { ac: this.ac, stream: this.stream, source: this.source };
    this.ac = null;
    this.stream = null;
    this.source = null;
    this.primed = false;
    return handover;
  },
};

function _getLangA() { return currentLanguagePair.split(',')[0]; }
function _getLangB() { return currentLanguagePair.split(',')[1]; }
// A meeting is monolingual iff ``currentLanguagePair`` has a single code
// (e.g. "en") rather than the bilingual "ja,en" shape. Used to hide the
// translation column, the popout A/Both/B toggle, and the translated
// slides pane so the presentation gets the freed space.
function _isMonolingual() { return !_getLangB(); }

// Script-based language routing. ASR occasionally mislabels segments
// (Japanese tagged "en", English tagged "ja"), which leaks text into
// the wrong column. We treat the script as authoritative:
//   - CJK characters present → route to the CJK lang in the pair (ja/zh/ko)
//   - Hangul present → route to ko if present
//   - otherwise → route to the non-CJK lang in the pair (or fall back to
//     the ASR label if it's in-pair, else langA)
function _routeLangByScript(text, asrLang, langA, langB) {
  const t = text || '';
  // Kanji/Hiragana/Katakana/CJK Compatibility + fullwidth punct
  const hasJaKana = /[\u3040-\u309F\u30A0-\u30FF]/.test(t);
  const hasCJK = /[\u3400-\u9FFF\uF900-\uFAFF]/.test(t);
  const hasHangul = /[\uAC00-\uD7AF\u1100-\u11FF]/.test(t);
  const pair = [langA, langB];
  const findInPair = (code) => pair.includes(code) ? code : null;

  if (hasHangul) {
    const ko = findInPair('ko');
    if (ko) return ko;
  }
  if (hasJaKana) {
    const ja = findInPair('ja');
    if (ja) return ja;
  }
  if (hasCJK) {
    // Ambiguous Han — prefer ja then zh then ko in the pair
    for (const c of ['ja', 'zh', 'ko']) {
      const m = findInPair(c);
      if (m) return m;
    }
  }
  // No CJK at all → must NOT land in a CJK column
  const cjkCodes = new Set(['ja', 'zh', 'ko']);
  if (asrLang && pair.includes(asrLang) && !cjkCodes.has(asrLang)) return asrLang;
  const nonCjk = pair.find(l => !cjkCodes.has(l));
  return nonCjk || langA;
}

function _updateColumnHeaders() {
  const langA = _getLangA();
  const langB = _getLangB();
  const nameA = _languageNames[langA]?.native_name || _languageNames[langA]?.name || langA.toUpperCase();
  const codeA = langA.toUpperCase();
  const el = (id) => document.getElementById(id);
  // Reflect monolingual on <body> so the transcript-grid CSS collapses
  // the second column and the A/B column toggle buttons hide.
  document.body.classList.toggle('monolingual', !langB);
  if (!langB) {
    if (el('col-lang-a-label')) el('col-lang-a-label').textContent = nameA;
    if (el('col-lang-a-hint')) el('col-lang-a-hint').textContent = `${codeA} original`;
    // B-column content stays empty; CSS hides it via .monolingual.
    const btnA = el('btn-col-a');
    const btnB = el('btn-col-b');
    if (btnA) { btnA.textContent = _languageNames[langA]?.name || codeA; btnA.style.display = ''; }
    if (btnB) btnB.style.display = 'none';
    return;
  }
  const nameB = _languageNames[langB]?.native_name || _languageNames[langB]?.name || langB.toUpperCase();
  const codeB = langB.toUpperCase();
  if (el('col-lang-a-label')) el('col-lang-a-label').textContent = nameA;
  if (el('col-lang-a-hint')) el('col-lang-a-hint').textContent = `${codeA} original + ${codeB}\u2192${codeA}`;
  if (el('col-lang-b-label')) el('col-lang-b-label').textContent = nameB;
  if (el('col-lang-b-hint')) el('col-lang-b-hint').textContent = `${codeB} original + ${codeA}\u2192${codeB}`;
  // Update column selector buttons in control bar
  const shortA = _languageNames[langA]?.name || codeA;
  const shortB = _languageNames[langB]?.name || codeB;
  const btnA = el('btn-col-a');
  const btnB = el('btn-col-b');
  if (btnA) { btnA.textContent = shortA; btnA.style.display = ''; }
  if (btnB) { btnB.textContent = shortB; btnB.style.display = ''; }
}

/** Find all segments belonging to a speaker. Matches by identity,
 *  display_name, AND raw cluster_id (via the "Speaker N" pattern OR the
 *  client-side registry's seq_index). Sorted by start_ms so the modal
 *  renders in chronological order regardless of the transcript view's
 *  direction.
 */
function findSpeakerSegments(speakerName) {
  // Resolve the speakerName back to a cluster_id via the display registry
  // so we also catch entries whose speakers[0].display_name was never
  // populated (older journals).
  let targetClusterId = null;
  for (const [cid, entry] of _speakerRegistry.clusters) {
    if (entry?.displayName === speakerName) { targetClusterId = cid; break; }
    if (!entry?.displayName && `Speaker ${entry?.seqIndex}` === speakerName) {
      targetClusterId = cid; break;
    }
  }
  const nameMatch = speakerName.match(/^Speaker (\d+)$/);
  const nameClusterStr = nameMatch ? nameMatch[1] : null;

  const out = [];
  for (const ev of store.segments.values()) {
    const s = ev.speakers?.[0];
    if (!s) {
      // Attributed elsewhere (e.g. just the display name from finalize)
      if ((ev.speaker_name || ev.display_name) === speakerName) out.push(ev);
      continue;
    }
    if (s.identity === speakerName || s.display_name === speakerName) { out.push(ev); continue; }
    if (targetClusterId != null && s.cluster_id === targetClusterId) { out.push(ev); continue; }
    if (nameClusterStr && String(s.cluster_id) === nameClusterStr) { out.push(ev); continue; }
  }
  out.sort((a, b) => (a.start_ms || 0) - (b.start_ms || 0));
  return out;
}

// Pop-out window reference (shared between active + review mode Live buttons)
let _livePopout = null;

function _openPopout(btnEl, meetingId) {
  if (_livePopout && !_livePopout.closed) { _livePopout.focus(); return; }
  const hash = meetingId ? `#meeting/${meetingId}` : '';
  _livePopout = window.open(
    `${location.origin}${location.pathname}?popout=view${hash}`,
    'scribe-view',
    `width=${Math.round(window.screen.availWidth * 0.8)},height=${Math.round(window.screen.availHeight * 0.6)},menubar=no,toolbar=no,location=no,status=no`
  );
  if (btnEl) btnEl.classList.add('active-toggle');
  const checkClosed = setInterval(() => {
    if (_livePopout?.closed) {
      clearInterval(checkClosed);
      _livePopout = null;
      if (btnEl) btnEl.classList.remove('active-toggle');
    }
  }, 1000);
}

// ─── Finalization Tracking ───────────────────────────────────
// Track meetings currently being finalized so we can reopen the modal
// when navigating back to them.
const _finalizingMeetings = new Map(); // meetingId → { step, label, ws }

// ─── Room Setup — Data-Driven Tables + Presets ──────────────

const SPEAKER_COLORS = ['#c45d20', '#1a6fb5', '#2a8540', '#9b2d7b', '#8b6914', '#b52d2d', '#2d6b5e', '#6b3fa0'];
let _persistTimer = null;

// Canonical speaker color assignment — ONE function used everywhere.
// Keyed by cluster_id so the same speaker always gets the same color
// across the timeline lanes, transcript blocks, speaker modals, and room strip.
// Using cluster_id directly (not sorted order) ensures stability across re-sorts.
function getSpeakerColor(clusterId) {
  if (clusterId == null) return SPEAKER_COLORS[0];
  const idx = Math.abs(parseInt(clusterId, 10) || 0) % SPEAKER_COLORS.length;
  return SPEAKER_COLORS[idx];
}

// ─── Speaker Display Registry ─────────────────────────────────
// Maps raw cluster_ids (which may be large pseudo-IDs like 100, 101 from
// the time-proximity fallback, or arbitrary numbers from diarization) to
// friendly sequential labels: "Speaker 1", "Speaker 2"... in first-seen
// order. Also honors user-assigned names (from rename modal).
//
// Used everywhere a speaker needs to be displayed so the UI is consistent.
const _speakerRegistry = {
  // clusterId → {seqIndex, displayName}
  clusters: new Map(),
  nextIndex: 1,
};

/** True if a server-supplied name is a generic "Speaker <number>" label
 *  (e.g. "Speaker 0", "Speaker 101") that carries no human context and
 *  should be regenerated from the client-side sequential registry instead.
 */
function _isGenericSpeakerLabel(name) {
  if (!name) return true;
  // Strict: matches "Speaker 12", "Speaker 101", ignores anything else.
  return /^Speaker\s+\d+$/.test(String(name).trim());
}

/** True if the cluster_id is a pseudo / placeholder cluster (>=100).
 *  DEAD CODE AFTER 2026-04 SPEAKER-SEPARATION REFACTOR — the server no
 *  longer emits pseudo clusters. `_process_event` broadcasts events with
 *  `speakers: []` when diarization hasn't caught up, and
 *  `_speaker_catchup_loop` rebroadcasts a revised event once pyannote
 *  has a real cluster. This check is kept as defensive handling of any
 *  residual pseudo-cluster events from a rolling restart (old backend
 *  still running while the new frontend loads). Safe to remove in a
 *  future cleanup pass once no pseudo-cluster events are seen in the
 *  wild for a few meetings.
 */
function _isPseudoCluster(clusterId) {
  return clusterId != null && Number(clusterId) >= 100;
}

/** Get the display name for a cluster_id.
 *  Precedence:
 *    1. Human-assigned name (from rename modal) — persistent across renders
 *    2. Explicit identity from backend (real name like "Brad", "田中")
 *    3. Sequential "Speaker N" from the registry (first-seen order)
 *    4. "Speaker ?" for null / pseudo / unresolved clusters
 *
 *  Events with no speaker attribution (empty `speakers` array on the
 *  wire) arrive with clusterId === null. After the 2026-04 speaker
 *  separation refactor, that is the common case for the first 0–2s of
 *  any segment — ASR has produced text but diarization hasn't caught up
 *  yet. The catch-up loop will send a revised event within a few hundred
 *  ms with the real cluster_id, and the registry assigns a sequential
 *  index at that point. Show "Speaker ?" in the interim, not "Unknown",
 *  so the placeholder is visually identical to pseudo-cluster rendering
 *  and users don't see a scary "Unknown" flash.
 */
function getSpeakerDisplayName(clusterId, explicitName) {
  if (clusterId == null) return 'Speaker ?';
  // Pseudo-cluster: transient placeholder — don't allocate a seq index.
  if (_isPseudoCluster(clusterId)) {
    const existing = _speakerRegistry.clusters.get(clusterId);
    if (existing?.displayName) return existing.displayName;
    return 'Speaker ?';
  }
  let entry = _speakerRegistry.clusters.get(clusterId);
  if (!entry) {
    entry = { seqIndex: _speakerRegistry.nextIndex++, displayName: null };
    _speakerRegistry.clusters.set(clusterId, entry);
  }
  // Trust the server-supplied name. Post-2026-04 the server remaps raw
  // diarize cluster_ids to a stable seq_index at finalize, so
  // "Speaker 3" from the server IS the canonical label for cluster 3
  // across the transcript, timeline lanes, participants panel, and
  // summary — NOT a generic fallback to be renumbered client-side.
  // The previous _isGenericSpeakerLabel gate was silently renumbering
  // speakers based on UI iteration order, which diverged from the
  // server's seq_index. Real user-set names still win because they
  // flow through `renameSpeaker(cluster_id, name)` on `speaker_rename`
  // WS events and get stamped into entry.displayName below.
  if (explicitName) {
    entry.displayName = explicitName;
  }
  return entry.displayName || `Speaker ${entry.seqIndex}`;
}

/** Get the sequential number assigned to this cluster (1, 2, 3…) — useful
 *  for coloring/avatar text regardless of whether the speaker has been named.
 *  Pseudo-clusters return 0 (no seat).
 */
function getSpeakerSeqIndex(clusterId) {
  if (clusterId == null) return 0;
  if (_isPseudoCluster(clusterId)) return 0;
  const entry = _speakerRegistry.clusters.get(clusterId);
  if (entry) return entry.seqIndex;
  // Auto-register
  const seq = _speakerRegistry.nextIndex++;
  _speakerRegistry.clusters.set(clusterId, { seqIndex: seq, displayName: null });
  return seq;
}

/** Assign a user-chosen display name to a cluster. Returns the new name. */
function renameSpeaker(clusterId, newName) {
  if (clusterId == null) return null;
  const trimmed = (newName || '').trim();
  if (!trimmed) return null;
  const entry = _speakerRegistry.clusters.get(clusterId) || {};
  entry.displayName = trimmed;
  if (entry.seqIndex == null) entry.seqIndex = _speakerRegistry.nextIndex++;
  _speakerRegistry.clusters.set(clusterId, entry);
  return trimmed;
}

/** Walk through all REAL speakers (not pseudo-clusters) — used by the
 *  virtual table to render one seat per detected cluster in first-seen order.
 */
function getAllSpeakers() {
  return [..._speakerRegistry.clusters.entries()]
    .filter(([clusterId]) => !_isPseudoCluster(clusterId))
    .map(([clusterId, entry]) => ({
      clusterId,
      seqIndex: entry.seqIndex,
      displayName: entry.displayName || `Speaker ${entry.seqIndex}`,
      hasCustomName: !!entry.displayName,
    }))
    .sort((a, b) => a.seqIndex - b.seqIndex);
}

/** Reset the registry (between meetings). */
function _resetSpeakerRegistry() {
  _speakerRegistry.clusters.clear();
  _speakerRegistry.nextIndex = 1;
}

// ─── Modal System ───────────────────────────────────────────
//
// Single-overlay stack model. At most one modal is visible at a time,
// but nested calls (e.g. the Meeting Actions modal opening a confirm
// dialog) are supported by hiding the outer card in place — its DOM
// and event listeners stay alive — and appending a new sibling card
// that becomes the active one. closeModal() pops the stack: it
// removes the topmost card (or clears the root card at the bottom),
// fires its _onClose hook, and re-activates the previous card.
//
// Interaction rules (mirrored in STYLING.md):
//   - Escape closes ONLY the topmost card. Outer cards survive.
//   - Backdrop click behaves the same as Escape.
//   - _onClose fires on ANY close path (button, Escape, backdrop,
//     explicit closeModal / closeAllModals). Dialog primitives use it
//     to resolve their promise with a cancel value so awaiting code
//     never hangs if the user dismisses a dialog by keyboard instead
//     of clicking Cancel.
//   - Dialog primitives (alertDialog / confirmDialog / promptDialog)
//     clear _onClose before calling closeModal() on an explicit
//     confirm, so they don't double-resolve.

function showModal(html, cssClass = '') {
  const overlay = document.getElementById('modal-overlay');
  const rootCard = document.getElementById('modal-card');

  // Hide any currently-active card so the new one takes over. Hidden
  // cards stay in the DOM (and keep their listeners) until they pop.
  const prevActive = overlay.querySelector('.modal-card-active');
  if (prevActive) {
    prevActive.classList.remove('modal-card-active');
    prevActive.setAttribute('aria-hidden', 'true');
    prevActive.style.display = 'none';
  }

  // First modal: reuse the static root card. Stacked modal: create a
  // new sibling so the outer card's DOM + handlers survive untouched.
  let card;
  if (!prevActive) {
    card = rootCard;
  } else {
    card = document.createElement('div');
    overlay.appendChild(card);
  }
  card.className = `modal-card modal-card-active ${cssClass}`;
  card.innerHTML = html;
  card.style.display = '';
  card.removeAttribute('aria-hidden');
  card._onClose = null; // set by dialog primitives for cancel cleanup

  overlay.style.display = '';
  overlay.onclick = (e) => { if (e.target === overlay) closeModal(); };
  document.addEventListener('keydown', _modalEscHandler);
  return card;
}

function closeModal() {
  const overlay = document.getElementById('modal-overlay');
  const rootCard = document.getElementById('modal-card');
  const active = overlay.querySelector('.modal-card-active');
  if (!active) {
    // Nothing active — ensure overlay is down and bail.
    overlay.style.display = 'none';
    document.removeEventListener('keydown', _modalEscHandler);
    overlay.onclick = null;
    return;
  }

  // Fire the cleanup hook BEFORE we tear DOM down so handlers can
  // inspect the card if they need to. Errors here must never block
  // the close — a hanging stack would trap the user.
  if (typeof active._onClose === 'function') {
    try { active._onClose(); } catch (e) { console.error('modal _onClose threw', e); }
  }
  active._onClose = null;
  active.classList.remove('modal-card-active');

  if (active === rootCard) {
    // Root card stays in the DOM (it's the static shell). Clear it.
    active.innerHTML = '';
    active.style.display = 'none';
  } else {
    // Stacked sibling: remove entirely.
    active.remove();
  }

  // Pop: re-activate the most recently hidden card with real content,
  // or close the overlay if the stack is empty.
  const all = Array.from(overlay.querySelectorAll('.modal-card'));
  const hidden = all.filter(c => c.style.display === 'none' && c.innerHTML.trim() !== '');
  if (hidden.length > 0) {
    const top = hidden[hidden.length - 1];
    top.classList.add('modal-card-active');
    top.style.display = '';
    top.removeAttribute('aria-hidden');
  } else {
    overlay.style.display = 'none';
    document.removeEventListener('keydown', _modalEscHandler);
    overlay.onclick = null;
  }
}

// Escape-key handler. Only closes the TOP modal (closeModal pops one).
// Repeated Escape presses walk the stack down to empty.
function _modalEscHandler(e) {
  if (e.key !== 'Escape') return;
  // If focus is inside an input/select/textarea, let the element handle
  // Escape itself first. promptDialog's input attaches its own keydown
  // on the input and will cancel there; we stop the document handler
  // from firing a duplicate closeModal.
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    return;
  }
  closeModal();
}

// Close every card in the stack. Used when an action mutates the
// underlying data in a way that makes outer cards stale (for example
// deleting a meeting while its tools modal is open).
function closeAllModals() {
  const overlay = document.getElementById('modal-overlay');
  // Bounded loop: each iteration pops one card, overlay hides when empty.
  // The cap protects against any future bug that fails to advance state.
  for (let i = 0; i < 16; i++) {
    if (overlay.style.display === 'none') break;
    if (!overlay.querySelector('.modal-card-active')) break;
    closeModal();
  }
}
window.closeAllModals = closeAllModals;

// Styled Yes/No confirm. Resolves true on confirm, false on cancel,
// Escape, backdrop click, or any non-confirm close. Uses the card
// _onClose hook so every dismissal path converges on a single cancel
// value — the awaiting caller never hangs.
async function confirmDialog(title, message, confirmText = 'Delete', danger = true) {
  return new Promise(resolve => {
    const card = showModal(`
      <div class="modal-confirm-title">${title}</div>
      <div class="modal-confirm-message">${message}</div>
      <div class="modal-confirm-actions">
        <button class="modal-btn" id="modal-cancel">Cancel</button>
        <button class="modal-btn ${danger ? 'danger' : ''}" id="modal-confirm">${confirmText}</button>
      </div>
    `, 'confirm');
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    card._onClose = () => finish(false);
    card.querySelector('#modal-cancel').onclick = () => { closeModal(); };
    card.querySelector('#modal-confirm').onclick = () => {
      card._onClose = null;  // skip default cancel-resolution
      closeModal();
      finish(true);
    };
    card.querySelector('#modal-confirm').focus();
  });
}

// Styled single-button notice. Replaces window.alert so every diagnostic
// lives in the same visual language as the rest of the UI. Resolves when
// the user acknowledges (button, Enter, Escape, backdrop click — all
// treated as dismissals since there is nothing to cancel).
//
// Auto-promotes to a wider "pre" variant when the message looks like a
// stack trace / error body (multi-line, contains JSON/traceback markers,
// or exceeds a short-prose length). The pre variant wraps the message
// in a mono, scrollable, selectable region and adds a Copy button so
// the full string can be captured for bug reports.
async function alertDialog(title, message, okText = 'OK') {
  const msg = String(message ?? '');
  const isLong = msg.length > 160 || msg.includes('\n');
  const looksTraceback = /Traceback|^\s*at |\bError:|\bException:|HTTP \d{3}|<\?xml|^\s*\{/m.test(msg);
  const pre = isLong || looksTraceback;
  return new Promise(resolve => {
    const body = pre
      ? `<div class="modal-confirm-message pre">${esc(msg)}</div>`
      : `<div class="modal-confirm-message">${esc(msg).replace(/\n/g, '<br>')}</div>`;
    const copyBtn = pre
      ? `<button type="button" class="modal-copy-btn" id="modal-copy" title="Copy message to clipboard">
           <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
           <span>Copy</span>
         </button>`
      : '';
    const cssClass = pre ? 'confirm wide' : 'confirm';
    const card = showModal(`
      <div class="modal-confirm-title">${esc(title)}</div>
      ${body}
      <div class="modal-confirm-actions">
        ${copyBtn}
        <button class="modal-btn primary" id="modal-ok">${esc(okText)}</button>
      </div>
    `, cssClass);
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    // _onClose ensures Enter-via-focused-OK, Escape, backdrop click,
    // and explicit closeModal() all converge on resolve(). No
    // document-level keydown listener is needed — the standard
    // _modalEscHandler pops the stack and triggers this hook.
    card._onClose = finish;
    card.querySelector('#modal-ok').onclick = () => {
      card._onClose = null;
      closeModal();
      finish();
    };
    card.querySelector('#modal-ok').focus();
    const copy = card.querySelector('#modal-copy');
    if (copy) {
      copy.onclick = async (ev) => {
        ev.stopPropagation();
        const label = copy.querySelector('span');
        const fallback = () => {
          const ta = document.createElement('textarea');
          ta.value = msg;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); } finally { document.body.removeChild(ta); }
        };
        try {
          if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(msg);
          } else {
            fallback();
          }
          copy.classList.add('copied');
          if (label) label.textContent = 'Copied';
          setTimeout(() => { copy.classList.remove('copied'); if (label) label.textContent = 'Copy'; }, 1400);
        } catch {
          if (label) label.textContent = 'Copy failed';
        }
      };
    }
  });
}

// Styled text-input prompt. Resolves to the trimmed string on confirm,
// or null on cancel / Escape / backdrop click (matching the window.prompt
// contract so callers can do `if (raw === null) return` unchanged).
// Options: placeholder, initialValue, confirmText, type ('text'|'number'),
// min/max (for type=number), inputMode, help (extra hint text).
async function promptDialog(title, message, options = {}) {
  const {
    placeholder = '',
    initialValue = '',
    confirmText = 'OK',
    cancelText = 'Cancel',
    type = 'text',
    min,
    max,
    inputMode,
    help = '',
  } = options;
  return new Promise(resolve => {
    const extraAttrs = [
      type ? `type="${esc(type)}"` : '',
      placeholder ? `placeholder="${esc(placeholder)}"` : '',
      min != null ? `min="${esc(String(min))}"` : '',
      max != null ? `max="${esc(String(max))}"` : '',
      inputMode ? `inputmode="${esc(inputMode)}"` : '',
    ].filter(Boolean).join(' ');
    const card = showModal(`
      <div class="modal-confirm-title">${esc(title)}</div>
      <div class="modal-confirm-message">${esc(message).replace(/\n/g, '<br>')}</div>
      <input class="modal-input" id="modal-input" ${extraAttrs} value="${esc(initialValue)}" autocomplete="off" spellcheck="false">
      ${help ? `<div class="modal-input-help">${esc(help)}</div>` : ''}
      <div class="modal-confirm-actions">
        <button class="modal-btn" id="modal-cancel">${esc(cancelText)}</button>
        <button class="modal-btn primary" id="modal-confirm">${esc(confirmText)}</button>
      </div>
    `, 'confirm');
    const input = card.querySelector('#modal-input');
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    // Default on any non-confirm close (Escape from outside the input,
    // backdrop, explicit closeModal): resolve null, matching window.prompt.
    card._onClose = () => finish(null);
    const ok = () => {
      card._onClose = null;
      const v = input.value.trim();
      closeModal();
      finish(v);
    };
    const cancel = () => { closeModal(); /* _onClose resolves null */ };
    card.querySelector('#modal-confirm').onclick = ok;
    card.querySelector('#modal-cancel').onclick = cancel;
    // Input-scoped Enter/Escape. Escape inside the input takes priority
    // over the document-level _modalEscHandler (which ignores Escape
    // while focus is inside an input), so this cleanly cancels only
    // the prompt without walking farther up the stack.
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); ok(); }
      else if (e.key === 'Escape') { e.preventDefault(); cancel(); }
    });
    // Defer focus so the slide-up animation doesn't clobber the selection
    setTimeout(() => { input.focus(); input.select(); }, 50);
  });
}

// Expose to window so slide-viewer.js (loaded dynamically on reader.html)
// and inline onclick handlers can reach the same primitives without
// duplicating implementation.
window.alertDialog = alertDialog;
window.confirmDialog = confirmDialog;
window.promptDialog = promptDialog;

function showSpeakerModal(speakerName, speakerColor, segments, meetingId) {
  const totalSegs = segments.length;
  const totalMs = segments.reduce((s, e) => s + (e.end_ms - e.start_ms), 0);
  const totalSec = Math.round(totalMs / 1000);
  const languages = [...new Set(segments.map(e => e.language).filter(l => l !== 'unknown'))];
  const mid = meetingId || audioPlayer.meetingId || '';

  const segHtml = segments.map((e, i) => {
    const time = formatTime(e.start_ms);
    const lang = e.language !== 'unknown' ? `<span class="seg-lang">${e.language.toUpperCase()}</span>` : '';
    // Source body — prefer server-rendered ruby (`furigana_html`) over
    // plain esc()'d text so kanji shows pronunciation in the speaker modal.
    const srcBody = e.furigana_html || esc(e.text);
    const trBody = e.translation?.furigana_html || (e.translation?.text ? esc(e.translation.text) : '');
    const trans = e.translation?.text
      ? `<div class="speaker-seg-translation">${trBody}</div>`
      : '<div class="speaker-seg-translation" style="color:var(--text-muted);font-size:0.8rem">awaiting translation...</div>';
    const playBtn = mid
      ? `<button class="speaker-seg-play" data-idx="${i}" title="Play this segment">▶</button>`
      : '';
    return `<div class="speaker-segment" data-start="${e.start_ms}" data-end="${e.end_ms}" data-segment-id="${e.segment_id || ''}">
      <div class="speaker-seg-left">
        ${playBtn}
        <div>
          <div class="speaker-seg-time">${time} ${lang}</div>
          <div class="speaker-seg-text">${srcBody}</div>
        </div>
      </div>
      <div class="speaker-seg-right">${trans}</div>
    </div>`;
  }).join('');

  const playAllBtn = mid && totalSegs > 0
    ? `<button class="modal-btn" id="speaker-play-all" style="margin-left:auto">▶ Play All</button>`
    : '';

  const card = showModal(`
    <div class="speaker-modal-header">
      <div class="speaker-modal-dot" style="background:${speakerColor}"></div>
      <div class="speaker-modal-name" id="speaker-modal-current-name">${esc(speakerName)}</div>
      <button class="btn-ghost speaker-rename-btn" id="speaker-rename-btn" title="Rename speaker">Rename</button>
      <div class="speaker-modal-stats">
        <span>${totalSegs} segments</span>
        <span>${totalSec}s speaking</span>
        <span>${languages.join(', ') || '—'}</span>
        ${playAllBtn}
      </div>
      <button class="speaker-modal-close" id="speaker-modal-close-btn">&times;</button>
    </div>
    <div class="speaker-modal-body">${segHtml || `
      <div class="speaker-modal-empty">
        <h4>No segments found for ${esc(speakerName)}</h4>
        <div>This usually means the meeting hasn't been finalized yet,
          or the speaker label in the timeline lane no longer matches
          any transcript entry. Try refreshing the meeting view.</div>
      </div>`}</div>
  `, 'speaker-modal');

  // Close button
  card.querySelector('#speaker-modal-close-btn')?.addEventListener('click', closeModal);

  // Rename button
  // Inline rename UI
  const renameBtn = card.querySelector('#speaker-rename-btn');
  const nameEl = card.querySelector('#speaker-modal-current-name');
  renameBtn?.addEventListener('click', () => {
    const currentName = nameEl?.textContent?.trim() || speakerName;
    // Replace name + button with an input field and save/cancel
    const header = nameEl.parentElement;
    const inputHtml = `
      <input type="text" class="speaker-rename-input" id="speaker-rename-input" value="${esc(currentName)}" />
      <button class="btn-ghost" id="speaker-rename-save" style="color:var(--success);font-weight:600">Save</button>
      <button class="btn-ghost" id="speaker-rename-cancel" style="color:var(--text-muted)">Cancel</button>
    `;
    nameEl.style.display = 'none';
    renameBtn.style.display = 'none';
    const container = document.createElement('div');
    container.className = 'speaker-rename-row';
    container.innerHTML = inputHtml;
    nameEl.after(container);
    const input = container.querySelector('#speaker-rename-input');
    input.focus();
    input.select();

    const doRename = () => {
      const newName = input.value.trim();
      container.remove();
      nameEl.style.display = '';
      renameBtn.style.display = '';
      if (!newName || newName === currentName) return;

      // Update modal title
      nameEl.textContent = newName;

      // Collect every cluster_id this speaker maps to. Usually just
      // one, but a speaker can span multiple clusters after a diarize
      // remap (e.g. when the same person got split into two clusters).
      // Scanning the segments we already have is faster than hitting
      // the registry and works whether the events came from `store`
      // or from the server's speaker response.
      const clusterIds = new Set();
      for (const s of segments) {
        const sp = (s.speakers || [])[0];
        if (sp && sp.cluster_id != null) {
          clusterIds.add(sp.cluster_id);
        }
      }
      // Also sweep the in-memory store so we catch clusters whose
      // events are in store but weren't in the segments list passed
      // to this modal (e.g. revisions that landed after the modal
      // opened).
      for (const [, ev] of store.segments) {
        if (ev.speakers?.length > 0) {
          const s = ev.speakers[0];
          if (s.identity === currentName || s.display_name === currentName) {
            s.identity = newName;
            s.display_name = newName;
            if (s.cluster_id != null) clusterIds.add(s.cluster_id);
          }
        }
      }

      // Stamp the new name into the speaker registry for every
      // affected cluster. This is the authoritative map that
      // `getSpeakerDisplayName()` reads — WITHOUT this, rows in the
      // transcript keep their old name until a full reload. (Bug
      // 2026-04-14: only _openSpeakerRenameModal touched the registry
      // before this fix; showSpeakerModal's inline rename skipped it.)
      for (const cid of clusterIds) {
        renameSpeaker(cid, newName);
      }

      // Now propagate visually. Both helpers read from the registry,
      // so they see the new name and paint it into every rendered
      // transcript block + detected-speakers strip entry.
      _refreshTranscriptSpeakerLabels();
      _refreshDetectedSpeakersStrip();

      // Timeline lane labels are drawn once per meeting load — update
      // them in place too since they don't go through the registry.
      document.querySelectorAll('.speaker-timeline-lane-label').forEach(el => {
        if (el.textContent === currentName) el.textContent = newName;
      });
      // Persist via API — send old_name so server can update all meeting files
      if (mid && segments.length > 0) {
        // First call includes old_name to update detected_speakers.json + room.json
        const firstSeg = segments.find(s => s.segment_id);
        if (firstSeg) {
          fetch(`${API}/api/meetings/${mid}/events/${firstSeg.segment_id}/speaker`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({speaker_name: newName, old_name: currentName}),
          }).catch(() => {});
        }
        // Remaining segments just get journal corrections (no old_name needed)
        for (const seg of segments.slice(1)) {
          if (seg.segment_id) {
            fetch(`${API}/api/meetings/${mid}/events/${seg.segment_id}/speaker`, {
              method: 'PUT',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({speaker_name: newName}),
            }).catch(() => {});
          }
        }
      }
    };

    container.querySelector('#speaker-rename-save').addEventListener('click', doRename);
    container.querySelector('#speaker-rename-cancel').addEventListener('click', () => {
      container.remove();
      nameEl.style.display = '';
      renameBtn.style.display = '';
    });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') doRename();
      if (e.key === 'Escape') { container.remove(); nameEl.style.display = ''; renameBtn.style.display = ''; }
    });
  });

  // Wire play buttons
  if (mid) {
    card.querySelectorAll('.speaker-seg-play').forEach(btn => {
      btn.addEventListener('click', () => {
        const seg = btn.closest('.speaker-segment');
        const startMs = parseInt(seg.dataset.start);
        const endMs = parseInt(seg.dataset.end);
        const segmentId = seg.dataset.segmentId || undefined;
        audioPlayer.playSegment(mid, startMs, endMs, segmentId);
        // Highlight this segment
        card.querySelectorAll('.speaker-segment.playing').forEach(s => s.classList.remove('playing'));
        seg.classList.add('playing');
      });
    });

    // Play all sequentially
    card.querySelector('#speaker-play-all')?.addEventListener('click', async () => {
      for (const seg of card.querySelectorAll('.speaker-segment')) {
        const startMs = parseInt(seg.dataset.start);
        const endMs = parseInt(seg.dataset.end);
        card.querySelectorAll('.speaker-segment.playing').forEach(s => s.classList.remove('playing'));
        seg.classList.add('playing');
        seg.scrollIntoView({ behavior: 'smooth', block: 'center' });
        audioPlayer.playSegment(mid, startMs, endMs);
        // Wait for playback to finish
        await new Promise(r => {
          audioPlayer.audio.onended = r;
          setTimeout(r, (endMs - startMs) + 1000); // fallback timeout
        });
      }
    });
  }
}

/** Render Teams-style speaker timeline with colored lane blocks and zoom */
function renderSpeakerTimeline(speakerLanes, durationMs, speakers, meetingId) {
  const container = document.getElementById('speaker-timeline');
  const lanesEl = document.getElementById('speaker-timeline-lanes');
  const timesEl = document.getElementById('speaker-timeline-times');
  const cursorEl = document.getElementById('speaker-timeline-cursor');

  if (!container || !lanesEl) return;

  container.style.display = '';
  lanesEl.innerHTML = '';
  timesEl.innerHTML = '';

  // Zoom state
  let zoomLevel = 1;
  const LABEL_WIDTH = 70;

  // Sort speakers by total speaking time (most talkative first)
  const sortedSpeakers = Object.entries(speakerLanes)
    .map(([id, blocks]) => {
      const serverName = speakers.find(s => String(s.cluster_id) === id)?.display_name;
      // Use the unified registry: respects explicit names, ignores generic
      // "Speaker N" labels from the server, falls back to sequential numbering.
      const name = getSpeakerDisplayName(Number(id), serverName);
      return {
        id,
        blocks,
        totalMs: blocks.reduce((s, b) => s + (b.end_ms - b.start_ms), 0),
        name,
      };
    })
    .sort((a, b) => b.totalMs - a.totalMs);

  // Render each speaker lane
  sortedSpeakers.forEach((speaker) => {
    // Color keyed by cluster_id — matches transcript blocks exactly
    const color = getSpeakerColor(speaker.id);
    const lane = document.createElement('div');
    lane.className = 'speaker-timeline-lane';
    lane.dataset.clusterId = String(speaker.id);

    const label = document.createElement('div');
    label.className = 'speaker-timeline-lane-label';
    label.textContent = speaker.name;
    label.style.color = color;
    label.style.cursor = 'pointer';
    label.title = 'Click to view speaker details';
    label.addEventListener('click', () => {
      showSpeakerModal(speaker.name, color, findSpeakerSegments(speaker.name), meetingId);
    });
    lane.appendChild(label);

    const track = document.createElement('div');
    track.className = 'speaker-timeline-lane-track';

    // Render blocks
    for (const block of speaker.blocks) {
      const left = (block.start_ms / durationMs) * 100;
      const width = Math.max(0.15, ((block.end_ms - block.start_ms) / durationMs) * 100);
      const el = document.createElement('div');
      el.className = 'speaker-timeline-block';
      el.style.left = `${left}%`;
      el.style.width = `${width}%`;
      el.style.background = color;
      el.dataset.segmentId = block.segment_id || '';
      el.dataset.startMs = String(block.start_ms);
      el.dataset.endMs = String(block.end_ms);
      el.title = `${speaker.name}: ${formatTime(block.start_ms)} - ${formatTime(block.end_ms)} — click to view transcript`;
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        // Seek audio + scroll transcript + highlight — all synced
        if (audioPlayer._fullMeetingSrc) {
          audioPlayer.seekFullMeeting(block.start_ms, block.segment_id);
        } else {
          audioPlayer.playSegment(meetingId, block.start_ms, block.end_ms, block.segment_id);
        }
        // Ensure the corresponding transcript block scrolls into view even if
        // playback was already at this position (seekFullMeeting no-ops in that case)
        const row = document.querySelector(
          `.compact-block[data-segment-id="${block.segment_id}"]`
        ) || [...document.querySelectorAll('[data-segment-ids]')]
          .find(e => e.dataset.segmentIds?.split(',').includes(block.segment_id));
        if (row) {
          row.scrollIntoView({ behavior: 'smooth', block: 'center' });
          audioPlayer._clearHighlight();
          row.classList.add('playing');
        }
      });
      track.appendChild(el);
    }

    // Click on empty track area to seek
    track.addEventListener('click', (e) => {
      if (e.target === track) {
        const rect = track.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / (rect.width * zoomLevel);
        const seekMs = Math.min(pct, 1) * durationMs;
        if (audioPlayer._fullMeetingSrc) {
          audioPlayer.seekFullMeeting(seekMs);
        }
      }
    });

    lane.appendChild(track);
    lanesEl.appendChild(lane);
  });

  // Apply zoom via CSS transform on tracks
  const applyZoom = () => {
    lanesEl.querySelectorAll('.speaker-timeline-lane-track').forEach(track => {
      track.style.transform = `scaleX(${zoomLevel})`;
      track.style.transformOrigin = 'left';
    });
    updateTimeMarkers();
  };

  // Time markers (update on zoom)
  const updateTimeMarkers = () => {
    timesEl.innerHTML = '';
    const numMarkers = Math.max(4, Math.min(12, Math.round(6 * zoomLevel)));
    for (let i = 0; i <= numMarkers; i++) {
      const ms = (i / numMarkers) * durationMs;
      const span = document.createElement('span');
      span.textContent = formatTime(ms);
      timesEl.appendChild(span);
    }
    // Scale time markers container to match zoom
    timesEl.style.transform = `scaleX(${zoomLevel})`;
    timesEl.style.transformOrigin = 'left';
  };
  updateTimeMarkers();

  // Zoom controls: Ctrl+scroll wheel
  container.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.2 : 0.83;
      zoomLevel = Math.max(1, Math.min(20, zoomLevel * factor));
      applyZoom();
    }
  }, { passive: false });

  // Cursor follows audio playback
  const updateCursor = () => {
    if (audioPlayer.audio.duration && audioPlayer.audio.currentTime > 0) {
      const pct = audioPlayer.audio.currentTime / audioPlayer.audio.duration;
      const trackEl = lanesEl.querySelector('.speaker-timeline-lane-track');
      const baseWidth = trackEl ? (trackEl.offsetWidth / zoomLevel) : 1;
      const cursorX = LABEL_WIDTH + pct * baseWidth * zoomLevel;
      cursorEl.style.left = `${cursorX}px`;
      cursorEl.style.display = '';

      // Auto-scroll to keep cursor visible when zoomed
      if (zoomLevel > 1 && container.scrollWidth > container.clientWidth) {
        const viewLeft = container.scrollLeft;
        const viewRight = viewLeft + container.clientWidth;
        if (cursorX < viewLeft + 50 || cursorX > viewRight - 50) {
          container.scrollLeft = cursorX - container.clientWidth / 2;
        }
      }
    } else {
      cursorEl.style.display = 'none';
    }
    requestAnimationFrame(updateCursor);
  };
  requestAnimationFrame(updateCursor);
}

class RoomSetup {
  constructor() {
    this.tables = [];
    this.seats = [];
    this.preset = 'rectangle';
    this.selected = null; // { type: 'seat'|'table', id: string }
    this.mode = 'setup'; // 'setup' | 'live' | 'review'
    this.meetingId = null; // set when mounting for live/review
    this.defaultCanvas = document.getElementById('room-canvas');
    this.canvas = this.defaultCanvas;
    this.btnStart = document.getElementById('btn-start-meeting');
    this.hintEl = document.getElementById('setup-hint');

    document.getElementById('btn-add-seat').addEventListener('click', () => this.addSeatAtCenter());
    // Add Table now also places the requested number of seats around the
    // new table in a single click. Falls back to 0 extra seats if the input
    // is missing/invalid.
    document.getElementById('btn-add-table')?.addEventListener('click', () => {
      const input = document.getElementById('seat-count-input');
      let n = parseInt(input?.value, 10);
      if (!Number.isFinite(n) || n < 0) n = 0;
      if (n > 20) n = 20;
      this.addTableWithSeats(n);
    });
    document.getElementById('btn-clear-layout')?.addEventListener('click', () => this.clearLayout());
    this.btnStart.addEventListener('click', async () => {
      // Debounce: disable immediately so double-clicks can't fire two
      // concurrent startMeeting() calls. Button state is restored by
      // startMeeting's own try/finally regardless of success / failure
      // (see RoomSetup.startMeeting — this is the bulletproof path for
      // both sync and async errors from startRecording).
      if (this.btnStart.disabled) return;
      this.btnStart.disabled = true;
      this.btnStart.dataset.origText = this.btnStart.textContent;
      this.btnStart.textContent = 'Starting…';
      try {
        await this.startMeeting();
      } catch (e) {
        // startMeeting's finally already restored btnStart; just
        // surface the error for visibility (sentry-style) without
        // re-touching the DOM.
        console.error('startMeeting failed:', e);
      }
    });

    // Single-language meetings are only accessible via the "Quick start
    // English" button on the landing page — the setup screen is always
    // bilingual. If users want mono they go back to the landing page.

    // Clicking a preset button jumps the seat-count input to the layout's
    // typical size (read from data-default-seats on the button) and then
    // applies the preset. Users can still override by editing the number.
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const defaultN = parseInt(btn.dataset.defaultSeats, 10);
        if (Number.isFinite(defaultN) && defaultN > 0) {
          const input = document.getElementById('seat-count-input');
          if (input) input.value = String(defaultN);
        }
        this.applyPreset(btn.dataset.preset);
      });
    });

    // Live re-apply: changing the seat count re-runs the currently active
    // preset so users see the layout update as they adjust the number.
    const seatInput = document.getElementById('seat-count-input');
    seatInput?.addEventListener('input', () => {
      if (!this.preset) return;
      // Debounce so typing "12" doesn't re-render once for "1" then again
      // for "12"; users see the result after a short pause.
      clearTimeout(this._seatInputDebounce);
      this._seatInputDebounce = setTimeout(() => {
        if (this.preset) this.applyPreset(this.preset);
      }, 180);
    });

    // Click canvas to deselect — rebound in mount() if canvas changes
    this._bindCanvasClick();

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Don't handle if typing in an input
      if (e.target.tagName === 'INPUT') return;

      if ((e.key === 'Delete' || e.key === 'Backspace') && this.selected) {
        e.preventDefault();
        if (this.selected.type === 'seat') this.removeSeat(this.selected.id);
        else if (this.selected.type === 'table') this.removeTable(this.selected.id);
        this._deselect();
      }

      if (e.key === 'Escape') this._deselect();
    });

    this._loadLayout();
  }

  _bindCanvasClick() {
    // Remove previous listener if any (we reattach on mount)
    if (this._canvasClickHandler) {
      this.canvas.removeEventListener('click', this._canvasClickHandler);
    }
    this._canvasClickHandler = (e) => {
      if (e.target === this.canvas) this._deselect();
    };
    this.canvas.addEventListener('click', this._canvasClickHandler);
  }

  /**
   * Reparent the editor into a different container.
   * Mode determines behavior:
   *   'setup'  — default setup page, with toolbar + start button
   *   'live'   — mid-meeting overlay, live sync, persists to meeting
   *   'review' — past-meeting edit, cluster chips, persists to meeting
   */
  mount(containerEl, mode = 'setup', opts = {}) {
    this.mode = mode;
    this.meetingId = opts.meetingId || null;

    // Reparent: move every rendered DOM element into the new container
    if (containerEl && containerEl !== this.canvas) {
      containerEl.innerHTML = '';
      for (const t of this.tables) {
        if (t.element) containerEl.appendChild(t.element);
      }
      for (const s of this.seats) {
        if (s.element) containerEl.appendChild(s.element);
      }
      this.canvas = containerEl;
      this._bindCanvasClick();
    }

    // Refresh the empty-canvas affordance for the new container
    this._updateEmptyOverlay();
  }

  /**
   * Restore the editor back to the default setup canvas.
   */
  unmount() {
    this.mount(this.defaultCanvas, 'setup');
  }

  // ── Selection ──────────────────────────────────────

  _select(type, id) {
    this._deselect();
    this.selected = { type, id };
    const el = type === 'seat'
      ? this.seats.find(s => s.seatId === id)?.element
      : this.tables.find(t => t.tableId === id)?.element;
    el?.classList.add('selected');
  }

  _deselect() {
    this.canvas.querySelectorAll('.selected').forEach(el => el.classList.remove('selected'));
    this.selected = null;
  }

  // ── Presets ────────────────────────────────────────

  /**
   * Apply a room preset. The number of seats comes from the shared
   * seat-count input (the one alongside the Add Table button) so both
   * entry points — the canvas empty-state overlay and the footer preset
   * row — produce the same result. Pass an explicit `seatCount` to
   * override (used by legacy callers / tests).
   */
  applyPreset(preset, seatCount) {
    this.preset = preset;
    document.querySelectorAll('.preset-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.preset === preset)
    );

    // Clear current scene
    this.tables.forEach(t => t.element?.remove());
    this.seats.forEach(s => s.element?.remove());
    this.tables = [];
    this.seats = [];

    // Read seat count from the shared input if not provided explicitly
    let n = seatCount;
    if (n == null) {
      const input = document.getElementById('seat-count-input');
      const parsed = parseInt(input?.value, 10);
      n = Number.isFinite(parsed) && parsed > 0 ? parsed : 6;
    }
    if (n < 1) n = 1;
    if (n > 20) n = 20;

    const P = this._presetData(preset, n);
    P.tables.forEach(t => { this.tables.push(t); this._renderTable(t); this._applyTable(t); });
    P.seats.forEach(s => { this.seats.push(s); this._renderSeat(s); this._applySeat(s); });
    this._updateHint();
    this._persistLayout();
  }

  _id() { return crypto.randomUUID(); }

  _presetData(preset, n = 6) {
    const T = (x, y, w, h, br, label) => ({
      tableId: this._id(), x, y, width: w, height: h, borderRadius: br, label: label || '', element: null
    });
    const S = (x, y) => ({
      seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null
    });

    switch (preset) {
      case 'boardroom':
        return { tables: [T(50, 50, 50, 28, 50, '')], seats: this._ellipseSeats(50, 50, 35, 22, n) };
      case 'round':
        return { tables: [T(50, 50, 30, 30, 50, '')], seats: this._ellipseSeats(50, 50, 26, 26, n) };
      case 'square':
        return { tables: [T(50, 50, 28, 28, 3, '')], seats: this._rectSeats(50, 50, 14, 14, n) };
      case 'rectangle':
        return { tables: [T(50, 50, 44, 22, 3, '')], seats: this._rectSeats(50, 50, 22, 11, n) };
      case 'classroom': {
        // Rows of up to 4 desks facing a front table. Each row is centred
        // on its own actual seat count so a partial last row doesn't end
        // up flush-left, and a single seat lands at the horizontal centre
        // instead of x=20.
        const perRow = Math.min(4, Math.max(1, Math.ceil(n / 2)));
        const numRows = Math.ceil(n / perRow);
        const rowWidth = 60; // usable horizontal span (20..80)
        const seats = [];
        for (let i = 0; i < n; i++) {
          const row = Math.floor(i / perRow);
          const col = i % perRow;
          const isLastRow = row === numRows - 1;
          const rowSeats = isLastRow && n % perRow !== 0 ? n % perRow : perRow;
          const step = rowWidth / rowSeats;
          const x = 20 + step * (col + 0.5);
          const y = 35 + row * 18;
          seats.push(S(x, Math.min(90, y)));
        }
        return { tables: [T(50, 15, 60, 5, 2, 'FRONT')], seats };
      }
      case 'u_shape': {
        // Three linear tables forming a U that opens upwards. Participants
        // sit INSIDE the U facing the centre — seats previously landed on
        // the outside (x=12, x=88, y=85), which placed them off-canvas
        // relative to the three tables. The inside-facing coordinates are
        // x=30 (right of left table), y=68 (above bottom table), and x=70
        // (left of right table).
        let nLeft = Math.round(n * 0.4);
        let nRight = Math.round(n * 0.4);
        let nBottom = n - nLeft - nRight;
        if (nBottom < 0) { nBottom = 0; nLeft = Math.floor(n / 2); nRight = n - nLeft; }
        const seats = [];
        // Left wall, walking top→bottom so seat order feels natural
        for (let i = 0; i < nLeft; i++) {
          const y = 30 + (i + 0.5) * (40 / Math.max(nLeft, 1));
          seats.push(S(30, y));
        }
        // Bottom of the U (inside, above the horizontal table)
        for (let i = 0; i < nBottom; i++) {
          const x = 30 + (i + 0.5) * (40 / Math.max(nBottom, 1));
          seats.push(S(x, 68));
        }
        // Right wall, walking bottom→top so the outer ring wraps cleanly
        for (let i = 0; i < nRight; i++) {
          const y = 70 - (i + 0.5) * (40 / Math.max(nRight, 1));
          seats.push(S(70, y));
        }
        return {
          tables: [T(22, 50, 8, 45, 2, ''), T(50, 76, 48, 6, 2, ''), T(78, 50, 8, 45, 2, '')],
          seats,
        };
      }
      case 'pods': {
        // Four pods in a 2x2 grid. The previous version stepped through
        // (top, right, bottom) only (indexInPod * π/2 - π/2), so each pod
        // topped out at 3 seats in an L-shape. Now each pod owns its
        // share of seats (round-robined so n=5 becomes 2/1/1/1) and the
        // seats are distributed evenly around the pod's full circle.
        const podCenters = [[30, 35], [70, 35], [30, 70], [70, 70]];
        const seats = [];
        for (let pod = 0; pod < 4; pod++) {
          const seatsInPod = Math.floor(n / 4) + (pod < n % 4 ? 1 : 0);
          if (seatsInPod === 0) continue;
          const [cx, cy] = podCenters[pod];
          for (let j = 0; j < seatsInPod; j++) {
            const angle = (2 * Math.PI * j) / seatsInPod - Math.PI / 2;
            seats.push(S(
              Math.max(5, Math.min(95, cx + 14 * Math.cos(angle))),
              Math.max(5, Math.min(95, cy + 12 * Math.sin(angle))),
            ));
          }
        }
        return {
          tables: [T(30, 35, 20, 16, 3, 'A'), T(70, 35, 20, 16, 3, 'B'), T(30, 70, 20, 16, 3, 'C'), T(70, 70, 20, 16, 3, 'D')],
          seats,
        };
      }
      case 'freeform':
      default: {
        // Free-form: sprinkle N seats in a loose grid, no table. Using
        // (i + 0.5) slot-centring so a single seat lands at (50, 50)
        // instead of the top-left corner.
        const cols = Math.ceil(Math.sqrt(n));
        const rows = Math.ceil(n / cols);
        const seats = [];
        for (let i = 0; i < n; i++) {
          const col = i % cols;
          const row = Math.floor(i / cols);
          const x = 20 + ((col + 0.5) / cols) * 60;
          const y = 25 + ((row + 0.5) / rows) * 50;
          seats.push(S(x, y));
        }
        return { tables: [], seats };
      }
    }
  }

  _ellipseSeats(cx, cy, rx, ry, n) {
    const off = n === 2 ? Math.PI / 2 : -Math.PI / 2;
    return Array.from({ length: n }, (_, i) => {
      const a = off + (2 * Math.PI * i) / n;
      return { seatId: this._id(), x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a), name: '', enrolled: false, enrollmentId: null, element: null };
    });
  }

  _rectSeats(cx, cy, hw, hh, n) {
    if (n <= 1) return [{ seatId: this._id(), x: cx, y: cy - hh - 6, name: '', enrolled: false, enrollmentId: null, element: null }];
    // Walk the perimeter with the slot-centre offset `(i + 0.5) / n`, so
    // n=4 on a square yields one seat at the midpoint of each edge
    // (top, right, bottom, left) instead of four corner-biased points.
    const perim = 2 * (2 * hw + 2 * hh);
    return Array.from({ length: n }, (_, i) => {
      let d = ((i + 0.5) / n) * perim, x, y;
      if (d < 2 * hw) {
        // Top edge
        x = cx - hw + d;
        y = cy - hh - 6;
      } else if (d < 2 * hw + 2 * hh) {
        // Right edge
        d -= 2 * hw;
        x = cx + hw + 6;
        y = cy - hh + d;
      } else if (d < 4 * hw + 2 * hh) {
        // Bottom edge
        d -= 2 * hw + 2 * hh;
        x = cx + hw - d;
        y = cy + hh + 6;
      } else {
        // Left edge
        d -= 4 * hw + 2 * hh;
        x = cx - hw - 6;
        y = cy + hh - d;
      }
      return { seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null };
    });
  }

  // ── Table management ───────────────────────────────

  addTable() {
    const t = { tableId: this._id(), x: 50, y: 50, width: 22, height: 14, borderRadius: 8, label: '', element: null };
    this.tables.push(t);
    this._renderTable(t);
    this._applyTable(t);
    this._persistLayout();
    this._updateHint();
    return t;
  }

  /**
   * Wipe all tables and seats. Shows the empty-state overlay again so
   * users can pick a fresh layout. Also clears the active preset so the
   * seat-count input stops live-re-rendering against a stale preset.
   */
  clearLayout() {
    this.tables.forEach(t => t.element?.remove());
    this.seats.forEach(s => s.element?.remove());
    this.tables = [];
    this.seats = [];
    this.preset = '';
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    this._persistLayout();
    this._updateHint();
  }

  /**
   * Add a new table and arrange `seatCount` seats evenly around it in a
   * single action, so users can say "I want a round table with 6 seats"
   * without clicking Add Seat repeatedly.
   *
   * Reuses _nextSeatPosition() so the placement algorithm matches the
   * existing auto-layout used when reconciling detected speakers.
   */
  addTableWithSeats(seatCount) {
    const table = this.addTable();
    if (!table || !seatCount || seatCount <= 0) return;
    // _nextSeatPosition computes positions against this.tables[0]. The new
    // table was just pushed so it's only this.tables[0] when it's the first
    // table; otherwise place seats around it explicitly using the same
    // ellipse layout as _nextSeatPosition.
    const isFirst = this.tables[0] === table;
    for (let i = 0; i < seatCount; i++) {
      let x, y;
      if (isFirst) {
        ({ x, y } = this._nextSeatPosition(i, seatCount));
      } else {
        const cx = table.x + table.width / 2;
        const cy = table.y + table.height / 2;
        const rx = table.width / 2 + 8;
        const ry = table.height / 2 + 10;
        const angle = (2 * Math.PI * i) / seatCount - Math.PI / 2;
        x = Math.max(5, Math.min(95, cx + rx * Math.cos(angle)));
        y = Math.max(5, Math.min(95, cy + ry * Math.sin(angle)));
      }
      const seat = {
        seatId: this._id(),
        x, y,
        name: '',
        enrolled: false,
        enrollmentId: null,
        element: null,
      };
      this.seats.push(seat);
      this._renderSeat(seat);
      this._applySeat(seat);
    }
    this._persistLayout();
    this._updateHint();
  }

  /**
   * Ensure the canvas has one seat per detected speaker and that the seat
   * names match the registry. Called when the room editor is opened in
   * "live" mode and whenever a new speaker is detected while the editor
   * is open. Seats are added around the first table (or in a grid if
   * no table exists). Existing seats are kept and bound to speakers in
   * first-seen order.
   */
  reconcileSeatsToDetectedSpeakers() {
    if (typeof getAllSpeakers !== 'function') return;
    const speakers = getAllSpeakers();
    if (speakers.length === 0) return;

    // 1. If we have fewer seats than speakers, ADD seats around the first
    //    table (or in a grid if there's no table).
    while (this.seats.length < speakers.length) {
      const idx = this.seats.length;
      const { x, y } = this._nextSeatPosition(idx, speakers.length);
      const seat = {
        seatId: this._id(),
        x, y,
        name: '',
        enrolled: false,
        enrollmentId: null,
        element: null,
      };
      this.seats.push(seat);
      this._renderSeat(seat);
      this._applySeat(seat);
    }

    // 2. Bind each seat to the speaker at the same index (first-seen order)
    //    so the seat label reflects the live speaker name. We keep existing
    //    names if the seat was already bound to that cluster.
    speakers.forEach((sp, i) => {
      const seat = this.seats[i];
      if (!seat) return;
      if (seat.name !== sp.displayName) {
        seat.name = sp.displayName;
        const nameEl = seat.element?.querySelector('.seat-name');
        if (nameEl) nameEl.textContent = sp.displayName;
      }
      // Stamp the cluster_id on the DOM so downstream code can map seat→speaker
      if (seat.element) {
        seat.element.dataset.clusterId = String(sp.clusterId);
      }
    });

    this._persistLayout();
    this._updateHint();
  }

  /** Compute the next seat position when auto-adding seats around a table. */
  _nextSeatPosition(index, totalWanted) {
    // Prefer arranging around the first table
    const table = this.tables[0];
    if (table) {
      const cx = table.x + table.width / 2;
      const cy = table.y + table.height / 2;
      const rx = table.width / 2 + 8;
      const ry = table.height / 2 + 10;
      // Spread evenly around an ellipse
      const n = Math.max(totalWanted, 1);
      const angle = (2 * Math.PI * index) / n - Math.PI / 2;
      const x = Math.max(5, Math.min(95, cx + rx * Math.cos(angle)));
      const y = Math.max(5, Math.min(95, cy + ry * Math.sin(angle)));
      return { x, y };
    }
    // No table — grid layout
    const cols = Math.ceil(Math.sqrt(totalWanted));
    const rows = Math.ceil(totalWanted / cols);
    const col = index % cols;
    const row = Math.floor(index / cols);
    const x = 15 + (70 / Math.max(cols - 1, 1)) * col;
    const y = 15 + (70 / Math.max(rows - 1, 1)) * row;
    return { x, y };
  }

  removeTable(tableId) {
    const idx = this.tables.findIndex(t => t.tableId === tableId);
    if (idx === -1) return;
    this.tables[idx].element?.remove();
    this.tables.splice(idx, 1);
    this._persistLayout();
    this._updateHint();
  }

  _renderTable(table) {
    const el = document.createElement('div');
    el.className = 'table-obj';
    el.innerHTML = `
      <button class="table-remove" title="Remove table">&times;</button>
      <div class="table-surface"><div class="table-grain"></div><div class="table-label">${table.label}</div></div>
      <div class="table-resize"></div>
    `;
    this.canvas.appendChild(el);
    table.element = el;

    el.querySelector('.table-remove').addEventListener('click', async e => {
      e.stopPropagation();
      const ok = await confirmDialog('Remove Table?', 'Remove this table from the room?', 'Remove');
      if (!ok) return;
      this.removeTable(table.tableId);
    });
    this._makeTableDraggable(table);
    this._makeTableResizable(table);
  }

  _applyTable(t) {
    if (!t.element) return;
    t.element.style.left = `${t.x}%`;
    t.element.style.top = `${t.y}%`;
    t.element.style.width = `${t.width}%`;
    t.element.style.height = `${t.height}%`;
    t.element.querySelector('.table-surface').style.borderRadius = `${t.borderRadius}%`;
  }

  _makeTableDraggable(table) {
    const surf = table.element.querySelector('.table-surface');
    let active = false, sx, sy, spx, spy;
    surf.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      this._select('table', table.tableId);
      active = true; surf.setPointerCapture(e.pointerId); table.element.classList.add('dragging');
      sx = e.clientX; sy = e.clientY; spx = table.x; spy = table.y;
    });
    surf.addEventListener('pointermove', e => {
      if (!active) return;
      const r = this.canvas.getBoundingClientRect();
      table.x = Math.max(5, Math.min(95, spx + (e.clientX - sx) / r.width * 100));
      table.y = Math.max(5, Math.min(95, spy + (e.clientY - sy) / r.height * 100));
      this._applyTable(table);
    });
    surf.addEventListener('pointerup', () => { if (!active) return; active = false; table.element.classList.remove('dragging'); this._persistLayout(); });
  }

  _makeTableResizable(table) {
    const h = table.element.querySelector('.table-resize');
    let active = false, sx, sy, sw, sh;
    h.addEventListener('pointerdown', e => {
      e.stopPropagation(); active = true; h.setPointerCapture(e.pointerId);
      sx = e.clientX; sy = e.clientY; sw = table.width; sh = table.height;
    });
    h.addEventListener('pointermove', e => {
      if (!active) return;
      const r = this.canvas.getBoundingClientRect();
      table.width = Math.max(6, Math.min(90, sw + (e.clientX - sx) / r.width * 200));
      table.height = Math.max(4, Math.min(90, sh + (e.clientY - sy) / r.height * 200));
      this._applyTable(table);
    });
    h.addEventListener('pointerup', () => { if (!active) return; active = false; this._persistLayout(); });
  }

  // ── Seat management ────────────────────────────────

  addSeatAtCenter() {
    // Generate candidate positions around ALL tables
    const candidates = [];
    for (const t of this.tables) {
      const isRound = t.borderRadius >= 30;
      if (isRound) {
        // Elliptical: generate 12 positions around the ellipse
        const rx = t.width / 2 + 7, ry = t.height / 2 + 7;
        for (let i = 0; i < 12; i++) {
          const a = (2 * Math.PI * i) / 12 - Math.PI / 2;
          candidates.push({ x: t.x + rx * Math.cos(a), y: t.y + ry * Math.sin(a) });
        }
      } else {
        // Rectangular: generate positions along edges with offset
        const hw = t.width / 2 + 7, hh = t.height / 2 + 7;
        // Long sides (top & bottom) get more candidates
        const longSide = t.width >= t.height;
        const nLong = 6, nShort = 3;
        // Top edge
        for (let i = 0; i < (longSide ? nLong : nShort); i++) {
          const frac = (i + 0.5) / (longSide ? nLong : nShort);
          candidates.push({ x: t.x - hw + frac * 2 * hw, y: t.y - hh });
        }
        // Bottom edge
        for (let i = 0; i < (longSide ? nLong : nShort); i++) {
          const frac = (i + 0.5) / (longSide ? nLong : nShort);
          candidates.push({ x: t.x - hw + frac * 2 * hw, y: t.y + hh });
        }
        // Left edge
        for (let i = 0; i < (longSide ? nShort : nLong); i++) {
          const frac = (i + 0.5) / (longSide ? nShort : nLong);
          candidates.push({ x: t.x - hw, y: t.y - hh + frac * 2 * hh });
        }
        // Right edge
        for (let i = 0; i < (longSide ? nShort : nLong); i++) {
          const frac = (i + 0.5) / (longSide ? nShort : nLong);
          candidates.push({ x: t.x + hw, y: t.y - hh + frac * 2 * hh });
        }
      }
    }

    // Fallback: if no tables, use grid
    if (candidates.length === 0) {
      for (let cx = 15; cx <= 85; cx += 10)
        for (let cy = 15; cy <= 85; cy += 10)
          candidates.push({ x: cx, y: cy });
    }

    // Clamp to canvas bounds
    const clamped = candidates.map(c => ({
      x: Math.max(5, Math.min(95, c.x)),
      y: Math.max(5, Math.min(95, c.y)),
    }));

    // Pick the candidate farthest from all existing seats
    let x = 50, y = 50;
    if (this.seats.length > 0) {
      let best = -1;
      for (const c of clamped) {
        const d = Math.min(...this.seats.map(s => Math.hypot(s.x - c.x, s.y - c.y)));
        if (d > best) { best = d; x = c.x; y = c.y; }
      }
    } else {
      x = clamped[0].x; y = clamped[0].y;
    }

    const s = { seatId: this._id(), x, y, name: '', enrolled: false, enrollmentId: null, element: null };
    this.seats.push(s);
    this._renderSeat(s);
    this._applySeat(s);
    this._updateHint();
    this._persistLayout();
  }

  removeSeat(seatId) {
    const idx = this.seats.findIndex(s => s.seatId === seatId);
    if (idx === -1) return;
    this.seats[idx].element?.remove();
    this.seats.splice(idx, 1);
    this._updateHint();
    this._persistLayout();
  }

  _renderSeat(seat) {
    const ci = this.seats.indexOf(seat);
    // design-time: no cluster_id yet, color by seat index
    const color = SPEAKER_COLORS[ci % SPEAKER_COLORS.length];
    const circ = 2 * Math.PI * 34;
    const n = document.createElement('div');
    n.className = 'seat-node';
    n.dataset.seatId = seat.seatId;
    n.style.setProperty('--seat-color', color);
    n.innerHTML = `
      <button class="seat-remove" title="Remove">&times;</button>
      <div class="seat-avatar" title="Click chair to enroll voice">
        <span class="seat-index">${ci + 1}</span>
        <svg class="seat-progress" viewBox="0 0 72 72">
          <circle class="progress-bg" cx="36" cy="36" r="34"/>
          <circle class="progress-fill" cx="36" cy="36" r="34" stroke-dasharray="${circ}" stroke-dashoffset="${circ}"/>
        </svg>
      </div>
      <span class="seat-name-label" title="Click name to edit">${seat.name || 'Click chair to enroll'}</span>
      <span class="seat-status-label">${seat.enrolled ? 'Enrolled' : ''}</span>
    `;
    this.canvas.appendChild(n);
    seat.element = n;
    if (seat.enrolled) n.classList.add('enrolled');
    // Name label is always clickable (both pre- and post-enrollment) — gives
    // a fast "just type a name" path without interrupting voice capture.
    const nameLabel = n.querySelector('.seat-name-label');
    nameLabel.style.cursor = 'pointer';
    nameLabel.addEventListener('click', (e) => {
      e.stopPropagation();
      this._editName(seat);
    });
    n.querySelector('.seat-remove').addEventListener('click', async e => {
      e.stopPropagation();
      if (seat.enrolled) {
        const ok = await confirmDialog('Remove Seat?', `Remove ${seat.name || 'this seat'} from the room?`, 'Remove');
        if (!ok) return;
      }
      this.removeSeat(seat.seatId);
    });
    this._makeSeatDraggable(seat);
  }

  _applySeat(s) { if (s.element) { s.element.style.left = `${s.x}%`; s.element.style.top = `${s.y}%`; } }

  _makeSeatDraggable(seat) {
    const av = seat.element.querySelector('.seat-avatar');
    let on = false, sx, sy, spx, spy, moved;
    av.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      this._select('seat', seat.seatId);
      on = true; moved = 0; av.setPointerCapture(e.pointerId);
      sx = e.clientX; sy = e.clientY; spx = seat.x; spy = seat.y;
      seat.element.style.transition = 'none'; seat.element.style.zIndex = '10';
    });
    av.addEventListener('pointermove', e => {
      if (!on) return;
      const r = this.canvas.getBoundingClientRect();
      const dx = e.clientX - sx, dy = e.clientY - sy;
      moved += Math.abs(dx) + Math.abs(dy);
      seat.x = Math.max(3, Math.min(97, spx + dx / r.width * 100));
      seat.y = Math.max(3, Math.min(97, spy + dy / r.height * 100));
      seat.element.style.left = `${seat.x}%`; seat.element.style.top = `${seat.y}%`;
    });
    av.addEventListener('pointerup', () => {
      if (!on) return; on = false;
      seat.element.style.transition = ''; seat.element.style.zIndex = '';
      if (moved < 10) this._seatAction(seat); else this._persistLayout();
    });
  }

  // ── Seat actions ────────────────────────────────────

  async _seatAction(seat) {
    // Chair tap → always start voice enrollment. The name label handles
    // name-only edits on its own click path. Tapping an already-enrolled
    // chair re-enrolls (overwrites) so a user who mis-captured can retry
    // without finding a separate "re-enroll" button.
    this._enrollSeat(seat);
  }

  async _enrollSeat(seat) {
    // Dynamic-length enrollment: record continuously, probe the server for
    // a self-stated name every ~1.2s, and stop as soon as the name is
    // detected (or the user clicks the seat again, or we hit the safety
    // ceiling). Replaces the old fixed 2s capture.
    const nameLabel = seat.element.querySelector('.seat-name-label');
    const statusLabel = seat.element.querySelector('.seat-status-label');
    const pf = seat.element.querySelector('.progress-fill');
    const circ = 2 * Math.PI * 34;
    // Probe early and often. Detection latency is dominated by ASR round-trip
    // (~300-500 ms on GB10), so a 500 ms probe interval naturally throttles
    // to back-to-back probes without overlap. First probe at 1000 ms gives
    // enough audio for a self-introduction in any language (Japanese names
    // like "私は田中です" need ~1s minimum for ASR to produce usable text;
    // the server's detect-name endpoint requires ≥0.8s / 25600 bytes).
    // MAX_MS is a safety ceiling — a silent mic can't hang the UI.
    const MIN_MS = 1000;
    const MAX_MS = 15000;
    const PROBE_INTERVAL_MS = 500;

    seat.element.classList.remove('enrolled');
    seat.element.classList.add('enrolling');
    nameLabel.textContent = 'Say your name...';
    statusLabel.textContent = 'listening';
    pf.style.strokeDashoffset = circ;

    // Manual-stop: clicking the seat while enrolling finalizes immediately
    // with whatever's been captured so far. Bind once, and clear on exit.
    //
    // The seat is started from a pointerup handler on .seat-avatar. The
    // browser then dispatches a *synthetic click* on the same element right
    // after pointerup. Without a guard, that synthetic click hits the
    // capture-phase stopHandler we register here and sets manualStop=true on
    // the very first 100 ms loop tick, so enrollment ends instantly. Ignore
    // any click that lands within the first 400 ms — that is comfortably
    // longer than the synthetic-click delay but well below the time it would
    // take a user to deliberately tap the seat a second time to abort.
    let manualStop = false;
    const stopArmedAt = performance.now() + 400;
    const stopHandler = (e) => {
      if (performance.now() < stopArmedAt) return;
      e.stopPropagation();
      manualStop = true;
    };
    seat.element.addEventListener('click', stopHandler, { capture: true });

    // Reuse the warmed-up mic when available so the user gets instant capture
    // instead of waiting ~500-1500 ms for getUserMedia + AudioContext boot.
    // Falls back to a one-shot acquisition if priming hasn't completed yet
    // (e.g. very fast click before the warm-up promise resolved).
    let proc = null;
    let usedWarmup = false;
    let localAc = null;
    let localStream = null;
    let localSource = null;
    if (micWarmup.primed) {
      usedWarmup = true;
      if (micWarmup.ac.state === 'suspended') {
        try { await micWarmup.ac.resume(); } catch {}
      }
    }
    const acRef = () => usedWarmup ? micWarmup.ac : localAc;
    const sourceRef = () => usedWarmup ? micWarmup.source : localSource;

    const chunks = [];
    // Build an Int16 PCM buffer from the accumulated Float32 chunks at 16kHz
    const buildPcm16 = () => {
      const totLen = chunks.reduce((s, c) => s + c.length, 0);
      if (!totLen) return new Int16Array(0);
      const full = new Float32Array(totLen);
      let off = 0;
      for (const c of chunks) { full.set(c, off); off += c.length; }
      const ratio = acRef().sampleRate / 16000;
      const outLen = Math.floor(totLen / ratio);
      const a16k = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) a16k[i] = full[Math.min(Math.floor(i * ratio), totLen - 1)];
      const s16 = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const v = Math.max(-1, Math.min(1, a16k[i]));
        s16[i] = v < 0 ? v * 32768 : v * 32767;
      }
      return s16;
    };

    try {
      if (!usedWarmup) {
        localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        localAc = new AudioContext();
        if (localAc.state === 'suspended') await localAc.resume();
        localSource = localAc.createMediaStreamSource(localStream);
      }
      proc = acRef().createScriptProcessor(4096, 1, 1);
      const startedAt = performance.now();
      proc.onaudioprocess = e => {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        const elMs = performance.now() - startedAt;
        // Ring fill: grows toward MAX_MS, resetting visually when name lands
        pf.style.strokeDashoffset = circ * (1 - Math.min(1, elMs / MAX_MS));
      };
      sourceRef().connect(proc);
      proc.connect(acRef().destination);

      // Probe loop — runs concurrently with capture. As soon as the probe
      // returns a name, we paint the chair with that name *immediately* and
      // wake the loop via a Promise so the user sees instant feedback rather
      // than waiting for the next 100 ms tick.
      let detectedName = null;
      let lastProbeAt = 0;
      let probeInFlight = false;
      let wakeLoop;
      const wakePromise = () => new Promise(r => { wakeLoop = r; });
      let waiter = wakePromise();
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));

      const onNameDetected = (name) => {
        if (detectedName) return;
        detectedName = name;
        // Instant feedback — paint the chair before we even stop the mic so
        // the user knows we heard them. Final classes/state get applied below
        // after the embedding extraction returns.
        nameLabel.textContent = name;
        statusLabel.textContent = 'captured';
        pf.style.strokeDashoffset = 0;
        if (wakeLoop) wakeLoop();
      };

      while (!manualStop && !detectedName) {
        const elMs = performance.now() - startedAt;
        if (elMs >= MAX_MS) break;

        if (elMs >= MIN_MS && !probeInFlight && (elMs - lastProbeAt) >= PROBE_INTERVAL_MS) {
          probeInFlight = true;
          lastProbeAt = elMs;
          const s16 = buildPcm16();
          statusLabel.textContent = `${(elMs / 1000).toFixed(1)}s · listening`;
          (async () => {
            try {
              const resp = await fetch(`${API}/api/room/enroll/detect-name`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: s16.buffer,
              });
              if (resp.ok) {
                const j = await resp.json();
                if (j.name) onNameDetected(j.name);
              }
            } catch {}
            probeInFlight = false;
            if (wakeLoop) wakeLoop();
          })();
        } else if (!detectedName) {
          statusLabel.textContent = `${(elMs / 1000).toFixed(1)}s · listening`;
        }
        // Wake on probe completion or after 100 ms, whichever comes first.
        await Promise.race([waiter, sleep(100)]);
        waiter = wakePromise();
      }

      // Stop capture. Detach the ScriptProcessor but leave the warmed-up
      // mic stream + AudioContext alive so the next chair tap is also
      // instant. Local (non-warm-up) resources are torn down in `finally`.
      try { proc.disconnect(); } catch {}
      try { sourceRef()?.disconnect(proc); } catch {}

      const s16final = buildPcm16();
      if (s16final.length < 16000) {
        // <1s captured — abort rather than poison the embedding store
        throw new Error('Too short');
      }

      // If we already painted a detected name, leave it visible during the
      // embedding extraction instead of flashing "Processing..." over it.
      if (!detectedName) {
        nameLabel.textContent = 'Processing...';
        statusLabel.textContent = '';
      } else {
        statusLabel.textContent = 'saving voice…';
      }
      // If we already have a name from the probe, pin it so the final
      // enrollment call doesn't re-run ASR unnecessarily.
      const q = detectedName ? `?name=${encodeURIComponent(detectedName)}` : '';
      const resp = await fetch(`${API}/api/room/enroll${q}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: s16final.buffer,
      });
      if (!resp.ok) throw new Error((await resp.json()).error || 'Failed');
      const result = await resp.json();
      seat.enrolled = true;
      seat.enrollmentId = result.enrollment_id;
      seat.name = result.name;
      seat.element.classList.remove('enrolling');
      seat.element.classList.add('enrolled');
      nameLabel.textContent = result.name;
      nameLabel.title = 'Click to edit name';
      nameLabel.style.cursor = 'pointer';
      // The persistent click-to-edit listener was attached in _renderSeat;
      // no need to re-add here.
      statusLabel.textContent = 'Enrolled';
      pf.style.strokeDashoffset = 0;
    } catch (err) {
      nameLabel.textContent = 'Tap to retry';
      statusLabel.textContent = err.message || 'Error';
      seat.element.classList.remove('enrolling');
      pf.style.strokeDashoffset = circ;
      // Ensure we released any lingering media resources on error paths
      try { proc?.disconnect(); } catch {}
      try { sourceRef()?.disconnect(); } catch {}
    } finally {
      seat.element.removeEventListener('click', stopHandler, { capture: true });
      // Tear down the local fallback resources only — leave the warm-up
      // singleton alive so the next chair tap is also instant.
      if (!usedWarmup) {
        try { localSource?.disconnect(); } catch {}
        if (localStream) localStream.getTracks().forEach(t => t.stop());
        if (localAc && localAc.state !== 'closed') {
          try { await localAc.close(); } catch {}
        }
      }
    }
    this._updateHint();
    this._persistLayout();
  }

  _editName(seat) {
    const lbl = seat.element.querySelector('.seat-name-label');
    const cur = seat.name || '';
    const inp = document.createElement('input');
    inp.className = 'seat-name-input';
    inp.value = cur;
    inp.placeholder = 'Enter name';
    inp.maxLength = 20;
    lbl.replaceWith(inp);
    inp.focus();
    if (cur) inp.select();
    let cancelled = false;
    const fin = () => {
      const nm = cancelled ? cur : (inp.value.trim() || cur);
      seat.name = nm;
      // Promote the seat to "set up" when a name is entered on a previously
      // blank chair — matches the old modal's "Just enter a name" path.
      if (nm && !seat.enrolled) {
        seat.enrolled = true;
        seat.element.classList.add('enrolled');
        const status = seat.element.querySelector('.seat-status-label');
        if (status) status.textContent = 'Named';
      }
      const l = document.createElement('span');
      l.className = 'seat-name-label';
      l.textContent = nm || 'Click chair to enroll';
      l.title = nm ? 'Click to edit name' : 'Click chair to enroll voice';
      l.style.cursor = 'pointer';
      l.addEventListener('click', (e) => {
        e.stopPropagation();
        this._editName(seat);
      });
      inp.replaceWith(l);
      if (seat.enrollmentId && nm && nm !== cur) {
        fetch(`${API}/api/room/enroll/rename?id=${seat.enrollmentId}&name=${encodeURIComponent(nm)}`, { method: 'POST' }).catch(() => {});
      }
      this._updateHint();
      this._persistLayout();
    };
    inp.addEventListener('blur', fin);
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter') inp.blur();
      if (e.key === 'Escape') { cancelled = true; inp.blur(); }
    });
  }

  // ── Persist / Load ─────────────────────────────────

  _persistLayout() {
    clearTimeout(_persistTimer);
    _persistTimer = setTimeout(() => {
      const body = {
        preset: this.preset,
        tables: this.tables.map(t => ({
          table_id: t.tableId, x: +t.x.toFixed(1), y: +t.y.toFixed(1),
          width: +t.width.toFixed(1), height: +t.height.toFixed(1),
          border_radius: +t.borderRadius.toFixed(1), label: t.label || '',
        })),
        seats: this.seats.map(s => ({
          seat_id: s.seatId, x: +s.x.toFixed(1), y: +s.y.toFixed(1),
          enrollment_id: s.enrollmentId || null, speaker_name: s.name || '',
        })),
      };
      // In live/review mode, persist to the meeting's room.json.
      // In setup mode, persist to the session draft.
      const url = (this.mode === 'live' || this.mode === 'review') && this.meetingId
        ? `${API}/api/meetings/${this.meetingId}/room/layout`
        : `${API}/api/room/layout`;
      fetch(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).catch(() => {});
    }, 300);
  }

  async _loadLayout() {
    try {
      const resp = await fetch(`${API}/api/room/layout`);
      const data = await resp.json();
      if (data.tables?.length > 0 || data.seats?.length > 0) {
        this.preset = data.preset || 'boardroom';
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.toggle('active', b.dataset.preset === this.preset));
        (data.tables || []).forEach(t => {
          const table = { tableId: t.table_id, x: t.x, y: t.y, width: t.width, height: t.height, borderRadius: t.border_radius, label: t.label || '', element: null };
          this.tables.push(table); this._renderTable(table); this._applyTable(table);
        });
        (data.seats || []).forEach(s => {
          const seat = { seatId: s.seat_id, x: s.x, y: s.y, name: s.speaker_name || '', enrolled: !!s.enrollment_id, enrollmentId: s.enrollment_id || null, element: null };
          this.seats.push(seat); this._renderSeat(seat); this._applySeat(seat);
        });
      } else {
        // Empty canvas by default — user adds a table/preset or starts directly
        this.preset = '';
        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      }
    } catch {
      this.preset = '';
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    }
    this._updateHint();
  }

  _updateHint() {
    const en = this.seats.filter(s => s.enrolled).length, tot = this.seats.length;
    this.btnStart.disabled = false; // Always allow starting
    if (tot === 0 && this.tables.length === 0) {
      this.hintEl.textContent = 'Click Start Meeting, or pick a layout / add seats first';
    }
    else if (tot === 0) { this.hintEl.textContent = 'Add seats or start directly'; }
    else if (en === tot) { this.hintEl.textContent = 'All enrolled — ready to start'; }
    else if (en > 0) { this.hintEl.textContent = `${en}/${tot} enrolled — click seats to set up, or start now`; }
    else { this.hintEl.textContent = 'Click seats to enroll, or start meeting now'; }
    // Refresh empty-canvas affordance in sync with hint
    this._updateEmptyOverlay();
  }

  _updateEmptyOverlay() {
    if (!this.canvas) return;
    const isEmpty = this.tables.length === 0 && this.seats.length === 0;
    let overlay = this.canvas.querySelector('.canvas-empty-overlay');

    if (!isEmpty) {
      overlay?.remove();
      return;
    }
    if (overlay) return; // Already present

    overlay = document.createElement('div');
    overlay.className = 'canvas-empty-overlay';
    overlay.innerHTML = `
      <div class="empty-title">Start with a layout</div>
      <div class="empty-sub">Pick a preset, add a table, or start an empty meeting</div>
      <div class="empty-actions">
        <button class="empty-btn" data-action="rectangle" title="Conference table">
          <svg width="26" height="18" viewBox="0 0 18 12" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="1" y="1" width="16" height="10" rx="1.5"/></svg>
          <span>Conference</span>
        </button>
        <button class="empty-btn" data-action="round" title="Round table">
          <svg width="22" height="22" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="7" cy="7" r="6"/></svg>
          <span>Round</span>
        </button>
        <button class="empty-btn" data-action="boardroom" title="Oval boardroom">
          <svg width="26" height="18" viewBox="0 0 18 12" fill="none" stroke="currentColor" stroke-width="1.2"><ellipse cx="9" cy="6" rx="8" ry="5"/></svg>
          <span>Boardroom</span>
        </button>
        <button class="empty-btn" data-action="add-table" title="Add a single table">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="6" width="18" height="12" rx="3"/><path d="M12 10v4M10 12h4"/></svg>
          <span>+ Table</span>
        </button>
        <button class="empty-btn" data-action="add-seat" title="Add a single seat">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="8" r="4"/><path d="M4 21v-1a8 8 0 0 1 16 0v1"/></svg>
          <span>+ Seat</span>
        </button>
        <button class="empty-btn empty-btn-primary" data-action="start-empty" title="Start without a layout">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
          <span>Start Empty</span>
        </button>
      </div>
    `;
    overlay.querySelectorAll('.empty-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const action = btn.dataset.action;
        if (action === 'add-table') {
          // Use the bulk add so the shared seat-count input applies
          // regardless of whether the user clicked here or in the footer.
          const input = document.getElementById('seat-count-input');
          let n = parseInt(input?.value, 10);
          if (!Number.isFinite(n) || n < 0) n = 0;
          if (n > 20) n = 20;
          this.addTableWithSeats(n);
        } else if (action === 'add-seat') {
          this.addSeatAtCenter();
        } else if (action === 'start-empty') {
          this.startMeeting();
        } else {
          this.applyPreset(action);
        }
      });
    });
    this.canvas.appendChild(overlay);
  }

  async startMeeting() {
    // Mark the fresh-start window BEFORE the async start POST. The
    // reconciler's shared transport predicate reads
    // _startInitiatedByThisTab (set via claimOwnership after the POST
    // returns) but `body.starting` is the sibling signal that keeps
    // navigation guards / reconcile's rehydration branch inert until
    // body.recording is set.
    document.body.classList.add('starting');

    // Transition to meeting mode. #landing-mode is hidden unconditionally
    // here (in addition to room-setup / view-mode) so a mid-start click
    // on Home never leaves landing visible on top of a running meeting.
    document.getElementById('landing-mode').style.display = 'none';
    document.getElementById('room-setup').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    document.getElementById('meeting-mode').style.display = '';
    document.getElementById('control-bar').style.display = '';
    document.body.classList.add('meeting-active');

    // Virtual table hidden by default on new meetings — user can add it
    // ad-hoc via the "Table" button in the control bar. Keeps the interface
    // clean at meeting start.
    document.body.classList.add('hide-table');
    const tableBtn = document.getElementById('btn-toggle-table');
    if (tableBtn) tableBtn.classList.remove('active-toggle');

    // Clear any stale timeline/player from a previously viewed meeting
    document.getElementById('speaker-timeline').style.display = 'none';
    document.getElementById('player-bar').style.display = 'none';
    audioPlayer.hide();

    // Render miniature table strip (hidden by default via hide-table class,
    // but rendered so it's instantly ready when the user toggles it on)
    this._renderTableStrip();

    // Initialize transcript column renderers (global subscription handles delivery)
    window._gridRenderer = new CompactGridRenderer(document.getElementById('transcript-grid'));

    try {
      await startRecording(false);
      // Success signal: startRecording adds body.recording on success
      // and swallows its own errors via its catch block. Only claim
      // ownership when we can observe that class — that's the one
      // reliable indicator a real recorder pipeline is now up on this
      // tab. (window.current_meeting_id may be stale from a previous
      // start if the current attempt failed without clearing it.)
      if (document.body.classList.contains('recording')
       && window.current_meeting_id
       && reconciler) {
        reconciler.claimOwnership(window.current_meeting_id);
      }
    } finally {
      // Bulletproof button restore: even if startRecording threw after
      // advancing past the sync try/catch in the outer click handler,
      // this block always clears .starting. If the start succeeded we
      // are now in body.recording; if it failed, startRecording's own
      // catch block has rolled the meeting-mode UI back to room-setup.
      document.body.classList.remove('starting');
      if (this.btnStart) {
        this.btnStart.disabled = false;
        this.btnStart.textContent = this.btnStart.dataset.origText || 'Start Meeting';
      }
    }
  }

  _renderTableStrip() {
    const strip = document.getElementById('meeting-table-strip');
    strip.innerHTML = '';

    // Create an inner container that matches the setup canvas aspect ratio,
    // then scale it to fit the strip. This keeps proportions identical.
    const inner = document.createElement('div');
    inner.className = 'strip-inner';
    strip.appendChild(inner);

    // Render tables
    for (const t of this.tables) {
      const el = document.createElement('div');
      el.className = 'strip-table';
      el.style.left = `${t.x}%`;
      el.style.top = `${t.y}%`;
      el.style.width = `${t.width}%`;
      el.style.height = `${t.height}%`;
      el.style.borderRadius = `${t.borderRadius}%`;
      inner.appendChild(el);
    }

    // SEAT RENDERING — two modes:
    //
    // 1. Active meeting: overlay detected speakers (from _speakerRegistry)
    //    onto the layout seats in first-seen order. Falls back to design-time
    //    placeholder names if no speakers are detected yet.
    //
    // 2. Design-time (setup mode): use the layout's own seat data as-is.
    const inMeeting = document.body.classList.contains('recording')
                   || document.body.classList.contains('meeting-active');
    const detected = inMeeting ? getAllSpeakers() : [];

    // Make sure we have enough seat SLOTS for every detected speaker —
    // if detection finds 6 speakers but the layout only has 4, we still
    // show all 6 by overflowing into synthesized positions.
    const layoutSeats = this.seats;
    const totalSlots = Math.max(layoutSeats.length, detected.length);

    for (let i = 0; i < totalSlots; i++) {
      const layoutSeat = layoutSeats[i];
      const detectedSpeaker = detected[i];

      // Position: prefer layout seat coords; overflow speakers get
      // auto-positioned along the bottom edge.
      let x, y;
      if (layoutSeat) {
        x = layoutSeat.x;
        y = layoutSeat.y;
      } else {
        const overflow = i - layoutSeats.length;
        x = 10 + (overflow * 18) % 80;
        y = 88;
      }

      // Seat content: detected speaker (if active meeting) else layout name
      const clusterId = detectedSpeaker?.clusterId ?? null;
      const seqIndex = detectedSpeaker?.seqIndex ?? (i + 1);
      const nameFromSpeaker = detectedSpeaker?.displayName;
      const nameFromLayout = layoutSeat?.name;
      const displayName = nameFromSpeaker || nameFromLayout || '';
      // A seat is considered "enrolled" (has a real human name) when the
      // display name isn't the fallback "Speaker N" pattern. Layout
      // enrollment is older design-time state; we also honor it.
      const hasRealName = displayName && !/^Speaker\s+\d+$/i.test(displayName.trim());
      const isEnrolled = layoutSeat?.enrolled || hasRealName;
      // design-time: no cluster_id yet — fall back to seat index colour
      const color = clusterId != null ? getSpeakerColor(clusterId) : SPEAKER_COLORS[i % SPEAKER_COLORS.length];

      const el = document.createElement('div');
      el.className = `strip-seat${isEnrolled ? ' enrolled' : ''}${detectedSpeaker ? ' detected' : ''}`;
      el.dataset.speakerId = String(i);
      if (clusterId != null) el.dataset.clusterId = String(clusterId);
      el.style.left = `${x}%`;
      el.style.top = `${y}%`;
      el.style.setProperty('--seat-color', color);

      const hoverHint = detectedSpeaker
        ? (isEnrolled
            ? `Click to rename ${displayName}`
            : `Click to name Speaker ${seqIndex}`)
        : 'Click to edit seat';
      el.title = hoverHint;

      // Use the seq index as the avatar number so the seat matches the
      // "Speaker 1 / 2 / 3" naming in the transcript. Name label is
      // always rendered: the CSS adds a "tap to name" hint when empty.
      el.innerHTML = `<span class="strip-seat-num">${seqIndex}</span>` +
        `<span class="strip-seat-name">${hasRealName ? esc(displayName) : ''}</span>`;

      el.addEventListener('click', () => {
        if (detectedSpeaker) {
          // Active-meeting path: open rename modal tied to cluster_id
          _openSpeakerRenameModal(clusterId, displayName, color);
        } else {
          // Design-time or historical fallback
          const seatName = layoutSeat?.name || `Speaker ${i+1}`;
          showSpeakerModal(seatName, color,
            findSpeakerSegments(seatName), window._gridRenderer?._meetingId);
        }
      });
      inner.appendChild(el);
    }
  }
}

const roomSetup = new RoomSetup();

// ─── Segment Store ───────────────────────────────────────────
// Reactive store keyed by segment_id. Only highest revision shown.

import { SegmentStore } from './segment-store.js';
import { createReconciler } from './meeting-reconcile.js';

// SPEAKER_COLORS defined in Room Setup section above

// ─── Audio Player ───────────────────────────────────────────

class AudioPlayer {
  constructor() {
    this.audio = new Audio();
    this.audio.preload = 'none';
    this.meetingId = null;
    this.currentSegmentId = null;
    this._fullMeetingSrc = null;  // track full-meeting audio URL
    this._bar = document.getElementById('player-bar');
    this._playBtn = document.getElementById('player-play');
    this._scrub = document.getElementById('player-scrub');
    this._current = document.getElementById('player-current');
    this._durationEl = document.getElementById('player-duration');
    this._speed = document.getElementById('player-speed');

    this._playBtn.addEventListener('click', () => this.togglePlay());
    this._scrub.addEventListener('input', () => {
      if (this.audio.duration) this.audio.currentTime = (this._scrub.value / 100) * this.audio.duration;
    });
    this._speed.addEventListener('change', () => { this.audio.playbackRate = parseFloat(this._speed.value); });
    this.audio.addEventListener('timeupdate', () => this._onTimeUpdate());
    this.audio.addEventListener('ended', () => this._onEnded());
  }

  /**
   * Find the transcript row for a segment id. Compact blocks merge multiple
   * segments into one row, so we have to check both the single-segment
   * `data-segment-id` attribute and the merged `data-segment-ids` list.
   * The plain `querySelector([data-segment-id=X])` path misses any segment
   * that isn't the *first* one in its merged block — that's the root cause
   * of the "text/audio mapping breaks after direction toggle" bug, because
   * the primary id of a merged block no longer always matches the playhead.
   */
  _findSegmentRow(segmentId) {
    if (!segmentId) return null;
    let row = document.querySelector(`.compact-block[data-segment-id="${segmentId}"]`);
    if (row) return row;
    for (const el of document.querySelectorAll('[data-segment-ids]')) {
      const ids = el.dataset.segmentIds;
      if (ids && ids.split(',').includes(segmentId)) return el;
    }
    return null;
  }

  /** Play a specific segment from a meeting */
  playSegment(meetingId, startMs, endMs, segmentId) {
    this.meetingId = meetingId;
    this._segmentOffset = startMs;  // absolute offset for timeline sync
    this.audio.src = `${API}/api/meetings/${meetingId}/audio?start_ms=${startMs}&end_ms=${endMs}`;
    this.audio.playbackRate = parseFloat(this._speed.value);
    this.audio.play();
    this._highlightRow(segmentId);
    this._playBtn.textContent = '⏸';
    this._playBtn.classList.add('playing');

    // Immediately scroll transcript to the segment
    if (segmentId && window._gridRenderer) {
      const row = this._findSegmentRow(segmentId);
      if (row) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  /** Load full meeting audio for podcast-style playback */
  loadMeeting(meetingId, durationMs, timeline) {
    this.meetingId = meetingId;
    this._timeline = timeline || null;
    this._segmentOffset = 0;  // full meeting — no offset
    this._fullMeetingSrc = `${API}/api/meetings/${meetingId}/audio`;
    this.audio.src = this._fullMeetingSrc;
    this._durationEl.textContent = this._fmt(durationMs);
    this._scrub.value = 0;
    this._current.textContent = '00:00';
    this._bar.style.display = '';
    document.body.classList.add('has-player');

    // Generate speaker color bands on scrub bar
    if (timeline && timeline.length && durationMs > 0) {
      // Use canonical cluster_id-based coloring so scrub bar matches
      // the speaker timeline lanes and transcript blocks
      const stops = timeline.map(seg => {
        const color = getSpeakerColor(seg.speaker_id);
        const s = (seg.start_ms / durationMs * 100).toFixed(2);
        const e = (seg.end_ms / durationMs * 100).toFixed(2);
        return `${color} ${s}%, ${color} ${e}%`;
      });
      this._scrub.style.setProperty('--speaker-gradient',
        `linear-gradient(to right, var(--bg-raised) 0%, ${stops.join(', ')}, var(--bg-raised) 100%)`);
    } else {
      this._scrub.style.removeProperty('--speaker-gradient');
    }
  }

  /** Seek within the full meeting audio, restoring it if playSegment changed the src */
  seekFullMeeting(ms, segmentId) {
    if (this._fullMeetingSrc && this.audio.src !== this._fullMeetingSrc) {
      this.audio.src = this._fullMeetingSrc;
      this._segmentOffset = 0;
    }
    this._segmentOffset = 0;
    this.audio.currentTime = ms / 1000;
    this.audio.playbackRate = parseFloat(this._speed.value);
    this.audio.play();
    this._highlightRow(segmentId);
    this._playBtn.textContent = '⏸';
    this._playBtn.classList.add('playing');
    if (segmentId) {
      const row = this._findSegmentRow(segmentId);
      if (row) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  togglePlay() {
    if (this.audio.paused) { this.audio.play(); this._playBtn.textContent = '⏸'; this._playBtn.classList.add('playing'); }
    else { this.audio.pause(); this._playBtn.textContent = '▶'; this._playBtn.classList.remove('playing'); }
  }

  hide() {
    this.audio.pause();
    this._bar.style.display = 'none';
    document.body.classList.remove('has-player');
    this._clearHighlight();
  }

  _onTimeUpdate() {
    if (!this.audio.duration) return;
    const pct = (this.audio.currentTime / this.audio.duration) * 100;
    this._scrub.value = pct;
    this._current.textContent = this._fmt(this.audio.currentTime * 1000);

    // Scroll transcript to current playback position, highlight active block,
    // and sync the active timeline block so all three views stay in lockstep.
    if (this._timeline && window._gridRenderer) {
      const currentMs = this.audio.currentTime * 1000 + (this._segmentOffset || 0);
      const seg = this._timeline.find(s => currentMs >= s.start_ms && currentMs <= s.end_ms);
      if (seg && seg.segment_id) {
        const row = this._findSegmentRow(seg.segment_id);
        // Track the *segment id* we last highlighted, not a DOM class, so
        // that toggling transcript direction (which leaves `.playing` on the
        // same element but visually repositions it) re-triggers scrolling.
        if (row && this.currentSegmentId !== seg.segment_id) {
          this._clearHighlight();
          row.classList.add('playing');
          row.scrollIntoView({ behavior: 'smooth', block: 'center' });
          this.currentSegmentId = seg.segment_id;
        }
        this._highlightTimelineBlock(seg.segment_id);
        this._highlightSpeakingSeat(seg.speaker_id);
      } else {
        this._clearSpeakingSeats();
      }
    }
  }

  _onEnded() {
    this._playBtn.textContent = '▶';
    this._playBtn.classList.remove('playing');
    this._clearHighlight();
  }

  _highlightRow(segmentId) {
    this._clearHighlight();
    if (segmentId) {
      this.currentSegmentId = segmentId;
      const row = this._findSegmentRow(segmentId);
      if (row) row.classList.add('playing');
    }
  }

  _highlightTimelineBlock(segmentId) {
    // Clear previously active timeline block
    document.querySelectorAll('.speaker-timeline-block.playing').forEach(el => {
      el.classList.remove('playing');
    });
    if (!segmentId) return;
    const blockEl = document.querySelector(`.speaker-timeline-block[data-segment-id="${segmentId}"]`);
    if (blockEl) blockEl.classList.add('playing');
  }

  _highlightSpeakingSeat(speakerId) {
    this._clearSpeakingSeats();
    if (speakerId == null) return;
    // Match by cluster_id (canonical) — seats are tagged with data-cluster-id
    // by the meeting loader. Fall back to legacy data-speaker-id for live meetings.
    const target = String(speakerId);
    document.querySelectorAll('.strip-seat').forEach(seat => {
      if (seat.dataset.clusterId === target || seat.dataset.speakerId === target) {
        seat.classList.add('speaking');
      }
    });
  }

  _clearSpeakingSeats() {
    document.querySelectorAll('.strip-seat.speaking').forEach(el => el.classList.remove('speaking'));
  }

  _clearHighlight() {
    // Scoped to transcript + timeline so we don't clobber .player-play.playing
    document.querySelectorAll(
      '.compact-block.playing, .speaker-timeline-block.playing'
    ).forEach(r => r.classList.remove('playing'));
    this.currentSegmentId = null;
  }

  _fmt(ms) {
    const s = Math.floor((ms || 0) / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }
}

const audioPlayer = new AudioPlayer();
window.audioPlayer = audioPlayer;

// ─── Compact Grid Renderer (sole active transcript renderer) ───
// Groups consecutive segments by speaker into single blocks.
// Each block shows speaker name once, then accumulates text.
// New block on: speaker change, >5s gap, or language direction change.

class CompactGridRenderer {
  constructor(gridEl, meetingId) {
    this.gridEl = gridEl;
    this._meetingId = meetingId || null;
    this._autoScroll = true;
    this._newestFirst = true;  // newest at top (reverse scroll)
    this._currentBlock = null;  // { speakerKey, lang, enEl, jaEl, endMs, segIds }
    this._segmentMap = new Map(); // segmentId → block reference
    this.gridEl.addEventListener('scroll', () => {
      if (this._newestFirst) {
        this._autoScroll = this.gridEl.scrollTop < 60;
      } else {
        const { scrollTop, scrollHeight, clientHeight } = this.gridEl;
        this._autoScroll = scrollHeight - scrollTop - clientHeight < 60;
      }
    });
  }

  toggleDirection() {
    this._newestFirst = !this._newestFirst;
    // Reverse all children
    const children = [...this.gridEl.children];
    this.gridEl.innerHTML = '';
    children.reverse().forEach(c => this.gridEl.appendChild(c));
    this._autoScroll = true;

    // After reversing, re-anchor scroll to the block corresponding to the
    // audio player's *current* position. We must resolve this live from
    // audio.currentTime + the timeline, NOT from ap.currentSegmentId, which
    // is the last-highlighted id and is typically one tick stale — scrolling
    // to it first and then snapping to the next segment on the following
    // timeupdate is what the "scrolls away then fixes itself" glitch is.
    const ap = window.audioPlayer;
    if (!ap) return this._newestFirst;

    let liveId = null;
    if (ap._timeline && ap.audio && !isNaN(ap.audio.currentTime)) {
      const currentMs = ap.audio.currentTime * 1000 + (ap._segmentOffset || 0);
      const liveSeg = ap._timeline.find(
        s => currentMs >= s.start_ms && currentMs <= s.end_ms
      );
      if (liveSeg) liveId = liveSeg.segment_id;
    }
    if (!liveId) liveId = ap.currentSegmentId;

    const anchorEl = liveId
      ? ap._findSegmentRow(liveId)
      : this.gridEl.querySelector('.compact-block.playing');

    if (anchorEl) {
      // Move the .playing highlight to match the live segment (in case it
      // drifted from where the audio actually is), then jump scroll.
      if (liveId) {
        ap._clearHighlight();
        anchorEl.classList.add('playing');
        ap.currentSegmentId = liveId;
      }
      anchorEl.scrollIntoView({ behavior: 'auto', block: 'center' });
    } else {
      this.gridEl.scrollTop = this._newestFirst ? 0 : this.gridEl.scrollHeight;
    }
    return this._newestFirst;
  }

  update(segmentId, event) {
    if (!segmentId) { this._clear(); return; }
    if (!event.text) return;

    let lang = event.language;
    const tr = event.translation?.text || '';
    const langA = _getLangA();
    const langB = _getLangB();
    // STRICT language-column routing by script. We never trust the ASR
    // label alone because it occasionally mislabels (Japanese tagged "en",
    // English tagged "ja"), and a single leaked character in the wrong
    // column is visibly broken. The script check is authoritative.
    lang = _routeLangByScript(event.text, lang, langA, langB);

    const speakerKey = this._getSpeakerKey(event);
    const startMs = event.start_ms || 0;
    const endMs = event.end_ms || startMs;

    // Keep speakers[] on the segment record so _rebuildBlock can refresh
    // the header if diarization / self-intro updates the attribution later.
    const segRecord = {
      text: event.text,
      tr,
      lang,
      isFinal: event.is_final,
      speakers: event.speakers || [],
      ts: Date.now(),
      // Ruby HTML for kanji segments. The server broadcasts a revision
      // event with furigana_html attached ~100ms after the initial final.
      furiganaHtml: event.furigana_html || null,
      trFuriganaHtml: event.translation?.furigana_html || null,
    };

    // Retroactive update path: if we already rendered this segment, update
    // the EXISTING block in place — the speaker catch-up loop on the server
    // re-broadcasts events with cluster_ids attached after diarization finishes.
    if (this._segmentMap.has(segmentId)) {
      const existingBlock = this._segmentMap.get(segmentId);
      existingBlock.segments.set(segmentId, segRecord);
      existingBlock.endMs = Math.max(existingBlock.endMs, endMs);

      // If the speaker changed (e.g., "unknown" → real cluster), update the
      // block's speaker key + color so the UI reflects the new attribution.
      if (existingBlock.speakerKey !== speakerKey) {
        existingBlock.speakerKey = speakerKey;
        const clusterId = event.speakers?.[0]?.cluster_id;
        if (clusterId != null) {
          const color = getSpeakerColor(clusterId);
          existingBlock.row.style.setProperty('--speaker-color', color);
          existingBlock.row.dataset.clusterId = String(clusterId);
          // Also update the speaker color used by the header text (re-rendered
          // by _rebuildBlock via the stored speakers[] on the segment record)
        }
      }
      this._rebuildBlock(existingBlock);
      return;
    }

    // New segment: check if we should merge into the last-created block.
    // Cap merging by three conditions:
    //   - same speaker key and language
    //   - gap < 5s since last segment
    //   - block's total duration < 45s (don't let one block swallow minutes
    //     of audio just because diarization returned one cluster for everything)
    //   - block holds fewer than 12 segments (visual readability cap)
    const blockDurationMs = this._currentBlock
      ? (this._currentBlock.endMs - this._currentBlock.startMs)
      : 0;
    const blockSegmentCount = this._currentBlock
      ? this._currentBlock.segments.size
      : 0;
    const shouldMerge = this._currentBlock
      && this._currentBlock.speakerKey === speakerKey
      && this._currentBlock.lang === lang
      && (startMs - this._currentBlock.endMs) < 5000
      && blockDurationMs < 45000
      && blockSegmentCount < 12;

    if (shouldMerge) {
      this._currentBlock.segments.set(segmentId, segRecord);
      this._segmentMap.set(segmentId, this._currentBlock);
      this._currentBlock.endMs = Math.max(this._currentBlock.endMs, endMs);
      this._rebuildBlock(this._currentBlock);
    } else {
      // Create new block
      const block = this._createBlock(speakerKey, lang, startMs, event);
      block.segments.set(segmentId, segRecord);
      this._segmentMap.set(segmentId, block);
      this._currentBlock = block;
      this._rebuildBlock(block);
    }

    if (this._autoScroll) {
      requestAnimationFrame(() => {
        this.gridEl.scrollTop = this._newestFirst ? 0 : this.gridEl.scrollHeight;
      });
    }
  }

  _getSpeakerKey(event) {
    if (!event.speakers?.length) return 'unknown';
    const s = event.speakers[0];
    return s.identity || s.display_name || `speaker-${s.cluster_id || 0}`;
  }

  _createBlock(speakerKey, lang, startMs, event) {
    const row = document.createElement('div');
    row.className = 'compact-block';

    // Speaker color — canonical mapping by cluster_id (same as timeline)
    const clusterId = event.speakers?.[0]?.cluster_id;
    const color = getSpeakerColor(clusterId);
    row.style.setProperty('--speaker-color', color);
    row.dataset.clusterId = String(clusterId ?? '');

    const langA = _getLangA();
    const langB = _getLangB();

    // Speaker label (shown once per block). If diarization detected
    // overlap, render an extra chip for each co-speaker so the UI
    // surfaces cross-talk instead of silently hiding it.
    const primary = event.speakers?.[0];
    const speakerName = primary?.identity || primary?.display_name || speakerKey;
    const overlapChips = (event.speakers || []).slice(1).map(s => {
      const cid = s.cluster_id;
      const c = getSpeakerColor(cid);
      const n = s.identity || s.display_name || getSpeakerDisplayName(cid, null);
      return `<span class="compact-speaker-overlap" style="--c:${c}" title="Overlapping: ${esc(n)}">+ ${esc(n)}</span>`;
    }).join('');
    const timeStr = formatTime(startMs);

    row.innerHTML = `
      <div class="compact-block-header">
        <span class="compact-speaker" style="color:${color}">${esc(speakerName)}</span>${overlapChips}
        <span class="compact-time">${timeStr}</span>
      </div>
      <div class="compact-columns">
        <div class="compact-col compact-col-a"></div>
        <div class="compact-col compact-col-b"></div>
      </div>
    `;

    if (this._newestFirst) {
      this.gridEl.prepend(row);
    } else {
      this.gridEl.appendChild(row);
    }

    const block = {
      speakerKey,
      lang,
      row,
      colA: row.querySelector('.compact-col-a'),
      colB: row.querySelector('.compact-col-b'),
      speakerEl: row.querySelector('.compact-speaker'),
      startMs,
      endMs: startMs,
      segments: new Map(),
    };

    // Click-to-seek is REVIEW-ONLY. During a live meeting we don't let
    // the UI play any audio out — the operator is typically in the room
    // with the speaker and any audio from the admin interface creates a
    // feedback loop / confuses the meeting. A finished meeting (no
    // `recording` body class) gets the normal seek-to-point behaviour.
    const isLiveNow = () => document.body.classList.contains('recording');
    row.addEventListener('click', (ev) => {
      if (isLiveNow()) return;
      if (ev.target.closest('.compact-speaker, .compact-speaker-overlap, .compact-time, button, a, input')) return;
      if (!this._meetingId) return;
      const sMs = block.startMs;
      const eMs = Math.max(block.endMs, sMs + 500);
      const firstSeg = block.segments.keys().next().value || null;
      if (audioPlayer._fullMeetingSrc) {
        audioPlayer.seekFullMeeting(sMs, firstSeg);
      } else {
        audioPlayer.playSegment(this._meetingId, sMs, eMs, firstSeg);
      }
    });
    // Only expose the clickable affordance when not live. Toggled by
    // start/stop handlers that manage the body.recording class.
    row.dataset.clickToSeek = '1';

    return block;
  }

  _rebuildBlock(block) {
    const langA = _getLangA();
    const langB = _getLangB();
    const cssA = _languageNames[langA]?.css_font_class || '';
    const cssB = _languageNames[langB]?.css_font_class || '';

    // Monolingual meetings: route everything into column A and leave
    // column B empty. CSS (body.monolingual / .popout-mode.monolingual)
    // collapses the empty column so the visible one claims full width.
    if (!langB) {
      const textOnly = [];
      let anyRuby = false;
      for (const [, seg] of block.segments) {
        if (seg.furiganaHtml) { textOnly.push(seg.furiganaHtml); anyRuby = true; }
        else textOnly.push(esc(seg.text));
      }
      const newA = textOnly.join(' ');
      // Only touch innerHTML when the rendered string actually changed.
      // Late-arriving speaker/furigana/translation events re-call rebuild
      // every few hundred ms, and writing innerHTML wipes any live text
      // selection — so a user dragging to copy loses their selection mid-
      // drag. This guard is load-bearing for Ctrl+C in live meetings.
      if (block.colA.innerHTML !== newA) block.colA.innerHTML = newA;
      if (block.colB.innerHTML !== '') block.colB.innerHTML = '';
      block.colA.classList.toggle('has-ruby', anyRuby);
      if (cssA) block.colA.classList.add(cssA);
      return;
    }

    let textA = [];
    let textB = [];

    // STRICT column routing. Both the source text AND the translation
    // are script-checked before being placed, so a translation that came
    // back in the wrong script (translator hiccup, JA→EN returning JA)
    // is dropped rather than leaking across columns.
    const cjkCodes = new Set(['ja', 'zh', 'ko']);
    const matchesColumn = (text, col) => {
      const routed = _routeLangByScript(text, col, langA, langB);
      return routed === col;
    };
    // Cells accumulate HTML fragments — plain text is escaped, furigana
    // ruby markup is passed through. We use innerHTML on assign so the
    // <ruby>/<rt> tags render instead of appearing as literal text.
    let anyRubyA = false;
    let anyRubyB = false;
    for (const [, seg] of block.segments) {
      const srcCol = seg.lang === langA ? 'A' : seg.lang === langB ? 'B' : null;
      if (!srcCol) continue;
      if (srcCol === 'A' && matchesColumn(seg.text, langA)) {
        if (seg.furiganaHtml) { textA.push(seg.furiganaHtml); anyRubyA = true; }
        else textA.push(esc(seg.text));
        if (seg.tr && matchesColumn(seg.tr, langB)) {
          if (seg.trFuriganaHtml) { textB.push(seg.trFuriganaHtml); anyRubyB = true; }
          else textB.push(esc(seg.tr));
        } else if (!seg.isFinal) textB.push('...');
      } else if (srcCol === 'B' && matchesColumn(seg.text, langB)) {
        if (seg.furiganaHtml) { textB.push(seg.furiganaHtml); anyRubyB = true; }
        else textB.push(esc(seg.text));
        if (seg.tr && matchesColumn(seg.tr, langA)) {
          if (seg.trFuriganaHtml) { textA.push(seg.trFuriganaHtml); anyRubyA = true; }
          else textA.push(esc(seg.tr));
        } else if (!seg.isFinal) textA.push('...');
      }
    }

    const classA = `compact-col compact-col-a ${cssA}${anyRubyA ? ' has-ruby' : ''}`;
    const classB = `compact-col compact-col-b ${cssB}${anyRubyB ? ' has-ruby' : ''}`;
    if (block.colA.className !== classA) block.colA.className = classA;
    if (block.colB.className !== classB) block.colB.className = classB;
    // Only touch innerHTML when the rendered string actually changed.
    // See the same guard in the monolingual branch above — this is what
    // keeps live text selection from being nuked by late-arriving
    // speaker/furigana/translation events on the same segment.
    const newA = textA.join(' ');
    const newB = textB.join(' ');
    if (block.colA.innerHTML !== newA) block.colA.innerHTML = newA;
    if (block.colB.innerHTML !== newB) block.colB.innerHTML = newB;

    // Refresh speaker label from the latest segment — late-arriving events
    // may carry updated speaker attribution (diarization, self-intro).
    if (block.speakerEl) {
      let latestSpeakers = null;
      let latestTs = 0;
      for (const [, seg] of block.segments) {
        if (seg.speakers?.length && (seg.ts || 0) >= latestTs) {
          latestSpeakers = seg.speakers;
          latestTs = seg.ts || 0;
        }
      }
      if (latestSpeakers?.length) {
        const s = latestSpeakers[0];
        const name = getSpeakerDisplayName(s.cluster_id, s.identity || s.display_name);
        if (block.speakerEl.textContent !== name) {
          block.speakerEl.textContent = name;
          // Also refresh color if cluster changed
          const color = getSpeakerColor(s.cluster_id);
          block.speakerEl.style.color = color;
          block.row.style.setProperty('--speaker-color', color);
          block.row.dataset.clusterId = String(s.cluster_id ?? '');
        }
      }
    }

    // Mark block with segment IDs for timeline scroll lookup
    const segIds = [...block.segments.keys()];
    block.row.dataset.segmentId = segIds[0] || '';
    block.row.dataset.segmentIds = segIds.join(',');

    // Dim partial text
    const hasPartial = [...block.segments.values()].some(s => !s.isFinal);
    block.row.classList.toggle('partial', hasPartial);
  }

  _clear(force = false) {
    // Never clear during active recording unless forced (e.g. user clicked Clear)
    if (!force && document.body.classList.contains('recording')) return;
    this._currentBlock = null;
    this._segmentMap.clear();
    this.gridEl.innerHTML = '';
  }
}

// ─── Audio Capture ───────────────────────────────────────────

class AudioPipeline {
  constructor() { this.ws = null; this.audioCtx = null; this.workletNode = null; this.stream = null; this.analyser = null; this.running = false; }

  async start(onEvent) {
    // If the setup-panel mic warm-up is still primed, consume it so the
    // recorder starts capturing audio with zero device-init latency. The
    // warm-up uses the same constraints we'd request below, so the handover
    // is loss-less. Otherwise acquire fresh.
    const warm = micWarmup.consume();
    let source;
    if (warm) {
      this.stream = warm.stream;
      this.audioCtx = warm.ac;
      source = warm.source;
    } else {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
      });
      this.audioCtx = new AudioContext();
    }
    if (this.audioCtx.state === 'suspended') await this.audioCtx.resume();
    const sampleRate = this.audioCtx.sampleRate;
    this.analyser = this.audioCtx.createAnalyser(); this.analyser.fftSize = 256;
    await this.audioCtx.audioWorklet.addModule('/static/js/audio-worklet.js');
    this.workletNode = new AudioWorkletNode(this.audioCtx, 'scribe-audio-processor', { processorOptions: { sampleRate } });
    if (!source) source = this.audioCtx.createMediaStreamSource(this.stream);
    source.connect(this.analyser); source.connect(this.workletNode);
    this.ws = new WebSocket(WS_URL); this.ws.binaryType = 'arraybuffer';
    this.ws.onmessage = (evt) => {
      wsMessageCount++;
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'tts_audio' && msg.audio_url) {
          // Auto-play TTS audio (interpretation-audio track)
          if (!document.getElementById('tts-mute')?.checked) {
            audioPlayer.audio.src = `${API}${msg.audio_url}`;
            audioPlayer.audio.play().catch(() => {});
          }
        } else if (msg.type === 'seat_update') {
          // Update specific seat name in the table strip
          const seats = document.querySelectorAll('.strip-seat');
          let updated = false;
          seats.forEach(s => {
            const nameSpan = s.querySelector('.strip-seat-name');
            if (nameSpan && !nameSpan.textContent.trim() && !updated) {
              nameSpan.textContent = msg.speaker_name;
              s.classList.add('enrolled');
              updated = true;
            }
          });
        } else if (msg.type === 'speaker_pulse') {
          // Update speaker pulse indicators on seats
          const seats = document.querySelectorAll('.strip-seat');
          const activeNames = new Set((msg.active_speakers || []).map(s => s.name).filter(Boolean));
          seats.forEach(s => {
            const nameSpan = s.querySelector('.strip-seat-name');
            const name = nameSpan?.textContent?.trim();
            if (name && activeNames.has(name)) {
              s.classList.add('speaking');
            } else {
              s.classList.remove('speaking');
            }
          });
        } else if (msg.type === 'room_layout_update') {
          // Mid-meeting layout change from another client — re-render
          window._onRoomLayoutUpdate?.(msg.layout);
        } else if (msg.type === 'speaker_rename') {
          // Another client renamed a speaker — sync the registry and
          // refresh all UI surfaces so names stay consistent.
          if (msg.cluster_id != null && msg.display_name) {
            renameSpeaker(msg.cluster_id, msg.display_name);
            _refreshDetectedSpeakersStrip();
            _refreshTranscriptSpeakerLabels();
          }
        } else if (msg.type === 'speaker_remap') {
          // Backend collapsed raw cluster_ids when diarize centroids merged
          // (fix 4). The surviving id takes over the retired id's label and
          // color so "Speaker 41" no longer leaks into the live view after
          // a consolidation pass. See _speaker_catchup_loop.
          if (msg.renames && typeof msg.renames === 'object') {
            for (const [retiredStr, survivor] of Object.entries(msg.renames)) {
              const retired = parseInt(retiredStr, 10);
              if (!Number.isFinite(retired)) continue;
              const survivorEntry = _speakerRegistry.clusters.get(survivor);
              const retiredEntry = _speakerRegistry.clusters.get(retired);
              if (retiredEntry && survivorEntry) {
                // Carry over any human-set name from retired → survivor.
                if (retiredEntry.displayName && !survivorEntry.displayName) {
                  survivorEntry.displayName = retiredEntry.displayName;
                }
              }
              _speakerRegistry.clusters.delete(retired);
            }
            _refreshDetectedSpeakersStrip();
            _refreshTranscriptSpeakerLabels();
          }
        } else if (msg.type === 'summary_regenerated') {
          // Server rebuilt summary.json (e.g. after a rename). Poke any
          // review view open on this meeting so the topics / action items
          // refresh.
          window._onSummaryRegenerated?.(msg.meeting_id);
        } else if (msg.type === 'meeting_warning') {
          // Silence watchdog (fix 3): audio hasn't landed in 10s+.
          console.warn('meeting_warning', msg);
          const s = document.getElementById('status-line');
          if (s) s.textContent = `Warning: ${msg.reason} (${msg.age_s}s)`;
        } else if (msg.type === 'meeting_warning_cleared') {
          // Reset banner if we set one.
          // (no-op beyond clearing — the next normal status update overwrites)
        } else if (msg.type === 'meeting_stopped') {
          console.info('meeting_stopped', msg);
        } else if (msg.type === 'meeting_cancelled') {
          console.info('meeting_cancelled', msg);
          // Server cancelled the meeting — clean up UI
          document.body.classList.remove('recording');
          document.body.classList.remove('meeting-active');
          const _cb = document.getElementById('control-bar');
          if (_cb) _cb.style.display = 'none';
          const _sl = document.getElementById('status-line');
          if (_sl) _sl.textContent = 'Meeting cancelled';
          store.clear();
        } else if (msg.type === 'dev_reset') {
          // DEV mode: server reset the meeting — clear transcript UI
          store.clear();
          if (window._gridRenderer) window._gridRenderer._clear(true);
          _resetSpeakerRegistry();
          _refreshDetectedSpeakersStrip();
          timer.reset(); timer.start();
          document.getElementById('status-line').textContent = 'DEV: Reset — speak to test';
        } else {
          onEvent(msg);
        }
      } catch(e) { console.warn('WS parse:', e); }
    };
    await new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; });
    this.workletNode.port.onmessage = (evt) => { if (this.ws?.readyState === WebSocket.OPEN) { audioChunkCount++; this.ws.send(evt.data); } };
    this.running = true;
  }

  async stop() {
    this.running = false; this.workletNode?.disconnect();
    if (this.audioCtx) await this.audioCtx.close();
    this.stream?.getTracks().forEach(t => t.stop()); this.ws?.close();
    this.workletNode = null; this.audioCtx = null; this.stream = null; this.ws = null; this.analyser = null;
  }

  getLevel() {
    if (!this.analyser) return 0;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0; for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    return Math.min(Math.sqrt(sum / data.length) * 8, 1);
  }
}

// ─── Audio-Out Listener (Real-Time Interpretation) ──────────

class AudioOutListener {
  constructor() {
    this.ws = null;
    this.audioCtx = null;
    this.enabled = false;
    this.preferredLanguage = '';
    this.mode = 'translation'; // 'translation' or 'full'
    this._queue = [];
    this._playing = false;
    // Auto-reconnect state. The server's event loop sometimes blocks long
    // enough that the WebSocket keepalive can't fire and the browser kills
    // the connection from under us. Without auto-reconnect that means
    // "audio worked then went silent forever until you re-tap Listen".
    // With auto-reconnect, the same scenario heals itself in <2s and the
    // user never notices.
    this._reconnectAttempts = 0;
    this._intentionalStop = false;
    this._reconnectTimer = null;
  }

  /**
   * Create the AudioContext SYNCHRONOUSLY inside the click handler that
   * triggered Listen, so the browser counts the resume as a user-gesture
   * action. After the first `await` in the click handler the gesture
   * context is gone and a deferred resume may silently no-op, leaving the
   * context permanently suspended — every decoded WAV plays into a muted
   * destination and the user hears nothing despite the server delivering
   * audio successfully. Idempotent: safe to call multiple times.
   */
  primeAudioContext() {
    if (this.audioCtx && this.audioCtx.state !== 'closed') return;
    try {
      this.audioCtx = new AudioContext();
      // Synchronous resume() returns a Promise but starts the work
      // immediately while the gesture is still hot. We do NOT await it
      // here — that would push the work past the gesture boundary.
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume().catch(() => {});
      }
    } catch (e) {
      console.warn('Failed to prime audio-out context:', e);
    }
  }

  async start(language, mode) {
    this.preferredLanguage = language || '';
    this.mode = mode || 'translation';
    this._intentionalStop = false;
    this._reconnectAttempts = 0;
    // Reuse the context primed by primeAudioContext() in the click handler
    // if it's still live. Only build a fresh one if priming was skipped.
    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      this.audioCtx = new AudioContext();
    }
    if (this.audioCtx.state === 'suspended') {
      try { await this.audioCtx.resume(); } catch {}
    }
    this._connect();
  }

  _connect() {
    if (this._intentionalStop) return;
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    try {
      this.ws = new WebSocket(`${wsProto}//${location.host}/api/ws/audio-out`);
    } catch (e) {
      this._lastErr = `ws ctor: ${e.message}`;
      this._scheduleReconnect();
      return;
    }
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.enabled = true;
      this._reconnectAttempts = 0;  // reset backoff on a successful open
      if (this.preferredLanguage) {
        this.ws.send(JSON.stringify({ type: 'set_language', language: this.preferredLanguage }));
      }
      this.ws.send(JSON.stringify({ type: 'set_mode', mode: this.mode }));
    };

    this.ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        this._bytesIn = (this._bytesIn || 0) + evt.data.byteLength;
        this._blobsIn = (this._blobsIn || 0) + 1;
        this._queue.push(evt.data);
        if (!this._playing) this._playNext();
      } else if (evt.data instanceof Blob) {
        this._lastErr = 'blob arrival converted';
        evt.data.arrayBuffer().then((ab) => {
          this._bytesIn = (this._bytesIn || 0) + ab.byteLength;
          this._blobsIn = (this._blobsIn || 0) + 1;
          this._queue.push(ab);
          if (!this._playing) this._playNext();
        });
      }
    };

    this.ws.onerror = () => { this._lastErr = 'ws error'; };
    this.ws.onclose = (e) => {
      this.enabled = false;
      this._lastErr = `ws close ${e?.code || ''}`;
      this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    if (this._intentionalStop) return;
    if (this._reconnectTimer) return;  // already pending
    // Exponential backoff capped at 5 s, with the first retry after ~500 ms
    // so a momentary server hiccup heals nearly instantly. The user never
    // re-taps Listen unless they explicitly stop it.
    const delay = Math.min(5000, 500 * Math.pow(2, this._reconnectAttempts));
    this._reconnectAttempts++;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._connect();
    }, delay);
  }

  setLanguage(lang) {
    this.preferredLanguage = lang;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_language', language: lang }));
    }
  }

  setMode(mode) {
    this.mode = mode;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_mode', mode: mode }));
    }
  }

  async _playNext() {
    if (this._queue.length === 0) { this._playing = false; return; }
    this._playing = true;
    const wavData = this._queue.shift();

    try {
      if (!this.audioCtx) { this._lastErr = 'play: ctx null'; return; }
      if (this.audioCtx.state === 'suspended') {
        try { await this.audioCtx.resume(); } catch (e) { this._lastErr = `resume: ${e.message}`; }
      }
      const audioBuffer = await this.audioCtx.decodeAudioData(wavData.slice(0));
      if (!audioBuffer || audioBuffer.length === 0) {
        this._lastErr = 'decoded empty';
        this._playNext();
        return;
      }
      this._decoded = (this._decoded || 0) + 1;
      const source = this.audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioCtx.destination);
      source.onended = () => { this._played = (this._played || 0) + 1; this._playNext(); };
      source.start();
    } catch (e) {
      this._decodeErr = (this._decodeErr || 0) + 1;
      this._lastErr = `decode: ${e.message || e.name}`;
      console.warn('Audio-out playback error:', e);
      this._playNext();
    }
  }

  stop() {
    this._intentionalStop = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this.enabled = false;
    this._queue = [];
    this._playing = false;
    this._bytesIn = 0;
    this._blobsIn = 0;
    this._decoded = 0;
    this._decodeErr = 0;
    this._played = 0;
    this._lastErr = '';
    if (this.ws) { this.ws.close(); this.ws = null; }
    if (this.audioCtx) { this.audioCtx.close().catch(() => {}); this.audioCtx = null; }
  }
}

const audioOutListener = new AudioOutListener();

// ── Audio listener telemetry → server ──────────────────────────────────
// Push the live state of the admin Listen pipeline to /api/diag/listener
// every 2 s so an operator can see what's happening with `scripts/
// scribe_trace.py --listeners` without asking the user to read text off
// the screen. Same wire shape as guest.html's poster so both surfaces
// show up in the same trace view.
const _ADMIN_CLIENT_ID = (() => {
  let id = sessionStorage.getItem('scribe_client_id');
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : `c-${Math.random().toString(36).slice(2)}`);
    sessionStorage.setItem('scribe_client_id', id);
  }
  return id;
})();
const _ADMIN_UA_SHORT = (() => {
  const ua = navigator.userAgent || '';
  if (/iPhone|iPad|iPod/.test(ua)) return 'iOS';
  if (/Android/.test(ua)) return 'Android';
  if (/Macintosh/.test(ua)) return 'Mac';
  if (/Windows/.test(ua)) return 'Win';
  return 'Other';
})();
function _adminAudioDiagSnapshot() {
  const ctx = audioOutListener.audioCtx;
  const ws = audioOutListener.ws;
  return {
    client_id: _ADMIN_CLIENT_ID,
    page: 'admin',
    ua_short: _ADMIN_UA_SHORT,
    ctx_state: ctx ? ctx.state : 'null',
    ctx_rate: ctx ? ctx.sampleRate : 0,
    ws_state: ws ? ['CONNECTING','OPEN','CLOSING','CLOSED'][ws.readyState] : 'NULL',
    primed: !!(ctx && ctx.state === 'running'),
    queue: audioOutListener._queue?.length || 0,
    bytes_in: audioOutListener._bytesIn || 0,
    blobs_in: audioOutListener._blobsIn || 0,
    decoded: audioOutListener._decoded || 0,
    decode_err: audioOutListener._decodeErr || 0,
    played: audioOutListener._played || 0,
    last_err: audioOutListener._lastErr || '',
  };
}
setInterval(() => {
  try {
    fetch(`${API}/api/diag/listener`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_adminAudioDiagSnapshot()),
      keepalive: true,
    }).catch(() => {});
  } catch {}
}, 2000);

// ─── Timer + Helpers ─────────────────────────────────────────

class Timer {
  constructor(el) { this.el = el; this.startTime = 0; this.interval = null; }
  start() { this.startTime = Date.now(); this.interval = setInterval(() => this._tick(), 1000); }
  stop() { clearInterval(this.interval); }
  reset() { this.stop(); this.el.textContent = '00:00'; }
  _tick() {
    const s = Math.floor((Date.now() - this.startTime) / 1000);
    this.el.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }
}

function esc(text) { const d = document.createElement('div'); d.textContent = text; return d.innerHTML; }
// Meeting start wall clock (set when recording begins)
let _meetingStartWallMs = 0;

function formatTime(offsetMs) {
  if (_meetingStartWallMs > 0) {
    // Show wall clock time in browser timezone (e.g., JST)
    const wallMs = _meetingStartWallMs + (offsetMs || 0);
    const d = new Date(wallMs);
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    const s = String(d.getSeconds()).padStart(2, '0');
    return `${h}:${m}:${s}`;
  }
  // Fallback: elapsed time (mm:ss)
  const total = Math.floor((offsetMs || 0) / 1000);
  const m = String(Math.floor(total / 60)).padStart(2, '0');
  const s = String(total % 60).padStart(2, '0');
  return `${m}:${s}`;
}

// ─── Init ────────────────────────────────────────────────────

const store = new SegmentStore();
// Live-ingestion gate. Set to false by exitLiveMeetingView when the
// user navigates away from the meeting view while recording. WS
// events routed through ingestFromLiveWs are dropped while the flag
// is off; historical replay (journal fetch) keeps going through
// store.ingest directly, so rehydration is unaffected.
store._liveEnabled = true;
// Browser-test hook. Exposed only when `?test=1` is in the URL — keeps
// the store off `window` in normal sessions but lets Playwright tests
// observe `window.__test_store` and friends without monkey-patching
// modules. Used by tests/browser/test_cross_window_sync.py.
if (new URLSearchParams(location.search).get('test') === '1') {
  window.__test_store = store;
  window.__test_ingest_count = 0;
  window.__test_msg_log = [];
}
function ingestFromLiveWs(event) {
  if (window.__test_msg_log) {
    window.__test_ingest_count++;
    window.__test_msg_log.push({
      seg: event?.segment_id || null,
      type: event?.type || null,
      text: (event?.text || '').slice(0, 30),
    });
  }
  // Always dispatch a window event for non-store consumers (admin slide
  // bar, etc.) — even if store ingestion is paused. Slide-related
  // messages are control plane, not transcript data.
  try {
    window.dispatchEvent(new CustomEvent('scribe-ws-message', { detail: event }));
  } catch {}
  if (!store._liveEnabled) return;
  store.ingest(event);
}
function setStoreLive(enabled) {
  store._liveEnabled = !!enabled;
}
// Reconciler is wired up once all its primitive deps (startRecording,
// showLanding, etc.) are declared below. Until then it is null; the
// callsites inside checkStatus() guard on `if (reconciler) ...`.
let reconciler = null;
const audio = new AudioPipeline();
const timer = new Timer(document.getElementById('timer'));

// Single global subscription that delegates to active column renderers.
// Guard against null — the `#segment-count` chip lives in the admin
// meter UI; popout mode renders no such element. Without the guard
// this listener used to throw on every store fan-out, which (before
// SegmentStore's per-listener try/catch) aborted iteration and
// silently prevented every subsequent subscriber — including the
// CompactGridRenderer one — from ever running.
store.subscribe(() => {
  const el = document.getElementById('segment-count');
  if (el) el.textContent = `${store.count} segments`;
});

// Subscribe to track detected speakers — any new REAL cluster_id seen in
// an event registers in the speaker registry, which drives the
// virtual-table seat list. Pseudo-clusters (time_proximity fallbacks) are
// ignored so they don't cause a transient "Speaker 1 → Speaker 2" shuffle
// when the catch-up loop resolves them to real diarization clusters.
store.subscribe((id, evt) => {
  if (!evt || !evt.speakers?.length) return;
  const s = evt.speakers[0];
  const cid = s.cluster_id;
  if (cid == null) return;
  if (_isPseudoCluster(cid)) return;  // transient — don't register
  const wasKnown = _speakerRegistry.clusters.has(cid);
  const prevName = _speakerRegistry.clusters.get(cid)?.displayName;
  // Register / update (honors explicit names too)
  const newName = getSpeakerDisplayName(cid, s.identity || s.display_name);
  if (!wasKnown || prevName !== newName) {
    // New speaker detected OR renamed — refresh the live speaker strip
    _refreshDetectedSpeakersStrip();
    // Also refresh already-rendered transcript blocks so their speaker
    // labels reflect the current registry state
    _refreshTranscriptSpeakerLabels();
  }
});

/** Refresh the virtual-table strip with the latest speaker registry state. */
function _refreshDetectedSpeakersStrip() {
  try {
    if (typeof roomSetup !== 'undefined' && roomSetup._renderTableStrip) {
      roomSetup._renderTableStrip();
    }
  } catch (e) {
    console.warn('refresh strip failed:', e);
  }
}

/** Walk rendered transcript blocks and update their speaker labels + colors
 *  from the current speaker registry. Used after a rename so previously
 *  rendered blocks show the new name.
 */
function _refreshTranscriptSpeakerLabels() {
  document.querySelectorAll('.compact-block[data-cluster-id]').forEach(row => {
    const cid = parseInt(row.dataset.clusterId, 10);
    if (isNaN(cid)) return;
    const newName = getSpeakerDisplayName(cid);
    const speakerEl = row.querySelector('.compact-speaker');
    if (speakerEl && speakerEl.textContent !== newName) {
      speakerEl.textContent = newName;
    }
    // Color too, in case the cluster_id was re-registered
    const color = getSpeakerColor(cid);
    row.style.setProperty('--speaker-color', color);
    if (speakerEl) speakerEl.style.color = color;
  });
}

/** Open a modal to rename a speaker cluster. On confirm, POSTs to the
 *  cluster rename endpoint and updates the client-side registry.
 */
function _openSpeakerRenameModal(clusterId, currentName, color) {
  const meetingId = window._gridRenderer?._meetingId;
  const seqIndex = getSpeakerSeqIndex(clusterId);
  showModal(`
    <div class="modal-card-header"><h2>Rename speaker</h2></div>
    <form id="rename-speaker-form" style="padding:1rem 1.25rem">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem">
        <div style="width:40px;height:40px;border-radius:50%;background:${color};color:#fff;
                    display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem">
          ${seqIndex}
        </div>
        <div>
          <div style="font-size:0.85rem;font-weight:600">Currently: ${esc(currentName)}</div>
          <div style="font-size:0.7rem;color:var(--text-muted)">Cluster ${clusterId}</div>
        </div>
      </div>
      <label style="font-size:0.75rem;font-weight:500;color:var(--text-secondary);
                    display:block;margin-bottom:0.4rem">New name</label>
      <input type="text" id="rename-speaker-input" value="${esc(currentName)}"
             style="width:100%;padding:0.5rem 0.7rem;font-size:0.9rem;
                    border:1px solid var(--border);border-radius:4px;
                    background:var(--bg-surface);color:var(--text-primary)" />
      <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem">
        <button type="button" class="btn btn-ghost" onclick="closeModal()">Cancel</button>
        <button type="submit" class="btn btn-primary">Rename</button>
      </div>
    </form>
  `);
  const input = document.getElementById('rename-speaker-input');
  input?.focus();
  input?.select();
  document.getElementById('rename-speaker-form')?.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const newName = input.value.trim();
    if (!newName || newName === currentName) {
      closeModal();
      return;
    }
    // Optimistic client-side update
    renameSpeaker(clusterId, newName);
    _refreshDetectedSpeakersStrip();
    _refreshTranscriptSpeakerLabels();
    closeModal();

    // Persist to server (writes speaker_correction entries for every
    // affected segment + updates detected_speakers.json + broadcasts)
    if (meetingId) {
      try {
        const resp = await fetch(
          `${API}/api/meetings/${meetingId}/clusters/${clusterId}/name`,
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speaker_name: newName }),
          },
        );
        if (!resp.ok) {
          console.warn('Rename persistence failed:', resp.status, await resp.text());
        }
      } catch (e) {
        console.warn('Rename request failed:', e);
      }
    }
  });
}

store.subscribe((id, evt) => {
  if (window._gridRenderer) window._gridRenderer.update(id, evt);
  // Active-speaker highlight on the table strip is driven exclusively by
  // the server's speaker_pulse broadcast (every 200ms) so there is one
  // authoritative source. A previous per-event adder with a 2.5s sticky
  // timer used to race the pulse loop and leave the previous speaker lit
  // during back-and-forth — don't add a second path here.
});

let meterRaf = null, wsMessageCount = 0, audioChunkCount = 0;

function updateMeter() {
  const pct = Math.round(audio.getLevel() * 100);
  document.getElementById('meter-bar').style.width = `${pct}%`;
  document.getElementById('segment-count').textContent = `${store.count} segs | ${wsMessageCount} ws | ${audioChunkCount} chunks | ${pct}% mic`;
  if (audio.running) meterRaf = requestAnimationFrame(updateMeter);
}

async function checkStatus() {
  try {
    const resp = await fetch(`${API}/api/status`);
    const data = await resp.json();
    const pills = document.getElementById('backend-pills');
    const details = data.backend_details || {};
    const backendEntries = [
      ['ASR', 'asr', 'Speech recognition'],
      ['Translate', 'translate', 'Translation model'],
      ['Diarize', 'diarize', 'Speaker diarization'],
      ['TTS', 'tts', 'Interpretation audio'],
    ];
    // TTS health overlay — surfaces real-time lag/queue/drop state on top
    // of the basic "backend ready" flag. Computed server-side by
    // _tts_health_evaluator; this UI only renders the committed state.
    const ttsMetrics = (data.metrics && data.metrics.tts) || {};
    const ttsHealth = ttsMetrics.health || 'healthy';
    const ttsLagP95 = (ttsMetrics.end_to_end_lag_ms || {}).p95;
    const ttsDrops = ttsMetrics.drops || {};
    const ttsDropTotal = Object.values(ttsDrops).reduce((a, b) => a + (b || 0), 0);

    pills.innerHTML = backendEntries
      .map(([label, key, desc]) => {
        const d = details[key] || { ready: data.backends[key], status: data.backends[key] ? 'active' : 'down' };
        const ok = d.ready;
        let pillClass, icon, title;
        if (ok) {
          pillClass = 'active';
          icon = '';
          title = d.detail ? `${desc} — ${d.detail}` : `${desc} — active`;
        } else if (d.status === 'loading') {
          pillClass = 'loading';
          icon = ' ◐';
          title = d.detail ? `${desc} — ${d.detail}` : `${desc} — loading`;
        } else if (d.status === 'error') {
          pillClass = 'down';
          icon = ' ⚠';
          title = d.detail ? `${desc} — ERROR: ${d.detail}` : `${desc} — error`;
        } else if (d.status === 'restarting') {
          pillClass = 'loading';
          icon = ' ↻';
          title = `${desc} — restarting`;
        } else if (d.status === 'not started') {
          pillClass = 'down';
          icon = ' ✕';
          title = `${desc} — not started`;
        } else {
          pillClass = 'down';
          icon = ' ✕';
          title = d.detail ? `${desc} — ${d.detail}` : `${desc} — ${d.status || 'unavailable'}`;
        }
        // TTS health state overrides color when the backend is nominally
        // up but falling behind — degraded (amber) or stalled (red).
        if (key === 'tts' && ok) {
          if (ttsHealth === 'stalled') {
            pillClass = 'down tts-stalled';
            icon = ' ⚠';
          } else if (ttsHealth === 'degraded') {
            pillClass = 'loading tts-degraded';
            icon = ' ⚠';
          }
          const lagStr = (ttsLagP95 != null) ? ` · lag p95=${Math.round(ttsLagP95)}ms` : '';
          const q = ttsMetrics.queue_depth || 0;
          const qMax = ttsMetrics.queue_maxsize || 0;
          const busy = ttsMetrics.workers_busy || 0;
          const conc = ttsMetrics.container_concurrency || 0;
          title = `TTS — ${ttsHealth}${lagStr} · queue ${q}/${qMax} · busy ${busy}/${conc} · drops ${ttsDropTotal}`;
        }
        const detailSpan = (!ok && d.detail) ? `<span class="pill-detail">${d.detail}</span>` : '';
        return `<span class="pill ${pillClass}" title="${title}">${label}${icon}${detailSpan}</span>`;
      }).join('');

    // Stall toast: when TTS flips to "stalled", show a prominent one-line
    // warning. Auto-clears when health returns to healthy. We track the
    // last-seen state on a module-level var so transitions fire only once.
    if (window._lastTtsHealth !== ttsHealth) {
      if (ttsHealth === 'stalled') {
        const tb = document.getElementById('tts-toast');
        if (tb) {
          tb.textContent = 'TTS stalled — Listen may miss audio until backend recovers';
          tb.classList.add('visible');
        }
      } else if (window._lastTtsHealth === 'stalled' && ttsHealth === 'healthy') {
        const tb = document.getElementById('tts-toast');
        if (tb) { tb.classList.remove('visible'); tb.textContent = ''; }
      }
      window._lastTtsHealth = ttsHealth;
    }

    // ── Pre-emptive Start-button gate (W4 Fix #3) ─────────────────
    // Disable the Start button + show a "Backends warming up…" banner
    // while any REQUIRED backend (ASR or translate) is not yet ready.
    // The operator gets a visible "wait" affordance instead of clicking
    // Start during cold-start and discovering the 503 the hard way.
    // The backend's wait-with-deadline preflight covers the subtler
    // case of a backend that's "ready" by /v1/models but slow on the
    // first inference call; together they hide cold-start latency from
    // the operator entirely.
    {
      const asrReady = !!(details.asr && details.asr.ready);
      const translateReady = !!(details.translate && details.translate.ready);
      const allRequiredReady = asrReady && translateReady;
      const banner = document.getElementById('backends-warming-banner');
      const bannerDetail = document.getElementById('backends-warming-detail');
      const startBtn = document.getElementById('btn-start-meeting');
      if (banner && startBtn) {
        if (allRequiredReady) {
          banner.style.display = 'none';
          // Don't override `disabled` set by other gates (e.g. mid-start
          // double-click guard) — only re-enable if the gate set it.
          if (startBtn.dataset.gatedByWarmup === '1') {
            startBtn.disabled = false;
            delete startBtn.dataset.gatedByWarmup;
          }
        } else {
          const waiting = [];
          if (!asrReady) waiting.push(`ASR (${(details.asr && details.asr.detail) || 'loading'})`);
          if (!translateReady) waiting.push(`Translate (${(details.translate && details.translate.detail) || 'loading'})`);
          if (bannerDetail) bannerDetail.textContent = waiting.join(' · ');
          banner.style.display = '';
          startBtn.disabled = true;
          startBtn.dataset.gatedByWarmup = '1';
        }
      }
    }

    // Crash red-dot on the server pill (if backend reported an unhandled
    // exception from a background task). Sanitised — only ts/component/code.
    const crash = (data.metrics || {}).crash;
    if (crash && window._lastCrashCode !== crash.code) {
      const tb = document.getElementById('tts-toast');
      if (tb) {
        tb.textContent = `server crash in ${crash.component} · code ${crash.code}`;
        tb.classList.add('visible', 'crash');
      }
      window._lastCrashCode = crash.code;
    }
    // ── Backend detail panel ──────────────────────────────────
    const bdPanel = document.getElementById('backend-detail-panel');
    if (bdPanel) {
      bdPanel.innerHTML = backendEntries.map(([label, key, desc]) => {
        const d = details[key] || {};
        const dotCls = d.ready ? 'active' : (d.status === 'loading' ? 'loading' : (d.status === 'error' ? 'error' : 'down'));
        let rows = '';
        if (d.model) rows += `<div class="bd-row"><span class="bd-label">Model</span><span class="bd-val">${d.model}</span></div>`;
        if (d.url) rows += `<div class="bd-row"><span class="bd-label">URL</span><span class="bd-val">${d.url}</span></div>`;
        if (d.consecutive_failures > 0) rows += `<div class="bd-row"><span class="bd-label">Failures</span><span class="bd-val" style="color:#ef4444">${d.consecutive_failures}</span></div>`;
        // TTS-specific metrics
        if (key === 'tts') {
          const t = ttsMetrics;
          rows += `<div class="bd-row"><span class="bd-label">Queue</span><span class="bd-val">${t.queue_depth || 0}/${t.queue_maxsize || 0}</span></div>`;
          rows += `<div class="bd-row"><span class="bd-label">Workers</span><span class="bd-val">${t.workers_busy || 0}/${t.workers_total || 0}</span></div>`;
          rows += `<div class="bd-row"><span class="bd-label">Delivered</span><span class="bd-val">${t.delivered || 0}</span></div>`;
          if (ttsDropTotal > 0) rows += `<div class="bd-row"><span class="bd-label">Drops</span><span class="bd-val" style="color:#ef4444">${ttsDropTotal}</span></div>`;
          const lag = (t.end_to_end_lag_ms || {}).p95;
          if (lag != null) rows += `<div class="bd-row"><span class="bd-label">Lag P95</span><span class="bd-val">${Math.round(lag)}ms</span></div>`;
        }
        rows += `<div class="bd-row"><span class="bd-label">Status</span><span class="bd-val">${d.status || 'unknown'}</span></div>`;
        const errDiv = (!d.ready && d.detail) ? `<div class="bd-error">${d.detail}</div>` : '';
        return `<div class="bd-card"><div class="bd-name"><span class="bd-dot ${dotCls}"></span>${label}</div>${rows}${errDiv}</div>`;
      }).join('');
    }

    // ── System resources panel ─────────────────────────────────
    const sysPanel = document.getElementById('system-resources-panel');
    const sys = data.system;
    if (sysPanel && sys) {
      const cpuCls = sys.cpu_pct > 90 ? ' style="color:#ef4444"' : sys.cpu_pct > 70 ? ' style="color:#f59e0b"' : '';
      const memCls = sys.mem_pct > 90 ? ' style="color:#ef4444"' : sys.mem_pct > 75 ? ' style="color:#f59e0b"' : '';
      let sysHtml = `<div class="bd-card"><div class="bd-name"><span class="bd-dot active"></span>System</div>`;
      sysHtml += `<div class="bd-row"><span class="bd-label">CPU</span><span class="bd-val"${cpuCls}>${sys.cpu_pct}%</span></div>`;
      sysHtml += `<div class="bd-row"><span class="bd-label">Memory</span><span class="bd-val"${memCls}>${Math.round(sys.mem_used_mb/1024)}/${Math.round(sys.mem_total_mb/1024)} GB (${sys.mem_pct}%)</span></div>`;
      sysHtml += `<div class="bd-row"><span class="bd-label">Load</span><span class="bd-val">${sys.load.join(' / ')}</span></div>`;
      const days = Math.floor(sys.uptime_s / 86400);
      const hrs = Math.floor((sys.uptime_s % 86400) / 3600);
      sysHtml += `<div class="bd-row"><span class="bd-label">Uptime</span><span class="bd-val">${days}d ${hrs}h</span></div>`;
      sysHtml += `</div>`;

      // Per-container resource cards
      if (sys.containers && sys.containers.length > 0) {
        for (const c of sys.containers) {
          const cCpu = c.cpu_pct > 200 ? ' style="color:#ef4444"' : c.cpu_pct > 100 ? ' style="color:#f59e0b"' : '';
          sysHtml += `<div class="bd-card"><div class="bd-name">${c.name}</div>`;
          sysHtml += `<div class="bd-row"><span class="bd-label">CPU</span><span class="bd-val"${cCpu}>${c.cpu_pct}%</span></div>`;
          sysHtml += `<div class="bd-row"><span class="bd-label">Memory</span><span class="bd-val">${c.mem_mb > 1024 ? (c.mem_mb/1024).toFixed(1) + ' GB' : c.mem_mb + ' MB'}</span></div>`;
          sysHtml += `<div class="bd-row"><span class="bd-label">PIDs</span><span class="bd-val">${c.pids}</span></div>`;
          sysHtml += `</div>`;
        }
      }
      sysPanel.innerHTML = sysHtml;
    }

    // Track DEV mode state — changes stop button behavior
    _devMode = !!data.dev_mode;
    const forceStopBtn = document.getElementById('btn-force-stop');
    if (_devMode && document.body.classList.contains('recording')) {
      if (forceStopBtn) forceStopBtn.style.display = '';
    } else {
      if (forceStopBtn) forceStopBtn.style.display = 'none';
    }

    const m = data.metrics || {};
    // REQUIRED = ASR + Translation. Diarize/TTS are nice-to-have but not gating.
    // This matches the server-side gate in /api/meeting/start.
    const ready = data.backends.asr && data.backends.translate;
    btnRecord.disabled = !ready;
    if (!_devMode || !document.body.classList.contains('recording')) {
      btnRecord.title = ready
        ? 'Start recording'
        : 'Required backends not ready yet — meeting start is blocked';
    }

    // Show clear message when backends are loading (any of the 4, not just required)
    const notReady = backendEntries
      .filter(([, key]) => !(details[key] || {}).ready);
    if (notReady.length > 0) {
      const waiting = notReady
        .map(([label, key]) => {
          const d = details[key] || {};
          const err = d.status === 'error' ? ' ERROR' : '';
          return d.detail ? `${label}${err}: ${d.detail}` : `${label}${err}`;
        }).join(' · ');
      const prefix = ready
        ? 'Ready (optional backends loading)'
        : 'Waiting for backends';
      document.getElementById('status-line').textContent = `${prefix}: ${waiting}`;
    }

    // State transitions (rehydration, banner, title, ownership) are
    // delegated to the reconciler; see static/js/meeting-reconcile.js.
    // When backends aren't ready yet, keep the old status text so the
    // user sees "Warming up..." instead of a silent idle state.
    if (!(data.meeting?.state === 'recording') && !ready) {
      document.getElementById('status-line').textContent = 'Warming up...';
    } else if (!(data.meeting?.state === 'recording')) {
      document.getElementById('status-line').textContent = 'Ready';
    }
    if (reconciler) {
      await reconciler.reconcile(data);
    }
  } catch (err) { console.error('checkStatus failed:', err); document.getElementById('status-line').textContent = 'Server unreachable'; btnRecord.disabled = true; }
}

// ─── Pop-Out View Mode ──────────────────────────────────────

if (POPOUT_MODE) {
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
      </select>
      <button class="popout-btn" id="popout-qr-btn" title="Show WiFi QR code">QR</button>
      <div class="popout-qr" id="popout-qr" style="display:none"></div>
    </div>
  `;
  document.body.insertBefore(popHeader, document.querySelector('main'));

  // Initialize compact renderer for pop-out (groups by speaker turn)
  window._gridRenderer = new CompactGridRenderer(document.getElementById('transcript-grid'));

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
        (langData.languages || []).forEach(l => { _languageNames[l.code] = l; });
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
          const meetResp = await fetch(`${API}/api/meetings/${mid}`);
          const meetData = await meetResp.json();
          if (meetData.meta?.language_pair?.length >= 1) {
            currentLanguagePair = meetData.meta.language_pair.join(',');
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
      // from `currentLanguagePair`. Called on init AND on every refresh
      // poll, so the popout picks up a meeting-language change (e.g.
      // user picked Dutch but the popout was opened pre-meeting and was
      // showing the default ja↔en pair).
      function _updatePopoutLangDisplay() {
        const langLabel = document.getElementById('popout-lang-label');
        const a = _getLangA();
        const b = _getLangB();
        const nameA = _languageNames[a]?.name || a.toUpperCase();
        if (_isMonolingual()) {
          // Single language — no arrow, no B button label.
          if (langLabel) langLabel.textContent = nameA;
          return;
        }
        const nameB = _languageNames[b]?.name || b.toUpperCase();
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
          const m = await fetch(`${API}/api/meetings/${liveMid}`);
          if (!m.ok) return;
          const md = await m.json();
          const lp = md?.meta?.language_pair;
          if (Array.isArray(lp) && lp.length >= 1 && lp.length <= 2) {
            const next = lp.join(',');
            if (next !== currentLanguagePair) {
              currentLanguagePair = next;
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
        const nameA = (_languageNames[a]?.name || a.toUpperCase());
        const nameB = (_languageNames[b]?.name || b.toUpperCase());
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

      // Load QR code with retry + re-render on SSID change so this view
      // stays in sync with the main admin view during/after rotation.
      let _popoutLastSsid = null;
      async function refreshPopoutQR(retries = 15, delayMs = 2000) {
        try {
          const wifiResp = await fetch(`${API}/api/meeting/wifi`);
          if (!wifiResp.ok) {
            if (retries > 0) setTimeout(() => refreshPopoutQR(retries - 1, delayMs), delayMs);
            return;
          }
          const wifiData = await wifiResp.json();
          if (!wifiData.wifi_qr_svg) {
            if (retries > 0) setTimeout(() => refreshPopoutQR(retries - 1, delayMs), delayMs);
            return;
          }
          if (wifiData.ssid === _popoutLastSsid) return;
          _popoutLastSsid = wifiData.ssid;
          const qrEl = document.getElementById('popout-qr');
          if (qrEl) {
            qrEl.innerHTML = `
              ${wifiData.wifi_qr_svg}
              <div class="popout-qr-expanded">
                ${wifiData.wifi_qr_svg}
                <div class="wifi-qr-label">Scan to join · ${esc(wifiData.ssid || '')}</div>
              </div>
            `;
            // Re-attach click handler for QR toggle (only once)
            const btn = document.getElementById('popout-qr-btn');
            if (btn && !btn.dataset.bound) {
              btn.dataset.bound = '1';
              btn.addEventListener('click', () => {
                qrEl.style.display = qrEl.style.display === 'none' ? '' : 'none';
              });
            }
          }
        } catch (qrErr) {
          console.warn('Popout QR load failed:', qrErr);
          if (retries > 0) setTimeout(() => refreshPopoutQR(retries - 1, delayMs), delayMs);
        }
      }
      refreshPopoutQR();
      // Periodic consistency poll — match the main view's cadence so both
      // converge to the same live SSID within 10s.
      setInterval(() => refreshPopoutQR(0, 0), 10000);

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
              s.src = '/static/js/terminal-panel.js?v=1';
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
              await fetch(`${API}/api/meetings/${smid}/decks/active`, {
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
        fetch(`${API}/api/meetings/${navMid}/slides/current`, {
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
          origImg.src = `${API}/api/meetings/${smid}/slides/${_slideState.current}/${leftEndpoint}${bust}`;
          origImg.alt = `Slide in ${langA?.toUpperCase() || 'A'}`;
        }
        if (transImg) {
          transImg.src = `${API}/api/meetings/${smid}/slides/${_slideState.current}/${rightEndpoint}${bust}`;
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
          const r = await fetch(`${API}/api/meetings/${smid}/decks`);
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
              const r = await fetch(`${API}/api/meetings/${smid}/slides`);
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
              origImg.src = `${API}/api/meetings/${smid}/slides/${data.index}/${data.kind}${cb}`;
            }
          } else {
            const transImg = document.getElementById('sv-trans');
            const transStatus = document.getElementById('sv-trans-status');
            if (transImg) {
              transImg.src = `${API}/api/meetings/${smid}/slides/${data.index}/${data.kind}${cb}`;
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
            const resp = await fetch(`${API}/api/meetings/${uploadMid}/slides/upload`, { method: 'POST', body: formData });
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
              const r = await fetch(`${API}/api/meetings/${checkMid}/slides`);
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
        // of the AudioPipeline's dev_reset/cancel handling — popout
        // doesn't have an audio WS so it must do the cleanup itself.
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
                const r = await fetch(`${API}/api/meetings/${liveMid}`);
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
                msg.type === 'meeting_cancelled' ||
                msg.type === 'dev_reset'
              ) {
                _resetPopoutMeetingState(null);
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

// ─── Admin slide controls ─────────────────────────────────────────
// Lets the operator advance/upload slides from the main admin view
// without opening the popout. Server broadcasts `slide_change` so the
// popout (and every other connected viewer) follows.
//
// Use case: operator views the original (English) on their own screen,
// the popout on a guest display shows just the translated transcript +
// translated slides — but the OPERATOR clicks Next to advance.
if (!POPOUT_MODE) {
  const _adminSlideBar = document.getElementById('admin-slide-bar');
  const _adminSlideThumb = document.getElementById('admin-slide-thumb');
  const _adminSlideLabel = document.getElementById('admin-slide-label');
  const _adminSlidePrev = document.getElementById('admin-slide-prev');
  const _adminSlideNext = document.getElementById('admin-slide-next');
  const _adminSlideUploadBtn = document.getElementById('admin-slide-upload-btn');
  const _adminSlideUploadInput = document.getElementById('admin-slide-upload-input');

  const _adminSlideState = { meetingId: null, deckId: null, total: 0, current: 0 };

  function _adminSlidesGetMeetingId() {
    const cm = window._currentMeetingId;
    if (cm) return cm;
    const url = location.hash.match(/^#meeting\/(.+)/);
    return url ? url[1] : null;
  }

  function _adminSlideRefresh() {
    if (!_adminSlideThumb) return;
    if (!_adminSlideState.deckId || !_adminSlideState.meetingId) {
      _adminSlideThumb.removeAttribute('src');
      _adminSlideLabel.textContent = '— / —';
      if (_adminSlidePrev) _adminSlidePrev.disabled = true;
      if (_adminSlideNext) _adminSlideNext.disabled = true;
      return;
    }
    const bust = `?d=${_adminSlideState.deckId}`;
    _adminSlideThumb.src =
      `${API}/api/meetings/${_adminSlideState.meetingId}/slides/${_adminSlideState.current}/original${bust}`;
    _adminSlideLabel.textContent =
      `${_adminSlideState.current + 1} / ${_adminSlideState.total}`;
    if (_adminSlidePrev) _adminSlidePrev.disabled = _adminSlideState.current <= 0;
    if (_adminSlideNext) _adminSlideNext.disabled = _adminSlideState.current >= _adminSlideState.total - 1;
  }

  async function _adminSlideAdvance(delta) {
    const newIdx = _adminSlideState.current + delta;
    if (newIdx < 0 || newIdx >= _adminSlideState.total) return;
    if (!_adminSlideState.meetingId) return;
    try {
      await fetch(`${API}/api/meetings/${_adminSlideState.meetingId}/slides/current`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: newIdx }),
      });
      // Optimistic update — the broadcast will confirm.
      _adminSlideState.current = newIdx;
      _adminSlideRefresh();
    } catch (e) {
      console.warn('admin slide advance failed:', e);
    }
  }

  if (_adminSlidePrev) _adminSlidePrev.addEventListener('click', () => _adminSlideAdvance(-1));
  if (_adminSlideNext) _adminSlideNext.addEventListener('click', () => _adminSlideAdvance(1));

  if (_adminSlideUploadBtn && _adminSlideUploadInput) {
    _adminSlideUploadBtn.addEventListener('click', () => _adminSlideUploadInput.click());
    _adminSlideUploadInput.addEventListener('change', async (ev) => {
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const mid = _adminSlidesGetMeetingId();
      if (!mid) {
        alertDialog?.('No active meeting', 'Start a meeting first, then upload slides.');
        return;
      }
      const fd = new FormData();
      fd.append('file', file);
      try {
        const r = await fetch(`${API}/api/meetings/${mid}/slides/upload`, {
          method: 'POST',
          body: fd,
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${r.status}`);
        }
        // Server will broadcast slide_change events as the deck renders.
      } catch (e) {
        console.warn('admin slide upload failed:', e);
        alertDialog?.('Upload failed', String(e.message || e));
      } finally {
        _adminSlideUploadInput.value = '';
      }
    });
  }

  // Watch the global WS event stream for slide events. We attach via
  // window.addEventListener since scribe-app dispatches a custom event
  // bus for these — fall back to polling /api/meetings/{id}/slides if
  // the bus isn't there yet (defensive against load-order changes).
  function _adminHandleSlideMsg(msg) {
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'slide_deck_ready' || msg.type === 'slide_deck_changed') {
      _adminSlideState.meetingId = msg.meeting_id || _adminSlideState.meetingId || _adminSlidesGetMeetingId();
      _adminSlideState.deckId = msg.deck_id;
      _adminSlideState.total = msg.total_slides || msg.total || 0;
      _adminSlideState.current = msg.current_slide_index || 0;
      if (_adminSlideBar) _adminSlideBar.style.display = '';
      _adminSlideRefresh();
    } else if (msg.type === 'slide_change') {
      if (typeof msg.slide_index === 'number') {
        _adminSlideState.current = msg.slide_index;
        if (msg.deck_id) _adminSlideState.deckId = msg.deck_id;
        _adminSlideRefresh();
      }
    }
  }
  window.addEventListener('scribe-ws-message', (e) => _adminHandleSlideMsg(e.detail));
  // Also check current meeting on load — if a deck is already active,
  // populate the bar immediately rather than waiting for a broadcast.
  (async () => {
    try {
      const sresp = await fetch(`${API}/api/status`);
      const sd = await sresp.json();
      const mid = sd?.meeting?.id;
      if (!mid) return;
      window._currentMeetingId = mid;
      const dresp = await fetch(`${API}/api/meetings/${mid}/slides`);
      if (!dresp.ok) return;
      const dd = await dresp.json();
      if (!dd.deck_id) return;
      _adminSlideState.meetingId = mid;
      _adminSlideState.deckId = dd.deck_id;
      _adminSlideState.total = dd.total_slides || 0;
      _adminSlideState.current = dd.current_slide_index || 0;
      if (_adminSlideBar) _adminSlideBar.style.display = '';
      _adminSlideRefresh();
    } catch {}
  })();
}

const btnRecord = document.getElementById('btn-record');
let _devMode = false;

btnRecord.addEventListener('click', async () => {
  if (document.body.classList.contains('recording')) {
    const confirmed = await confirmDialog(
      'Stop meeting?',
      'Recording will end and the meeting will be finalized with a summary. You won’t be able to add more audio.',
      'Stop Meeting',
      true,
    );
    if (!confirmed) return;
    await stopRecording();
  } else {
    await startRecording(false);
  }
});

// DEV mode reset button — clears transcript without stopping the meeting
const btnForceStop = document.getElementById('btn-force-stop');
if (btnForceStop) {
  btnForceStop.addEventListener('click', async () => {
    await devResetMeeting();
  });
}

// Cancel meeting button — discard everything without finalization
const btnCancelMeeting = document.getElementById('btn-cancel-meeting');
if (btnCancelMeeting) {
  btnCancelMeeting.addEventListener('click', async () => {
    if (!document.body.classList.contains('recording')) return;
    const confirmed = await confirmDialog(
      'Cancel meeting?',
      'All audio, transcript, and data will be permanently deleted. This cannot be undone.',
      'Cancel Meeting',
      true,
    );
    if (!confirmed) return;

    btnCancelMeeting.disabled = true;
    btnCancelMeeting.textContent = 'Cancelling…';
    try {
      // Disconnect audio hardware immediately
      if (audio && audio.running) {
        audio.stop();
      }
      const resp = await fetch(`${API}/api/meeting/cancel`, { method: 'POST' });
      const data = await resp.json();
      if (resp.ok) {
        document.body.classList.remove('recording');
        document.body.classList.remove('meeting-active');
        document.body.classList.remove('starting');
        reconciler?.releaseOwnership();
        reconciler?.clearReconnectState();
        document.getElementById('control-bar').style.display = 'none';
        const statusEl = document.getElementById('status-line');
        if (statusEl) statusEl.textContent = 'Meeting cancelled';
        // Close WS connections
        if (audio && audio.ws) { audio.ws.close(); audio.ws = null; }
        stopAnalytics();
        _resetSpeakerRegistry();
        if (typeof meetingsMgr !== 'undefined' && meetingsMgr) meetingsMgr.refresh();
      } else {
        console.error('Cancel failed:', data);
        showModal(`
          <div class="modal-confirm-title">Cancel failed</div>
          <div class="modal-confirm-message">${esc(data.error || 'Unknown error')}</div>
          <div class="modal-confirm-actions">
            <button class="modal-btn" onclick="closeModal()">Close</button>
          </div>
        `, 'confirm');
      }
    } catch (e) {
      console.error('Cancel error:', e);
      showModal(`
        <div class="modal-confirm-title">Cancel failed</div>
        <div class="modal-confirm-message">${esc(e.message)}</div>
        <div class="modal-confirm-actions">
          <button class="modal-btn" onclick="closeModal()">Close</button>
        </div>
      `, 'confirm');
    } finally {
      btnCancelMeeting.disabled = false;
      btnCancelMeeting.textContent = 'Cancel';
    }
  });
}

async function validateMic() {
  const el = document.getElementById('status-line'); el.textContent = 'Checking mic...';
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext(); if (ctx.state === 'suspended') await ctx.resume();
  const src = ctx.createMediaStreamSource(stream); const an = ctx.createAnalyser(); an.fftSize = 2048; src.connect(an);
  let max = 0; const buf = new Float32Array(an.fftSize);
  for (let i = 0; i < 10; i++) { await new Promise(r => setTimeout(r, 50)); an.getFloatTimeDomainData(buf); max = Math.max(max, buf.reduce((m, v) => Math.max(m, Math.abs(v)), 0)); }
  stream.getTracks().forEach(t => t.stop()); await ctx.close();
  console.log('Mic validation: max amplitude =', max);
  if (max === 0) {
    el.textContent = 'No audio detected — check mic is selected and input volume is turned up';
    throw new Error('Mic silent (zero signal)');
  } else if (max < 0.0001) {
    el.textContent = `Mic too quiet (${(max * 100).toFixed(3)}%) — increase input volume in system settings`;
    throw new Error('Mic silent');
  }
  el.textContent = `Mic OK (${(max * 100).toFixed(1)}%)`; return true;
}

async function _resumeMeeting(meetingId) {
  const statusEl = document.getElementById('status-line');
  const btnResume = document.getElementById('btn-resume');
  try {
    // Immediate feedback
    statusEl.textContent = `Resuming ${meetingId}...`;
    if (btnResume) { btnResume.disabled = true; btnResume.textContent = 'Resuming...'; }

    const resp = await fetch(`${API}/api/meetings/${meetingId}/resume`, {method: 'POST'});
    const data = await resp.json();
    if (!data.resumed) {
      statusEl.textContent = data.error || 'Resume failed';
      if (btnResume) { btnResume.disabled = false; btnResume.textContent = 'Resume'; }
      return;
    }

    // Switch to meeting mode
    currentLanguagePair = (data.language_pair || ['en','ja']).join(',');
    _updateColumnHeaders();
    document.getElementById('landing-mode').style.display = 'none';
    document.getElementById('room-setup').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    document.getElementById('meeting-mode').style.display = '';
    document.getElementById('control-bar').style.display = '';
    document.body.classList.add('meeting-active');

    // Set up transcript renderer and load existing events
    window._gridRenderer = new CompactGridRenderer(document.getElementById('transcript-grid'), meetingId);
    statusEl.textContent = `Loading transcript...`;
    const meetResp = await fetch(`${API}/api/meetings/${meetingId}`);
    const meetData = await meetResp.json();
    if (meetData.events) {
      for (const ev of meetData.events) store.ingest(ev);
    }
    statusEl.textContent = `Starting mic...`;

    // Render table strip from saved layout
    roomSetup._renderTableStrip();

    // Start audio capture
    await startRecording(true);
    statusEl.textContent = `Recording — resumed ${meetingId}`;
  } catch (err) {
    console.error('Resume failed:', err);
    statusEl.textContent = `Resume failed: ${err.message}`;
    if (btnResume) { btnResume.disabled = false; btnResume.textContent = 'Resume'; }
  }
}

async function startRecording(isResume) {
  try {
    if (!isResume) {
      await validateMic();
      // Fresh meeting — clear the speaker registry so we don't carry
      // detected speakers from a previous meeting into this one.
      _resetSpeakerRegistry();
    }
    const resp = await fetch(`${API}/api/meeting/start`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({language_pair: currentLanguagePair})
    });
    // Read as text first so a non-JSON 500 ("Internal Server Error") gives
    // us a useful error instead of "Unexpected token I in JSON".
    const rawBody = await resp.text();
    let data;
    try {
      data = rawBody ? JSON.parse(rawBody) : {};
    } catch {
      throw new Error(
        `Server returned ${resp.status} ${resp.statusText}: ${rawBody || '(empty body)'}`,
      );
    }

    // Server refused because backends aren't deep-healthy
    if (resp.status === 503) {
      const notReady = (data.not_ready || []).map(x => {
        return x.detail ? `${x.backend}: ${x.detail}` : x.backend;
      }).join(' · ');
      const msg = data.message || 'Backends not ready';
      document.getElementById('status-line').textContent = `${msg} (${notReady})`;
      // Show a modal so the user notices
      const items = (data.not_ready || []).map(x => `
        <div class="backends-not-ready-list-item">
          <span class="backend-name">${esc(x.backend)}</span>
          <span class="backend-detail">${esc(x.detail || 'not ready')}</span>
        </div>
      `).join('');
      const banner = `
        <div class="backends-not-ready-header">
          <div class="backends-not-ready-icon">!</div>
          <h2>Can't start meeting</h2>
        </div>
        <div class="backends-not-ready-body">
          <p class="backends-not-ready-message">${esc(msg)}</p>
          <div class="backends-not-ready-list">${items}</div>
          <p class="backends-not-ready-hint">
            Wait for all backend pills in the header to turn green, then try again.
          </p>
        </div>
        <div class="backends-not-ready-footer">
          <button class="modal-btn primary" onclick="closeModal()">OK</button>
        </div>
      `;
      showModal(banner, 'backends-not-ready');
      throw new Error('Backends not ready');
    }

    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    // Update language pair from server response (may differ on resume)
    if (data.language_pair) {
      currentLanguagePair = data.language_pair.join(',');
      _updateColumnHeaders();
    }
    if (!isResume && !data.resumed) store.clear();
    wsMessageCount = 0; audioChunkCount = 0;
    document.body.classList.add('recording');
    document.getElementById('status-line').textContent = `Recording ${data.meeting_id}`;
    if (window._gridRenderer) window._gridRenderer._meetingId = data.meeting_id;
    audioPlayer.meetingId = data.meeting_id;
    window.current_meeting_id = data.meeting_id;
    _meetingStartWallMs = Date.now();
    timer.start();
    await audio.start((event) => ingestFromLiveWs(event));
    updateMeter(); startAnalytics(); refreshWifiQR();
  } catch (err) {
    console.error('Start failed:', err);
    document.getElementById('status-line').textContent = `Error: ${err.message}`;
    // Roll back to room-setup so the user can retry. RoomSetup.startMeeting
    // owns btnStart restore via its own finally; we just clean the body
    // classes and unhide room-setup here.
    document.body.classList.remove('meeting-active');
    document.body.classList.remove('starting');
    reconciler?.releaseOwnership();
    document.getElementById('meeting-mode').style.display = 'none';
    document.getElementById('control-bar').style.display = 'none';
    document.getElementById('room-setup').style.display = '';
  }
}

async function stopRecording() {
  timer.stop(); cancelAnimationFrame(meterRaf);
  document.getElementById('meter-bar').style.width = '0%';
  audio.workletNode?.disconnect(); audio.stream?.getTracks().forEach(t => t.stop());
  if (audio.audioCtx) await audio.audioCtx.close();
  audio.audioCtx = null; audio.stream = null; audio.workletNode = null; audio.analyser = null;

  // Stop audio-out listener if active
  if (audioOutListener.enabled) {
    audioOutListener.stop();
    const listenBtn = document.getElementById('btn-listen');
    if (listenBtn) listenBtn.classList.remove('active');
    const listenLang = document.getElementById('listen-lang');
    if (listenLang) listenLang.style.display = 'none';
  }

  // Get meeting ID before stopping
  const statusResp = await fetch(`${API}/api/status`);
  const statusData = await statusResp.json();
  const meetingId = statusData.meeting?.id;

  // Track this meeting as finalizing
  if (meetingId) _finalizingMeetings.set(meetingId, { step: 0, label: 'Starting...', ws: audio.ws });

  // Show finalization modal with progress steps — matches the 6 steps
  // emitted by /api/meeting/stop on the server (which is now identical
  // to the /finalize pipeline including full-audio diarization).
  const steps = [
    { label: 'Flushing speech recognition', icon: 'mic' },
    { label: 'Completing translations', icon: 'translate' },
    { label: 'Saving speaker data', icon: 'waveform' },
    { label: 'Running full-audio diarization', icon: 'speakers' },
    { label: 'Generating timeline', icon: 'timeline' },
    { label: 'Generating meeting summary', icon: 'summary' },
  ];

  const card = showModal(`
    <div class="finalize-modal">
      <div class="finalize-header">
        <div class="finalize-header-content">
          <div class="finalize-pulse"></div>
          <div>
            <h3>Finalizing Meeting</h3>
            <p class="finalize-subtitle" id="finalize-subtitle">Wrapping up — please wait</p>
          </div>
        </div>
        <button class="finalize-close" id="finalize-close-btn" title="Close">&times;</button>
      </div>
      <div class="finalize-progress-track">
        <div class="finalize-progress-fill" id="finalize-progress-fill"></div>
      </div>
      <div class="finalize-steps" id="finalize-steps">
        ${steps.map((s, i) => `
          <div class="finalize-step" data-step="${i + 1}">
            <div class="step-indicator">
              <div class="step-ring">
                <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="none" stroke-width="1.5"/></svg>
                <span class="step-check">&#10003;</span>
              </div>
              ${i < steps.length - 1 ? '<div class="step-connector"></div>' : ''}
            </div>
            <span class="step-label">${s.label}</span>
          </div>
        `).join('')}
      </div>
      <div class="finalize-eta" id="finalize-eta"></div>
      <div class="finalize-summary" id="finalize-summary" style="display:none"></div>
    </div>
  `, 'finalize');

  // Track whether we received the final step via WS
  let receivedFinalStep = false;

  // Close button handler — close modal and reload meeting view
  const closeAndReload = () => {
    closeModal();
    if (meetingId) {
      setTimeout(() => meetingsMgr?.viewMeeting(meetingId), 300);
    }
  };
  card.querySelector('#finalize-close-btn')?.addEventListener('click', closeAndReload);

  const _updateProgress = (step, { allDone = false } = {}) => {
    const pct = allDone ? 100 : Math.min(100, (step / 6) * 100);
    const fill = card.querySelector('#finalize-progress-fill');
    if (fill) fill.style.width = `${pct}%`;

    card.querySelectorAll('.finalize-step').forEach(el => {
      const s = parseInt(el.dataset.step);
      // When allDone, every step is marked done (including the final one).
      // Otherwise a step is "done" only once a LATER step has started.
      el.classList.toggle('done', allDone ? true : s < step);
      el.classList.toggle('active', allDone ? false : s === step);
    });
  };

  // Listen for progress events on the still-open WS
  if (audio.ws) {
    audio.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'finalize_progress') {
          // The server sends TWO step=6 messages: the first marks the START
          // of summary generation (no summary field yet), the second carries
          // the generated summary and signals real completion. Treat only
          // the summary-bearing message as the terminal event — otherwise
          // we close the WS before the summary arrives and the user sees
          // "Generating meeting summary" spinning forever.
          const isCompletion = msg.step >= 6 && (msg.summary !== undefined || msg.meeting_id);

          _updateProgress(msg.step, { allDone: isCompletion });

          // Update tracking map
          if (meetingId) {
            const tracker = _finalizingMeetings.get(meetingId);
            if (tracker) { tracker.step = msg.step; tracker.label = msg.label || ''; }
          }

          // Update subtitle with current step label
          const subtitle = card.querySelector('#finalize-subtitle');
          if (subtitle && msg.label) {
            subtitle.textContent = msg.label;
          }

          // Show ETA
          const etaEl = card.querySelector('#finalize-eta');
          if (etaEl && msg.eta_seconds > 0) {
            etaEl.textContent = `Estimated ${msg.eta_seconds}s remaining`;
            etaEl.classList.add('visible');
          } else if (etaEl) {
            etaEl.classList.remove('visible');
          }

          // Only finalize the modal once the completion message arrives.
          if (isCompletion) {
            receivedFinalStep = true;
            // Transition to complete state
            const pulse = card.querySelector('.finalize-pulse');
            if (pulse) pulse.classList.add('complete');
            if (subtitle) subtitle.textContent = 'Meeting finalized';

            if (msg.summary) {
              _renderFinalizationSummary(msg.summary, meetingId);
            }
            // Now safe to clean up WS
            _finalizeCleanup();
            // Modal stays open; the user dismisses it via the × close
            // button (which calls closeAndReload → viewMeeting) or the
            // "View Meeting" button inside the rendered summary.
          }
        } else {
          ingestFromLiveWs(msg);
        }
      } catch {}
    };
  }

  const _finalizeCleanup = () => {
    audio.ws?.close(); audio.ws = null; audio.running = false;
    stopAnalytics(); refreshWifiQR(); document.body.classList.remove('recording');
    document.body.classList.remove('meeting-active');
    document.body.classList.remove('starting');
    reconciler?.releaseOwnership();
    reconciler?.clearReconnectState();
    // Reset speaker registry so the next meeting starts fresh
    _resetSpeakerRegistry();
    document.getElementById('control-bar').style.display = 'none';
    document.getElementById('status-line').textContent = meetingId
      ? `Meeting complete — ${meetingId}`
      : 'Stopped';
    if (meetingId) _finalizingMeetings.delete(meetingId);
    meetingsMgr?.refresh();
  };

  // Fire stop — WS handler manages the progress; cleanup happens on step 6
  document.getElementById('status-line').textContent = 'Finalizing...';
  fetch(`${API}/api/meeting/stop`, { method: 'POST' }).then(() => {
    // HTTP response arrived — if WS already delivered step 6, nothing to do.
    // If WS died or never sent step 6, clean up after a short grace period.
    if (!receivedFinalStep) {
      setTimeout(() => {
        if (!receivedFinalStep) {
          _updateProgress(6, { allDone: true });
          const subtitle = card.querySelector('#finalize-subtitle');
          if (subtitle) subtitle.textContent = 'Meeting finalized';
          const pulse = card.querySelector('.finalize-pulse');
          if (pulse) pulse.classList.add('complete');
          _finalizeCleanup();
          // Try to load summary from API since WS didn't deliver it
          fetch(`${API}/api/meetings/${meetingId}/summary`).then(r => r.json()).then(summary => {
            if (summary && !summary.error) _renderFinalizationSummary(summary, meetingId);
          }).catch(() => {});
          // Modal stays open; user dismisses via close button or
          // "View Meeting" button inside the summary card.
        }
      }, 2000);
    }
  }).catch(() => {
    document.getElementById('status-line').textContent = 'Stop failed';
    const subtitle = card.querySelector('#finalize-subtitle');
    if (subtitle) { subtitle.textContent = 'Finalization failed'; subtitle.classList.add('error'); }
  });
}

async function devResetMeeting() {
  document.getElementById('status-line').textContent = 'DEV: Resetting…';
  try {
    const resp = await fetch(`${API}/api/meeting/dev-reset`, { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      document.getElementById('status-line').textContent = data.error || 'Reset failed';
    }
    // UI cleanup is handled by the WS 'dev_reset' broadcast handler
  } catch (e) {
    document.getElementById('status-line').textContent = 'Reset error: ' + e.message;
  }
}

function _renderFinalizationSummary(summary, meetingId) {
  const el = document.getElementById('finalize-summary');
  if (!el) return;
  if (summary.error) {
    el.innerHTML = `<p class="finalize-error">${esc(summary.error || 'Summary generation failed')}</p>`;
    el.style.display = '';
    return;
  }

  // Collapse the progress steps with animation
  const stepsEl = el.closest('.finalize-modal')?.querySelector('#finalize-steps');
  if (stepsEl) stepsEl.classList.add('collapsed');
  const etaEl = el.closest('.finalize-modal')?.querySelector('#finalize-eta');
  if (etaEl) etaEl.classList.remove('visible');

  const meta = summary.metadata || {};
  const durationMin = meta.duration_min || 0;
  const durationStr = durationMin >= 60
    ? `${Math.floor(durationMin / 60)}h ${Math.round(durationMin % 60)}m`
    : `${Math.round(durationMin)}m`;
  const numSpeakers = meta.num_speakers || (summary.speaker_stats || []).length || 0;
  const isV2 = !!summary.key_insights;

  // ── Shared components ──

  const speakerColors = ['var(--speaker-1)', 'var(--speaker-2)', 'var(--speaker-3)', 'var(--speaker-4)', '#8b6914', '#b52d2d'];
  const stats = (summary.speaker_stats || []).map((s, i) => {
    const barColor = speakerColors[i % speakerColors.length];
    return `<div class="speaker-stat" style="animation-delay:${i * 80}ms">
      <div class="speaker-stat-dot" style="background:${barColor}"></div>
      <span class="speaker-stat-name">${esc(s.name)}</span>
      <div class="speaker-stat-bar">
        <div class="speaker-stat-fill" style="width:${Math.max(3, s.pct)}%;background:${barColor}"></div>
      </div>
      <span class="speaker-stat-pct">${Math.round(s.pct)}%</span>
    </div>`;
  }).join('');

  // ── Build main content based on schema version ──

  let mainContent = '';

  if (isV2) {
    // ── V2: Rich insights, categorized actions, named entities, attributed quotes ──

    // Key Insights
    const insights = (summary.key_insights || []).map((insight, i) => {
      const speakerPills = (insight.speakers || []).map(sp =>
        `<span class="insight-speaker-pill">${esc(sp)}</span>`
      ).join('');
      // Convert newlines in description to paragraphs
      const descParas = esc(insight.description || '').split(/\n\n|\n/).filter(Boolean).map(p =>
        `<p>${p}</p>`
      ).join('');
      return `<div class="insight-card" style="animation-delay:${i * 80}ms">
        <div class="insight-header">
          <span class="insight-number">${String(i + 1).padStart(2, '0')}</span>
          <h5 class="insight-title">${esc(insight.title)}</h5>
        </div>
        <div class="insight-body">${descParas}</div>
        ${speakerPills ? `<div class="insight-speakers">${speakerPills}</div>` : ''}
      </div>`;
    }).join('');

    // Named Entities
    const entities = summary.named_entities || {};
    const entitySections = [];
    for (const [category, items] of Object.entries(entities)) {
      if (items && items.length > 0) {
        const label = category.charAt(0).toUpperCase() + category.slice(1);
        const pills = items.map(item => `<span class="entity-pill entity-${esc(category)}">${esc(item)}</span>`).join('');
        entitySections.push(`<div class="entity-group">
          <span class="entity-label">${esc(label)}</span>
          <div class="entity-pills">${pills}</div>
        </div>`);
      }
    }
    const entitiesHtml = entitySections.join('');

    // Categorized Action Items
    const actionsByCategory = {};
    for (const a of (summary.action_items || [])) {
      const cat = a.category || 'General';
      if (!actionsByCategory[cat]) actionsByCategory[cat] = [];
      actionsByCategory[cat].push(a);
    }
    let actionIdx = 0;
    const categorizedActions = Object.entries(actionsByCategory).map(([cat, items]) => {
      const itemsHtml = items.map(a => {
        actionIdx++;
        return `<li class="action-item" style="animation-delay:${actionIdx * 50}ms">
          <span class="action-check"></span>
          <div class="action-body">
            <span>${esc(a.task)}</span>
            ${a.assignee ? `<span class="action-assignee">${esc(a.assignee)}</span>` : ''}
          </div>
        </li>`;
      }).join('');
      return `<div class="action-category-group">
        <div class="action-category-header">${esc(cat)}</div>
        <ul class="action-list">${itemsHtml}</ul>
      </div>`;
    }).join('');

    // Key Quotes (attributed)
    const quotes = (summary.key_quotes || []).map((q, i) => {
      // Support both v2 object format and v1 string format
      const text = typeof q === 'string' ? q : q.text;
      const speaker = typeof q === 'object' ? q.speaker : null;
      const context = typeof q === 'object' ? q.context : null;
      return `<div class="quote-card" style="animation-delay:${i * 60}ms">
        <div class="quote-mark">&ldquo;</div>
        <div class="quote-body">
          <p class="quote-text">${esc(text || '')}</p>
          ${speaker || context ? `<div class="quote-attribution">
            ${speaker ? `<span class="quote-speaker">${esc(speaker)}</span>` : ''}
            ${context ? `<span class="quote-context">${esc(context)}</span>` : ''}
          </div>` : ''}
        </div>
      </div>`;
    }).join('');

    // Decisions
    const decisions = (summary.decisions || []).map((d, i) =>
      `<li style="animation-delay:${i * 60}ms"><span class="decision-bullet"></span>${esc(d)}</li>`
    ).join('');

    // Topics
    const topics = (summary.topics || []).map((t, i) =>
      `<li class="topic-item" style="animation-delay:${i * 60}ms">
        <span class="topic-marker">${String(i + 1).padStart(2, '0')}</span>
        <div>
          <strong>${esc(t.title)}</strong>
          <p>${esc(t.description || '')}</p>
        </div>
      </li>`
    ).join('');

    mainContent = `
      <div class="summary-scroll summary-v2">
        <div class="summary-section summary-overview">
          <h4>Executive Summary</h4>
          <p>${esc(summary.executive_summary || '')}</p>
        </div>
        ${entitiesHtml ? `<div class="summary-section summary-entities"><h4>Mentioned</h4><div class="entities-grid">${entitiesHtml}</div></div>` : ''}
        ${insights ? `<div class="summary-section summary-insights"><h4>Key Insights</h4><div class="insights-grid">${insights}</div></div>` : ''}
        ${categorizedActions ? `<div class="summary-section summary-actions-section"><h4>Action Items</h4>${categorizedActions}</div>` : ''}
        ${decisions ? `<div class="summary-section"><h4>Decisions</h4><ul class="decision-list">${decisions}</ul></div>` : ''}
        ${quotes ? `<div class="summary-section summary-quotes"><h4>Key Quotes</h4><div class="quotes-grid">${quotes}</div></div>` : ''}
        ${topics ? `<div class="summary-section"><h4>Topics</h4><ul class="topic-list">${topics}</ul></div>` : ''}
        ${stats ? `<div class="summary-section"><h4>Speaker Participation</h4><div class="speaker-stats-grid">${stats}</div></div>` : ''}
      </div>`;
  } else {
    // ── V1 fallback: simple flat rendering (kept during backfill transition) ──

    const topics = (summary.topics || []).map((t, i) =>
      `<li class="topic-item" style="animation-delay:${i * 60}ms">
        <span class="topic-marker">${String(i + 1).padStart(2, '0')}</span>
        <div>
          <strong>${esc(t.title)}</strong>
          <p>${esc(t.description || '')}</p>
        </div>
      </li>`
    ).join('');

    const decisions = (summary.decisions || []).map((d, i) =>
      `<li style="animation-delay:${i * 60}ms"><span class="decision-bullet"></span>${esc(d)}</li>`
    ).join('');

    const actions = (summary.action_items || []).map((a, i) =>
      `<li class="action-item" style="animation-delay:${i * 60}ms">
        <span class="action-check"></span>
        <div class="action-body">
          <span>${esc(a.task)}</span>
          ${a.assignee ? `<span class="action-assignee">${esc(a.assignee)}</span>` : ''}
        </div>
      </li>`
    ).join('');

    mainContent = `
      <div class="summary-scroll">
        <div class="summary-section summary-overview">
          <h4>Summary</h4>
          <p>${esc(summary.executive_summary || '')}</p>
        </div>
        ${topics ? `<div class="summary-section"><h4>Topics</h4><ul class="topic-list">${topics}</ul></div>` : ''}
        ${decisions ? `<div class="summary-section"><h4>Decisions</h4><ul class="decision-list">${decisions}</ul></div>` : ''}
        ${actions ? `<div class="summary-section"><h4>Action Items</h4><ul class="action-list">${actions}</ul></div>` : ''}
        ${stats ? `<div class="summary-section"><h4>Speaker Participation</h4><div class="speaker-stats-grid">${stats}</div></div>` : ''}
      </div>`;
  }

  // Meeting ID — click to copy.
  el.innerHTML = `
    <div class="summary-meta-bar">
      <span class="meta-chip">${durationStr}</span>
      <span class="meta-chip">${numSpeakers} speaker${numSpeakers !== 1 ? 's' : ''}</span>
      <span class="meta-chip">${meta.num_segments || '—'} segments</span>
      <span class="meta-chip">${(meta.languages || []).map(l => l.toUpperCase()).join(' / ')}</span>
      ${meetingId ? `<span class="meta-chip meta-chip-id" id="finalize-meeting-id" title="Click to copy meeting ID" style="cursor:pointer;font-family:ui-monospace,Menlo,monospace">id: ${esc(meetingId)}</span>` : ''}
    </div>
    ${mainContent}
    <div class="summary-qa" id="summary-qa">
      <div class="qa-messages" id="qa-messages"></div>
      <div class="qa-input-row">
        <input type="text" id="qa-input" placeholder="Ask about this meeting..." autocomplete="off" />
        <button class="qa-send-btn" id="qa-send">Ask</button>
      </div>
    </div>
    <div class="summary-actions">
      <div class="summary-actions-left">
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${meetingId}/export?format=md')">Markdown</button>
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${meetingId}/export?format=txt&lang=${_getLangA()}')">${_getLangA().toUpperCase()}</button>
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${meetingId}/export?format=txt&lang=${_getLangB()}')">${_getLangB().toUpperCase()}</button>
        <button class="modal-btn btn-download" onclick="window.open('/api/meetings/${meetingId}/export?format=zip')">ZIP</button>
      </div>
      <div class="summary-actions-right">
        <button class="modal-btn btn-primary" id="finalize-done-btn">View Meeting</button>
      </div>
    </div>
  `;
  el.style.display = '';

  // ── Wire up Q&A ──
  _initQaPanel(meetingId);

  // ── Wire up View Meeting button ──
  el.querySelector('#finalize-done-btn')?.addEventListener('click', () => {
    closeModal();
    if (meetingId) {
      setTimeout(() => meetingsMgr?.viewMeeting(meetingId), 300);
    }
  });

  // Click-to-copy for the meeting-id chip
  const idChip = el.querySelector('#finalize-meeting-id');
  if (idChip && meetingId) {
    idChip.addEventListener('click', async () => {
      const originalText = idChip.textContent;
      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(meetingId);
        } else {
          const ta = document.createElement('textarea');
          ta.value = meetingId;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          ta.remove();
        }
        idChip.textContent = 'copied ✓';
        setTimeout(() => { idChip.textContent = originalText; }, 1200);
      } catch (e) {
        idChip.textContent = 'copy failed';
        setTimeout(() => { idChip.textContent = originalText; }, 1200);
      }
    });
  }
}

// ─── Q&A Panel ───────────────────────────────────────────────

function _initQaPanel(meetingId) {
  const input = document.getElementById('qa-input');
  const sendBtn = document.getElementById('qa-send');
  const messagesEl = document.getElementById('qa-messages');
  if (!input || !sendBtn || !messagesEl) return;

  let qaInFlight = false;

  const submitQuestion = async () => {
    const question = input.value.trim();
    if (!question || qaInFlight) return;

    // Show user message
    const userMsg = document.createElement('div');
    userMsg.className = 'qa-msg qa-msg-user';
    userMsg.textContent = question;
    messagesEl.appendChild(userMsg);
    input.value = '';
    qaInFlight = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '...';

    // Create assistant message bubble
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'qa-msg qa-msg-assistant';
    assistantMsg.textContent = '';
    messagesEl.appendChild(assistantMsg);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
      const resp = await fetch(`${API}/api/meetings/${meetingId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: 'Request failed' }));
        assistantMsg.textContent = err.error || 'Request failed';
        assistantMsg.classList.add('qa-msg-error');
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'chunk') {
              assistantMsg.textContent += data.text;
              messagesEl.scrollTop = messagesEl.scrollHeight;
            } else if (data.type === 'error') {
              assistantMsg.textContent = data.text;
              assistantMsg.classList.add('qa-msg-error');
            }
          } catch {}
        }
      }
    } catch (e) {
      assistantMsg.textContent = 'Connection error: ' + e.message;
      assistantMsg.classList.add('qa-msg-error');
    } finally {
      qaInFlight = false;
      sendBtn.disabled = false;
      sendBtn.textContent = 'Ask';
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  };

  sendBtn.addEventListener('click', submitQuestion);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuestion();
    }
  });
}


/**
 * Re-show the finalization summary modal for any completed meeting.
 * Useful for reviewing past meetings or regenerating summaries.
 */
async function showFinalizationSummaryFor(meetingId) {
  const card = showModal(`
    <div class="finalize-modal">
      <div class="finalize-header">
        <div class="finalize-header-content">
          <div class="finalize-pulse complete"></div>
          <div>
            <h3>Meeting Summary</h3>
            <p class="finalize-subtitle" id="finalize-subtitle">Loading...</p>
          </div>
        </div>
        <div class="finalize-header-right" style="display:flex;align-items:center;gap:0.5rem">
          <div class="finalize-lang-toggle" id="finalize-lang-toggle" role="tablist" aria-label="Summary language" style="display:none;gap:2px;border:1px solid var(--border);border-radius:6px;overflow:hidden"></div>
          <button class="finalize-close" id="finalize-close-btn" title="Close">&times;</button>
        </div>
      </div>
      <div class="finalize-summary" id="finalize-summary"></div>
    </div>
  `, 'finalize');
  card.querySelector('#finalize-close-btn')?.addEventListener('click', closeModal);

  const subtitle = card.querySelector('#finalize-subtitle');
  const summaryEl = card.querySelector('#finalize-summary');
  const langToggle = card.querySelector('#finalize-lang-toggle');

  // Resolve the meeting's language pair so we can render a toggle.
  // Cached locally — we render once and rebind on every fetch result.
  let _meetingPair = null;
  try {
    const mr = await fetch(`${API}/api/meetings/${meetingId}`);
    if (mr.ok) {
      const md = await mr.json();
      const lp = md?.meta?.language_pair;
      if (Array.isArray(lp) && lp.length === 2) _meetingPair = lp;
    }
  } catch {}

  let _activeLang = null; // null = default summary language
  function _renderLangToggle() {
    if (!langToggle || !_meetingPair) return;
    langToggle.style.display = '';
    langToggle.innerHTML = '';
    const opts = [{ code: null, label: 'Default' }, ..._meetingPair.map((c) => ({ code: c, label: (_languageNames[c]?.name || c.toUpperCase()) }))];
    for (const opt of opts) {
      const btn = document.createElement('button');
      btn.className = 'popout-btn popout-lang-btn';
      btn.textContent = opt.label;
      btn.style.cssText = 'border:none;border-right:1px solid var(--border);border-radius:0;padding:2px 10px;font-size:0.7rem;background:none;cursor:pointer';
      if ((_activeLang || null) === (opt.code || null)) {
        btn.style.background = 'var(--text-primary)';
        btn.style.color = 'var(--bg-surface)';
        btn.style.fontWeight = '600';
      }
      btn.addEventListener('click', () => _loadSummary(opt.code));
      langToggle.appendChild(btn);
    }
  }

  async function _loadSummary(lang) {
    _activeLang = lang;
    _renderLangToggle();
    if (subtitle) subtitle.textContent = lang ? `Translating to ${(_languageNames[lang]?.name || lang.toUpperCase())}…` : 'Loading…';
    summaryEl.innerHTML = '<p style="padding:2rem;text-align:center;color:var(--text-secondary)">Working…</p>';
    const url = `${API}/api/meetings/${meetingId}/summary${lang ? `?lang=${encodeURIComponent(lang)}` : ''}`;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      const summary = await r.json();
      if (!summary || summary.error) throw new Error(summary?.error || 'No summary');
      if (subtitle) subtitle.textContent = lang ? `Translated · ${(_languageNames[lang]?.name || lang.toUpperCase())}` : 'Meeting finalized';
      summaryEl.innerHTML = '';
      _renderFinalizationSummary(summary, meetingId);
    } catch (e) {
      summaryEl.innerHTML = `<p class="finalize-error">${esc(String(e.message || e))}</p>`;
    }
  }

  try {
    const resp = await fetch(`${API}/api/meetings/${meetingId}/summary`);
    if (resp.ok) {
      const summary = await resp.json();
      if (summary && !summary.error) {
        if (subtitle) subtitle.textContent = 'Meeting finalized';
        _renderLangToggle();
        _renderFinalizationSummary(summary, meetingId);
        return;
      }
    }
    // Summary missing or errored — offer to regenerate
    if (subtitle) subtitle.textContent = 'No summary available';
    summaryEl.style.display = '';
    summaryEl.innerHTML = `
      <div class="summary-actions">
        <p style="padding:1rem;color:var(--text-secondary)">
          This meeting doesn't have a summary yet. Generate one now?
        </p>
        <div class="summary-actions-right">
          <button class="modal-btn btn-primary" id="finalize-regenerate-btn">Generate Summary</button>
        </div>
      </div>
    `;
    card.querySelector('#finalize-regenerate-btn')?.addEventListener('click', async () => {
      if (subtitle) subtitle.textContent = 'Generating summary...';
      summaryEl.innerHTML = '<p style="padding:2rem;text-align:center;color:var(--text-secondary)">Working...</p>';
      try {
        const finalizeResp = await fetch(
          `${API}/api/meetings/${meetingId}/finalize?force=true`,
          { method: 'POST' }
        );
        if (finalizeResp.ok) {
          const sumResp = await fetch(`${API}/api/meetings/${meetingId}/summary`);
          if (sumResp.ok) {
            const summary = await sumResp.json();
            if (summary && !summary.error) {
              if (subtitle) subtitle.textContent = 'Meeting finalized';
              _renderFinalizationSummary(summary, meetingId);
              return;
            }
          }
        }
        summaryEl.innerHTML = '<p class="finalize-error">Summary generation failed</p>';
      } catch (e) {
        summaryEl.innerHTML = `<p class="finalize-error">Error: ${esc(String(e))}</p>`;
      }
    });
  } catch (e) {
    if (subtitle) subtitle.textContent = 'Failed to load summary';
    summaryEl.style.display = '';
    summaryEl.innerHTML = `<p class="finalize-error">${esc(String(e))}</p>`;
  }
}

// Expose globally for inline onclick handlers
window.showFinalizationSummaryFor = showFinalizationSummaryFor;


// ── Meeting tools modal ──────────────────────────────────────
// Houses Re-diarize, Reprocess, Versions, Delete behind a single ⋯
// button on each meeting row. Each action is presented with a clear
// description, an estimate of how long it takes, and a confirm step
// for destructive ones. Keeps the row clean and the consequences obvious.
function _openMeetingToolsModal(meeting, mgr) {
  const m = meeting || {};
  const card = showModal(`
    <div class="finalize-modal">
      <div class="finalize-header">
        <div class="finalize-header-content">
          <div>
            <h3>Meeting actions</h3>
            <p class="finalize-subtitle">${esc(m.meeting_id || '')}</p>
          </div>
        </div>
        <button class="finalize-close" id="tools-close-btn" title="Close">&times;</button>
      </div>
      <div class="finalize-summary" id="tools-body">
        <div class="meeting-tool-list">
          <div class="meeting-tool-item" data-action="rediarize">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Re-diarize speakers</div>
              <div class="meeting-tool-desc">
                Re-runs full-audio diarization + speaker consolidation on the existing
                transcript. Use when speaker labels look wrong (over-clustered, missing
                speakers, fragments). Optional: pin the expected speaker count.
                <span class="meeting-tool-meta">~2-3 min for 60-min audio · keeps a snapshot</span>
              </div>
            </div>
            <button class="modal-btn" data-action="rediarize-go">Run</button>
          </div>
          <div class="meeting-tool-item" data-action="reprocess">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Full reprocess from raw audio</div>
              <div class="meeting-tool-desc">
                Re-runs ASR + translation + diarization end-to-end from
                <code>recording.pcm</code>. Use when transcript text quality is poor
                (e.g. wrong language detected). Auto-snapshots the current run so you
                can compare via Versions afterwards.
                <span class="meeting-tool-meta">~10-15 min for 60-min audio · destructive but versioned</span>
              </div>
            </div>
            <button class="modal-btn" data-action="reprocess-go">Run</button>
          </div>
          <div class="meeting-tool-item" data-action="versions">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Versions / Compare runs</div>
              <div class="meeting-tool-desc">
                Each reprocess auto-snapshots the prior outputs. View past versions and
                see a per-dimension diff (segment count, language tags, translation
                coverage, speaker count, summary structure) so you can judge whether a
                change actually improved quality.
              </div>
            </div>
            <button class="modal-btn" data-action="versions-go">Open</button>
          </div>
          <div class="meeting-tool-item meeting-tool-danger" data-action="delete">
            <div class="meeting-tool-text">
              <div class="meeting-tool-title">Delete meeting</div>
              <div class="meeting-tool-desc">
                Permanently removes the meeting directory — transcript, audio, slides,
                summary, all snapshots. Cannot be undone.
              </div>
            </div>
            <button class="modal-btn modal-btn-danger" data-action="delete-go">Delete</button>
          </div>
        </div>
      </div>
    </div>
  `, 'finalize');
  card.querySelector('#tools-close-btn')?.addEventListener('click', closeModal);

  async function _runRediarize() {
    const raw = await promptDialog(
      'Re-diarize meeting',
      'Pin a speaker count when known (recommended for over-clustered meetings). Leave blank to let the model decide.',
      {
        placeholder: 'Speaker count (1–12) or blank',
        confirmText: 'Re-diarize',
        type: 'number',
        inputMode: 'numeric',
        min: 1,
        max: 12,
        help: 'Runs diarization + speaker consolidation on the existing transcript. Keeps a snapshot.',
      }
    );
    if (raw === null) return;
    const expected = raw === '' ? null : parseInt(raw, 10);
    if (raw !== '' && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
      // Validation failed — alert stacks on top of tools modal so the
      // user drops back here after dismissing and can pick something else.
      await alertDialog('Invalid count', 'Speaker count must be a number between 1 and 12, or blank.');
      return;
    }
    // Inputs validated. Dismiss the tools modal before the long-running
    // fetch so the UI isn't frozen behind a dialog for 2–3 minutes; the
    // completion / failure alert at the end is the only modal the user
    // needs to see for this operation.
    closeAllModals();
    try {
      const qs = expected != null ? `?expected_speakers=${expected}` : '';
      const resp = await fetch(`${API}/api/meetings/${m.meeting_id}/finalize${qs}`, { method: 'POST' });
      if (!resp.ok) throw new Error(await resp.text());
      const result = await resp.json();
      mgr?.refresh();
      await alertDialog(
        'Re-diarize complete',
        `${result?.diarization?.unique_speakers ?? '?'} speakers detected from ${result?.diarization?.segments ?? '?'} diarize segments.`,
      );
    } catch (err) {
      await alertDialog('Re-diarize failed', String(err.message || err));
    }
  }

  async function _runReprocess() {
    const raw = await promptDialog(
      'Full reprocess from raw audio',
      'Re-runs ASR + translation + diarization for a higher-quality transcript. Slow: about 10–15 minutes for a 60-minute meeting. Snapshots the current journal automatically.',
      {
        placeholder: 'Speaker count (1–12) or blank',
        confirmText: 'Reprocess',
        type: 'number',
        inputMode: 'numeric',
        min: 1,
        max: 12,
        help: 'Pin a speaker count when known, or leave blank to let pyannote decide.',
      }
    );
    if (raw === null) return;
    const expected = raw === '' ? null : parseInt(raw, 10);
    if (raw !== '' && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
      await alertDialog('Invalid count', 'Speaker count must be a number between 1 and 12, or blank.');
      return;
    }
    // Dismiss the tools modal before the long (10–15 min) fetch so
    // the rest of the UI is usable while the server works. The final
    // completion / failure alert is the user's only required feedback.
    closeAllModals();
    try {
      const qs = expected != null ? `?expected_speakers=${expected}` : '';
      const resp = await fetch(`${API}/api/meetings/${m.meeting_id}/reprocess${qs}`, { method: 'POST' });
      if (!resp.ok) throw new Error(await resp.text());
      const result = await resp.json();
      mgr?.refresh();
      const segs = result?.segments ?? '?';
      const tr = result?.translated ?? '?';
      const sp = result?.speakers ?? '?';
      await alertDialog(
        'Reprocess complete',
        `${segs} segments, ${tr} translated, ${sp} speakers detected. The previous run is saved as a version; open Tools > Versions to compare.`,
      );
    } catch (err) {
      await alertDialog('Reprocess failed', String(err.message || err));
    }
  }

  async function _showVersions() {
    closeAllModals();
    let resp;
    try {
      resp = await fetch(`${API}/api/meetings/${m.meeting_id}/versions`);
    } catch (e) {
      await alertDialog('Versions unavailable', String(e));
      return;
    }
    const data = await resp.json().catch(() => ({}));
    const versions = (data && data.versions) || [];
    if (!versions.length) {
      await alertDialog(
        'No versions yet',
        'Run a reprocess first. Each reprocess auto-snapshots the prior run so you can compare them here.',
      );
      return;
    }
    let diffHtml = '';
    try {
      const dResp = await fetch(`${API}/api/meetings/${m.meeting_id}/versions/diff`);
      if (dResp.ok) {
        const dData = await dResp.json();
        const dims = (dData && dData.diff && dData.diff.dimensions) || {};
        const totals = (dData && dData.diff && dData.diff.totals) || {};
        const sign = { better: '▲', worse: '▼', same: '·' };
        const color = { better: '#10b981', worse: '#ef4444', same: '#9a9aa2' };
        const rows = Object.entries(dims).map(([k, v]) => {
          const pct = (v.delta_rel * 100).toFixed(1);
          return `<tr>
            <td style="padding:0.25rem 0.5rem;color:${color[v.verdict]}">${sign[v.verdict]}</td>
            <td style="padding:0.25rem 0.5rem;font-family:var(--font-mono);font-size:0.78rem">${esc(k)}</td>
            <td style="padding:0.25rem 0.5rem;text-align:right;font-family:var(--font-mono)">${esc(String(v.baseline))}</td>
            <td style="padding:0.25rem 0.5rem;text-align:center;color:#9a9aa2">→</td>
            <td style="padding:0.25rem 0.5rem;text-align:right;font-family:var(--font-mono)">${esc(String(v.compare))}</td>
            <td style="padding:0.25rem 0.5rem;text-align:right;color:${color[v.verdict]};font-family:var(--font-mono)">${pct}%</td>
          </tr>`;
        }).join('');
        diffHtml = `
          <div style="margin-top:1rem">
            <div style="font-weight:600;margin-bottom:0.4rem">Diff: ${esc(dData.baseline)} → ${esc(dData.compare)}</div>
            <table style="width:100%;border-collapse:collapse;font-size:0.78rem;">${rows}</table>
            <div style="margin-top:0.5rem;font-size:0.75rem;color:var(--text-secondary)">
              <span style="color:#10b981">▲ ${totals.better||0} better</span> ·
              <span style="color:#ef4444">▼ ${totals.worse||0} worse</span> ·
              <span style="color:#9a9aa2">· ${totals.same||0} same</span>
            </div>
          </div>`;
      }
    } catch {}
    const versionsHtml = versions.map(v => {
      const m = v.manifest || {};
      const inputs = m.inputs || {};
      return `<div style="padding:0.4rem 0.5rem;border-bottom:1px solid var(--border);font-size:0.78rem">
        <div style="font-family:var(--font-mono);font-weight:600">${esc(v.name)}</div>
        <div style="color:var(--text-secondary);margin-top:0.15rem">
          ${esc(m.snapshot_at_utc || '')} · git=${esc((m.git_commit||'').slice(0,8) || '-')} · pair=${esc((inputs.language_pair||[]).join(','))} · expected_speakers=${esc(String(inputs.expected_speakers || '-'))}
        </div>
      </div>`;
    }).join('');
    showModal(`
      <div class="finalize-modal">
        <div class="finalize-header">
          <div class="finalize-header-content"><div><h3>Versions</h3><p class="finalize-subtitle">${esc(m.meeting_id || '')}</p></div></div>
          <button class="finalize-close" onclick="closeModal()" title="Close">&times;</button>
        </div>
        <div class="finalize-summary">
          ${diffHtml}
          <div style="margin-top:1rem;font-weight:600">Snapshots (${versions.length})</div>
          ${versionsHtml}
        </div>
      </div>
    `, 'finalize');
  }

  async function _confirmDelete() {
    const go = await confirmDialog(
      'Delete meeting?',
      `<code style="font-family:var(--font-mono);font-size:0.82em">${esc(m.meeting_id || '')}</code><br><br>This permanently removes the transcript, audio, slides, summary, and all snapshots. The action cannot be undone.`,
      'Delete',
      true,
    );
    if (!go) return;
    // Drop the whole stack — tools modal would show a stale meeting
    // row between the fetch starting and mgr.refresh() removing it.
    closeAllModals();
    try {
      const r = await fetch(`${API}/api/meetings/${m.meeting_id}`, { method: 'DELETE' });
      if (!r.ok) throw new Error(await r.text());
      mgr?.refresh();
    } catch (e) {
      await alertDialog('Delete failed', String(e.message || e));
      return;
    }
  }

  card.querySelector('[data-action="rediarize-go"]').addEventListener('click', _runRediarize);
  card.querySelector('[data-action="reprocess-go"]').addEventListener('click', _runReprocess);
  card.querySelector('[data-action="versions-go"]').addEventListener('click', _showVersions);
  card.querySelector('[data-action="delete-go"]').addEventListener('click', _confirmDelete);
}

document.getElementById('btn-clear').addEventListener('click', () => { if (window._gridRenderer) window._gridRenderer._clear(true); store.clear(); timer.reset(); });

// Transcript scroll direction toggle — persisted in localStorage
(function() {
  const btn = document.getElementById('btn-scroll-dir');
  if (!btn) return;
  // Restore saved preference (default: newest first, matches CompactGridRenderer init)
  const saved = localStorage.getItem('scribe_transcript_direction');
  if (saved === 'oldest') {
    btn.textContent = '↓ Oldest first';
    btn.dataset.direction = 'oldest';
    // Apply after gridRenderer is initialized (it defaults to newestFirst=true)
    setTimeout(() => {
      if (window._gridRenderer && window._gridRenderer._newestFirst) {
        window._gridRenderer.toggleDirection();
      }
    }, 100);
  } else {
    btn.textContent = '↑ Newest first';
    btn.dataset.direction = 'newest';
  }
  btn.addEventListener('click', () => {
    if (!window._gridRenderer) return;
    const newestFirst = window._gridRenderer.toggleDirection();
    btn.textContent = newestFirst ? '↑ Newest first' : '↓ Oldest first';
    btn.dataset.direction = newestFirst ? 'newest' : 'oldest';
    localStorage.setItem('scribe_transcript_direction', newestFirst ? 'newest' : 'oldest');
  });
})();

let analyticsInterval = null;
function startAnalytics() {
  analyticsInterval = setInterval(async () => {
    try {
      const data = (await (await fetch(`${API}/api/status`)).json());
      // Update the togglable metrics dashboard if it's visible. The header
      // has no inline metric chips anymore — everything lives in the
      // dashboard panel (live meetings) or the finalization modal (past).
      if (document.getElementById('metrics-dashboard').style.display !== 'none') {
        updateMetricsDashboard(data);
      }
    } catch {}
  }, 2000);
}
function stopAnalytics() { clearInterval(analyticsInterval); }

// ─── Reconciler wiring + tiered /api/status polling ───────────

// Inject or locate the sticky "Meeting in progress" / "Reconnecting"
// banner. Styled by static/css/style.css (.meeting-banner and its
// .return / .reconnecting / .visible modifiers).
function _ensureBannerEl() {
  let el = document.getElementById('meeting-banner');
  if (el) return el;
  el = document.createElement('div');
  el.id = 'meeting-banner';
  el.className = 'meeting-banner';
  el.setAttribute('role', 'status');
  el.setAttribute('aria-live', 'polite');
  el.innerHTML = `
    <span class="meeting-banner-dot" aria-hidden="true"></span>
    <span class="meeting-banner-label"></span>
    <button class="meeting-banner-btn" type="button"></button>
  `;
  document.body.appendChild(el);
  return el;
}

function _showBanner(state) {
  const el = _ensureBannerEl();
  if (!state) {
    el.classList.remove('visible', 'return', 'reconnecting');
    el.querySelector('.meeting-banner-btn').onclick = null;
    return;
  }
  el.classList.add('visible');
  el.classList.toggle('return', state.mode === 'return');
  el.classList.toggle('reconnecting', state.mode === 'reconnecting');
  el.querySelector('.meeting-banner-label').textContent = state.label || '';
  const btn = el.querySelector('.meeting-banner-btn');
  if (state.button) {
    btn.textContent = state.button;
    btn.style.display = '';
    btn.setAttribute('aria-hidden', 'false');
    btn.onclick = state.onClick || null;
  } else {
    btn.style.display = 'none';
    btn.setAttribute('aria-hidden', 'true');
    btn.onclick = null;
  }
}

function _getAudioWsState() {
  const ws = audio && audio.ws;
  if (!ws) return 'closed';
  switch (ws.readyState) {
    case WebSocket.OPEN: return 'open';
    case WebSocket.CONNECTING: return 'connecting';
    default: return 'closed';
  }
}

// View-only WS lifecycle. Tracked so the reconciler can attach/detach
// on take-over and upgrade paths without knowing the WS protocol.
let _viewOnlyWs = null;
let _viewOnlyKeepAlive = null;
function _attachViewOnlyWs(meetingId) {
  if (_viewOnlyWs && _viewOnlyWs.readyState !== WebSocket.CLOSED) return;
  VIEW_ONLY = true;
  document.body.classList.add('view-only');
  const viewWs = new WebSocket(`${WS_PROTO}//${location.host}/api/ws/view`);
  viewWs.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'speaker_rename' && msg.cluster_id != null) {
        renameSpeaker(msg.cluster_id, msg.display_name);
        _refreshDetectedSpeakersStrip();
        _refreshTranscriptSpeakerLabels();
      } else {
        ingestFromLiveWs(msg);
      }
    } catch {}
  };
  viewWs.onopen = () => {
    document.getElementById('status-line').textContent = `Viewing ${meetingId} (read-only)`;
  };
  _viewOnlyWs = viewWs;
  _viewOnlyKeepAlive = setInterval(() => {
    if (viewWs.readyState === WebSocket.OPEN) viewWs.send('ping');
    else { clearInterval(_viewOnlyKeepAlive); _viewOnlyKeepAlive = null; }
  }, 30000);
}
function _detachViewOnlyWs() {
  if (_viewOnlyWs) { _viewOnlyWs.close(); _viewOnlyWs = null; }
  if (_viewOnlyKeepAlive) { clearInterval(_viewOnlyKeepAlive); _viewOnlyKeepAlive = null; }
  VIEW_ONLY = false;
  document.body.classList.remove('view-only');
}

async function _loadMeetingJournal(meetingId) {
  const resp = await fetch(`${API}/api/meetings/${meetingId}`);
  const data = await resp.json();
  if (data.meta?.created_at) {
    _meetingStartWallMs = new Date(data.meta.created_at).getTime();
  }
  if (data.events) {
    // Direct store.ingest — bypasses the live-WS gate since this is
    // authoritative journal replay, not a live event stream.
    for (const ev of data.events) store.ingest(ev);
  }
  return data;
}

function _resetLiveStore() {
  store.clear();
  _resetSpeakerRegistry();
  window._gridRenderer = new CompactGridRenderer(document.getElementById('transcript-grid'));
}

function _showMeetingMode() {
  document.getElementById('landing-mode').style.display = 'none';
  document.getElementById('room-setup').style.display = 'none';
  document.getElementById('view-mode').style.display = 'none';
  document.getElementById('meeting-mode').style.display = '';
  document.getElementById('control-bar').style.display = '';
  document.body.classList.add('hide-table');
}

reconciler = createReconciler({
  doc: document,
  storage: window.sessionStorage,
  fetchFn: (...a) => fetch(...a),
  getAudioWsState: _getAudioWsState,
  startRecording: (resume) => startRecording(resume),
  attachViewOnlyWs: _attachViewOnlyWs,
  detachViewOnlyWs: _detachViewOnlyWs,
  loadMeetingJournal: _loadMeetingJournal,
  resetStore: _resetLiveStore,
  setStoreLive,
  renderTableStrip: () => roomSetup._renderTableStrip(),
  showMeetingMode: _showMeetingMode,
  showBanner: _showBanner,
  setTitle: (t) => { document.title = t; },
  onFinalizeCleanup: () => {
    // The live meeting ended server-side. Clear client-side pipeline.
    try { audio.ws?.close(); } catch {}
    if (audio) { audio.ws = null; audio.running = false; }
    _detachViewOnlyWs();
    document.body.classList.remove('recording', 'meeting-active', 'starting');
  },
  apiBase: API,
  popoutMode: POPOUT_MODE,
});
window._reconciler = reconciler;  // dev-console introspection

// Tiered status polling — 2 s while recording/starting, 10 s idle.
// Paused when the tab is hidden; immediate tick on visibilityresume.
let _pollTimer = null;
function _scheduleNextStatusTick() {
  if (document.hidden || POPOUT_MODE) return;
  const busy = document.body.classList.contains('recording')
            || document.body.classList.contains('starting');
  const delay = busy ? 2000 : 10000;
  _pollTimer = setTimeout(_statusTick, delay);
}
async function _statusTick() {
  try { await checkStatus(); } catch {}
  _scheduleNextStatusTick();
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) return;
  if (_pollTimer) { clearTimeout(_pollTimer); _pollTimer = null; }
  _statusTick();
});

if (!POPOUT_MODE) { checkStatus().then(_scheduleNextStatusTick); }

// ─── beforeunload warning ────────────────────────────────────
// Warn only while recording or mid-start. Server-side finalization
// completes regardless of browser presence so no warning then.
if (!POPOUT_MODE) {
  window.addEventListener('beforeunload', (e) => {
    if (document.body.classList.contains('recording')
     || document.body.classList.contains('starting')) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
}

// ─── Landing Page Navigation ─────────────────────────────────

function showLanding() {
  document.getElementById('landing-mode').style.display = '';
  document.getElementById('room-setup').style.display = 'none';
  document.getElementById('meeting-mode').style.display = 'none';
  document.getElementById('view-mode').style.display = 'none';
  document.getElementById('speaker-timeline').style.display = 'none';
  document.getElementById('meeting-summary-panel').style.display = 'none';
  document.getElementById('player-bar').style.display = 'none';
  // Release the warmed-up mic so the browser indicator turns off when the
  // user backs out of setup without starting a meeting.
  micWarmup.release();
  // Don't clobber the status line here — checkStatus() polls backend state
  // every 2s and owns the text so it stays consistent whether the page was
  // loaded at `/` or navigated to via a click on the Home button (`/#home`).
}

function hideLanding() {
  document.getElementById('landing-mode').style.display = 'none';
}

// Leave the live meeting view while the meeting is still running on
// the server. Suppresses live ingestion so landing / review / setup
// can't be corrupted by incoming WS events, but does NOT touch the
// AudioPipeline / audio.ws — the meeting keeps recording. The
// reconciler's return banner will drive the user back when they want
// to resume the live view; enterLiveMeetingMode re-enables
// ingestion and re-fetches the journal so the rebuild is clean.
function exitLiveMeetingView() {
  setStoreLive(false);
  document.body.classList.remove('meeting-active');
  document.body.classList.add('off-meeting-view');
  document.getElementById('meeting-mode').style.display = 'none';
}

// Home icon — always navigates to the landing page. Disabled during an
// active recording so users don't accidentally drop the meeting UI.
document.getElementById('btn-home')?.addEventListener('click', (e) => {
  e.preventDefault();
  // Navigation is always allowed — even while recording. The "Meeting
  // in progress" banner is the return affordance; exitLiveMeetingView
  // keeps the live pipeline alive while the user is off the meeting
  // view, and detaches rendering so events don't corrupt other views.
  if (document.body.classList.contains('recording')
   || document.body.classList.contains('meeting-active')) {
    exitLiveMeetingView();
  }
  history.pushState(null, '', '#home');
  showLanding();
});

// Header pulse icon — shortcut to start or stop a recording. When idle
// it navigates to /#setup (the room layout where the user configures
// and starts a meeting). When a meeting is already recording it
// dispatches to the existing control-bar record button so the stop
// path stays unified in one place.
document.getElementById('btn-logo-record')?.addEventListener('click', (e) => {
  e.preventDefault();
  if (document.body.classList.contains('recording')) {
    document.getElementById('btn-record')?.click();
    return;
  }
  history.pushState(null, '', '#setup');
  meetingsMgr?.showSetup();
});

document.getElementById('landing-start-btn')?.addEventListener('click', () => {
  hideLanding();
  history.pushState(null, '', '#setup');
  meetingsMgr?.showSetup();
});

// Quick-start English: skip the setup screen entirely and fire a
// monolingual English meeting. Useful for the common "I just want to
// record a solo session" case where picking seats and enrolling voices
// is overkill. Sets ``currentLanguagePair`` to "en" before calling
// ``roomSetup.startMeeting()`` — that method handles the UI transition
// and fires ``startRecording(false)``, which POSTs the new
// ``currentLanguagePair`` to /api/meeting/start. The server's strict
// parser accepts the single code and creates the meeting in
// monolingual mode.
document.getElementById('landing-quick-english-btn')?.addEventListener('click', (e) => {
  const btn = e.currentTarget;
  if (btn.disabled) return;
  btn.disabled = true;
  const origText = btn.textContent;
  btn.textContent = 'Starting…';
  try {
    // Quick start bypasses the setup screen entirely. We set
    // currentLanguagePair to a single code (mono) and POST it directly —
    // the dropdowns don't need to reflect this because the user won't
    // see the setup screen unless they explicitly navigate back, at
    // which point showSetup() resets the pair to the bilingual default.
    currentLanguagePair = 'en';
    _updateColumnHeaders();

    hideLanding();
    history.pushState(null, '', '#meeting');
    // startMeeting lives on RoomSetup (not MeetingsManager) — it does
    // the UI transition + kicks off the actual recording pipeline.
    roomSetup.startMeeting();
  } catch (err) {
    btn.disabled = false;
    btn.textContent = origText;
    throw err;
  }
});

// ─── Hamburger Menu ──────────────────────────────────────────

const hamburgerBtn = document.getElementById('btn-hamburger');
const meetingsPanel = document.getElementById('meetings-panel');
const meetingsBackdrop = document.getElementById('meetings-backdrop');

function toggleMeetingsPanel() {
  const open = meetingsPanel.classList.toggle('open');
  meetingsBackdrop.classList.toggle('open', open);
}

hamburgerBtn.addEventListener('click', toggleMeetingsPanel);
meetingsBackdrop.addEventListener('click', toggleMeetingsPanel);

// ─── Meetings History ────────────────────────────────────────

class MeetingsManager {
  constructor() {
    this.listEl = document.getElementById('meetings-list');
    this.btnNew = document.getElementById('btn-new-meeting');
    this.viewingMeetingId = null;
    // Tab state — "all" keeps strict chronological order so the timeline
    // doesn't reshuffle when a meeting is favorited. "favorites" is a
    // separate view for the user's curated demo/reference picks.
    this.activeTab = (() => {
      try { return localStorage.getItem('meetings_tab') || 'all'; } catch { return 'all'; }
    })();
    this._lastMeetings = [];

    // Wire the tab switcher buttons
    const tabs = document.querySelectorAll('.meetings-tab');
    tabs.forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.tab === this.activeTab);
      btn.setAttribute('aria-selected', btn.dataset.tab === this.activeTab ? 'true' : 'false');
      btn.addEventListener('click', () => {
        this.activeTab = btn.dataset.tab;
        try { localStorage.setItem('meetings_tab', this.activeTab); } catch {}
        tabs.forEach((b) => {
          b.classList.toggle('active', b.dataset.tab === this.activeTab);
          b.setAttribute('aria-selected', b.dataset.tab === this.activeTab ? 'true' : 'false');
        });
        this._render(this._lastMeetings);
      });
    });

    this.btnNew.addEventListener('click', () => {
      // Free navigation: even while recording the user can reach the
      // setup screen. The server's idempotent start fast-path handles
      // any accidental re-start attempts; the return banner remains
      // visible so the user can always get back to the live view.
      if (document.body.classList.contains('recording')
       || document.body.classList.contains('meeting-active')) {
        exitLiveMeetingView();
      }
      this.showSetup(); toggleMeetingsPanel();
    });
    this.refresh();
    setInterval(() => this.refresh(), 10000);
  }

  async refresh() {
    try {
      const resp = await fetch(`${API}/api/meetings`);
      const data = await resp.json();
      this._lastMeetings = data.meetings || [];
      this._render(this._lastMeetings);
    } catch {}
  }

  _render(meetings) {
    this.listEl.innerHTML = '';

    // Apply the active tab filter. The list comes from the server already
    // sorted newest-first; we DON'T re-sort favorites to the top here so
    // toggling a star never reshuffles the timeline. Favorites tab just
    // narrows the view to the starred subset.
    if (this.activeTab === 'favorites') {
      meetings = meetings.filter((m) => m.is_favorite);
      if (meetings.length === 0) {
        this.listEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);font-size:0.75rem;text-align:center;">No starred meetings yet — tap a star to add one.</div>';
        return;
      }
    }

    if (meetings.length === 0) {
      this.listEl.innerHTML = '<div style="padding:1rem;color:var(--text-muted);font-size:0.75rem;text-align:center;">No meetings yet</div>';
      return;
    }

    for (const m of meetings) {
      const item = document.createElement('div');
      item.className = `meeting-item${this.viewingMeetingId === m.meeting_id ? ' active' : ''}`;
      const date = m.created_at ? new Date(m.created_at) : null;
      const dateStr = date ? `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}` : 'Unknown';
      // Build action buttons
      let summaryBtn = '';
      if (m.state === 'complete') {
        if (m.has_summary) {
          summaryBtn = '<button class="meeting-summary btn-ghost" title="View finalization summary"><span class="check-mark">✓</span> Summary</button>';
        } else if (m.event_count > 0) {
          summaryBtn = '<button class="meeting-summary btn-ghost meeting-no-summary" title="No summary — click to generate">⊘ Summary</button>';
        }
      }
      const viewBtn = m.event_count > 0 && m.state !== 'recording'
        ? '<button class="meeting-view btn-ghost" title="View meeting">View</button>'
        : '';
      const resumeBtn = m.state === 'interrupted'
        ? '<button class="meeting-resume btn-ghost" title="Resume recording">Resume</button>'
        : '';
      // Interrupted meetings with events but no summary can be finalized
      // without resuming — just close them out and run the summarizer.
      const finalizeBtn = m.state === 'interrupted' && m.event_count > 0 && !m.has_summary
        ? '<button class="meeting-finalize btn-ghost" title="End and finalize (summary + cleanup)">End</button>'
        : '';

      // Short summary preview — truncated to ~120 chars, full text in tooltip
      const fullSummary = m.executive_summary || '';
      const topicsStr = (m.topics || []).join(' • ');
      const shortSummary = fullSummary
        ? (fullSummary.length > 120 ? fullSummary.slice(0, 117) + '…' : fullSummary)
        : '';
      const summaryTitle = fullSummary
        + (topicsStr ? `\n\nTopics: ${topicsStr}` : '');
      const summaryPreview = shortSummary
        ? `<div class="meeting-item-summary" title="${esc(summaryTitle)}">${esc(shortSummary)}</div>`
        : '';

      const starBtn = `<button class="meeting-star${m.is_favorite ? ' starred' : ''}" title="${m.is_favorite ? 'Unstar — remove from favorites' : 'Star — mark as useful demo / reference'}" aria-pressed="${m.is_favorite ? 'true' : 'false'}">${m.is_favorite ? '★' : '☆'}</button>`;
      // Reprocess buttons — only on complete meetings with events.
      // - Re-diarize: fast, just re-runs full-audio diarization + speaker
      //   collapse on the existing transcript (~2-3 min for 60-min audio).
      // - Reprocess: slow, re-runs ASR + translation + diarization from
      //   raw audio (~10-15 min for 60-min audio). Use when you want
      //   higher-quality transcript text in addition to speaker fixes.
      const canReprocess = m.state === 'complete' && m.event_count > 0;
      // Show the tools menu (⋯) for any non-live meeting with a journal,
      // so a meeting whose reprocess was killed mid-flight can still be
      // recovered (Retry reprocess, view versions, delete). Without this
      // the user has no escape hatch when state gets stuck at
      // 'reprocessing' or 'interrupted'.
      const canShowTools = m.event_count > 0 && m.state !== 'recording';
      const rediarizeBtn = canReprocess
        ? '<button class="meeting-rediarize btn-ghost" title="Re-run diarization + speaker consolidation (fast, ~2-3 min for 60-min audio). Optional speaker count.">Re-diarize</button>'
        : '';
      const reprocessBtn = canReprocess
        ? '<button class="meeting-reprocess btn-ghost" title="Full reprocess: re-run ASR + translation + diarization from raw audio (slow, ~10-15 min for 60-min audio). Use for higher-quality transcript.">Reprocess</button>'
        : '';
      // Slides button: only when this meeting has a deck on disk. Opens
      // the original-language PDF in a new tab — uses the same deck that
      // was uploaded during the meeting.
      const slidesBtn = m.has_slides
        ? '<button class="meeting-slides btn-ghost" title="Open the slide deck uploaded during this meeting">Slides</button>'
        : '';
      // Tools menu: hides the destructive / heavy admin actions
      // (Re-diarize, Reprocess, Versions, Delete) behind a single "⋯"
      // button so the row stays uncluttered and the primary actions
      // (View / Summary / Slides) stay obvious.
      const toolsBtn = '<button class="meeting-tools btn-ghost" title="More actions: Re-diarize, Reprocess, Versions, Delete" aria-label="More actions">⋯</button>';
      item.innerHTML = `
        <div class="meeting-item-row">
          ${starBtn}
          <div class="meeting-item-content">
            <div class="meeting-item-head">
              <span class="meeting-item-date">${dateStr}</span>
              <span class="meeting-item-info">
                <span class="meeting-item-state ${m.state}">${m.state}</span>
                ${m.event_count > 0 ? `${m.event_count} events` : ''}
              </span>
            </div>
            ${summaryPreview}
          </div>
          ${viewBtn}
          ${summaryBtn}
          ${slidesBtn}
          ${resumeBtn}
          ${finalizeBtn}
          ${canShowTools ? toolsBtn : '<button class="meeting-delete" title="Delete meeting">&times;</button>'}
        </div>
      `;
      item.querySelector('.meeting-item-content').addEventListener('click', () => this.viewMeeting(m.meeting_id));
      item.querySelector('.meeting-view')?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.viewMeeting(m.meeting_id);
      });
      item.querySelector('.meeting-summary')?.addEventListener('click', (e) => {
        e.stopPropagation();
        showFinalizationSummaryFor(m.meeting_id);
      });
      item.querySelector('.meeting-resume')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        await _resumeMeeting(m.meeting_id);
        if (meetingsPanel.classList.contains('open')) toggleMeetingsPanel();
        this.refresh();
      });
      item.querySelector('.meeting-finalize')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Ending…';
        try {
          const resp = await fetch(`${API}/api/meetings/${m.meeting_id}/finalize`, {
            method: 'POST',
          });
          if (!resp.ok) throw new Error(await resp.text());
          // Refresh the list — the meeting state should transition to "complete"
          // once the summarizer finishes. Poll every 3s.
          this.refresh();
          const poll = setInterval(async () => {
            await this.refresh();
            const resp2 = await fetch(`${API}/api/meetings`);
            const data = await resp2.json();
            const updated = (data.meetings || []).find(x => x.meeting_id === m.meeting_id);
            if (updated && updated.state === 'complete') {
              clearInterval(poll);
            }
          }, 3000);
          setTimeout(() => clearInterval(poll), 120000); // hard stop after 2 min
        } catch (err) {
          showModal(`<div class="modal-confirm-title">Finalization failed</div><div class="modal-confirm-message">${esc(String(err.message || err))}</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm');
          btn.disabled = false;
          btn.textContent = 'End';
        }
      });
      item.querySelector('.meeting-delete')?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteMeeting(m.meeting_id);
      });
      // Tools menu (⋯) — opens an actions modal that explains each
      // operation. Keeps Re-diarize / Reprocess / Versions / Delete out
      // of the row so the row stays clean.
      item.querySelector('.meeting-tools')?.addEventListener('click', (e) => {
        e.stopPropagation();
        _openMeetingToolsModal(m, this);
      });
      // Slides: open the deck PDF for this past meeting in a new tab.
      // PDF is the original deck (lossless). Translated PNGs are also
      // available per-slide via /api/meetings/{id}/slides/{N}/translated
      // if a richer viewer is wired in later.
      item.querySelector('.meeting-slides')?.addEventListener('click', (e) => {
        e.stopPropagation();
        window.open(`${API}/api/meetings/${m.meeting_id}/slides/original.pdf`, '_blank', 'noopener');
      });

      // Re-diarize: fast path — just re-runs diarize + speaker consolidation
      item.querySelector('.meeting-rediarize')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const raw = await promptDialog(
          'Re-diarize meeting',
          'Pin a speaker count when known (recommended for over-clustered meetings). Leave blank to let the model decide.',
          {
            placeholder: 'Speaker count (1–12) or blank',
            confirmText: 'Re-diarize',
            type: 'number',
            inputMode: 'numeric',
            min: 1,
            max: 12,
            help: 'Keeps the transcript text. Replaces speaker labels only.',
          }
        );
        if (raw === null) return;
        const expected = raw === '' ? null : parseInt(raw, 10);
        if (raw !== '' && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
          await alertDialog('Invalid count', 'Speaker count must be a number between 1 and 12, or blank.');
          return;
        }
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Re-diarizing…';
        try {
          const qs = expected != null ? `?expected_speakers=${expected}` : '';
          const resp = await fetch(`${API}/api/meetings/${m.meeting_id}/finalize${qs}`, { method: 'POST' });
          if (!resp.ok) throw new Error(await resp.text());
          const result = await resp.json();
          this.refresh();
          await alertDialog(
            'Re-diarize complete',
            `${result?.diarization?.unique_speakers ?? '?'} speakers detected from ${result?.diarization?.segments ?? '?'} diarize segments.`,
          );
        } catch (err) {
          await alertDialog('Re-diarize failed', String(err.message || err));
        } finally {
          btn.disabled = false;
          btn.textContent = 'Re-diarize';
        }
      });

      // Reprocess: slow path — re-runs ASR + translation + diarize
      item.querySelector('.meeting-reprocess')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const raw = await promptDialog(
          'Full reprocess from raw audio',
          'Re-runs ASR + translation + diarization for a higher-quality transcript. Slow: about 10–15 minutes for a 60-minute meeting. The current journal is backed up as journal.jsonl.bak.',
          {
            placeholder: 'Speaker count (1–12) or blank',
            confirmText: 'Reprocess',
            type: 'number',
            inputMode: 'numeric',
            min: 1,
            max: 12,
            help: 'Pin a speaker count when known, or leave blank to let pyannote decide.',
          }
        );
        if (raw === null) return;
        const expected = raw === '' ? null : parseInt(raw, 10);
        if (raw !== '' && (!Number.isFinite(expected) || expected < 1 || expected > 12)) {
          await alertDialog('Invalid count', 'Speaker count must be a number between 1 and 12, or blank.');
          return;
        }
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Reprocessing…';
        try {
          const qs = expected != null ? `?expected_speakers=${expected}` : '';
          // No timeout on the fetch — server holds the connection until
          // the full pipeline completes, which can take 10+ minutes.
          const resp = await fetch(`${API}/api/meetings/${m.meeting_id}/reprocess${qs}`, { method: 'POST' });
          if (!resp.ok) throw new Error(await resp.text());
          const result = await resp.json();
          this.refresh();
          const segs = result?.segments ?? '?';
          const tr = result?.translated ?? '?';
          const sp = result?.speakers ?? '?';
          await alertDialog('Reprocess complete', `${segs} segments, ${tr} translated, ${sp} speakers detected.`);
        } catch (err) {
          await alertDialog('Reprocess failed', String(err.message || err));
        } finally {
          btn.disabled = false;
          btn.textContent = 'Reprocess';
        }
      });
      item.querySelector('.meeting-star')?.addEventListener('click', async (e) => {
        e.stopPropagation();
        const btn = e.currentTarget;
        const next = !m.is_favorite;
        // Optimistic update so the star feels instant — revert on failure.
        btn.disabled = true;
        btn.classList.toggle('starred', next);
        btn.textContent = next ? '★' : '☆';
        btn.setAttribute('aria-pressed', next ? 'true' : 'false');
        try {
          const resp = await fetch(`${API}/api/meetings/${m.meeting_id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_favorite: next }),
          });
          if (!resp.ok) throw new Error(await resp.text());
          m.is_favorite = next;
          // Re-render so favorites re-sort to the top of the list
          this.refresh();
        } catch (err) {
          // Revert optimistic UI and surface the error
          btn.classList.toggle('starred', !next);
          btn.textContent = !next ? '★' : '☆';
          btn.setAttribute('aria-pressed', !next ? 'true' : 'false');
          showModal(`<div class="modal-confirm-title">Could not update favorite</div><div class="modal-confirm-message">${esc(String(err.message || err))}</div><div class="modal-confirm-actions"><button class="modal-btn" onclick="closeModal()">OK</button></div>`, 'confirm');
        } finally {
          btn.disabled = false;
        }
      });
      this.listEl.appendChild(item);
    }
  }

  async viewMeeting(meetingId) {
    // Navigation is allowed while a meeting is recording — the live
    // pipeline stays alive, the reconciler's return banner lets the
    // user come back. We detach live rendering so the review view
    // can safely own the shared store.
    if (document.body.classList.contains('recording')
     || document.body.classList.contains('meeting-active')) {
      exitLiveMeetingView();
    }

    // If this meeting is still being finalized, reopen the finalization modal
    if (_finalizingMeetings.has(meetingId)) {
      this._reopenFinalizationModal(meetingId);
      return;
    }
    this.viewingMeetingId = meetingId;
    this.refresh();
    // Close the panel if it's open (don't open it if closed)
    if (meetingsPanel.classList.contains('open')) toggleMeetingsPanel();

    try {
      const resp = await fetch(`${API}/api/meetings/${meetingId}`);
      const data = await resp.json();

      // Clear stale state from previous meeting view. Cluster_ids are
      // meeting-local, so without resetting the speaker registry the
      // first-seen-order sequential labels ("Speaker 1", "Speaker 2"...)
      // and any custom-renamed names from the previous meeting leak into
      // this one. We also have to drop the one-on-one mode assignments
      // (also keyed on the previous meeting's cluster_ids) and null the
      // audio player's timeline so a fast meeting switch can't cross-wire
      // old segments with new audio.
      _resetSpeakerRegistry();
      if (typeof _oneOnOneSpeakers !== 'undefined') {
        _oneOnOneSpeakers.left = null;
        _oneOnOneSpeakers.right = null;
      }
      audioPlayer._timeline = null;
      audioPlayer.currentSegmentId = null;
      store.clear();

      document.getElementById('speaker-timeline').style.display = 'none';
      document.getElementById('speaker-timeline-lanes').innerHTML = '';
      document.getElementById('speaker-timeline-times').innerHTML = '';
      document.getElementById('meeting-summary-panel').style.display = 'none';
      document.getElementById('player-bar').style.display = 'none';
      audioPlayer.hide();

      // Metrics split-view is only meaningful for live meetings — clear it on
      // entering review so the 320px right-rail reservation doesn't hide the
      // new top summary bar under the pinned dashboard.
      document.body.classList.remove('metrics-split');
      const metricsDash = document.getElementById('metrics-dashboard');
      if (metricsDash) metricsDash.style.display = 'none';
      const metricsBtn = document.getElementById('btn-metrics');
      if (metricsBtn) metricsBtn.classList.remove('active-toggle');

      // Show meeting mode with the transcript grid for past meeting replay
      document.getElementById('landing-mode').style.display = 'none';
      document.getElementById('room-setup').style.display = 'none';
      document.getElementById('view-mode').style.display = 'none';
      document.getElementById('meeting-mode').style.display = '';

      // Initialize grid renderer with meeting ID for playback
      window._gridRenderer = new CompactGridRenderer(document.getElementById('transcript-grid'), meetingId);
      // Finished-meeting default: ALWAYS oldest-first with the viewport
      // anchored to the FIRST segment.
      //
      // Two things need to happen here that the live-view code path gets
      // for free:
      //
      // 1. Flip the renderer direction so new blocks get appended at the
      //    end (oldest-to-newest reading order).
      // 2. Disable auto-scroll. The renderer's default update() calls
      //    `scrollTop = scrollHeight` after each event ingest so the
      //    LATEST block stays visible — that's right for a live stream
      //    but wrong for review mode, where we want the oldest block to
      //    stay at the top while every later event fills in below.
      //    Without this, the last ingested event during initial load
      //    yanks the viewport to the bottom and the user sees "newest
      //    at bottom" while scrolled to it, which reads as "reversed".
      {
        const isFinished0 = data.meta?.state && data.meta.state !== 'recording';
        if (isFinished0 && window._gridRenderer) {
          window._gridRenderer.toggleDirection();
          window._gridRenderer._autoScroll = false;
          requestAnimationFrame(() => {
            const g = document.getElementById('transcript-grid');
            if (g) g.scrollTop = 0;
          });
        }
      }

      // Show podcast player + speaker timeline if audio/timeline exists
      try {
        const tlResp = await fetch(`${API}/api/meetings/${meetingId}/timeline`);
        if (tlResp.ok) {
          const tl = await tlResp.json();
          if (tl.duration_ms > 0) {
            audioPlayer.loadMeeting(meetingId, tl.duration_ms, tl.segments);
            // Render speaker timeline lanes
            if (tl.speaker_lanes && Object.keys(tl.speaker_lanes).length > 0) {
              renderSpeakerTimeline(tl.speaker_lanes, tl.duration_ms, tl.speakers || [], meetingId);
            }
          }
        }
      } catch {}

      // Set meeting start time for wall clock display
      if (data.meta?.created_at) {
        _meetingStartWallMs = new Date(data.meta.created_at).getTime();
      }

      // Set language pair from meeting metadata. Length 1 = monolingual
      // (store a single-code string so downstream helpers can decide mode).
      if (data.meta?.language_pair && data.meta.language_pair.length >= 1) {
        currentLanguagePair = data.meta.language_pair.join(',');
        _updateColumnHeaders();
      }

      // Load events into store → flows to all transcript columns
      if (data.events?.length > 0) {
        for (const event of data.events) {
          store.ingest(event);
        }
      }

      // Render table strip if room layout exists
      const strip = document.getElementById('meeting-table-strip');
      strip.innerHTML = '';
      if (data.room && (data.room.tables?.length || data.room.seats?.length)) {
        strip.style.display = '';
        for (const t of data.room.tables || []) {
          const el = document.createElement('div');
          el.className = 'strip-table';
          el.style.cssText = `left:${t.x}%;top:${t.y}%;width:${t.width}%;height:${t.height}%;border-radius:${t.border_radius}%`;
          strip.appendChild(el);
        }
        // Build name→cluster_id map from events so seat colors match the transcript
        const nameToClusterId = {};
        for (const e of data.events || []) {
          const sp = e.speakers?.[0];
          if (sp) {
            const key = sp.identity || sp.display_name;
            if (key && !(key in nameToClusterId)) {
              nameToClusterId[key] = sp.cluster_id;
            }
          }
        }
        for (let si = 0; si < (data.room.seats || []).length; si++) {
          const s = data.room.seats[si];
          // Color by cluster_id (canonical) — falls back to seat index if unknown
          const clusterId = nameToClusterId[s.speaker_name];
          const color = clusterId != null ? getSpeakerColor(clusterId) : SPEAKER_COLORS[si % SPEAKER_COLORS.length];
          const el = document.createElement('div');
          el.className = `strip-seat enrolled`;
          el.dataset.speakerId = String(si);
          el.dataset.clusterId = String(clusterId ?? '');
          el.style.cssText = `left:${s.x}%;top:${s.y}%;--seat-color:${color}`;
          el.innerHTML = `<span class="strip-seat-num">${esc(s.speaker_name?.[0] || (si+1).toString())}</span><span class="strip-seat-name">${esc(s.speaker_name || '')}</span>`;
          // Click to show speaker's segments
          const seatName = s.speaker_name || `Speaker ${si+1}`;
          el.addEventListener('click', () => {
            showSpeakerModal(seatName, color, findSpeakerSegments(seatName), meetingId);
          });
          strip.appendChild(el);
        }
      } else {
        strip.style.display = 'none';
      }

      // Hide control bar for replay — review tools live in the summary bar
      document.getElementById('control-bar').style.display = 'none';
      const stateLabel = data.meta?.state === 'interrupted' ? ' · interrupted' : '';
      const nameA = _languageNames[_getLangA()]?.name || _getLangA().toUpperCase();
      const nameB = _languageNames[_getLangB()]?.name || _getLangB().toUpperCase();
      const savedDir = localStorage.getItem('scribe_transcript_direction');
      const dirInitialLabel = savedDir === 'oldest' ? '↓ Oldest' : '↑ Newest';

      // Status line is now a single short label — the event count and meeting
      // id live inside the summary modal where they belong.
      document.getElementById('status-line').textContent = `Review${stateLabel}`;

      // Populate the top summary bar with the review toolbar. The bar is
      // always shown in review mode, even if no summary exists yet; the
      // "View Summary" button is appended afterwards by _renderSummaryPanel
      // only when the summary fetch succeeds.
      //
      // The "id: <uuid>" chip is a click-to-copy so users reviewing a
      // meeting can identify which one they were looking at when
      // something goes wrong (re-finalize failure, download link issue).
      const summaryPanel = document.getElementById('meeting-summary-panel');
      summaryPanel.innerHTML =
        `<span class="summary-bar-icon" aria-hidden="true">📄</span>` +
        `<span class="summary-bar-title">Meeting Summary</span>` +
        `<span class="summary-bar-id" id="rv-meeting-id" title="Click to copy meeting ID" style="cursor:pointer;font-family:ui-monospace,Menlo,monospace;font-size:0.78rem;opacity:0.7;margin-right:0.5rem">id: ${esc(meetingId)}</span>` +
        `<div class="summary-bar-tools">` +
          `<button class="btn-ghost" id="rv-col-a" title="Show only ${nameA}">${nameA}</button>` +
          `<button class="btn-ghost" id="rv-col-b" title="Show only ${nameB}">${nameB}</button>` +
          `<button class="btn-ghost" id="rv-table" title="Toggle virtual table">Table</button>` +
          `<button class="btn-ghost" id="rv-compact" title="Compact horizontal view">Compact</button>` +
          `<button class="btn-ghost" id="rv-scroll-dir" title="Toggle transcript order (newest first / oldest first)">${dirInitialLabel}</button>` +
          `<button class="btn-ghost" id="rv-live" title="Open pop-out translation view">Pop-out</button>` +
          `<button class="btn-ghost" id="rv-edit-layout" title="Edit layout + assign detected voices to seats">Edit Layout</button>` +
          `<button class="btn-ghost" id="btn-reprocess" title="Re-run ASR, diarization, translation and regenerate summary from the original audio">Re-finalize</button>` +
          (data.meta?.state === 'interrupted' ? `<button class="btn-ghost" id="btn-resume" style="background:var(--success);color:#fff">Resume</button>` : '') +
        `</div>`;
      summaryPanel.style.display = 'flex';

      // Column selectors — toggle between: both → only-a → only-b → both
      document.getElementById('rv-col-a')?.addEventListener('click', (e) => {
        const wasActive = document.body.classList.contains('show-only-a');
        document.body.classList.remove('show-only-a', 'show-only-b');
        document.getElementById('rv-col-b')?.classList.remove('active-toggle');
        if (!wasActive) {
          document.body.classList.add('show-only-a');
          e.target.classList.add('active-toggle');
        } else {
          e.target.classList.remove('active-toggle');
        }
      });
      document.getElementById('rv-col-b')?.addEventListener('click', (e) => {
        const wasActive = document.body.classList.contains('show-only-b');
        document.body.classList.remove('show-only-a', 'show-only-b');
        document.getElementById('rv-col-a')?.classList.remove('active-toggle');
        if (!wasActive) {
          document.body.classList.add('show-only-b');
          e.target.classList.add('active-toggle');
        } else {
          e.target.classList.remove('active-toggle');
        }
      });

      document.getElementById('rv-live')?.addEventListener('click', (e) => {
        _openPopout(e.target, meetingId);
      });
      document.getElementById('rv-table')?.addEventListener('click', (e) => { document.body.classList.toggle('hide-table'); e.target.classList.toggle('active-toggle'); });
      document.getElementById('rv-compact')?.addEventListener('click', (e) => { document.body.classList.toggle('compact-mode'); e.target.classList.toggle('active-toggle'); });
      document.getElementById('rv-scroll-dir')?.addEventListener('click', (e) => {
        if (!window._gridRenderer) return;
        const newestFirst = window._gridRenderer.toggleDirection();
        e.target.textContent = newestFirst ? '↑ Newest' : '↓ Oldest';
        localStorage.setItem('scribe_transcript_direction', newestFirst ? 'newest' : 'oldest');
      });
      document.getElementById('rv-edit-layout')?.addEventListener('click', () => {
        openRoomEditor('review', meetingId);
      });
      document.getElementById('btn-reprocess')?.addEventListener('click', async () => {
        const btn = document.getElementById('btn-reprocess');
        btn.disabled = true;
        btn.textContent = 'Re-finalizing…';
        document.getElementById('status-line').textContent = 'Re-finalizing: diarization, speaker data, summary...';
        try {
          // Re-finalize = finalize again: full-audio diarization + regen
          // speaker_lanes + regen detected_speakers + regen summary.
          // Use this after fixing bugs/config — same pipeline as the initial
          // finalize but forced to regenerate everything.
          const resp = await fetch(`${API}/api/meetings/${meetingId}/finalize?force=true`, {
            method: 'POST',
          });
          if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${err}`);
          }
          const result = await resp.json();
          console.log('Re-finalize result:', result);

          // Warn clearly if the audio is corrupted at the recording level
          const aq = result.audio_quality;
          if (aq && !aq.usable) {
            showModal(`
              <div class="modal-card-header"><h2>Audio quality warning</h2></div>
              <div style="padding:1rem 1.25rem;font-size:0.85rem;line-height:1.5">
                <p style="margin-bottom:0.75rem">
                  <strong>This recording is ${aq.zero_fill_pct}% silence-filled</strong>,
                  with the longest gap being ${(aq.longest_zero_run_ms/1000).toFixed(1)}s.
                </p>
                <p style="margin-bottom:0.75rem">
                  An earlier audio writer inserted zero-gaps whenever WebSocket
                  chunks arrived later than wall clock. Speaker separation for
                  this meeting is fundamentally limited — only
                  ${result.diarization?.unique_speakers || '?'} speaker(s) could
                  be clustered from the corrupted audio.
                </p>
                <p style="color:var(--text-muted)">
                  New meetings recorded after the audio writer fix will be clean.
                </p>
              </div>
              <div class="modal-card-footer">
                <button class="btn btn-primary" onclick="closeModal()">OK</button>
              </div>
            `);
          }

          const dz = result.diarization || {};
          document.getElementById('status-line').textContent =
            `Re-finalized ${meetingId}: ${dz.segments || 0} diarization segments, ${dz.unique_speakers || 0} speakers`;
          this.viewMeeting(meetingId);
        } catch (e) {
          document.getElementById('status-line').textContent = `Re-finalize failed: ${e.message}`;
          btn.disabled = false;
          btn.textContent = 'Re-finalize';
        }
      });
      document.getElementById('btn-resume')?.addEventListener('click', () => _resumeMeeting(meetingId));

      // Click-to-copy for the review-mode meeting-id chip. Keep in sync
      // with the equivalent handler in _renderFinalizationSummary().
      document.getElementById('rv-meeting-id')?.addEventListener('click', async (e) => {
        const chip = e.currentTarget;
        const original = chip.textContent;
        try {
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(meetingId);
          } else {
            const ta = document.createElement('textarea');
            ta.value = meetingId;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
          }
          chip.textContent = 'copied ✓';
        } catch {
          chip.textContent = 'copy failed';
        }
        setTimeout(() => { chip.textContent = original; }, 1200);
      });

      // Load summary if available — must run AFTER the summary bar is
      // populated with rv-* tools above, because _renderSummaryPanel
      // appends the "View Summary" button to the existing
      // .summary-bar-tools row. Running it earlier would have nothing
      // to append to and then the later innerHTML= would wipe the row.
      try {
        const sumResp = await fetch(`${API}/api/meetings/${meetingId}/summary`);
        if (sumResp.ok) {
          const summary = await sumResp.json();
          _renderSummaryPanel(summary, meetingId);
        }
      } catch {}
    } catch (err) {
      document.getElementById('status-line').textContent = `Error: ${err.message}`;
    }
  }

  _reopenFinalizationModal(meetingId) {
    // Close meetings panel if open
    if (meetingsPanel.classList.contains('open')) toggleMeetingsPanel();

    const tracker = _finalizingMeetings.get(meetingId);
    if (!tracker) return;

    // Must stay in sync with the 6-step flow emitted by /api/meeting/stop
    // (see stopMeeting() above).
    const steps = [
      { label: 'Flushing speech recognition' },
      { label: 'Completing translations' },
      { label: 'Saving speaker data' },
      { label: 'Running full-audio diarization' },
      { label: 'Generating timeline' },
      { label: 'Generating meeting summary' },
    ];

    const card = showModal(`
      <div class="finalize-modal">
        <div class="finalize-header">
          <div class="finalize-header-content">
            <div class="finalize-pulse"></div>
            <div>
              <h3>Finalizing Meeting</h3>
              <p class="finalize-subtitle" id="finalize-subtitle">${tracker.label || 'Processing...'}</p>
            </div>
          </div>
          <button class="finalize-close" id="finalize-close-btn" title="Close">&times;</button>
        </div>
        <div class="finalize-progress-track">
          <div class="finalize-progress-fill" id="finalize-progress-fill" style="width:${Math.min(100, (tracker.step / 6) * 100)}%"></div>
        </div>
        <div class="finalize-steps" id="finalize-steps">
          ${steps.map((s, i) => `
            <div class="finalize-step${(i + 1) < tracker.step ? ' done' : (i + 1) === tracker.step ? ' active' : ''}" data-step="${i + 1}">
              <div class="step-indicator">
                <div class="step-ring">
                  <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" fill="none" stroke-width="1.5"/></svg>
                  <span class="step-check">&#10003;</span>
                </div>
                ${i < steps.length - 1 ? '<div class="step-connector"></div>' : ''}
              </div>
              <span class="step-label">${s.label}</span>
            </div>
          `).join('')}
        </div>
        <div class="finalize-eta" id="finalize-eta"></div>
        <div class="finalize-summary" id="finalize-summary" style="display:none"></div>
      </div>
    `, 'finalize');

    card.querySelector('#finalize-close-btn')?.addEventListener('click', () => closeModal());

    // Listen for updates via the original WS (if still alive)
    const ws = tracker.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      const origOnMessage = ws.onmessage;
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'finalize_progress') {
            // Server sends TWO step=6 messages: the first starts summary
            // generation (no summary field), the second carries the
            // summary and signals completion. Only the second one is
            // terminal — see the matching logic in stopMeeting().
            const isCompletion = msg.step >= 6 && (msg.summary !== undefined || msg.meeting_id);

            const pct = isCompletion ? 100 : Math.min(100, (msg.step / 6) * 100);
            const fill = card.querySelector('#finalize-progress-fill');
            if (fill) fill.style.width = `${pct}%`;

            card.querySelectorAll('.finalize-step').forEach(el => {
              const s = parseInt(el.dataset.step);
              el.classList.toggle('done', isCompletion ? true : s < msg.step);
              el.classList.toggle('active', isCompletion ? false : s === msg.step);
            });

            const subtitle = card.querySelector('#finalize-subtitle');
            if (subtitle && msg.label) subtitle.textContent = msg.label;

            const etaEl = card.querySelector('#finalize-eta');
            if (etaEl && msg.eta_seconds > 0) {
              etaEl.textContent = `Estimated ${msg.eta_seconds}s remaining`;
              etaEl.classList.add('visible');
            } else if (etaEl) {
              etaEl.classList.remove('visible');
            }

            // Update tracker
            tracker.step = msg.step;
            tracker.label = msg.label || '';

            if (isCompletion) {
              _finalizingMeetings.delete(meetingId);
              const pulse = card.querySelector('.finalize-pulse');
              if (pulse) pulse.classList.add('complete');
              if (subtitle) subtitle.textContent = 'Meeting finalized';
              if (msg.summary) _renderFinalizationSummary(msg.summary, meetingId);
            }
          }
        } catch {}
      };
    }
  }

  async deleteMeeting(meetingId) {
    const ok = await confirmDialog(
      'Delete Meeting?',
      'This will permanently delete the meeting and all its recordings, transcripts, and translations. This cannot be undone.',
    );
    if (!ok) return;
    try {
      const resp = await fetch(`${API}/api/meetings/${meetingId}`, { method: 'DELETE' });
      if (!resp.ok) { const e = await resp.json(); console.warn('Delete failed:', e.error); return; }
      if (this.viewingMeetingId === meetingId) this.showSetup();
      this.refresh();
    } catch (err) { console.warn('Delete error:', err); }
  }

  showSetup() {
    this.viewingMeetingId = null;
    this.refresh();
    // Prime the microphone the moment the setup panel becomes visible so
    // the first chair tap captures audio instantly instead of waiting on
    // getUserMedia + AudioContext init (~500-1500 ms cold). Fire-and-forget;
    // _enrollSeat falls back to a per-call acquisition if priming hasn't
    // resolved yet (or if the user denies the permission prompt).
    micWarmup.prime();
    document.getElementById('landing-mode').style.display = 'none';
    document.getElementById('room-setup').style.display = '';
    document.getElementById('meeting-mode').style.display = 'none';
    document.getElementById('view-mode').style.display = 'none';
    document.getElementById('speaker-timeline').style.display = 'none';
    document.getElementById('meeting-summary-panel').style.display = 'none';
    document.getElementById('metrics-dashboard').style.display = 'none';
    const metricsBtn = document.getElementById('btn-metrics');
    if (metricsBtn) metricsBtn.classList.remove('active-toggle');
    document.body.classList.remove('show-only-a', 'show-only-b', 'hide-live', 'hide-table', 'compact-mode', 'metrics-split');
    audioPlayer.hide();
    store.clear();
    document.getElementById('status-line').textContent = 'Ready';
    // Reset Start button in case a previous start attempt failed
    const startBtn = document.getElementById('btn-start-meeting');
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.textContent = startBtn.dataset.origText || 'Start Meeting';
    }
    // The setup screen is always bilingual. If the user is returning from
    // a mono meeting (started via the landing page's "Quick start English"
    // button), reset both dropdowns back to the server's configured default
    // pair — it's always possible to get back to a multi-language meeting
    // from here. Single-language is accessible ONLY via the landing page
    // quick-start, which bypasses this screen entirely.
    const selA = document.getElementById('lang-a-select');
    const selB = document.getElementById('lang-b-select');
    const selector = document.getElementById('language-selector');
    if (selA && selB) {
      const [defA, defB] = _defaultLanguagePair();
      const langCodes = [...selB.options].map(o => o.value).filter(Boolean);
      // Prefer server defaults; degrade to the first two distinct codes
      // the dropdowns know about if those aren't present (e.g. a very
      // restricted /api/languages response).
      const pickA = langCodes.includes(defA) ? defA : (langCodes[0] || defA);
      let pickB = langCodes.includes(defB) && defB !== pickA
        ? defB
        : (langCodes.find(v => v !== pickA) || defB);
      selA.value = pickA;
      selB.value = pickB;
      currentLanguagePair = `${pickA},${pickB}`;
      if (selector) selector.classList.remove('mono');
      _updateColumnHeaders();
    }
  }
}

const meetingsMgr = new MeetingsManager();

// ─── WiFi QR (main view — only during active meeting, hover to expand) ────────────
//
// Retries with exponential backoff so that opening the admin page BEFORE
// the hotspot rotation completes (meeting-scribe's _start_wifi_ap is
// fire-and-forget and takes ~5-30s to bring the AP up on the MT7925) still
// ends up rendering the fresh SSID. Also re-fetches when the SSID in the
// response differs from the currently-rendered one, so a mid-session
// rotation propagates to every open view.
let _lastRenderedWifiSsid = null;

async function refreshWifiQR(retries = 15, delayMs = 2000) {
  const qrEl = document.getElementById('header-qr');
  if (!qrEl) return;
  try {
    const statusResp = await fetch(`${API}/api/status`);
    const status = await statusResp.json();
    // Only show QR during active meeting
    if (!status.meeting?.id || status.meeting?.state !== 'recording') {
      qrEl.innerHTML = '';
      _lastRenderedWifiSsid = null;
      return;
    }
    const resp = await fetch(`${API}/api/meeting/wifi`);
    if (!resp.ok) {
      if (retries > 0) setTimeout(() => refreshWifiQR(retries - 1, delayMs), delayMs);
      return;
    }
    const data = await resp.json();
    if (!data.wifi_qr_svg) {
      if (retries > 0) setTimeout(() => refreshWifiQR(retries - 1, delayMs), delayMs);
      return;
    }
    // Only re-render when the SSID actually changed (avoid DOM churn).
    if (data.ssid !== _lastRenderedWifiSsid) {
      _lastRenderedWifiSsid = data.ssid;
      qrEl.innerHTML = `
        ${data.wifi_qr_svg}
        <div class="header-qr-expanded">
          ${data.wifi_qr_svg}
          <div class="wifi-qr-label">Scan to join · ${esc(data.ssid || '')}</div>
        </div>
      `;
    }
  } catch {
    if (retries > 0) setTimeout(() => refreshWifiQR(retries - 1, delayMs), delayMs);
  }
}
refreshWifiQR();

// Poll every 10s so every open view converges to the same live SSID.
// If rotation happens mid-session this catches up within 10s.
setInterval(() => refreshWifiQR(0, 0), 10000);

// ─── Settings panel (gear icon) ─────────────────────────────
// Dedicated slide-over with timezone + WiFi regulatory-domain dropdowns.
// Fetches option lists from GET /api/admin/settings on first open, writes
// via PUT on save. Keyboard: Enter saves, Escape closes. Click backdrop
// to dismiss. Focus is managed so sighted keyboard users land on the
// first field when the panel opens and return to the gear button on close.
(function initSettingsPanel() {
  const panel = document.getElementById('settings-panel');
  const backdrop = document.getElementById('settings-backdrop');
  const openBtn = document.getElementById('btn-settings');
  const closeBtn = document.getElementById('btn-settings-close');
  const saveBtn = document.getElementById('btn-settings-save');
  const wifiModeSelect = document.getElementById('setting-wifi-mode');
  const adminSsidInput = document.getElementById('setting-admin-ssid');
  const adminPasswordInput = document.getElementById('setting-admin-password');
  const adminPwToggle = document.getElementById('btn-toggle-admin-pw');
  const wifiLiveStatus = document.getElementById('wifi-live-status');
  const regdomainSelect = document.getElementById('setting-wifi-regdomain');
  const regdomainLive = document.getElementById('regdomain-live');
  const timezoneSelect = document.getElementById('setting-timezone');
  const devModeCheck = document.getElementById('setting-dev-mode');
  const voiceModeSelect = document.getElementById('setting-tts-voice-mode');
  const status = document.getElementById('settings-status');
  if (!panel || !openBtn || !regdomainSelect || !timezoneSelect) return;

  // Password show/hide toggle
  if (adminPwToggle && adminPasswordInput) {
    adminPwToggle.addEventListener('click', () => {
      const showing = adminPasswordInput.type === 'text';
      adminPasswordInput.type = showing ? 'password' : 'text';
      adminPwToggle.textContent = showing ? 'Show' : 'Hide';
    });
  }

  let loaded = false;
  let lastLoadedData = null;

  const setStatus = (msg, cls = '') => {
    status.textContent = msg || '';
    status.className = 'settings-status' + (cls ? ' ' + cls : '');
  };

  const populate = (data) => {
    lastLoadedData = data;

    // WiFi mode select
    if (wifiModeSelect) {
      wifiModeSelect.innerHTML = '';
      for (const opt of data.wifi_mode_options || []) {
        const el = document.createElement('option');
        el.value = opt.code;
        el.textContent = opt.name;
        if (opt.code === data.wifi_mode) el.selected = true;
        wifiModeSelect.appendChild(el);
      }
    }

    // Admin SSID
    if (adminSsidInput) adminSsidInput.value = data.admin_ssid || '';

    // Admin password — write-only, show placeholder if set
    if (adminPasswordInput) {
      adminPasswordInput.value = '';
      adminPasswordInput.placeholder = data.admin_password_set
        ? 'Leave blank to keep current'
        : 'Set a password (8-63 chars)';
    }

    // WiFi live status line
    if (wifiLiveStatus) {
      if (data.wifi_active && data.wifi_ssid) {
        const sec = data.wifi_security;
        const km = sec ? sec.key_mgmt || '?' : '?';
        wifiLiveStatus.textContent = `Live: ${data.wifi_ssid} (${km})`;
        wifiLiveStatus.classList.remove('mismatch');
      } else {
        wifiLiveStatus.textContent = 'WiFi AP is off';
        wifiLiveStatus.classList.add('mismatch');
      }
    }

    regdomainSelect.innerHTML = '';
    for (const opt of data.wifi_regdomain_options || []) {
      const el = document.createElement('option');
      el.value = opt.code;
      el.textContent = `${opt.code} · ${opt.name}`;
      if (opt.code === data.wifi_regdomain) el.selected = true;
      regdomainSelect.appendChild(el);
    }

    timezoneSelect.innerHTML = '';
    const blank = document.createElement('option');
    blank.value = '';
    blank.textContent = '— Server local time —';
    if (!data.timezone) blank.selected = true;
    timezoneSelect.appendChild(blank);
    for (const tz of data.timezone_options || []) {
      const el = document.createElement('option');
      el.value = tz;
      el.textContent = tz;
      if (tz === data.timezone) el.selected = true;
      timezoneSelect.appendChild(el);
    }

    if (devModeCheck) devModeCheck.checked = !!data.dev_mode;

    if (voiceModeSelect) {
      voiceModeSelect.innerHTML = '';
      for (const opt of data.tts_voice_mode_options || []) {
        const el = document.createElement('option');
        el.value = opt.code;
        el.textContent = opt.name;
        if (opt.code === data.tts_voice_mode) el.selected = true;
        voiceModeSelect.appendChild(el);
      }
    }

    if (data.wifi_regdomain_current && data.wifi_regdomain_current !== data.wifi_regdomain) {
      regdomainLive.textContent =
        `Live radio: ${data.wifi_regdomain_current} — will switch to ${data.wifi_regdomain} on next hotspot rotation`;
      regdomainLive.classList.add('mismatch');
    } else if (data.wifi_regdomain_current) {
      regdomainLive.textContent = `Live radio: ${data.wifi_regdomain_current}`;
      regdomainLive.classList.remove('mismatch');
    } else {
      regdomainLive.textContent = '';
    }
  };

  const load = async () => {
    setStatus('Loading…');
    try {
      const resp = await fetch(`${API}/api/admin/settings`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      populate(data);
      loaded = true;
      setStatus('');
    } catch (err) {
      setStatus('Could not load settings', 'error');
    }
  };

  const save = async () => {
    if (!loaded || !lastLoadedData) return;
    const body = {
      wifi_regdomain: regdomainSelect.value,
      timezone: timezoneSelect.value,
      dev_mode: devModeCheck ? devModeCheck.checked : false,
    };
    if (voiceModeSelect && voiceModeSelect.value) {
      body.tts_voice_mode = voiceModeSelect.value;
    }
    // WiFi mode + admin creds
    if (wifiModeSelect) body.wifi_mode = wifiModeSelect.value;
    if (adminSsidInput && adminSsidInput.value.trim()) {
      body.admin_ssid = adminSsidInput.value.trim();
    }
    if (adminPasswordInput && adminPasswordInput.value) {
      body.admin_password = adminPasswordInput.value;
    }

    // Warn if switching away from admin while likely connected over WiFi
    const oldMode = lastLoadedData.wifi_mode;
    const newMode = body.wifi_mode;
    if (oldMode === 'admin' && newMode !== 'admin') {
      const ok = await confirmDialog('Switch WiFi mode?', 'Switching away from admin mode will disconnect you from WiFi.', 'Continue', false);
      if (!ok) return;
    }

    setStatus('Applying…');
    saveBtn.disabled = true;
    try {
      const resp = await fetch(`${API}/api/admin/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json().catch(() => ({}));

      // 202 = async WiFi mode switch in progress
      if (resp.status === 202) {
        setStatus(`Switching WiFi to ${data.wifi_mode || newMode}…`, 'ok');
        // Poll until live state matches desired, or timeout
        let attempts = 0;
        const poll = setInterval(async () => {
          attempts++;
          try {
            const pollResp = await fetch(`${API}/api/admin/settings`);
            if (pollResp.ok) {
              const pollData = await pollResp.json();
              populate(pollData);
              if (pollData.wifi_active === (newMode !== 'off') || attempts > 20) {
                clearInterval(poll);
                setStatus('WiFi mode switched', 'ok');
                setTimeout(() => setStatus(''), 3000);
              }
            }
          } catch { /* ignore poll errors */ }
          if (attempts > 20) {
            clearInterval(poll);
            setStatus('WiFi switch may still be in progress', 'error');
          }
        }, 2000);
        return;
      }

      if (!resp.ok) {
        setStatus(data.error || `HTTP ${resp.status}`, 'error');
        return;
      }
      populate(data);
      if (data.runtime_ok === false) {
        setStatus(
          `Saved, but 'iw reg set ${data.wifi_regdomain}' did not take effect`,
          'error',
        );
      } else {
        setStatus('Saved', 'ok');
        setTimeout(() => {
          if (status.textContent === 'Saved') setStatus('');
        }, 2500);
      }
    } catch {
      setStatus('Network error', 'error');
    } finally {
      saveBtn.disabled = false;
    }
  };

  const open = async () => {
    panel.classList.add('open');
    backdrop.classList.add('open');
    panel.setAttribute('aria-hidden', 'false');
    openBtn.setAttribute('aria-expanded', 'true');
    if (!loaded) await load();
    // Defer focus to allow the transition to start
    setTimeout(() => {
      const firstField =
        regdomainSelect.disabled ? timezoneSelect : regdomainSelect;
      firstField.focus();
    }, 60);
  };

  const close = () => {
    panel.classList.remove('open');
    backdrop.classList.remove('open');
    panel.setAttribute('aria-hidden', 'true');
    openBtn.setAttribute('aria-expanded', 'false');
    openBtn.focus();
  };

  openBtn.addEventListener('click', () => {
    if (panel.classList.contains('open')) close();
    else open();
  });
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('click', close);
  saveBtn.addEventListener('click', save);

  panel.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'Enter' && !e.isComposing) {
      // Don't trigger on select (browser handles Enter to open dropdown).
      // Also skip the terminal-access card so typing Enter in the secret
      // input doesn't submit the whole settings form.
      if (e.target.tagName !== 'SELECT' && !e.target.closest('.term-access-card, .term-font-row')) {
        e.preventDefault();
        save();
      }
    }
  });

  // ── Terminal access ──────────────────────────────────────────
  // Surfaces the admin secret so the user can authorize this browser (or
  // any other) without jumping to /admin/bootstrap. Scope-gated server
  // side: admin-LAN = allowed, guest hotspot = 403.
  const termCard        = document.getElementById('term-access-card');
  const termTitle       = document.getElementById('term-access-title');
  const termMeta        = document.getElementById('term-access-meta');
  const termSecretInput = document.getElementById('term-access-secret-input');
  const termSecretPath  = document.getElementById('term-access-secret-path');
  const termRevealBtn   = document.getElementById('btn-term-secret-reveal');
  const termCopyBtn     = document.getElementById('btn-term-secret-copy');
  const termAuthorizeBtn   = document.getElementById('btn-term-authorize');
  const termDeauthorizeBtn = document.getElementById('btn-term-deauthorize');

  let termAccess = null;   // { secret, secret_path, cookie_set, cookie_max_age_seconds }
  let termAccessInflight = null;

  const paintTermAccess = (data) => {
    termAccess = data;
    if (!termCard) return;
    if (!data) {
      termCard.dataset.state = 'unknown';
      termTitle.textContent = 'Status unavailable';
      termMeta.textContent = '';
      return;
    }
    termCard.dataset.state = data.cookie_set ? 'authorized' : 'unauthorized';
    termTitle.textContent = data.cookie_set
      ? 'This browser is authorized'
      : 'This browser is not authorized';
    const days = Math.round((data.cookie_max_age_seconds || 0) / 86400);
    termMeta.textContent = data.cookie_set
      ? `cookie · ${days}d max`
      : 'no cookie · paste secret or click authorize';
    if (termSecretInput) termSecretInput.value = data.secret || '';
    if (termSecretPath) {
      termSecretPath.textContent = data.secret_path
        ? `Stored at ${data.secret_path}`
        : '';
    }
    if (termDeauthorizeBtn) termDeauthorizeBtn.hidden = !data.cookie_set;
    if (termAuthorizeBtn) {
      termAuthorizeBtn.textContent = data.cookie_set
        ? 'Re-authorize'
        : 'Authorize this browser';
    }
  };

  const loadTermAccess = async () => {
    if (termAccessInflight) return termAccessInflight;
    termAccessInflight = (async () => {
      try {
        const resp = await fetch(`${API}/api/admin/terminal-access`, {
          credentials: 'include',
        });
        if (resp.status === 403) {
          paintTermAccess({ secret: '', secret_path: '', cookie_set: false });
          termCard.dataset.state = 'denied';
          termTitle.textContent = 'Guest scope — terminal disabled';
          termMeta.textContent = 'admin LAN only';
          return;
        }
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        paintTermAccess(await resp.json());
      } catch {
        paintTermAccess(null);
      } finally {
        termAccessInflight = null;
      }
    })();
    return termAccessInflight;
  };

  if (termRevealBtn && termSecretInput) {
    termRevealBtn.addEventListener('click', () => {
      const showing = termSecretInput.type === 'text';
      termSecretInput.type = showing ? 'password' : 'text';
      termRevealBtn.textContent = showing ? 'Show' : 'Hide';
    });
  }

  if (termCopyBtn && termSecretInput) {
    termCopyBtn.addEventListener('click', async () => {
      const secret = termSecretInput.value;
      if (!secret) return;
      try { await navigator.clipboard.writeText(secret); }
      catch {
        termSecretInput.type = 'text';
        termSecretInput.select();
        document.execCommand && document.execCommand('copy');
      }
      const prev = termCopyBtn.textContent;
      termCopyBtn.textContent = 'Copied';
      termCopyBtn.classList.add('btn-icon-success');
      setTimeout(() => {
        termCopyBtn.textContent = prev;
        termCopyBtn.classList.remove('btn-icon-success');
      }, 1400);
    });
  }

  if (termAuthorizeBtn) {
    termAuthorizeBtn.addEventListener('click', async () => {
      if (!termAccess || !termAccess.secret) {
        await loadTermAccess();
      }
      if (!termAccess || !termAccess.secret) {
        setStatus('Cannot read admin secret', 'error');
        return;
      }
      termAuthorizeBtn.disabled = true;
      try {
        const resp = await fetch(`${API}/api/admin/authorize`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ secret: termAccess.secret }),
        });
        if (!resp.ok) {
          setStatus('Authorization failed', 'error');
          return;
        }
        setStatus('Terminal cookie minted', 'ok');
        setTimeout(() => { if (status.textContent === 'Terminal cookie minted') setStatus(''); }, 2500);
        await loadTermAccess();
        // Nudge any existing terminal panel to re-attach now that the
        // cookie is set. _connect() is idempotent if a ws is already live.
        try {
          if (window._terminalPanel) window._terminalPanel._connect();
        } catch {}
      } finally {
        termAuthorizeBtn.disabled = false;
      }
    });
  }

  if (termDeauthorizeBtn) {
    termDeauthorizeBtn.addEventListener('click', async () => {
      termDeauthorizeBtn.disabled = true;
      try {
        await fetch(`${API}/api/admin/deauthorize`, {
          method: 'POST', credentials: 'include',
        });
        setStatus('Terminal cookie cleared', 'ok');
        setTimeout(() => { if (status.textContent === 'Terminal cookie cleared') setStatus(''); }, 2500);
        await loadTermAccess();
      } finally {
        termDeauthorizeBtn.disabled = false;
      }
    });
  }

  // Font-size slider — keyboard shortcuts Ctrl+/−/0 still work inside xterm.
  const termFontSlider  = document.getElementById('setting-term-font-size');
  const termFontValue   = document.getElementById('term-font-value');
  const termFontDecBtn  = document.getElementById('btn-term-font-dec');
  const termFontIncBtn  = document.getElementById('btn-term-font-inc');

  const clampFont = (n) => Math.max(9, Math.min(26, Math.round(n)));
  const applyTerminalFont = (px, { push = true } = {}) => {
    const v = clampFont(px);
    if (termFontSlider && String(termFontSlider.value) !== String(v)) {
      termFontSlider.value = String(v);
    }
    if (termFontValue) termFontValue.textContent = `${v}px`;
    if (push) {
      try { localStorage.setItem('terminal_font_size', String(v)); } catch {}
      if (window._terminalPanel && typeof window._terminalPanel.setFontSize === 'function') {
        window._terminalPanel.setFontSize(v);
      }
    }
  };
  // Initial paint from localStorage.
  applyTerminalFont(parseInt(localStorage.getItem('terminal_font_size') || '13', 10), { push: false });
  if (termFontSlider) {
    termFontSlider.addEventListener('input', () => applyTerminalFont(termFontSlider.value));
  }
  if (termFontDecBtn) termFontDecBtn.addEventListener('click', () => applyTerminalFont(clampFont((termFontSlider && +termFontSlider.value) || 13) - 1));
  if (termFontIncBtn) termFontIncBtn.addEventListener('click', () => applyTerminalFont(clampFont((termFontSlider && +termFontSlider.value) || 13) + 1));
  // React when the terminal panel itself changes font size (e.g. Ctrl+=).
  window.addEventListener('terminal:font-size', (e) => {
    if (e && e.detail && typeof e.detail.size === 'number') {
      applyTerminalFont(e.detail.size, { push: false });
    }
  });

  // Refresh terminal-access card whenever the settings panel is opened.
  const origOpen = open;
  // Monkey-patch the outer open() to also refresh term access.
  // (Open is a local const — but it's closed over by the click handler so
  // we just chain a call here via an observer on aria-hidden.)
  const observer = new MutationObserver(() => {
    if (panel.getAttribute('aria-hidden') === 'false') loadTermAccess();
  });
  observer.observe(panel, { attributes: true, attributeFilter: ['aria-hidden'] });
  // Avoid unused-var lint on origOpen (we intentionally don't mutate the fn).
  void origOpen;
})();

// ─── Fetch Languages ───────────────────────────────────────
//
// Two independent dropdowns (Language A ↔ Language B) over the full set of
// TTS-capable languages. Every (a, b) with a != b is a valid pair — the old
// "popular pairs" curated list was removed because (1) it was English-centric
// (missing combos like de↔fr, es↔it) and (2) it rendered each option using
// the *native* name only, so "Deutsch ↔ English" was routinely misread as
// "Dutch ↔ English" and users ended up in a German meeting when they meant
// Dutch. Labels now show "English — native" (e.g. "German — Deutsch") so the
// false cognate can't strike again.
// Hard-coded fallback so the dropdowns never render empty even if the API
// is down or returns a stale shape. Kept in sync with the Python registry.
const _LANG_FALLBACK = [
  {code: 'en', name: 'English',    native_name: 'English',         tts_supported: true},
  {code: 'zh', name: 'Chinese',    native_name: '中文',             tts_supported: true},
  {code: 'ja', name: 'Japanese',   native_name: '日本語',           tts_supported: true},
  {code: 'ko', name: 'Korean',     native_name: '한국어',           tts_supported: true},
  {code: 'fr', name: 'French',     native_name: 'Français',        tts_supported: true},
  {code: 'de', name: 'German',     native_name: 'Deutsch',         tts_supported: true},
  {code: 'es', name: 'Spanish',    native_name: 'Español',         tts_supported: true},
  {code: 'it', name: 'Italian',    native_name: 'Italiano',        tts_supported: true},
  {code: 'pt', name: 'Portuguese', native_name: 'Português',       tts_supported: true},
  {code: 'ru', name: 'Russian',    native_name: 'Русский',         tts_supported: true},
  {code: 'nl', name: 'Dutch',      native_name: 'Nederlands',      tts_supported: false},
  {code: 'ar', name: 'Arabic',     native_name: 'العربية',          tts_supported: false},
  {code: 'th', name: 'Thai',       native_name: 'ไทย',             tts_supported: false},
  {code: 'vi', name: 'Vietnamese', native_name: 'Tiếng Việt',      tts_supported: false},
  {code: 'id', name: 'Indonesian', native_name: 'Bahasa Indonesia', tts_supported: false},
  {code: 'ms', name: 'Malay',      native_name: 'Bahasa Melayu',   tts_supported: false},
  {code: 'hi', name: 'Hindi',      native_name: 'हिन्दी',           tts_supported: false},
  {code: 'tr', name: 'Turkish',    native_name: 'Türkçe',          tts_supported: false},
  {code: 'pl', name: 'Polish',     native_name: 'Polski',          tts_supported: false},
  {code: 'uk', name: 'Ukrainian',  native_name: 'Українська',      tts_supported: false},
];

(async function loadLanguages() {
  const selA = document.getElementById('lang-a-select');
  const selB = document.getElementById('lang-b-select');

  const buildOption = (lang) => {
    const opt = document.createElement('option');
    opt.value = lang.code;
    // Show English name first so "Deutsch" can't be misread as "Dutch".
    let label = lang.native_name && lang.native_name !== lang.name
      ? `${lang.name} — ${lang.native_name}`
      : lang.name;
    if (lang.tts_supported === false) label += ' (text only)';
    opt.textContent = label;
    return opt;
  };

  const populate = (select, langs, selectedCode) => {
    select.innerHTML = '';
    for (const lang of langs) select.appendChild(buildOption(lang));
    if (selectedCode && [...select.options].some(o => o.value === selectedCode)) {
      select.value = selectedCode;
    }
  };

  // The right dropdown disables whichever language is currently on the
  // left, so the user can never pick the same language on both sides.
  // There is no __none__ sentinel on the setup screen — mono lives on
  // the landing page's quick-start path.
  const syncDisabled = () => {
    if (!selA || !selB) return;
    for (const opt of selA.options) opt.disabled = false;
    for (const opt of selB.options) {
      opt.disabled = (opt.value === selA.value);
    }
  };

  // If A and B collide, swap B to the next available language (NOT mono).
  const pickDifferent = (avoid, langs) => {
    return langs.find(l => l.code !== avoid)?.code || avoid;
  };

  let listenersAttached = false;
  let _langsCache = [];
  const wireUp = (langs, defaultPair) => {
    if (!selA || !selB || langs.length < 2) return;
    _langsCache = langs;
    for (const lang of langs) _languageNames[lang.code] = lang;

    // Promote the server default to bilingual for the setup screen. The
    // setup screen is always bilingual; mono-only defaults (legacy config)
    // still get paired here so the dropdowns are usable.
    const [defaultA, defaultB] = defaultPair.length >= 2
      ? defaultPair
      : [defaultPair[0], pickDifferent(defaultPair[0], langs)];
    _defaultLanguagePairCache = [defaultA, defaultB];

    // currentLanguagePair may be "en" (monolingual — user is returning
    // from a quick-start session) or "ja,en" (bilingual). Either way, the
    // setup screen renders a bilingual pair.
    const curParts = (currentLanguagePair || `${defaultA},${defaultB}`).split(',');
    const has = (code) => langs.some(l => l.code === code);
    const pickA = has(curParts[0]) ? curParts[0] : defaultA;
    let pickB;
    if (curParts.length >= 2 && has(curParts[1]) && curParts[1] !== pickA) {
      pickB = curParts[1];
    } else if (has(defaultB) && defaultB !== pickA) {
      pickB = defaultB;
    } else {
      pickB = pickDifferent(pickA, langs);
    }

    populate(selA, langs, pickA);
    populate(selB, langs, pickB);
    currentLanguagePair = `${selA.value},${selB.value}`;
    syncDisabled();
    // Setup screen is always bilingual — make sure the mono CSS never
    // sticks around from a previous session.
    const selector = document.getElementById('language-selector');
    if (selector) selector.classList.remove('mono');

    if (listenersAttached) return;
    listenersAttached = true;
    const onChangeA = () => {
      // Collision with B → swap B to a different language (NOT mono).
      if (selA.value === selB.value) {
        selB.value = pickDifferent(selA.value, _langsCache);
      }
      currentLanguagePair = `${selA.value},${selB.value}`;
      syncDisabled();
      _updateColumnHeaders();
    };
    const onChangeB = () => {
      // Collision with A (shouldn't happen due to disabled options) →
      // swap B to a different language.
      if (selB.value === selA.value) {
        selB.value = pickDifferent(selA.value, _langsCache);
      }
      currentLanguagePair = `${selA.value},${selB.value}`;
      syncDisabled();
      _updateColumnHeaders();
    };
    selA.addEventListener('change', onChangeA);
    selB.addEventListener('change', onChangeB);
  };

  // Render the fallback synchronously so the dropdowns are never blank, even
  // for a split second while the fetch is in flight or if the server is down.
  wireUp(_LANG_FALLBACK, ['en', 'ja']);

  try {
    const resp = await fetch(`${API}/api/languages`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const langs = (data.languages && data.languages.length >= 2) ? data.languages : _LANG_FALLBACK;
    const defaultPair = (
      Array.isArray(data.default_pair)
      && data.default_pair.length >= 1
      && data.default_pair.length <= 2
    ) ? data.default_pair : ['en', 'ja'];
    wireUp(langs, defaultPair);
  } catch (e) {
    console.warn('Failed to load /api/languages, using built-in fallback list:', e);
  }
  _updateColumnHeaders();
})();

// ─── Keyboard Shortcuts ─────────────────────────────────────

document.addEventListener('keydown', (e) => {
  // Ignore if typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  // Space bar: play/pause audio
  if (e.code === 'Space' && audioPlayer.audio.src) {
    e.preventDefault();
    audioPlayer.togglePlay();
  }
});

// ─── URL-based Meeting Navigation ───────────────────────────

// Hash-based routing: #home, #setup, #meeting/{id}
// Navigation is always honored — the live pipeline keeps streaming
// to the server regardless of which panel is visible, and the
// "Meeting in progress" banner drives the user back when needed.
function handleHashRoute() {
  const hash = location.hash || '#home';

  if (hash === '#home' || hash === '#' || hash === '') {
    if (document.body.classList.contains('recording')
     || document.body.classList.contains('meeting-active')) {
      exitLiveMeetingView();
    }
    showLanding();
    return;
  }

  if (hash === '#setup') {
    if (document.body.classList.contains('recording')
     || document.body.classList.contains('meeting-active')) {
      exitLiveMeetingView();
    }
    hideLanding();
    meetingsMgr?.showSetup();
    return;
  }

  const match = hash.match(/^#meeting\/(.+)/);
  if (match && match[1]) {
    const meetingId = match[1];
    setTimeout(() => meetingsMgr?.viewMeeting(meetingId), 500);
  }
}

// Set hash when navigating views
const _origViewMeeting = MeetingsManager.prototype.viewMeeting;
MeetingsManager.prototype.viewMeeting = async function(meetingId) {
  history.pushState(null, '', `#meeting/${meetingId}`);
  return _origViewMeeting.call(this, meetingId);
};

const _origShowSetup = MeetingsManager.prototype.showSetup;
MeetingsManager.prototype.showSetup = function() {
  history.pushState(null, '', '#setup');
  return _origShowSetup.call(this);
};

// Handle initial hash on page load + back/forward (skip in pop-out mode)
if (!POPOUT_MODE) {
  handleHashRoute();
  window.addEventListener('hashchange', handleHashRoute);
}

// ─── Summary Panel (for past meeting review) ────────────────
// In review mode the summary bar is already populated with the review
// toolbar (rv-* buttons) by viewMeeting(). This function appends the
// "View Summary" button to the existing tools row when a summary is
// available. The finalization modal is the single source of truth for
// summary content (stats, topics, decisions, actions, speaker breakdown,
// downloads); this button is just the entry point.

function _renderSummaryPanel(summary, meetingId) {
  const panel = document.getElementById('meeting-summary-panel');
  if (!panel || !summary || summary.error) return;
  const tools = panel.querySelector('.summary-bar-tools');
  if (!tools) return;

  // Don't inject twice if called again (e.g. after re-finalize)
  if (tools.querySelector('#summary-bar-open')) return;

  const btn = document.createElement('button');
  btn.className = 'btn-ghost summary-bar-btn';
  btn.id = 'summary-bar-open';
  btn.title = 'Open the full meeting summary with topics, decisions, action items, and downloads';
  btn.textContent = 'View Summary';
  btn.addEventListener('click', () => showFinalizationSummaryFor(meetingId));
  tools.appendChild(btn);
}

// ─── Listen Toggle (Real-Time Interpretation Audio) ─────────

// Auto-start Listen if navigated via /#listen (e.g. from /demo page).
// Admin scope is excluded — see _isAdminScope guard below.
if (location.hash === '#listen' && !(location.protocol === 'https:' || location.port === '8080')) {
  const _waitForRecording = setInterval(() => {
    if (document.body.classList.contains('recording')) {
      clearInterval(_waitForRecording);
      document.getElementById('btn-listen')?.click();
      history.replaceState(null, '', location.pathname);
    }
  }, 1000);
  setTimeout(() => clearInterval(_waitForRecording), 30000);
}

// Admin scope (HTTPS / port 8080) must never play interpretation audio —
// the operator sits next to the speaker and bleeding TTS back into the
// room creates a feedback loop. Hide the Listen button entirely on admin.
// Guests (HTTP / port 80) keep the button.
const _isAdminScope = location.protocol === 'https:' || location.port === '8080';
if (_isAdminScope) {
  const _hideListen = () => {
    ['btn-listen', 'listen-lang', 'listen-mode'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
  };
  _hideListen();
}

document.getElementById('btn-listen')?.addEventListener('click', async () => {
  if (_isAdminScope) return;  // belt + suspenders
  const btn = document.getElementById('btn-listen');
  const langSelect = document.getElementById('listen-lang');

  const modeSelect = document.getElementById('listen-mode');

  if (audioOutListener.enabled) {
    audioOutListener.stop();
    btn.classList.remove('active');
    langSelect.style.display = 'none';
    modeSelect.style.display = 'none';
  } else {
    // CRITICAL: prime the AudioContext SYNCHRONOUSLY here, BEFORE any
    // await. If we let `audioOutListener.start()` create the context
    // after the language fetch resolves, the user-gesture context is
    // already gone and the resume() runs in a deferred microtask that
    // browsers refuse to honor. The result is a permanently-suspended
    // AudioContext that decodes every incoming WAV into a muted
    // destination — server says "delivered", user hears nothing.
    audioOutListener.primeAudioContext();

    // Populate language selector if empty
    if (langSelect.options.length === 0) {
      try {
        const resp = await fetch(`${API}/api/languages`);
        const data = await resp.json();
        (data.languages || []).forEach(l => {
          const opt = document.createElement('option');
          opt.value = l.code;
          opt.textContent = l.native_name || l.name;
          langSelect.appendChild(opt);
        });
        // Default to the "other" language in the pair
        const langB = _getLangB();
        if (langB) langSelect.value = langB;
      } catch { /* ignore */ }
    }
    const lang = langSelect.value || _getLangB() || 'en';
    const mode = modeSelect.value || 'translation';
    await audioOutListener.start(lang, mode);
    btn.classList.add('active');
    langSelect.style.display = '';
    modeSelect.style.display = '';
  }
});

document.getElementById('listen-lang')?.addEventListener('change', (e) => {
  audioOutListener.setLanguage(e.target.value);
});

document.getElementById('listen-mode')?.addEventListener('change', (e) => {
  audioOutListener.setMode(e.target.value);
});

// ─── Metrics Dashboard ──────────────────────────────────────

document.getElementById('btn-metrics')?.addEventListener('click', (e) => {
  const dash = document.getElementById('metrics-dashboard');
  const isVisible = dash.style.display !== 'none';
  dash.style.display = isVisible ? 'none' : '';
  document.body.classList.toggle('metrics-split', !isVisible);
  e.target.classList.toggle('active-toggle', !isVisible);
});

// Layout toggle buttons — "Live" opens a pop-out translation view
document.getElementById('btn-toggle-live')?.addEventListener('click', (e) => {
  _openPopout(e.target);
});

// Column selectors (active meeting)
document.getElementById('btn-col-a')?.addEventListener('click', (e) => {
  const wasActive = document.body.classList.contains('show-only-a');
  document.body.classList.remove('show-only-a', 'show-only-b');
  document.getElementById('btn-col-b')?.classList.remove('active-toggle');
  if (!wasActive) { document.body.classList.add('show-only-a'); e.target.classList.add('active-toggle'); }
  else { e.target.classList.remove('active-toggle'); }
});
document.getElementById('btn-col-b')?.addEventListener('click', (e) => {
  const wasActive = document.body.classList.contains('show-only-b');
  document.body.classList.remove('show-only-a', 'show-only-b');
  document.getElementById('btn-col-a')?.classList.remove('active-toggle');
  if (!wasActive) { document.body.classList.add('show-only-b'); e.target.classList.add('active-toggle'); }
  else { e.target.classList.remove('active-toggle'); }
});

document.getElementById('btn-toggle-table')?.addEventListener('click', (e) => {
  // Flip hide-table, then set active-toggle to match VISIBLE state
  // (active = table is showing)
  const nowHidden = document.body.classList.toggle('hide-table');
  e.target.classList.toggle('active-toggle', !nowHidden);
});

// ─── Room Editor Overlay (mid-meeting + review) ─────────────

function openRoomEditor(mode, meetingId) {
  const overlay = document.getElementById('room-editor-overlay');
  const canvas = document.getElementById('room-editor-canvas');
  const sidebar = document.getElementById('room-editor-sidebar');
  const titleEl = document.getElementById('room-editor-title-text');
  const badge = document.getElementById('room-editor-mode-badge');

  if (!overlay || !canvas) return;

  titleEl.textContent = mode === 'review' ? 'Edit Layout & Assign Voices'
    : mode === 'live' ? 'Edit Room & Assign Live Voices'
    : 'Edit Room Layout';
  badge.className = `room-editor-mode-badge ${mode}`;
  badge.textContent = mode === 'live' ? 'LIVE' : mode === 'review' ? 'REVIEW' : '';
  // Show sidebar for BOTH live and review modes so the user can assign
  // detected voices to seats mid-meeting.
  sidebar.style.display = (mode === 'review' || mode === 'live') ? '' : 'none';

  if (mode === 'review' && meetingId) {
    // Load persisted room.json for this past meeting
    fetch(`${API}/api/meetings/${meetingId}/room`)
      .then(r => r.ok ? r.json() : null)
      .then(layout => {
        if (layout) {
          roomSetup.tables.forEach(t => t.element?.remove());
          roomSetup.seats.forEach(s => s.element?.remove());
          roomSetup.tables = [];
          roomSetup.seats = [];
          (layout.tables || []).forEach(t => {
            const tbl = { tableId: t.table_id, x: t.x, y: t.y, width: t.width, height: t.height, borderRadius: t.border_radius, label: t.label || '', element: null };
            roomSetup.tables.push(tbl);
            roomSetup._renderTable(tbl);
            roomSetup._applyTable(tbl);
          });
          (layout.seats || []).forEach(s => {
            const seat = { seatId: s.seat_id, x: s.x, y: s.y, name: s.speaker_name || '', enrolled: !!s.enrollment_id, enrollmentId: s.enrollment_id || null, element: null };
            roomSetup.seats.push(seat);
            roomSetup._renderSeat(seat);
            roomSetup._applySeat(seat);
          });
        }
        roomSetup.mount(canvas, mode, { meetingId });
        _populateClusterChips(meetingId);
      })
      .catch(() => {
        roomSetup.mount(canvas, mode, { meetingId });
      });
  } else if (mode === 'live' && meetingId) {
    // For an active meeting: keep the user's existing room layout but
    // (1) ensure there's a seat for every detected speaker, and
    // (2) populate the sidebar with currently detected voices.
    roomSetup.mount(canvas, mode, { meetingId });
    // If the user hasn't set up any table yet, give them a default one
    // so new seats can be arranged around it
    if (roomSetup.tables.length === 0) {
      roomSetup.addTable();
    }
    roomSetup.reconcileSeatsToDetectedSpeakers();
    _populateClusterChipsLive(meetingId);
  } else {
    roomSetup.mount(canvas, mode, { meetingId });
  }

  overlay.style.display = 'flex';

  overlay.querySelectorAll('.room-editor-actions .preset-btn').forEach(btn => {
    btn.onclick = () => roomSetup.applyPreset(btn.dataset.preset);
  });
  document.getElementById('room-editor-add-table').onclick = () => roomSetup.addTable();
  document.getElementById('room-editor-add-seat').onclick = () => roomSetup.addSeatAtCenter();
}

function closeRoomEditor() {
  const overlay = document.getElementById('room-editor-overlay');
  if (!overlay || overlay.style.display === 'none') return;
  overlay.style.display = 'none';
  roomSetup.unmount();
  // Stop the live-chip refresh loop if it was running
  if (window._liveChipRefreshTimer) {
    clearInterval(window._liveChipRefreshTimer);
    window._liveChipRefreshTimer = null;
  }
  if (document.getElementById('meeting-mode')?.style.display !== 'none') {
    roomSetup._renderTableStrip();
  }
}

/** Populate the cluster sidebar with LIVE detected speakers from the
 *  client-side speaker registry. Refreshes every 2s while the editor
 *  is open so new speakers appear as they're detected.
 */
function _populateClusterChipsLive(meetingId) {
  const chipsEl = document.getElementById('cluster-chips');
  if (!chipsEl) return;

  const render = () => {
    const speakers = getAllSpeakers();
    if (speakers.length === 0) {
      chipsEl.innerHTML =
        '<div style="color:var(--text-muted);font-size:0.75rem;padding:0.5rem">' +
        'No voices detected yet. Speakers will appear here as they speak.</div>';
      return;
    }

    const boundNames = new Set(roomSetup.seats.map(s => s.name).filter(Boolean));
    chipsEl.innerHTML = '';

    speakers.forEach((sp) => {
      const clusterId = sp.clusterId;
      const name = sp.displayName;
      const color = getSpeakerColor(clusterId);
      const isBound = boundNames.has(name);

      const chip = document.createElement('div');
      chip.className = `cluster-chip${isBound ? ' cluster-chip-bound' : ''}`;
      chip.draggable = !isBound;
      chip.dataset.clusterId = String(clusterId);
      chip.dataset.displayName = name;
      chip.style.setProperty('--chip-color', color);
      chip.innerHTML = `
        <div class="cluster-chip-swatch"></div>
        <div class="cluster-chip-body">
          <div class="cluster-chip-name">${esc(name)}</div>
          <div class="cluster-chip-stats">Speaker ${sp.seqIndex}${sp.hasCustomName ? ' · named' : ''}</div>
        </div>
        <button class="cluster-chip-rename" title="Rename this voice">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
        </button>
      `;

      chip.querySelector('.cluster-chip-rename')?.addEventListener('click', (e) => {
        e.stopPropagation();
        _openSpeakerRenameModal(clusterId, name, color);
      });

      if (!isBound) {
        chip.addEventListener('dragstart', (ev) => {
          chip.classList.add('dragging');
          ev.dataTransfer.setData('text/plain', JSON.stringify({
            cluster_id: clusterId,
            display_name: name,
          }));
          ev.dataTransfer.effectAllowed = 'move';
        });
        chip.addEventListener('dragend', () => {
          chip.classList.remove('dragging');
          document.querySelectorAll('.seat-node.drop-target').forEach(el =>
            el.classList.remove('drop-target')
          );
        });
      }

      chip.addEventListener('dblclick', () => {
        _openSpeakerRenameModal(clusterId, name, color);
      });

      chipsEl.appendChild(chip);
    });

    _wireSeatDropTargets(meetingId);
  };

  render();
  // Auto-refresh every 2s so new speakers show up as they join. Also
  // reconcile the canvas seats so each new speaker gets a new seat.
  if (window._liveChipRefreshTimer) clearInterval(window._liveChipRefreshTimer);
  window._liveChipRefreshTimer = setInterval(() => {
    roomSetup.reconcileSeatsToDetectedSpeakers();
    render();
    _wireSeatDropTargets(meetingId);
  }, 2000);
}

async function _populateClusterChips(meetingId) {
  const chipsEl = document.getElementById('cluster-chips');
  if (!chipsEl) return;
  chipsEl.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem">Loading...</div>';

  try {
    const resp = await fetch(`${API}/api/meetings/${meetingId}/speakers`);
    const data = await resp.json();
    const speakers = data.speakers || [];
    if (speakers.length === 0) {
      chipsEl.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem">No voices detected.</div>';
      return;
    }

    const boundNames = new Set(roomSetup.seats.map(s => s.name).filter(Boolean));

    chipsEl.innerHTML = '';
    speakers.forEach((sp, i) => {
      const clusterId = sp.cluster_id ?? sp.speaker_id ?? i;
      // Unified naming — ignores generic "Speaker N" server labels
      const name = getSpeakerDisplayName(Number(clusterId), sp.display_name);
      const color = getSpeakerColor(Number(clusterId));
      const segCount = sp.segment_count || 0;
      const totalMs = sp.total_speaking_ms || 0;
      const totalSec = Math.round(totalMs / 1000);
      const timeStr = totalSec >= 60 ? `${Math.floor(totalSec / 60)}m ${totalSec % 60}s` : `${totalSec}s`;
      const isBound = boundNames.has(name);

      const firstSeenMs = sp.first_seen_ms || 0;

      const chip = document.createElement('div');
      chip.className = `cluster-chip${isBound ? ' cluster-chip-bound' : ''}`;
      chip.draggable = !isBound;
      chip.dataset.clusterId = String(clusterId);
      chip.dataset.displayName = name;
      chip.dataset.firstSeenMs = String(firstSeenMs);
      chip.style.setProperty('--chip-color', color);
      chip.innerHTML = `
        <div class="cluster-chip-swatch"></div>
        <div class="cluster-chip-body">
          <div class="cluster-chip-name">${esc(name)}</div>
          <div class="cluster-chip-stats">${segCount} segments · ${timeStr}</div>
        </div>
        <button class="cluster-chip-play" title="Play a 4s sample">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
        </button>
      `;

      // Play a 4-second sample starting at first_seen_ms
      chip.querySelector('.cluster-chip-play')?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!audioPlayer?.audio) return;
        audioPlayer.seekFullMeeting?.(firstSeenMs);
        // Auto-pause after ~4s
        clearTimeout(window._clusterSampleTimer);
        window._clusterSampleTimer = setTimeout(() => {
          if (audioPlayer?.audio && !audioPlayer.audio.paused) {
            audioPlayer.audio.pause();
            const btn = document.getElementById('player-play');
            if (btn) { btn.textContent = '▶'; btn.classList.remove('playing'); }
          }
        }, 4000);
      });

      if (!isBound) {
        chip.addEventListener('dragstart', (e) => {
          chip.classList.add('dragging');
          e.dataTransfer.setData('text/plain', JSON.stringify({
            cluster_id: Number(clusterId),
            display_name: name,
          }));
          e.dataTransfer.effectAllowed = 'move';
        });
        chip.addEventListener('dragend', () => {
          chip.classList.remove('dragging');
          document.querySelectorAll('.seat-node.drop-target').forEach(el =>
            el.classList.remove('drop-target')
          );
        });
      }

      chip.addEventListener('dblclick', async () => {
        const newName = await promptDialog(
          'Rename voice',
          'Set a display name for this voice cluster. Used wherever the cluster appears (transcript, timeline, speaker lanes).',
          {
            initialValue: name,
            placeholder: 'Voice name',
            confirmText: 'Rename',
          },
        );
        if (!newName || newName === name) return;
        _assignCluster(meetingId, Number(clusterId), null, newName);
      });

      chipsEl.appendChild(chip);
    });

    _wireSeatDropTargets(meetingId);
  } catch (e) {
    chipsEl.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem">Error loading voices</div>';
  }
}

function _wireSeatDropTargets(meetingId) {
  document.querySelectorAll('#room-editor-canvas .seat-node').forEach(seatEl => {
    seatEl.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      seatEl.classList.add('drop-target');
    });
    seatEl.addEventListener('dragleave', () => {
      seatEl.classList.remove('drop-target');
    });
    seatEl.addEventListener('drop', async (e) => {
      e.preventDefault();
      seatEl.classList.remove('drop-target');
      try {
        const data = JSON.parse(e.dataTransfer.getData('text/plain'));
        const seatId = seatEl.dataset.seatId;
        if (!seatId) return;
        await _assignCluster(meetingId, data.cluster_id, seatId, data.display_name);
      } catch {}
    });

    // Right-click a bound seat → unbind the cluster
    seatEl.addEventListener('contextmenu', async (e) => {
      const seatId = seatEl.dataset.seatId;
      if (!seatId) return;
      const seat = roomSetup.seats.find(s => s.seatId === seatId);
      if (!seat || !seat.name) return; // Only unbind if there's something bound
      e.preventDefault();
      if (!await confirmDialog('Unbind speaker?', `Remove "${esc(seat.name)}" from this seat?`, 'Unbind', true)) return;
      await _unbindCluster(meetingId, seatId);
    });
  });
}

async function _unbindCluster(meetingId, seatId) {
  const seat = roomSetup.seats.find(s => s.seatId === seatId);
  if (!seat) return;

  // Fetch the current speakers list to find which cluster this name maps to
  let clusterId = null;
  try {
    const resp = await fetch(`${API}/api/meetings/${meetingId}/speakers`);
    const data = await resp.json();
    const match = (data.speakers || []).find(sp =>
      (sp.display_name || '') === seat.name
    );
    if (match) {
      clusterId = match.cluster_id ?? match.speaker_id ?? null;
    }
  } catch {}

  // Clear the seat name locally and persist via /room/layout
  seat.name = "";
  const nameEl = seat.element?.querySelector('.seat-name');
  if (nameEl) nameEl.textContent = '';
  roomSetup._persistLayout();

  // Reset the cluster's display name in the client registry, then sync
  // the backend with the sequential fallback produced by getSpeakerDisplayName.
  if (clusterId != null) {
    const cid = Number(clusterId);
    // Clear any user-assigned name and re-derive the sequential label
    const entry = _speakerRegistry.clusters.get(cid);
    if (entry) entry.displayName = null;
    _speakerRegistry.clusters.set(cid, entry || { seqIndex: _speakerRegistry.nextIndex++ });
    const defaultName = getSpeakerDisplayName(cid);
    try {
      await fetch(`${API}/api/meetings/${meetingId}/speakers/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cluster_id: cid,
          seat_id: null,
          display_name: defaultName,
        }),
      });
    } catch {}
    _refreshTranscriptSpeakerLabels();
    _refreshDetectedSpeakersStrip();
  }

  // Refresh chip sidebar
  await _populateClusterChips(meetingId);
}

async function _assignCluster(meetingId, clusterId, seatId, displayName) {
  try {
    const resp = await fetch(`${API}/api/meetings/${meetingId}/speakers/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        cluster_id: clusterId,
        seat_id: seatId,
        display_name: displayName,
      }),
    });
    if (!resp.ok) {
      console.error('Assign failed:', await resp.text());
      return;
    }
    if (seatId) {
      const seat = roomSetup.seats.find(s => s.seatId === seatId);
      if (seat) {
        seat.name = displayName;
        const nameEl = seat.element?.querySelector('.seat-name');
        if (nameEl) nameEl.textContent = displayName;
      }
    }
    // Also update the client-side registry so the transcript + seat strip
    // reflect the new name immediately.
    renameSpeaker(clusterId, displayName);
    _refreshTranscriptSpeakerLabels();
    // Refresh sidebar using whichever populator is appropriate
    const badge = document.getElementById('room-editor-mode-badge');
    const isLive = badge?.classList.contains('live');
    if (isLive) {
      _populateClusterChipsLive(meetingId);
    } else {
      await _populateClusterChips(meetingId);
    }
  } catch (e) {
    console.error('Assign error:', e);
  }
}

document.getElementById('room-editor-close')?.addEventListener('click', closeRoomEditor);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('room-editor-overlay')?.style.display === 'flex') {
    closeRoomEditor();
  }
});

document.getElementById('meeting-table-strip')?.addEventListener('click', () => {
  if (!document.body.classList.contains('meeting-active')) return;
  // Don't open for past-meeting viewing — only for the active live meeting
  if (document.body.classList.contains('view-only')) return;
  const mid = window.current_meeting_id || null;
  if (mid) openRoomEditor('live', mid);
});

// Refresh mini strip when the server broadcasts a layout update
window._onRoomLayoutUpdate = function(layout) {
  if (!layout) return;
  roomSetup.tables.forEach(t => t.element?.remove());
  roomSetup.seats.forEach(s => s.element?.remove());
  roomSetup.tables = [];
  roomSetup.seats = [];
  (layout.tables || []).forEach(t => {
    const tbl = { tableId: t.table_id, x: t.x, y: t.y, width: t.width, height: t.height, borderRadius: t.border_radius, label: t.label || '', element: null };
    roomSetup.tables.push(tbl);
    roomSetup._renderTable(tbl);
    roomSetup._applyTable(tbl);
  });
  (layout.seats || []).forEach(s => {
    const seat = { seatId: s.seat_id, x: s.x, y: s.y, name: s.speaker_name || '', enrolled: !!s.enrollment_id, enrollmentId: s.enrollment_id || null, element: null };
    roomSetup.seats.push(seat);
    roomSetup._renderSeat(seat);
    roomSetup._applySeat(seat);
  });
  roomSetup._renderTableStrip();
};

document.getElementById('btn-compact')?.addEventListener('click', (e) => {
  document.body.classList.toggle('compact-mode');
  e.target.classList.toggle('active-toggle');
});

// ─── 1:1 Conversation Mode ──────────────────────────────────

let _oneOnOneActive = false;
let _oneOnOneSwapped = false;
const _oneOnOneSpeakers = { left: null, right: null }; // cluster_id → side

document.getElementById('btn-one-on-one')?.addEventListener('click', (e) => {
  _oneOnOneActive = !_oneOnOneActive;
  e.target.classList.toggle('active-toggle', _oneOnOneActive);
  document.getElementById('one-on-one-container').style.display = _oneOnOneActive ? '' : 'none';
  document.querySelector('.transcript-col-headers').style.display = _oneOnOneActive ? 'none' : '';
  document.getElementById('transcript-grid').style.display = _oneOnOneActive ? 'none' : '';
  // Reset speaker assignments on re-enable
  if (_oneOnOneActive) {
    _oneOnOneSpeakers.left = null;
    _oneOnOneSpeakers.right = null;
    document.getElementById('oo-left').innerHTML = '';
    document.getElementById('oo-right').innerHTML = '';
  }
});

document.getElementById('oo-swap')?.addEventListener('click', () => {
  _oneOnOneSwapped = !_oneOnOneSwapped;
  // Swap the speaker labels
  const leftLabel = document.getElementById('oo-speaker-left');
  const rightLabel = document.getElementById('oo-speaker-right');
  const tmp = leftLabel.textContent;
  leftLabel.textContent = rightLabel.textContent;
  rightLabel.textContent = tmp;
  // Swap cluster_id assignments
  const tmpId = _oneOnOneSpeakers.left;
  _oneOnOneSpeakers.left = _oneOnOneSpeakers.right;
  _oneOnOneSpeakers.right = tmpId;
});

function _renderOneOnOneSegment(event) {
  if (!_oneOnOneActive || !event.is_final || !event.text) return;

  const clusterId = event.speakers?.[0]?.cluster_id ?? 0;
  const speakerName = getSpeakerDisplayName(
    clusterId,
    event.speakers?.[0]?.identity || event.speakers?.[0]?.display_name,
  );

  // Assign speaker to side on first appearance
  if (_oneOnOneSpeakers.left === null) {
    _oneOnOneSpeakers.left = clusterId;
    document.getElementById('oo-speaker-left').textContent = speakerName;
  } else if (_oneOnOneSpeakers.right === null && clusterId !== _oneOnOneSpeakers.left) {
    _oneOnOneSpeakers.right = clusterId;
    document.getElementById('oo-speaker-right').textContent = speakerName;
  }

  // Determine which pane
  let paneId;
  if (clusterId === _oneOnOneSpeakers.left) paneId = 'oo-left';
  else if (clusterId === _oneOnOneSpeakers.right) paneId = 'oo-right';
  else paneId = 'oo-left'; // Fallback for 3rd+ speakers

  const pane = document.getElementById(paneId);
  const tr = event.translation?.text || '';
  const langA = _getLangA();
  const langB = _getLangB();
  const cssClass = event.language === langA ? (_languageNames[langA]?.css_font_class || '') : (_languageNames[langB]?.css_font_class || '');
  const trCssClass = event.language === langA ? (_languageNames[langB]?.css_font_class || '') : (_languageNames[langA]?.css_font_class || '');
  // Prefer server-rendered ruby (`furigana_html`) over plain esc()'d text
  // for both source and translation, so kanji shows pronunciation in the
  // 1:1 view too. Falls back to esc() for non-JA languages or before the
  // furigana revision lands.
  const srcBody = event.furigana_html || esc(event.text);
  const trBody = event.translation?.furigana_html || (tr ? esc(tr) : '');
  const html = `<div class="oo-original ${cssClass}">${srcBody}</div>${tr ? `<div class="oo-translation ${trCssClass}">${trBody}</div>` : ''}`;

  // Dedup: server broadcasts the same segment_id multiple times (raw event,
  // in_progress translation, done translation). Update in place if we've
  // already rendered this segment — NEVER appendChild on every event.
  const selector = `[data-segment-id="${(window.CSS && CSS.escape) ? CSS.escape(event.segment_id) : event.segment_id}"]`;
  const existing = pane.querySelector(selector);
  if (existing) {
    existing.innerHTML = html;
    return;
  }

  const block = document.createElement('div');
  block.className = 'oo-block';
  block.dataset.segmentId = event.segment_id;
  block.innerHTML = html;
  pane.appendChild(block);
  pane.scrollTop = pane.scrollHeight;
}

// Hook into segment store to feed 1:1 mode
store.subscribe((segId, event) => {
  if (_oneOnOneActive && event.is_final) {
    _renderOneOnOneSegment(event);
  }
});


function updateMetricsDashboard(data) {
  const m = data.metrics || {};
  const gpu = data.gpu;
  const b = data.backends || {};

  const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };

  // ASR
  const asrEps = (m.asr_eps || 0).toFixed(1);
  set('mc-asr', `${m.asr_finals || 0}`);
  set('mc-asr-sub', `${asrEps} evt/s`);

  // Translation
  const avgMs = m.avg_translation_ms || 0;
  set('mc-trans', `${m.translations_completed || 0}/${m.translations_submitted || 0}`);
  set('mc-trans-sub', avgMs > 0 ? `avg ${avgMs}ms` : '');

  // VRAM bar
  const vramPct = gpu ? (gpu.vram_pct || 0) : 0;
  set('mc-vram', gpu ? `${gpu.vram_used_mb || 0}MB / ${gpu.vram_total_mb || 0}MB (${vramPct.toFixed(0)}%)` : '—');
  const vramBar = document.getElementById('mc-vram-bar');
  if (vramBar) {
    vramBar.style.width = `${vramPct}%`;
    vramBar.className = `metric-bar ${vramPct > 90 ? 'metric-bar-danger' : vramPct > 75 ? 'metric-bar-warn' : ''}`;
  }

  // Audio
  set('mc-audio', `${(m.audio_s || 0).toFixed(0)}s`);

  // Connections
  const wsCount = (data.connections || 0) + (data.audio_out_connections || 0);
  set('mc-health', `${wsCount}`);

  // Backends
  const backendStr = Object.entries(b)
    .map(([k, v]) => `<span class="backend-badge ${v ? 'backend-on' : 'backend-off'}">${k}</span>`)
    .join(' ');
  const backendsEl = document.getElementById('mc-backends');
  if (backendsEl) backendsEl.innerHTML = backendStr;

  // W5 — reliability tiles. Color-coded via CSS class so warn/crit
  // states pop visually before they kill a meeting.
  const setColored = (id, text, level) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = `metric-card-value ${level === 'crit' ? 'metric-value-crit' : level === 'warn' ? 'metric-value-warn' : ''}`;
  };

  // ASR RTT p95 — backend control-path latency.  Warn >800ms, crit >2000ms.
  const asrRtt = (m.asr_request_rtt_ms && m.asr_request_rtt_ms.p95) || null;
  if (asrRtt === null) {
    setColored('mc-asr-rtt-p95', '—', 'ok');
  } else {
    const lvl = asrRtt > 2000 ? 'crit' : asrRtt > 800 ? 'warn' : 'ok';
    setColored('mc-asr-rtt-p95', `${asrRtt.toFixed(0)}ms`, lvl);
  }

  // Watchdog fires per minute.  Warn >0.5/min, crit >2/min.
  const wdRate = m.watchdog_fires_per_min || 0;
  const wdLvl = wdRate > 2 ? 'crit' : wdRate > 0.5 ? 'warn' : 'ok';
  setColored('mc-watchdog-fires', `${wdRate}`, wdLvl);

  // Time since last ASR final.  Warn >5s, crit >15s.  null on a fresh
  // meeting (no finals yet) — render '—' rather than alarming.
  const tsf = m.time_since_last_final_s;
  if (tsf === null || tsf === undefined) {
    setColored('mc-since-final', '—', 'ok');
  } else {
    const lvl = tsf > 15 ? 'crit' : tsf > 5 ? 'warn' : 'ok';
    setColored('mc-since-final', `${tsf.toFixed(1)}s`, lvl);
  }

  // GPU free MB.  Warn <8GB, crit <4GB.  Inverse of the other tiles.
  const freeMb = gpu ? (gpu.vram_free_mb || 0) : null;
  if (freeMb === null) {
    setColored('mc-gpu-free', '—', 'ok');
  } else {
    const lvl = freeMb < 4096 ? 'crit' : freeMb < 8192 ? 'warn' : 'ok';
    const gb = (freeMb / 1024).toFixed(1);
    setColored('mc-gpu-free', `${gb} GB`, lvl);
  }
}
