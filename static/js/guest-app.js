const SPEAKER_COLORS = ['#c45d20', '#1a6fb5', '#2a8540', '#9b2d7b'];

// ─── Speaker Display Registry (port from scribe-app.js) ─────────
// Keeps the guest portal's "Speaker N" labels consistent with the host
// UI. Without this, the raw cluster_id is used as the display number,
// so pyannote pseudo cluster ids like 101 render as "Speaker 102" and
// real cluster 5 renders as "Speaker 6" — both wrong. Keep this in
// sync with the equivalent block in scribe-app.js (clusters map,
// _isPseudoCluster, getGuestSpeakerName, getGuestSpeakerColor).
const _guestSpeakerRegistry = {
  clusters: new Map(),  // clusterId → {seqIndex, displayName}
  nextIndex: 1,
};

function _isGuestPseudoCluster(clusterId) {
  // Server-side time_proximity fallback allocates cluster_ids >=100 as
  // transient placeholders until diarization catch-up resolves them.
  // Don't burn a sequential slot on these — show "Speaker ?".
  return clusterId != null && Number(clusterId) >= 100;
}

function _isGenericGuestSpeakerLabel(name) {
  // "Speaker 12" / "Speaker 101" — server-side generic labels that
  // should NOT override our first-seen-order sequential mapping.
  if (!name) return true;
  return /^Speaker\s+\d+$/.test(String(name).trim());
}

function getGuestSpeakerName(clusterId, explicitName) {
  if (clusterId == null) return 'Unknown';
  if (_isGuestPseudoCluster(clusterId)) {
    const existing = _guestSpeakerRegistry.clusters.get(clusterId);
    if (existing?.displayName) return existing.displayName;
    return 'Speaker ?';
  }
  let entry = _guestSpeakerRegistry.clusters.get(clusterId);
  if (!entry) {
    entry = { seqIndex: _guestSpeakerRegistry.nextIndex++, displayName: null };
    _guestSpeakerRegistry.clusters.set(clusterId, entry);
  }
  if (explicitName && !_isGenericGuestSpeakerLabel(explicitName)) {
    entry.displayName = explicitName;
  }
  return entry.displayName || `Speaker ${entry.seqIndex}`;
}

function getGuestSpeakerColor(clusterId) {
  if (clusterId == null) return SPEAKER_COLORS[0];
  // Pseudo-clusters get a muted/neutral slot so they don't flicker
  // colors when catch-up resolves them to a real cluster.
  if (_isGuestPseudoCluster(clusterId)) return '#8a8a94';
  // Color by the sequential index so speakers stay stable even if
  // pyannote hands out non-contiguous cluster_ids.
  const entry = _guestSpeakerRegistry.clusters.get(clusterId);
  const seq = entry?.seqIndex ?? 1;
  return SPEAKER_COLORS[(seq - 1) % SPEAKER_COLORS.length];
}

const API = location.origin;
// When served from HTTP port 80 proxy, WebSockets must go directly to
// the HTTPS backend (port 80 can't proxy WS with stdlib). The captive
// portal injects __SCRIBE_WS_OVERRIDE to handle this.
const _wsOverride = window.__SCRIBE_WS_OVERRIDE || null;
const WS_PROTO = _wsOverride ? 'wss:' : (location.protocol === 'https:' ? 'wss:' : 'ws:');
const WS_HOST = _wsOverride ? _wsOverride.replace(/^wss?:\/\//, '') : location.host;

let langFilter = localStorage.getItem('guest_langFilter') || 'all';
let newestFirst = (localStorage.getItem('guest_newestFirst') ?? 'true') !== 'false';
let languages = {};
let segments = new Map();
const _savedListenLang = localStorage.getItem('guest_listenLang') || '';
// Default to "full" mode so a listener whose selected language matches the
// speaker's language still hears something: the source-audio passthrough.
// "translation" mode (TTS-only) would give dead air in that case because no
// translation is ever produced for the source's own language — a footgun
// observed 2026-04-15 in the de↔en meeting logs (listener picked 'en',
// speaker spoke 'en', every TTS event logged "skipped_for_lang" because
// target was always 'de'). "full" sends passthrough+TTS so the listener
// always hears audio regardless of direction.
//
// One-time migration: existing devices that visited pre-2026-04-15 have
// `guest_listenMode=translation` stuck in localStorage and the default
// change below would not reach them. The `guest_listenMode_v2` key flips
// them to "full" once, without clobbering any later explicit choice.
if (!localStorage.getItem('guest_listenMode_v2')) {
  localStorage.setItem('guest_listenMode', 'full');
  localStorage.setItem('guest_listenMode_v2', '1');
}
const _savedListenMode = localStorage.getItem('guest_listenMode') || 'full';
const _savedListenOn = localStorage.getItem('guest_listenOn') === 'true';

const transcriptEl = document.getElementById('transcript');
const waitingEl = document.getElementById('waiting');
const statusEl = document.getElementById('status');
const pulseEl = document.getElementById('pulse');
const langBar = document.getElementById('lang-bar');
const langSelect = document.getElementById('lang-select');

const liveBar = document.getElementById('live-bar');
const liveBarText = document.getElementById('live-bar-text');

langSelect.addEventListener('change', () => {
  langFilter = langSelect.value;
  localStorage.setItem('guest_langFilter', langFilter);
  renderAll();
});
// Restore saved filter
if (langFilter !== 'all') langSelect.value = langFilter;

// Scroll-direction toggle
const btnDir = document.getElementById('btn-dir-guest');
function _syncDirButton() {
  if (!btnDir) return;
  btnDir.textContent = newestFirst ? '↑ Newest' : '↓ Oldest';
}
_syncDirButton();

btnDir?.addEventListener('click', () => {
  newestFirst = !newestFirst;
  localStorage.setItem('guest_newestFirst', newestFirst);
  _syncDirButton();
  // Reverse the existing rendered segments in place so the on-screen
  // order flips without a full re-render.
  const children = [...transcriptEl.children];
  transcriptEl.innerHTML = '';
  children.reverse().forEach(c => transcriptEl.appendChild(c));
  // Pin the viewport to the newest segment after toggle.
  requestAnimationFrame(() => {
    if (newestFirst) {
      transcriptEl.scrollTop = 0;
    } else {
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }
  });
});

function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}:${String(m % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderSegment(id, animate = false) {
  const evt = segments.get(id);
  if (!evt || !evt.text) return;
  // Don't filter here — the rendering logic below handles showing original or translation
  // based on langFilter. Filtering here skips translated segments.

  let el = document.querySelector(`[data-seg="${id}"]`);
  if (!el) {
    el = document.createElement('div');
    el.className = 'seg';
    el.dataset.seg = id;
    // Insert at the "newest" end according to the current direction.
    if (newestFirst) {
      transcriptEl.prepend(el);
    } else {
      transcriptEl.appendChild(el);
    }
    // Cap visible segments to ~1 minute of text (last 15 segments)
    const MAX_VISIBLE = 15;
    const children = transcriptEl.children;
    while (children.length > MAX_VISIBLE) {
      // In newest-first mode the oldest is at the bottom; in oldest-first
      // mode the oldest is at the top. Drop from the correct end.
      const old = newestFirst
        ? children[children.length - 1]
        : children[0];
      const oldId = old.dataset.seg;
      transcriptEl.removeChild(old);
      segments.delete(oldId);
    }
    // In oldest-first mode, keep the viewport pinned to the newest
    // segment at the bottom so users aren't stuck reading history.
    if (!newestFirst) {
      requestAnimationFrame(() => { transcriptEl.scrollTop = transcriptEl.scrollHeight; });
    }
  }

  el.className = `seg${evt.is_final ? '' : ' partial'}${animate ? '' : ' no-anim'}`;

  const time = formatTime(evt.start_ms || 0);
  let speaker = '';
  if (evt.speakers?.length) {
    const s = evt.speakers[0];
    // Use the guest-local speaker registry so pyannote pseudo-cluster ids
    // (>=100) and non-contiguous real cluster ids don't leak as "Speaker
    // 102"-style labels. Prefer explicit identity/display_name when
    // they're real human labels (the registry ignores generic "Speaker N"
    // strings from the server so they get re-normalized client-side).
    const explicit = s.identity || s.display_name;
    const name = getGuestSpeakerName(s.cluster_id, explicit);
    const color = getGuestSpeakerColor(s.cluster_id);
    speaker = `<span class="seg-speaker" style="color:${color}"><span class="seg-dot" style="background:${color}"></span>${esc(name)}</span>`;
  }

  const langClass = evt.language === 'ja' ? 'ja' : '';

  // Source-text body. When the server has annotated kanji with furigana,
  // it sends the ruby HTML in `furigana_html` (one revision after the
  // ASR final lands). We pass it through verbatim instead of esc()'ing
  // it, so the <ruby><rt> markup actually renders. Falling back to esc()
  // when furigana isn't present keeps the path safe for non-JA languages.
  const srcBody = evt.furigana_html || esc(evt.text);
  const trBody = evt.translation?.furigana_html || (evt.translation?.text ? esc(evt.translation.text) : '');

  // Build text content based on filter
  let textHtml = '';
  if (langFilter === 'all') {
    // Side by side: original + translation
    textHtml = `<div class="seg-text ${langClass}">${srcBody}</div>`;
    if (evt.translation?.text) {
      const trClass = evt.translation.target_language === 'ja' ? 'ja' : '';
      textHtml += `<div class="seg-text ${trClass}" style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.15rem">${trBody}</div>`;
    }
  } else {
    // Single language: show original if it matches, or translation if that matches
    if (evt.language === langFilter) {
      textHtml = `<div class="seg-text ${langClass}">${srcBody}</div>`;
    } else if (evt.translation?.target_language === langFilter) {
      const trClass = langFilter === 'ja' ? 'ja' : '';
      textHtml = `<div class="seg-text ${trClass}">${trBody}</div>`;
    }
  }

  if (!textHtml) { el.style.display = 'none'; return; }
  el.style.display = '';

  el.innerHTML = `<div class="seg-meta"><span>${time}</span>${speaker}</div>${textHtml}`;

  // Newest segments prepended at top — always visible, no scroll needed
}

function renderAll() {
  transcriptEl.innerHTML = '';
  for (const id of segments.keys()) renderSegment(id);
}

const _pending = new Map(); // Hold segments until translation arrives

// Merge `incoming` event onto `existing`, taking the union of additive
// dimensions (translation, furigana_html, speakers). The server pushes
// each dimension as an independent revision and the translation
// rebroadcast does NOT carry furigana — so naive overwrite drops it.
// Mirror of the admin SegmentStore.ingest merge, simplified for this view.
function _mergeSegment(existing, incoming) {
  if (!existing) return { ...incoming };
  const merged = { ...existing, ...incoming };
  // Keep the longer / more complete text + the highest revision.
  if (existing.text && (!incoming.text || incoming.text.length < existing.text.length)) {
    merged.text = existing.text;
  }
  merged.revision = Math.max(existing.revision || 0, incoming.revision || 0);
  // Furigana on source.
  if (!merged.furigana_html && existing.furigana_html) {
    merged.furigana_html = existing.furigana_html;
  }
  // Translation: prefer the most-complete version (one with .text > one without).
  const exTr = existing.translation;
  const inTr = incoming.translation;
  if (exTr?.text && !inTr?.text) {
    merged.translation = exTr;
  } else if (inTr?.text && !exTr?.text) {
    merged.translation = inTr;
  } else if (exTr?.text && inTr?.text) {
    // Both have text — take the new one but preserve furigana from either.
    merged.translation = { ...inTr };
  } else if (exTr || inTr) {
    merged.translation = inTr || exTr;
  }
  // Translation-side furigana (e.g. ja translation of an en source).
  if (exTr?.furigana_html && !merged.translation?.furigana_html) {
    merged.translation = {
      ...(merged.translation || {}),
      furigana_html: exTr.furigana_html,
    };
  }
  // Speakers: latest non-empty wins.
  if ((!merged.speakers || merged.speakers.length === 0) && existing.speakers?.length) {
    merged.speakers = existing.speakers;
  }
  return merged;
}

function ingest(evt) {
  const id = evt.segment_id;
  if (!id || !evt.text) return;

  // Partials → live bar only
  if (!evt.is_final) {
    liveBar.style.display = '';
    liveBarText.textContent = evt.text;
    return;
  }

  // Locate any prior copy of this segment — either already displayed or
  // still pending — and merge the new event onto it. This is what makes
  // furigana survive: a furigana revision arriving while the segment is
  // still in `_pending` (waiting for translation) gets folded into the
  // pending record, and the eventual promotion to `segments` carries the
  // furigana along instead of overwriting it with the translation event.
  const wasInSegments = segments.has(id);
  const wasInPending = _pending.has(id);
  const prior = wasInSegments ? segments.get(id) : (wasInPending ? _pending.get(id) : null);
  const merged = _mergeSegment(prior, evt);

  const hasTr = !!(merged.translation?.text && merged.translation.status === 'done');

  // Already displayed → just update in place.
  if (wasInSegments) {
    segments.set(id, merged);
    renderSegment(id, false);
    return;
  }

  // Ready to display: either we have the translation, or the user is in
  // single-language mode and doesn't need a translation to render.
  if (hasTr || langFilter !== 'all') {
    if (wasInPending) _pending.delete(id);
    segments.set(id, merged);
    liveBar.style.display = 'none';
    renderSegment(id, true);
    return;
  }

  // Hold in pending until translation arrives. Set a 4 s safety net so a
  // segment that never gets a translation still surfaces eventually.
  _pending.set(id, merged);
  liveBar.style.display = '';
  liveBarText.textContent = merged.text;
  if (!wasInPending) {
    setTimeout(() => {
      if (_pending.has(id) && !segments.has(id)) {
        segments.set(id, _pending.get(id));
        _pending.delete(id);
        liveBar.style.display = 'none';
        renderSegment(id, true);
      }
    }, 4000);
  }
}

// Populate language selector from API
async function loadLanguages() {
  try {
    const resp = await fetch(`${API}/api/languages`);
    const data = await resp.json();
    if (data.languages) {
      for (const lang of data.languages) {
        languages[lang.code] = lang;
        const opt = document.createElement('option');
        opt.value = lang.code;
        opt.textContent = `${lang.native_name || lang.name} only`;
        langSelect.appendChild(opt);
      }
      // Hide listen button when no TTS-capable languages exist
      const hasTts = data.languages.some(l => l.tts_supported !== false);
      const listenBtn = document.getElementById('btn-listen-guest');
      if (listenBtn) listenBtn.style.display = hasTts ? '' : 'none';
    }
  } catch {}

  // Restore saved selections
  if (langFilter !== 'all') langSelect.value = langFilter;
  const modeEl = document.getElementById('listen-mode-guest');
  if (modeEl && _savedListenMode) modeEl.value = _savedListenMode;
  // NOTE: do NOT auto-click the Listen button on reload. A programmatic
  // setTimeout click is not a user gesture, so the AudioContext.resume()
  // inside the click handler silently fails — every WAV blob then decodes
  // into a permanently-suspended context and the listener hears nothing
  // even though `listener.deliveries` is climbing on the server. Browsers
  // require ONE real tap per page load to start audio output. Listen
  // language / mode preferences are still restored above so the user only
  // needs to tap, not pick the language again.
}

async function poll() {
  try {
    const resp = await fetch(`${API}/api/status`);
    if (!resp.ok) throw new Error(resp.status);
    const data = await resp.json();

    if (data.meeting?.state === 'recording') {
      // Meeting active — load languages (if not yet) then connect
      if (langSelect.options.length <= 1) await loadLanguages();
      waitingEl.style.display = 'none';
      transcriptEl.style.display = '';
      langBar.style.display = '';
      pulseEl.classList.remove('idle');
      statusEl.textContent = 'Live';
      if (!_wsConnected) connectWs();
      return;
    }
  } catch {}

  // No active meeting or server unreachable — keep polling
  waitingEl.style.display = '';
  transcriptEl.style.display = 'none';
  langBar.style.display = 'none';
  statusEl.textContent = 'Waiting...';
  pulseEl.classList.add('idle');
  setTimeout(poll, 3000);
}

let _wsConnected = false;
function connectWs() {
  if (_wsConnected) return;
  _wsConnected = true;
  const ws = new WebSocket(`${WS_PROTO}//${WS_HOST}/api/ws/view`);
  ws.onmessage = (evt) => {
    try { ingest(JSON.parse(evt.data)); } catch {}
  };
  ws.onclose = () => {
    _wsConnected = false;
    statusEl.textContent = 'Disconnected — reconnecting...';
    pulseEl.classList.add('idle');
    setTimeout(poll, 3000);
  };
  ws.onopen = () => {
    statusEl.textContent = 'Live';
    pulseEl.classList.remove('idle');
  };
  const keepAlive = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    else clearInterval(keepAlive);
  }, 30000);
}

// Guest audio interpretation listener
let _guestAudioCtx = null;
let _guestAudioWs = null;
let _guestAudioQueue = [];
let _guestAudioPlaying = false;
// Auto-reconnect state. Server event-loop hiccups can drop the WS without
// any user action; this lets us heal in <2s instead of leaving the user
// stranded waiting for them to manually re-tap Listen.
let _guestIntentionalStop = false;
let _guestReconnectAttempts = 0;
let _guestReconnectTimer = null;
let _guestActiveLang = '';
let _guestActiveMode = '';

// ── MSE (Media Source Extensions) primary path ───────────────────────
// The first-class audio path. Uses a hidden <audio> element backed by
// a MediaSource, with fMP4/AAC fragments appendBuffer()'d as they
// arrive over the WS. This is the only path that can reach the
// "playback" audio session category on iPhone Safari — Web Audio
// alone cannot escape the "ambient" category which is silenced by the
// ringer switch. Falls back to Web Audio if MediaSource is unavailable.
let _guestUseMse = false;                // feature-detect result, set at load
let _guestMediaSource = null;            // MediaSource instance (per connection)
let _guestSourceBuffer = null;           // SourceBuffer hung off the MediaSource
let _guestMseInitAppended = false;       // has the current generation's init frame been appended?
let _guestAppendQueue = [];              // Uint8Array queue awaiting appendBuffer
let _guestSbBusy = false;                // true while appendBuffer OR remove is in flight
let _guestFormatAcked = false;           // did the server ack our set_format yet?
let _guestConnGeneration = 0;            // bumped on every reconnect — drops late frames
const _MSE_MAX_BUFFERED_S = 12.0;        // rolling window behind currentTime (trim beyond this)
const _MSE_RE_ANCHOR_S = 30.0;           // jump to live edge if drift exceeds this

// ── Audio diagnostic ─────────────────────────────────────────────────────
// Live state of every step in the audio chain. Surfaced two ways:
//   1. Tiny on-screen banner (read locally if devtools is unavailable)
//   2. Periodic server-side ping (POST /api/diag/listener) so an operator
//      can pull the same data from `scripts/scribe_trace.py --listeners`
//      WITHOUT asking the user to read text off a phone screen.
const _diag = {
  bytesIn: 0,
  blobsIn: 0,
  decodeOk: 0,
  decodeErr: 0,
  played: 0,
  lastErr: '',
  primed: false,
  path: 'unknown',                       // "mse" | "web-audio" | "web-audio-forced" | "unknown"
  formatAcked: false,
  mseBufferedStart: null,
  mseBufferedEnd: null,
  mediaError: null,
  connGeneration: 0,
};
const _CLIENT_ID = (() => {
  let id = sessionStorage.getItem('scribe_client_id');
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : `c-${Math.random().toString(36).slice(2)}`);
    sessionStorage.setItem('scribe_client_id', id);
  }
  return id;
})();
const _UA_SHORT = (() => {
  const ua = navigator.userAgent || '';
  if (/iPhone|iPad|iPod/.test(ua)) return 'iOS';
  if (/Android/.test(ua)) return 'Android';
  if (/Macintosh/.test(ua)) return 'Mac';
  if (/Windows/.test(ua)) return 'Win';
  return 'Other';
})();

// ── MSE feature detection + URL overrides ───────────────────────────
// Runs at page load, before any tap. Sets `_guestUseMse` to true when
// the browser supports MediaSource (or the iOS 17+ ManagedMediaSource
// variant) with the AAC-LC/fMP4 profile we encode server-side. The
// URL overrides are deliberately in implementation scope so we can
// verify the fallback path on iPhone without changing client builds:
//   ?force=mse        — force MSE even if detection disagrees
//   ?force=fallback   — force the Web Audio + silent-MP3 fallback path
(() => {
  const MSE_MIME = 'audio/mp4; codecs="mp4a.40.2"';
  const hasMse =
    (typeof window.ManagedMediaSource !== 'undefined' &&
     typeof ManagedMediaSource.isTypeSupported === 'function' &&
     ManagedMediaSource.isTypeSupported(MSE_MIME)) ||
    (typeof window.MediaSource !== 'undefined' &&
     typeof MediaSource.isTypeSupported === 'function' &&
     MediaSource.isTypeSupported(MSE_MIME));
  _guestUseMse = !!hasMse;
  try {
    const params = new URLSearchParams(location.search);
    const force = params.get('force');
    if (force === 'fallback') _guestUseMse = false;
    if (force === 'mse')      _guestUseMse = true;
  } catch (e) { /* URL parse error ignored */ }
  _diag.path = _guestUseMse ? 'mse' : 'web-audio';
})();
function _diagSnapshot() {
  const ctx = _guestAudioCtx;
  const ws = _guestAudioWs;
  // Sample live MSE state from the <audio> element so operators can
  // see buffered region + live-edge drift in real time.
  let mseStart = null, mseEnd = null, mediaErr = null;
  try {
    const el = document.getElementById('guest-audio-el');
    if (el) {
      if (el.buffered && el.buffered.length > 0) {
        mseStart = el.buffered.start(0);
        mseEnd = el.buffered.end(el.buffered.length - 1);
      }
      if (el.error) mediaErr = el.error.code;
    }
  } catch (e) { /* buffered access can throw before first append */ }
  return {
    client_id: _CLIENT_ID,
    page: 'guest',
    ua_short: _UA_SHORT,
    ctx_state: ctx ? ctx.state : 'null',
    ctx_rate: ctx ? ctx.sampleRate : 0,
    ws_state: ws ? ['CONNECTING','OPEN','CLOSING','CLOSED'][ws.readyState] : 'NULL',
    primed: _diag.primed,
    queue: _guestAudioQueue.length,
    bytes_in: _diag.bytesIn,
    blobs_in: _diag.blobsIn,
    decoded: _diag.decodeOk,
    decode_err: _diag.decodeErr,
    played: _diag.played,
    last_err: _diag.lastErr || '',
    path: _diag.path,
    format_acked: _diag.formatAcked,
    mse_buffered_start: mseStart,
    mse_buffered_end: mseEnd,
    mse_append_queue: _guestAppendQueue.length,
    media_error: mediaErr,
    conn_generation: _guestConnGeneration,
  };
}
function _diagRender() {
  // No-op on the guest portal — the on-screen banner used to render the
  // live audio-chain state here, but it covered the top of the page on
  // small phones. Keep the diag snapshot going to the server
  // (`_diagPushToServer`) so operators can still inspect it via
  // `scripts/scribe_trace.py --listeners`; just don't paint it over the
  // UI. Set `?diag=1` in the URL if you want it back visually.
  if (!location.search.includes('diag=1')) return;
  const el = document.getElementById('audio-diag');
  if (!el) return;
  el.style.display = '';
  const s = _diagSnapshot();
  el.textContent =
    `ws=${s.ws_state}  ctx=${s.ctx_state}@${s.ctx_rate}Hz  primed=${s.primed}  ` +
    `q=${s.queue}  bytes=${s.bytes_in}  blobs=${s.blobs_in}  ` +
    `decoded=${s.decoded}  decErr=${s.decode_err}  played=${s.played}` +
    (s.last_err ? `  ERR=${s.last_err}` : '');
}
function _diagPushToServer() {
  // Fire-and-forget POST. Failures are silent — diagnostic, not load-bearing.
  try {
    fetch(`${API}/api/diag/listener`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_diagSnapshot()),
      keepalive: true,
    }).catch(() => {});
  } catch {}
}
setInterval(_diagRender, 500);
setInterval(_diagPushToServer, 2000);

// iOS Safari workaround: the FIRST audio playback after a user tap must
// happen synchronously inside the touch handler. We play a 1-frame silent
// buffer immediately on tap to "unlock" the audio output. After this,
// subsequent buffer sources can fire from arbitrary async contexts.
function _primeGuestAudioForIOS(ctx) {
  try {
    const silent = ctx.createBuffer(1, 1, 22050);
    const src = ctx.createBufferSource();
    src.buffer = silent;
    src.connect(ctx.destination);
    src.start(0);
    _diag.primed = true;
  } catch (e) {
    _diag.lastErr = `prime: ${e.message}`;
  }
}

document.getElementById('btn-listen-guest')?.addEventListener('click', () => {
  // NOTE: this handler is intentionally NOT async. Every unlock call
  // (keeper.play(), guestAudioEl.play(), AudioContext.resume(),
  // _primeGuestAudioForIOS) runs synchronously inside the user-gesture
  // context. The moment we hit an `await`, the gesture window closes
  // and any deferred audio-unlock work silently no-ops in strict-
  // autoplay browsers. WebSocket connect / send do NOT need a gesture.
  const btn = document.getElementById('btn-listen-guest');
  const modeSelect = document.getElementById('listen-mode-guest');
  const listenLang = document.getElementById('listen-lang-guest');
  const speedSelect = document.getElementById('listen-speed-guest');
  const audioEl = document.getElementById('guest-audio-el');
  const keeperEl = document.getElementById('guest-audio-session-keeper');

  if (_guestAudioWs || _guestMediaSource) {
    _guestIntentionalStop = true;
    if (_guestReconnectTimer) { clearTimeout(_guestReconnectTimer); _guestReconnectTimer = null; }
    if (_guestAudioWs) {
      try { _guestAudioWs.close(); } catch (e) {}
      _guestAudioWs = null;
    }
    _teardownMediaSource();
    if (_guestAudioCtx) { _guestAudioCtx.close().catch(()=>{}); _guestAudioCtx = null; }
    try { keeperEl?.pause(); } catch (e) {}
    try { audioEl?.pause(); } catch (e) {}
    btn.classList.remove('active');
    modeSelect.style.display = 'none';
    listenLang.style.display = 'none';
    speedSelect.style.display = 'none';
    localStorage.removeItem('guest_listenOn');
    return;
  }

  // Populate listen language selector with TTS-capable languages only
  if (listenLang.options.length === 0) {
    for (const [code, lang] of Object.entries(languages)) {
      if (lang.tts_supported === false) continue;
      const opt = document.createElement('option');
      opt.value = code;
      opt.textContent = lang.native_name || lang.name;
      listenLang.appendChild(opt);
    }
    if (listenLang.options.length === 0) return;  // no TTS languages available
    listenLang.value = _savedListenLang || (langSelect.value !== 'all' ? langSelect.value : 'en');
  }

  // Restore saved speed
  speedSelect.value = localStorage.getItem('guest_listenSpeed') || '1.0';
  _guestPlaybackRate = parseFloat(speedSelect.value);

  // (0) ALWAYS start the silent keeper first — promotes the audio
  //     session to "playback" category on iOS Safari regardless of
  //     which audio path wins below. If MSE fails asynchronously
  //     later, the keeper is already running and the Web Audio
  //     fallback inherits the promoted session without a second tap.
  try { keeperEl?.play().catch(() => {}); } catch (e) {}

  // (1) Prime Web Audio inside the gesture window, in ALL paths.
  //     MSE doesn't need it to produce output, but the fallback does;
  //     priming here means a later MSE rejection can fall through to
  //     the Web Audio path without requiring the user to tap again.
  _guestAudioCtx = new AudioContext();
  if (_guestAudioCtx.state === 'suspended') {
    _guestAudioCtx.resume().catch((e) => { _diag.lastErr = `resume: ${e.message}`; });
  }
  _primeGuestAudioForIOS(_guestAudioCtx);

  _guestActiveLang = listenLang.value || 'en';
  _guestActiveMode = modeSelect.value || 'translation';
  _guestIntentionalStop = false;
  _guestReconnectAttempts = 0;
  _guestConnGeneration += 1;
  _diag.connGeneration = _guestConnGeneration;

  if (_guestUseMse) {
    // (2) MSE primary path. Build MediaSource + attach to <audio>,
    //     open WS, then call .play() (NOT awaited — the promise may
    //     stay pending until first appendBuffer, which is fine).
    _diag.path = 'mse';
    _buildMediaSource(audioEl);
    _guestConnectMse();
    // .play() runs LAST so the MSE pipeline is ready to receive
    // bytes the moment Safari unlocks the element.
    try {
      const p = audioEl.play();
      if (p && typeof p.catch === 'function') {
        p.catch((e) => {
          _diag.lastErr = `audio.play rejected: ${e.message}`;
          _diag.path = 'web-audio-forced';
          _teardownMseAndSwitchToFallback();
        });
      }
    } catch (e) {
      _diag.lastErr = `audio.play threw: ${e.message}`;
      _diag.path = 'web-audio-forced';
      _teardownMseAndSwitchToFallback();
    }
  } else {
    // (2b) Pure fallback path. Legacy Web Audio pipeline, plus the
    //      silent keeper already playing from step (0).
    _diag.path = 'web-audio';
    _guestConnect();
  }

  _diagRender();

  btn.classList.add('active');
  modeSelect.style.display = '';
  listenLang.style.display = '';
  speedSelect.style.display = '';
  localStorage.setItem('guest_listenOn', 'true');
  localStorage.setItem('guest_listenLang', listenLang.value);
  localStorage.setItem('guest_listenMode', modeSelect.value);
});

function _guestConnect() {
  if (_guestIntentionalStop) return;
  try {
    _guestAudioWs = new WebSocket(`${WS_PROTO}//${WS_HOST}/api/ws/audio-out`);
  } catch (e) {
    _diag.lastErr = `ws ctor: ${e.message}`;
    _scheduleGuestReconnect();
    return;
  }
  _guestAudioWs.binaryType = 'arraybuffer';
  _guestAudioWs.onopen = () => {
    _guestReconnectAttempts = 0;
    // Explicitly negotiate wav-pcm so we skip the server's grace
    // window default (legacy cached clients still work because the
    // server defaults to wav-pcm after the grace anyway).
    _guestAudioWs.send(JSON.stringify({ type: 'set_format', format: 'wav-pcm' }));
    _guestAudioWs.send(JSON.stringify({ type: 'set_language', language: _guestActiveLang }));
    _guestAudioWs.send(JSON.stringify({ type: 'set_mode', mode: _guestActiveMode }));
  };
  _guestAudioWs.onmessage = (evt) => {
    if (typeof evt.data === 'string') {
      // Consume the format_ack so it doesn't show up as "unknown msg
      // type" in diag. Legacy clients don't need to act on it.
      try {
        const msg = JSON.parse(evt.data);
        if (msg && msg.type === 'format_ack') {
          _guestFormatAcked = true;
          _diag.formatAcked = true;
        }
      } catch (e) { /* ignore */ }
      return;
    }
    if (evt.data instanceof ArrayBuffer) {
      _diag.bytesIn += evt.data.byteLength;
      _diag.blobsIn++;
      _guestAudioQueue.push(evt.data);
      if (!_guestAudioPlaying) _playGuestAudio();
    } else if (evt.data instanceof Blob) {
      _diag.lastErr = 'binaryType arrived as Blob — converting';
      evt.data.arrayBuffer().then((ab) => {
        _diag.bytesIn += ab.byteLength;
        _diag.blobsIn++;
        _guestAudioQueue.push(ab);
        if (!_guestAudioPlaying) _playGuestAudio();
      });
    } else {
      _diag.lastErr = `unknown msg type: ${typeof evt.data}`;
    }
    _diagRender();
  };
  _guestAudioWs.onerror = () => { _diag.lastErr = 'ws error'; _diagRender(); };
  _guestAudioWs.onclose = (e) => {
    _diag.lastErr = `ws close ${e?.code || ''}`;
    _diagRender();
    _scheduleGuestReconnect();
  };
}

// ── MSE: MediaSource + SourceBuffer lifecycle ─────────────────────────

function _buildMediaSource(audioEl) {
  // Creates a fresh MediaSource and attaches it to the <audio> element.
  // MUST run inside the tap gesture (on first tap) OR during a reconnect
  // initiated from a still-primed audio session — the silent keeper
  // keeps the page's audio session alive so the reconnect path doesn't
  // need a new user gesture.
  const MseCtor = window.ManagedMediaSource || window.MediaSource;
  _guestMediaSource = new MseCtor();
  audioEl.src = URL.createObjectURL(_guestMediaSource);
  _guestMseInitAppended = false;
  _guestAppendQueue = [];
  _guestSbBusy = false;
  _guestFormatAcked = false;
  _guestMediaSource.addEventListener('sourceopen', () => {
    try {
      _guestSourceBuffer = _guestMediaSource.addSourceBuffer(
        'audio/mp4; codecs="mp4a.40.2"'
      );
      _guestSourceBuffer.mode = 'sequence';
      _guestSourceBuffer.addEventListener('updateend', _onSourceBufferUpdateEnd);
      _guestSourceBuffer.addEventListener('error', (ev) => {
        _diag.lastErr = 'source_buffer error';
      });
      _mseDrainQueue();
    } catch (e) {
      _diag.lastErr = `addSourceBuffer: ${e.message}`;
    }
  }, { once: true });
}

function _teardownMediaSource() {
  if (_guestSourceBuffer) {
    try { _guestSourceBuffer.abort(); } catch (e) {}
    _guestSourceBuffer = null;
  }
  if (_guestMediaSource) {
    try {
      if (_guestMediaSource.readyState === 'open') _guestMediaSource.endOfStream();
    } catch (e) {}
    _guestMediaSource = null;
  }
  try {
    const audioEl = document.getElementById('guest-audio-el');
    if (audioEl && audioEl.src) {
      URL.revokeObjectURL(audioEl.src);
      audioEl.removeAttribute('src');
      audioEl.load();
    }
  } catch (e) {}
  _guestAppendQueue = [];
  _guestMseInitAppended = false;
  _guestSbBusy = false;
  _guestFormatAcked = false;
}

function _guestConnectMse() {
  // WS lifecycle for the MSE path. Rebuilds the MediaSource on
  // reconnect (the silent keeper stays alive so no new tap is needed).
  if (_guestIntentionalStop) return;
  const audioEl = document.getElementById('guest-audio-el');

  // If a reconnect fired while the old MediaSource is still around,
  // tear it down first — the server will send a fresh init segment
  // from a fresh encoder, which is incompatible with the existing
  // SourceBuffer timeline.
  if (_guestMediaSource) {
    _teardownMediaSource();
  }
  _buildMediaSource(audioEl);

  try {
    _guestAudioWs = new WebSocket(`${WS_PROTO}//${WS_HOST}/api/ws/audio-out`);
  } catch (e) {
    _diag.lastErr = `ws ctor: ${e.message}`;
    _scheduleGuestReconnect();
    return;
  }
  _guestAudioWs.binaryType = 'arraybuffer';
  _guestAudioWs.onopen = () => {
    _guestReconnectAttempts = 0;
    try {
      _guestAudioWs.send(JSON.stringify({ type: 'set_format', format: 'mse-fmp4-aac' }));
      _guestAudioWs.send(JSON.stringify({ type: 'set_language', language: _guestActiveLang }));
      _guestAudioWs.send(JSON.stringify({ type: 'set_mode', mode: _guestActiveMode }));
    } catch (e) {
      _diag.lastErr = `ws send handshake: ${e.message}`;
    }
  };
  _guestAudioWs.onmessage = _mseOnMessage;
  _guestAudioWs.onerror = () => { _diag.lastErr = 'ws error'; _diagRender(); };
  _guestAudioWs.onclose = _mseOnClose;
  // Reconnect path: ensure the <audio> element is still playing.
  // On the first tap it was already started; on reconnect we need
  // another play() call because .src changed.
  try {
    const p = audioEl.play();
    if (p && typeof p.catch === 'function') {
      p.catch((e) => { _diag.lastErr = `audio.play on reconnect: ${e.message}`; });
    }
  } catch (e) { /* ignore */ }
}

function _mseOnMessage(evt) {
  if (typeof evt.data === 'string') {
    try {
      const msg = JSON.parse(evt.data);
      if (msg && msg.type === 'format_ack' && msg.format === 'mse-fmp4-aac') {
        _guestFormatAcked = true;
        _diag.formatAcked = true;
      }
    } catch (e) { /* ignore malformed text */ }
    return;
  }
  if (!(evt.data instanceof ArrayBuffer)) {
    // Some browsers arrive with Blob even after binaryType='arraybuffer'.
    if (evt.data instanceof Blob) {
      evt.data.arrayBuffer().then((ab) => _mseOnMessage({ data: ab }));
    }
    return;
  }
  if (!_guestFormatAcked) {
    _diag.lastErr = 'binary frame before format_ack — server bug, dropped';
    return;
  }
  const view = new Uint8Array(evt.data);
  if (view.length < 2) {
    _diag.lastErr = `runt frame (${view.length}B)`;
    return;
  }
  const frameType = view[0];
  const payload = view.subarray(1);
  _diag.bytesIn += payload.length;
  _diag.blobsIn++;
  if (frameType === 0x49) {
    // Init segment ('I'). A fresh init on a SourceBuffer that already
    // has one means a new generation after reconnect — _mseOnClose
    // should have cleared _guestMseInitAppended first. Defensive drop
    // if it didn't.
    if (_guestMseInitAppended) {
      _diag.lastErr = 'duplicate init frame, dropped';
      return;
    }
    _guestMseInitAppended = true;
    _guestAppendQueue.unshift(payload);     // init goes to the FRONT
  } else if (frameType === 0x46) {
    _guestAppendQueue.push(payload);
  } else {
    _diag.lastErr = `unknown frame type 0x${frameType.toString(16)}`;
    return;
  }
  _mseDrainQueue();
}

function _onSourceBufferUpdateEnd() {
  // Sole owner of _guestSbBusy = false + re-entry into the drain loop.
  // Do NOT wire any other listener to updateend — the state machine
  // is defined in exactly one place.
  _guestSbBusy = false;
  _mseDrainQueue();
}

function _resolveLiveTarget(sb, ct) {
  // Returns { kind, start, end } for the buffered range relative to
  // currentTime. Handles all four states: live, forward (gap with a
  // range after ct), behind (all ranges before ct), none. Used by
  // _mseDrainQueue to make the right trim/seek decisions regardless
  // of whether playback is inside or between buffered regions.
  if (!sb.buffered || sb.buffered.length === 0) {
    return { kind: 'none', start: 0, end: 0 };
  }
  let forward = null, behind = null;
  for (let i = 0; i < sb.buffered.length; i++) {
    const s = sb.buffered.start(i);
    const e = sb.buffered.end(i);
    if (ct >= s && ct <= e) return { kind: 'live', start: s, end: e };
    if (s > ct && (forward === null || s < forward.start)) forward = { start: s, end: e };
    if (e < ct && (behind === null || e > behind.end))    behind  = { start: s, end: e };
  }
  if (forward) return { kind: 'forward', start: forward.start, end: forward.end };
  if (behind)  return { kind: 'behind',  start: behind.start,  end: behind.end };
  return { kind: 'none', start: 0, end: 0 };
}

function _mseDrainQueue() {
  if (!_guestSourceBuffer || _guestSourceBuffer.updating || _guestSbBusy) return;

  const audioEl = document.getElementById('guest-audio-el');
  if (!audioEl) return;
  const ct = audioEl.currentTime;
  const target = _resolveLiveTarget(_guestSourceBuffer, ct);

  // IMPORTANT: live-target resolution + gap-seek MUST run on every
  // updateend, even when the queue is empty. Otherwise the single-
  // fragment-resume case stays stalled: a long gap, one resumed
  // append lands, updateend fires, queue is empty — early return
  // would strand playback in the gap until another fragment arrives.
  // Resolving the target BEFORE the queue-empty guard means the seek
  // into the forward range happens as soon as it's visible.

  // STEP 1: trim ranges FULLY behind the rolling window (works for
  // every target kind — we only touch ranges uncontroversially old).
  for (let i = _guestSourceBuffer.buffered.length - 1; i >= 0; i--) {
    const s = _guestSourceBuffer.buffered.start(i);
    const e = _guestSourceBuffer.buffered.end(i);
    if (e < ct - _MSE_MAX_BUFFERED_S - 0.5) {
      try {
        _guestSbBusy = true;
        _guestSourceBuffer.remove(s, e);
        return;                              // updateend → re-enter
      } catch (err) { _guestSbBusy = false; }
    }
  }

  // STEP 2: target-state branches.
  if (target.kind === 'live') {
    if (target.start < ct - _MSE_MAX_BUFFERED_S - 0.5) {
      try {
        _guestSbBusy = true;
        _guestSourceBuffer.remove(target.start, ct - _MSE_MAX_BUFFERED_S);
        return;
      } catch (err) { _guestSbBusy = false; }
    }
    if (target.end - ct > _MSE_RE_ANCHOR_S) {
      audioEl.currentTime = target.end - 1.0;
    }
  } else if (target.kind === 'forward') {
    // currentTime is in a gap. The nearest forward range is where
    // live audio is sitting; jump into it so playback resumes.
    audioEl.currentTime = target.start + 0.05;
  } else if (target.kind === 'behind') {
    if (ct - target.end > 2.0) {
      _diag.lastErr = `behind-only drift=${(ct - target.end).toFixed(1)}s`;
    }
  }
  // target.kind === 'none' — first append hasn't happened yet; fall through.

  // STEP 3: only now check the queue. Trim/seek decisions above already
  // fired and will persist even if there's nothing to append.
  if (_guestAppendQueue.length === 0) return;
  const chunk = _guestAppendQueue.shift();
  _guestSbBusy = true;
  try {
    _guestSourceBuffer.appendBuffer(chunk);
  } catch (err) {
    _guestSbBusy = false;
    if (err.name === 'QuotaExceededError') {
      try {
        _guestSourceBuffer.remove(0, Math.max(0, ct - 2.0));
      } catch (e2) { /* ignore */ }
      _guestAppendQueue.unshift(chunk);
    } else {
      _diag.lastErr = `appendBuffer ${err.name}: ${err.message}`;
    }
  }
}

function _mseOnClose(evt) {
  _diag.lastErr = `ws close ${evt?.code || ''}`;
  _guestConnGeneration += 1;
  _diag.connGeneration = _guestConnGeneration;
  _diagRender();
  _teardownMediaSource();        // drop buffered media, rebuild on reconnect
  _scheduleGuestReconnect();
}

function _teardownMseAndSwitchToFallback() {
  // MSE play() rejected or the pipeline hit an unrecoverable error.
  // The silent keeper and Web Audio context are already primed from
  // the tap handler — just tear MSE down and let the legacy path
  // take over using the existing _guestConnect handshake.
  _diag.path = 'web-audio-forced';
  if (_guestAudioWs) {
    try { _guestAudioWs.close(); } catch (e) {}
    _guestAudioWs = null;
  }
  _teardownMediaSource();
  // Start the legacy WS (will negotiate wav-pcm via the grace default).
  _guestConnect();
}

function _scheduleGuestReconnect() {
  if (_guestIntentionalStop) return;
  if (_guestReconnectTimer) return;
  // Exponential backoff capped at 5 s. The first retry fires at ~500 ms
  // so a server hiccup heals before the user notices.
  const delay = Math.min(5000, 500 * Math.pow(2, _guestReconnectAttempts));
  _guestReconnectAttempts++;
  _guestReconnectTimer = setTimeout(() => {
    _guestReconnectTimer = null;
    if (_guestUseMse && _diag.path === 'mse') {
      _guestConnectMse();
    } else {
      _guestConnect();
    }
  }, delay);
}

// When the tab/page comes back to the foreground on iOS, Safari has
// usually suspended every setTimeout for the duration it was backgrounded.
// If our WS died while suspended, the scheduled reconnect may be stale
// or may never have fired. Force an immediate health-check and resume
// when visibility returns. Also triggers on page-focus (desktop) and on
// network-up (tab resumed from tethered sleep).
function _guestResumeIfDead() {
  if (_guestIntentionalStop) return;
  const btn = document.getElementById('btn-listen-guest');
  if (!btn || !btn.classList.contains('active')) return; // Listen not on

  // CRITICAL: without a live AudioContext we cannot play anything via
  // the Web Audio fallback path, and without the <audio> element's
  // gesture-authorized state the MSE path can't play either. Both
  // are authorized during the Listen tap. If either is gone, the
  // reconnect from here cannot recover them — un-stick the button and
  // prompt the user for a fresh tap.
  if (!_guestAudioCtx || _guestAudioCtx.state === 'closed') {
    btn.classList.remove('active');
    const status = document.getElementById('status');
    if (status) status.textContent = 'Tap Listen again to resume audio';
    if (_guestAudioWs) {
      try { _guestAudioWs.close(); } catch (e) {}
      _guestAudioWs = null;
    }
    _teardownMediaSource();
    return;
  }

  // MSE path: check the <audio> element's health. A closed MediaSource
  // plus a dead WS triggers a full rebuild via _guestConnectMse (the
  // silent keeper keeps the audio session alive so no re-tap needed).
  if (_guestUseMse && _diag.path === 'mse') {
    const audioEl = document.getElementById('guest-audio-el');
    if (audioEl && audioEl.error) {
      _diag.lastErr = `audio.error code=${audioEl.error.code}`;
      _teardownMediaSource();
      if (_guestAudioWs) {
        try { _guestAudioWs.close(); } catch (e) {}
        _guestAudioWs = null;
      }
      _scheduleGuestReconnect();
      return;
    }
    if (audioEl && audioEl.paused) {
      // Element was paused by iOS on backgrounding. Try to resume.
      try { audioEl.play().catch(() => {}); } catch (e) {}
    }
  }

  const ws = _guestAudioWs;
  const dead = !ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING;
  if (dead) {
    if (_guestReconnectTimer) { clearTimeout(_guestReconnectTimer); _guestReconnectTimer = null; }
    _guestReconnectAttempts = 0;
    if (_guestUseMse && _diag.path === 'mse') {
      _guestConnectMse();
    } else {
      _guestConnect();
    }
  }
  // Also un-suspend the AudioContext if Safari parked it while the tab
  // was hidden — otherwise newly-arrived buffers decode fine but never
  // reach the speaker.
  if (_guestAudioCtx && _guestAudioCtx.state === 'suspended') {
    _guestAudioCtx.resume().catch(() => {});
  }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') _guestResumeIfDead();
});
window.addEventListener('focus', _guestResumeIfDead);
window.addEventListener('online', _guestResumeIfDead);
// Belt-and-braces: every 5 s check if we think we're listening but the WS
// is silently dead. Catches edge cases where onclose never fires (iOS has
// been observed to drop WebSockets without calling onclose under heavy
// memory pressure).
setInterval(_guestResumeIfDead, 5000);

document.getElementById('listen-speed-guest')?.addEventListener('change', (e) => {
  _guestPlaybackRate = parseFloat(e.target.value);
  localStorage.setItem('guest_listenSpeed', e.target.value);
});

let _guestPlaybackRate = parseFloat(localStorage.getItem('guest_listenSpeed') || '1.0');

async function _playGuestAudio() {
  if (!_guestAudioQueue.length) { _guestAudioPlaying = false; _diagRender(); return; }
  _guestAudioPlaying = true;
  const wav = _guestAudioQueue.shift();
  try {
    if (!_guestAudioCtx) {
      _diag.lastErr = 'play: ctx is null';
      _diagRender();
      return;
    }
    // Some browsers leave the context "suspended" between buffer
    // playbacks if the page isn't focused. Try to resume defensively.
    if (_guestAudioCtx.state === 'suspended') {
      try { await _guestAudioCtx.resume(); } catch (e) {
        _diag.lastErr = `play.resume: ${e.message}`;
      }
    }
    const buf = await _guestAudioCtx.decodeAudioData(wav.slice(0));
    if (!buf || buf.length === 0) {
      _diag.lastErr = `decoded empty (${buf ? buf.duration.toFixed(2) + 's' : 'null'})`;
      _diagRender();
      _playGuestAudio();
      return;
    }
    _diag.decodeOk++;
    const src = _guestAudioCtx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = _guestPlaybackRate;
    // Anti-click fade: each streaming chunk arrives with its own WAV
    // header, and the join between two consecutive BufferSources rarely
    // lands on a zero crossing of the waveform — the discontinuity
    // pops audibly at every chunk boundary. A 3ms linear fade in at
    // the head and fade out at the tail, applied via a GainNode, masks
    // the step without noticeably muddying speech. Duration matches
    // the ~2–4 ms threshold where human hearing stops detecting the
    // envelope as "amplitude modulation" and just hears "clean start".
    const gain = _guestAudioCtx.createGain();
    const now = _guestAudioCtx.currentTime;
    const fadeMs = 0.003;
    const bufDur = buf.duration / _guestPlaybackRate;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(1, now + fadeMs);
    if (bufDur > fadeMs * 2) {
      gain.gain.setValueAtTime(1, now + bufDur - fadeMs);
      gain.gain.linearRampToValueAtTime(0, now + bufDur);
    }
    src.connect(gain);
    gain.connect(_guestAudioCtx.destination);
    // Schedule the NEXT buffer to start exactly when this one ends,
    // instead of waiting for onended → main-thread jitter (10–30 ms of
    // silence). Pre-pulling the next decode here keeps the audio graph
    // continuous and further suppresses boundary artifacts.
    src.onended = () => { _diag.played++; _diagRender(); _playGuestAudio(); };
    src.start(now);
    _diagRender();
  } catch (e) {
    _diag.decodeErr++;
    _diag.lastErr = `decode: ${e.message || e.name}`;
    _diagRender();
    _playGuestAudio();
  }
}

document.getElementById('listen-mode-guest')?.addEventListener('change', (e) => {
  localStorage.setItem('guest_listenMode', e.target.value);
  if (_guestAudioWs && _guestAudioWs.readyState === WebSocket.OPEN) {
    _guestAudioWs.send(JSON.stringify({ type: 'set_mode', mode: e.target.value }));
  }
});

document.getElementById('listen-lang-guest')?.addEventListener('change', (e) => {
  localStorage.setItem('guest_listenLang', e.target.value);
  if (_guestAudioWs && _guestAudioWs.readyState === WebSocket.OPEN) {
    _guestAudioWs.send(JSON.stringify({ type: 'set_language', language: e.target.value }));
  }
});

poll();
