// Meeting Scribe — Admin slide bar.
//
// The slide preview / Prev-Next nav / speaker-notes panel that the
// operator sees above the live transcript when a deck is uploaded
// for translation. Lives on the admin SPA only — popout windows
// render their own slide surface via the popout SPA. This file
// exports the boot function the bootstrap half calls; everything
// the module touches is DOM, localStorage, fetch, or the
// `scribe-ws-message` window event bus.
//
// State that ``startRecording`` needs to reset between meetings is
// stamped onto `window._adminSlideResetUI` during boot — the same
// surface the recording-lifecycle module already invokes when
// kicking off a fresh-start.

import { _enc } from "../lib/meeting-url.js";

const API = "";

export function bootAdminSlideBar() {
  const _adminSlideBar = document.getElementById("admin-slide-bar");
  const _adminSlideThumb = document.getElementById("admin-slide-thumb");
  const _adminSlideThumbTrans = document.getElementById("admin-slide-thumb-translated");
  const _adminSlideThumbTransStatus = document.getElementById("admin-slide-thumb-trans-status");
  const _adminSlideLabel = document.getElementById("admin-slide-label");
  const _adminSlidePrev = document.getElementById("admin-slide-prev");
  const _adminSlideNext = document.getElementById("admin-slide-next");
  const _adminSlideResize = document.getElementById("admin-slide-resize");
  const _adminSlideSplitter = document.getElementById("admin-slide-splitter");
  const _adminSlideNotes = document.getElementById("admin-slide-notes");
  const _adminSlideNotesBody = document.getElementById("admin-slide-notes-body");

  // Per-deck speaker-notes cache. Loaded once from
  // GET /api/meetings/<id>/slides/notes when a deck arrives, then
  // indexed by current slide as the operator navigates. Source-
  // language only for now.
  const _adminSlideNotesCache = { deckId: null, notes: [] };
  async function _adminFetchNotes(meetingId) {
    if (!meetingId) return;
    try {
      const r = await fetch(`${API}/api/meetings/${_enc(meetingId)}/slides/notes`);
      if (!r.ok) {
        _adminSlideNotesCache.notes = [];
        return;
      }
      const data = await r.json();
      _adminSlideNotesCache.notes = Array.isArray(data.notes) ? data.notes : [];
    } catch {
      _adminSlideNotesCache.notes = [];
    }
  }
  function _adminRenderNotes() {
    if (!_adminSlideNotes || !_adminSlideNotesBody) return;
    const idx = _adminSlideState.current;
    const text = (_adminSlideNotesCache.notes[idx] || "").trim();
    if (!text) {
      _adminSlideNotesBody.textContent = "No notes for this slide.";
      _adminSlideNotesBody.classList.add("is-empty");
    } else {
      _adminSlideNotesBody.textContent = text;
      _adminSlideNotesBody.classList.remove("is-empty");
    }
  }

  // Resize handle on the bottom edge of the admin slide bar. The
  // operator drags it up/down to adjust how much vertical real
  // estate the slide preview claims (vs. the transcript below).
  // Persisted as ``admin_slide_bar_height_vh``; double-click resets
  // to the 60vh default. Range 25–85 vh, clamped on read.
  if (_adminSlideBar && _adminSlideResize) {
    const STORAGE_KEY = "admin_slide_bar_height_vh";
    const DEFAULT_VH = 60;
    const MIN_VH = 25;
    const MAX_VH = 85;
    const _applyHeight = (vh) => {
      const clamped = Math.max(MIN_VH, Math.min(MAX_VH, vh));
      _adminSlideBar.style.setProperty("--admin-slide-bar-height", clamped + "vh");
      return clamped;
    };
    const stored = parseFloat(localStorage.getItem(STORAGE_KEY) || "");
    if (Number.isFinite(stored)) _applyHeight(stored);
    else _applyHeight(DEFAULT_VH);

    let dragging = false;
    let startY = 0;
    let startVh = DEFAULT_VH;
    const _onMove = (ev) => {
      if (!dragging) return;
      const dy = ev.clientY - startY;
      const dvh = (dy / window.innerHeight) * 100;
      const next = _applyHeight(startVh + dvh);
      ev.preventDefault();
      _adminSlideResize.dataset.lastVh = String(next);
    };
    const _onUp = () => {
      if (!dragging) return;
      dragging = false;
      _adminSlideResize.classList.remove("dragging");
      document.body.style.cursor = "";
      document.removeEventListener("mousemove", _onMove);
      document.removeEventListener("mouseup", _onUp);
      const lastVh = parseFloat(_adminSlideResize.dataset.lastVh || "");
      if (Number.isFinite(lastVh)) {
        try { localStorage.setItem(STORAGE_KEY, String(lastVh)); } catch {}
      }
    };
    _adminSlideResize.addEventListener("mousedown", (ev) => {
      dragging = true;
      startY = ev.clientY;
      const cssVh = parseFloat(getComputedStyle(_adminSlideBar)
        .getPropertyValue("--admin-slide-bar-height")) || DEFAULT_VH;
      startVh = cssVh;
      _adminSlideResize.classList.add("dragging");
      document.body.style.cursor = "ns-resize";
      document.addEventListener("mousemove", _onMove);
      document.addEventListener("mouseup", _onUp);
      ev.preventDefault();
    });
    _adminSlideResize.addEventListener("dblclick", () => {
      _applyHeight(DEFAULT_VH);
      try { localStorage.removeItem(STORAGE_KEY); } catch {}
    });
  }

  const _adminSlideState = { meetingId: null, deckId: null, total: 0, current: 0 };

  // Exposed for the fresh-meeting carry-over reset in ``startRecording``.
  // _adminSlideState is lexically scoped to this function; without an
  // exported handle the recording-lifecycle module cannot clear it,
  // and a previous meeting's slide thumbnails + numeric state bleed
  // into the next meeting's UI even though the server discards them.
  window._adminSlideResetUI = function _adminSlideResetUI() {
    _adminSlideState.meetingId = null;
    _adminSlideState.deckId = null;
    _adminSlideState.total = 0;
    _adminSlideState.current = 0;
    if (_adminSlideThumb) {
      _adminSlideThumb.removeAttribute("src");
      _adminSlideThumb.alt = "Current slide (original)";
    }
    if (_adminSlideThumbTrans) {
      _adminSlideThumbTrans.removeAttribute("src");
      _adminSlideThumbTrans.alt = "Current slide (translated)";
    }
    if (_adminSlideThumbTransStatus) {
      _adminSlideThumbTransStatus.style.display = "none";
      _adminSlideThumbTransStatus.textContent = "Translating…";
    }
    if (_adminSlideLabel) _adminSlideLabel.textContent = "0 / 0";
    if (_adminSlidePrev) _adminSlidePrev.disabled = true;
    if (_adminSlideNext) _adminSlideNext.disabled = true;
    if (_adminSlideBar) _adminSlideBar.style.display = "none";
  };

  function _adminSlidesGetMeetingId() {
    // Read the canonical meeting id ``window.current_meeting_id``
    // (written by ``startRecording`` and by the reload-rehydration
    // init below). The pre-fix code read ``window._currentMeetingId``,
    // a parallel one-off variable that was only populated by the
    // IIFE below at page load — so a fresh start in the *same* page
    // produced a null read here, the upload alerted "No active
    // meeting", and the user concluded uploads were broken.
    const cm = window.current_meeting_id;
    if (cm) return cm;
    const url = location.hash.match(/^#meeting\/(.+)/);
    return url ? url[1] : null;
  }

  function _adminSlideRefresh() {
    if (!_adminSlideThumb) return;
    if (!_adminSlideState.deckId || !_adminSlideState.meetingId) {
      _adminSlideThumb.removeAttribute("src");
      if (_adminSlideThumbTrans) _adminSlideThumbTrans.removeAttribute("src");
      if (_adminSlideThumbTransStatus) _adminSlideThumbTransStatus.classList.remove("is-hidden");
      _adminSlideLabel.textContent = "— / —";
      if (_adminSlidePrev) _adminSlidePrev.disabled = true;
      if (_adminSlideNext) _adminSlideNext.disabled = true;
      return;
    }
    const base = `${API}/api/meetings/${_enc(_adminSlideState.meetingId)}/slides/${_adminSlideState.current}`;
    const bust = `?d=${_adminSlideState.deckId}`;
    _adminSlideThumb.src = `${base}/original${bust}`;

    if (_adminSlideThumbTrans) {
      // Translation runs after rendering_original completes. While
      // we wait, the status overlay (``Translating…``) covers the
      // empty <img>. Once the load fires, hide the overlay; on
      // error keep it visible so the user knows it's still cooking
      // / failed rather than seeing a blank pane.
      if (_adminSlideThumbTransStatus) _adminSlideThumbTransStatus.classList.remove("is-hidden");
      _adminSlideThumbTrans.onload = () => {
        if (_adminSlideThumbTransStatus) _adminSlideThumbTransStatus.classList.add("is-hidden");
      };
      _adminSlideThumbTrans.onerror = () => {
        if (_adminSlideThumbTransStatus) _adminSlideThumbTransStatus.classList.remove("is-hidden");
      };
      _adminSlideThumbTrans.src = `${base}/translated${bust}`;
    }

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
      await fetch(`${API}/api/meetings/${_enc(_adminSlideState.meetingId)}/slides/current`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: newIdx }),
      });
      // Optimistic update — the broadcast will confirm.
      _adminSlideState.current = newIdx;
      _adminSlideRefresh();
    } catch (e) {
      console.warn("admin slide advance failed:", e);
    }
  }

  if (_adminSlidePrev) _adminSlidePrev.addEventListener("click", () => _adminSlideAdvance(-1));
  if (_adminSlideNext) _adminSlideNext.addEventListener("click", () => _adminSlideAdvance(1));

  // Splitter: drag horizontally to adjust slides : notes ratio.
  // Persists to localStorage as ``admin_slide_split_flex`` (an integer
  // 30–80 representing slide-pane percentage). Double-click resets.
  if (_adminSlideBar && _adminSlideSplitter) {
    const SPLIT_KEY = "admin_slide_split_flex";
    const SPLIT_DEFAULT = 70;
    const SPLIT_MIN = 30;
    const SPLIT_MAX = 80;
    const _applySplit = (pct) => {
      const clamped = Math.max(SPLIT_MIN, Math.min(SPLIT_MAX, pct));
      _adminSlideBar.style.setProperty("--admin-slide-split-flex", String(clamped));
      return clamped;
    };
    const stored = parseFloat(localStorage.getItem(SPLIT_KEY) || "");
    if (Number.isFinite(stored)) _applySplit(stored);
    else _applySplit(SPLIT_DEFAULT);

    let dragging = false;
    let startX = 0;
    let startPct = SPLIT_DEFAULT;
    let mainWidth = 0;
    const _onMove = (ev) => {
      if (!dragging || !mainWidth) return;
      const dx = ev.clientX - startX;
      const dpct = (dx / mainWidth) * 100;
      const next = _applySplit(startPct + dpct);
      ev.preventDefault();
      _adminSlideSplitter.dataset.lastPct = String(next);
    };
    const _onUp = () => {
      if (!dragging) return;
      dragging = false;
      _adminSlideSplitter.classList.remove("dragging");
      document.body.style.cursor = "";
      document.removeEventListener("mousemove", _onMove);
      document.removeEventListener("mouseup", _onUp);
      const lastPct = parseFloat(_adminSlideSplitter.dataset.lastPct || "");
      if (Number.isFinite(lastPct)) {
        try { localStorage.setItem(SPLIT_KEY, String(lastPct)); } catch {}
      }
    };
    _adminSlideSplitter.addEventListener("mousedown", (ev) => {
      const main = _adminSlideBar.querySelector(".admin-slide-main");
      mainWidth = main ? main.getBoundingClientRect().width : 0;
      if (!mainWidth) return;
      dragging = true;
      startX = ev.clientX;
      startPct = parseFloat(getComputedStyle(_adminSlideBar)
        .getPropertyValue("--admin-slide-split-flex")) || SPLIT_DEFAULT;
      _adminSlideSplitter.classList.add("dragging");
      document.body.style.cursor = "ew-resize";
      document.addEventListener("mousemove", _onMove);
      document.addEventListener("mouseup", _onUp);
      ev.preventDefault();
    });
    _adminSlideSplitter.addEventListener("dblclick", () => {
      _applySplit(SPLIT_DEFAULT);
      try { localStorage.removeItem(SPLIT_KEY); } catch {}
    });
  }

  // Slide events reach this handler via the `scribe-ws-message` bus
  // that `ingestFromLiveWs` dispatches; the bus carries every WS
  // message, including the control-plane slide_* events.
  function _adminHandleSlideMsg(msg) {
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "slide_deck_ready" || msg.type === "slide_deck_changed") {
      _adminSlideState.meetingId = msg.meeting_id || _adminSlideState.meetingId || _adminSlidesGetMeetingId();
      _adminSlideState.deckId = msg.deck_id;
      _adminSlideState.total = msg.total_slides || msg.total || 0;
      _adminSlideState.current = msg.current_slide_index || 0;
      if (_adminSlideBar) _adminSlideBar.style.display = "";
      _adminSlideRefresh();
      // Refresh speaker-notes cache when the deck arrives or
      // changes. Notes are static per-deck, so we only re-fetch
      // when deck_id flips.
      if (_adminSlideState.deckId !== _adminSlideNotesCache.deckId) {
        _adminSlideNotesCache.deckId = _adminSlideState.deckId;
        _adminFetchNotes(_adminSlideState.meetingId).then(_adminRenderNotes);
      } else {
        _adminRenderNotes();
      }
    } else if (msg.type === "slide_change") {
      if (typeof msg.slide_index === "number") {
        _adminSlideState.current = msg.slide_index;
        if (msg.deck_id) _adminSlideState.deckId = msg.deck_id;
        _adminSlideRefresh();
        _adminRenderNotes();
      }
    } else if (msg.type === "slide_partial_ready") {
      // Per-slide PNG landed on disk. Listening for this is what
      // unblocks the initial-load case where the deck broadcast
      // arrives before the first slide's PNG has finished rendering;
      // without it the admin used to show a blank thumbnail until
      // the operator clicked Next.
      if (!msg.deck_id || msg.deck_id !== _adminSlideState.deckId) return;
      if (typeof msg.index !== "number" || msg.index !== _adminSlideState.current) return;
      const base = `${API}/api/meetings/${_enc(_adminSlideState.meetingId)}/slides/${msg.index}`;
      const cb = `?d=${_adminSlideState.deckId}&t=${Date.now()}`;
      if (msg.kind === "original" && _adminSlideThumb) {
        _adminSlideThumb.src = `${base}/original${cb}`;
      } else if (msg.kind === "translated" && _adminSlideThumbTrans) {
        _adminSlideThumbTrans.src = `${base}/translated${cb}`;
        if (_adminSlideThumbTransStatus) {
          _adminSlideThumbTrans.onload = () => _adminSlideThumbTransStatus.classList.add("is-hidden");
        }
      }
    }
  }
  window.addEventListener("scribe-ws-message", (e) => _adminHandleSlideMsg(e.detail));

  // Reload-rehydration: if a deck is already active when the admin
  // tab loads (operator refresh mid-meeting), populate the bar
  // immediately rather than waiting for the next broadcast.
  (async () => {
    try {
      const sresp = await fetch(`${API}/api/status`);
      const sd = await sresp.json();
      const mid = sd?.meeting?.id;
      if (!mid) return;
      // Write the canonical ``current_meeting_id`` so the reload
      // path produces the same global state as the start-meeting
      // path. Other callers (claimOwnership, the slide-upload PPTX
      // path, the audio-out listener wiring) all read this variable;
      // pre-fix the reload path populated only the parallel
      // ``_currentMeetingId`` and they all saw stale state.
      window.current_meeting_id = mid;
      const dresp = await fetch(`${API}/api/meetings/${_enc(mid)}/slides`);
      if (!dresp.ok) return;
      const dd = await dresp.json();
      if (!dd.deck_id) return;
      _adminSlideState.meetingId = mid;
      _adminSlideState.deckId = dd.deck_id;
      _adminSlideState.total = dd.total_slides || 0;
      _adminSlideState.current = dd.current_slide_index || 0;
      if (_adminSlideBar) _adminSlideBar.style.display = "";
      _adminSlideRefresh();
      _adminSlideNotesCache.deckId = _adminSlideState.deckId;
      await _adminFetchNotes(mid);
      _adminRenderNotes();
    } catch {}
  })();
}
