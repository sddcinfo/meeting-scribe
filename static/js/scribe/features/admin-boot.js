/**
 * Meeting Scribe — main application controller.
 *
 * Event-driven architecture:
 *   AudioWorklet → WebSocket (binary) → Server ASR/Translate → WebSocket (JSON) → SegmentStore → DOM
 *
 * SegmentStore: keyed by segment_id, tracks highest revision per segment.
 * DOM updates are batched via requestAnimationFrame for smooth rendering.
 */

// esc() — HTML-escape helper used pervasively for innerHTML
// interpolation. Lives in scribe/lib/escape.js so feature modules
// can import it cleanly.
import { esc } from "../lib/escape.js";
// micWarmup singleton — primed when the room-setup view appears,
// consumed by AudioPipeline when the meeting starts. See
// features/mic-warmup.js for the lifecycle contract.
import { micWarmup } from "./mic-warmup.js";
// Shared cross-module state — currentLanguagePair, current_meeting_id, …
// All reads/writes go through ``state.<key>`` (no double-backed module
// scope copy). See state.js for the inventory + window-contract bindings.
// `store` is the SegmentStore singleton; `audio` is the AudioPipeline
// singleton — both live in state.js so every feature module shares a
// single instance via a clean import path.
import { audio, state, store, timer } from "../state.js";
// Modal system — showModal/closeModal/closeAllModals plus the three
// dialog primitives. Window-contract surfaces (alertDialog,
// confirmDialog, promptDialog, closeAllModals) are stamped by
// features/modal-system.bootstrap.js after this file evaluates.
import {
  alertDialog,
  closeAllModals,
  closeModal,
  confirmDialog,
  promptDialog,
  showModal,
} from "./modal-system.js";
// Speaker registry — the three handles this boot orchestrator calls
// directly (other surfaces are imported by their respective feature
// modules).
import {
  _speakerRegistry,
  getSpeakerDisplayName,
  isPseudoCluster as _isPseudoCluster,
  refreshTranscriptSpeakerLabels as _refreshTranscriptSpeakerLabels,
} from "./speaker-registry.js";
// Q&A panel — streaming chat-style ask-the-meeting flow, mounted by
// finalize-summary into the review modal.
import { initQaPanel as _initQaPanel } from "./qa-panel.js";
// Finalize-summary modal renderer. The pure module takes a hooks bag
// for lang codes + onViewMeeting; this file wraps it with the actual
// values bound here.
import {
  renderFinalizationSummary as _renderFinalizationSummaryRaw,
  renderSummaryPanel as _renderSummaryPanelRaw,
} from "./finalize-summary.js";
function _renderFinalizationSummary(summary, meetingId) {
  return _renderFinalizationSummaryRaw(summary, meetingId, {
    langA: _getLangA(),
    langB: _getLangB(),
    onViewMeeting: (id) => meetingsMgr?.viewMeeting(id),
  });
}
// Shared language helpers — used here only for the
// _renderFinalizationSummary wrapper hooks; isMonolingual + route-by-script
// are pulled directly by feature modules.
import {
  getLangA as _getLangA,
  getLangB as _getLangB,
} from "../lib/lang-helpers.js";
// MeetingTimer class is imported by state.js (where the singleton
// `timer` is constructed); this file consumes `timer` via the named
// import from state.js below.
// AudioPipeline — recorder + WebSocket + AudioWorklet stack that
// drives ASR/translate during a live meeting. The instance `audio`
// below feeds segments back via ingestFromLiveWs(); hooks plumb
// speaker-registry refreshes and the meeting-cancelled cleanup back
// into this file since the refresh helpers + segment store live here.
// Tiny URL helpers — `_enc(s)` = encodeURIComponent shorthand,
// `_meetingsUrl(mid, suffix)` builds `/api/meetings/<encoded-mid><suffix>`.
import { _enc, _meetingsUrl } from "../lib/meeting-url.js";
// Speaker-review modal + timeline + segment finder. The functions
// take their state dependency (the SegmentStore singleton) via the
// `hooks.store` parameter, and the post-rename refresh via the
// `hooks.onSpeakerRegistryChanged` callback. Thin closures below
// bind `store` and the refresh helpers so call sites stay 4-arg.
import {
  findSpeakerSegments as _findSpeakerSegmentsRaw,
  showSpeakerModal as _showSpeakerModalRaw,
  renderSpeakerTimeline as _renderSpeakerTimelineRaw,
  openSpeakerRenameModal as _openSpeakerRenameModalRaw,
} from "./speaker-review.js";
function _openSpeakerRenameModal(clusterId, currentName, color) {
  return _openSpeakerRenameModalRaw(clusterId, currentName, color, {
    onSpeakerRegistryChanged: () => {
      _refreshDetectedSpeakersStrip();
      _refreshTranscriptSpeakerLabels();
    },
  });
}
// Background-finalize toast plumbing — `setMeetingsMgrRef` is called
// right after constructing `meetingsMgr` so the module can repaint the
// list on terminal-success.
import { setMeetingsMgrRef } from "./bg-finalize-toast.js";
// Pop-out window opener — spawns `?popout=view` in a fresh tab.
import { openPopout as _openPopout } from "./popout-window.js";
// MeetingsManager — owns the slide-over history panel + review entry
// path. Constructed below with a deps bag carrying the recording-
// lifecycle helpers + summary panel renderer + language wrappers + the
// three speaker-review wrappers that bind `store` + a refresh hook.
import { MeetingsManager } from "./meetings-manager.js";
// Baseline SegmentStore subscribers (`#segment-count` chip + detected-
// speakers tracker) + the `?test=1` window-contract hooks live in
// features/segment-store-subscribers.bootstrap.js. We call
// `configureSegmentStoreSubscribers({...})` once with a getter for
// `_refreshDetectedSpeakersStrip` (which closes over roomSetup).
import { configureSegmentStoreSubscribers } from "./segment-store-subscribers.bootstrap.js";
// RoomSetup — singleton constructed below so we can pass `startRecording`
// + a lazy reconciler getter into the constructor.
import { RoomSetup } from "./room-setup.js";
// Live-session pure helpers (validateMic / setStoreLive /
// loadMeetingJournal / resetLiveStore / showMeetingMode). Aliased to
// underscored names that match the reconciler-init deps bag shape.
import {
  ingestFromLiveWs,
  loadMeetingJournal as _loadMeetingJournal,
  resetLiveStore as _resetLiveStore,
  setStoreLive,
  showMeetingMode as _showMeetingMode,
  validateMic,
} from "./live-session.js";
// View-only WS lifecycle (read-only stream for observer tabs) and the
// recording-WS readyState probe. Configured once below so the speaker-
// rename branch can refresh the table strip.
import {
  attachViewOnlyWs as _attachViewOnlyWs,
  configureViewOnlyWs,
  detachViewOnlyWs as _detachViewOnlyWs,
  getAudioWsState as _getAudioWsState,
} from "./view-only-ws.js";
// `exitLiveMeetingView` is the navigation helper passed to
// MeetingsManager's constructor below. The hideLanding / showLanding
// helpers are consumed directly by the admin-navigation bootstrap.
import { exitLiveMeetingView } from "./navigation.js";
// Room-editor overlay + cluster sidebar — full module under
// features/room-editor.{js,bootstrap.js}. The two callback hooks
// (`_openSpeakerRenameModal`, `_refreshDetectedSpeakersStrip`) reach
// the editor module via the one-time `configureRoomEditor` setter
// below; the bootstrap half wires the DOM event listeners +
// window._onRoomLayoutUpdate publish.
import { configureRoomEditor } from "./room-editor.js";
// Recording-lifecycle helpers (resumeMeeting, startRecording,
// stopRecording, showFinalizationSummaryFor). The deps that live in
// this file (the reconciler + meetingsMgr singletons, the refresh
// helpers, the summary wrapper) are handed in via
// configureRecordingLifecycle() once every dep target has been
// declared.
import {
  configureRecordingLifecycle,
  resumeMeeting as _resumeMeeting,
  showFinalizationSummaryFor,
  startRecording,
} from "./recording-lifecycle.js";
// Meeting controls (btn-record / btn-takeover-recording /
// btn-cancel-meeting). Same lazy-getter pattern for reconciler /
// meetingsMgr as the recording-lifecycle module.
import {
  bootMeetingControls,
  configureMeetingControls,
} from "./meeting-controls.js";
// Admin SPA navigation bootstrap — wires btn-home / landing-start /
// landing-quick-english / hash router + MeetingsManager prototype
// patches. Booted right after meetingsMgr + roomSetup are constructed
// (lazy getter deps).
import {
  bootAdminNavigation,
  configureAdminNavigation,
} from "./admin-navigation.bootstrap.js";
// `showBanner` plumbs the "Meeting in progress · Return" /
// "Reconnecting…" surfaces into createReconciler's deps bag.
import { showBanner as _showBanner } from "./meeting-banner.js";
// `/api/status` poll handler — `checkStatus()` repaints every backend
// pill, the TTS-stalled badge, the live stats footer, then hands off
// to `reconciler.reconcile(data)`. The status-poll module owns the
// 401-redirect helper too.
import {
  checkStatus,
  configureStatusPoll,
} from "./status-poll.js";
import { updateColumnHeaders as _updateColumnHeadersRaw } from "./column-headers.js";
// Setup-screen language A/B dropdown loader. The IIFE that hydrates
// `state.languageNames` + caches the server's default pair lives in
// the bootstrap half so it fires after this orchestrator.
import { getDefaultLanguagePair as _defaultLanguagePair } from "./language-loader.js";
function findSpeakerSegments(speakerName) {
  return _findSpeakerSegmentsRaw(speakerName, store);
}
function showSpeakerModal(speakerName, color, segments, meetingId) {
  return _showSpeakerModalRaw(speakerName, color, segments, meetingId, {
    store,
    onSpeakerRegistryChanged: () => {
      _refreshTranscriptSpeakerLabels();
      _refreshDetectedSpeakersStrip();
    },
  });
}
function renderSpeakerTimeline(speakerLanes, durationMs, speakers, meetingId) {
  return _renderSpeakerTimelineRaw(speakerLanes, durationMs, speakers, meetingId, {
    store,
    onSpeakerRegistryChanged: () => {
      _refreshTranscriptSpeakerLabels();
      _refreshDetectedSpeakersStrip();
    },
  });
}

const API = '';
const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTO}//${location.host}/api/ws`;

// View-only mode: determined by server (only the recording session is admin)
// Pop-out mode: ?popout=view opens a clean 2-column translation window.
// The `view-only` body class carries the read-only state the rest of
// the SPA inspects (CSS rules + the takeover-bar visibility).
const POPOUT_MODE = new URLSearchParams(location.search).get('popout');
if (POPOUT_MODE) {
  document.body.classList.add('popout-view');
  document.body.classList.add('view-only');
}

// Server-configured default pair (see language-loader.js) is consumed
// by `_defaultLanguagePair` via the named import above. The IIFE that
// hydrates from /api/languages + writes to the cache lives in the
// bootstrap half. Mono-language meetings are NOT exposed via this
// default — they live behind the landing page's "Quick start English"
// button; the helper resets the setup-screen dropdowns back to a
// bilingual pair every time the user returns to setup, so a prior
// mono meeting can't leak into the next one.

function _updateColumnHeaders() {
  return _updateColumnHeadersRaw(state);
}



// ─── Construct singletons + boot wiring ────────────────────────
//
// This orchestrator owns:
//
//   * Construction of the three singletons — `roomSetup`,
//     `reconciler`, `meetingsMgr` — that feature modules reach via
//     lazy getters in their configure() deps bags.
//   * Tiny pure-data caches + thin wrappers (e.g. `_renderSummaryPanel`)
//     that bind feature-module imports to constructor deps with the
//     shape MeetingsManager expects.
//
// `reconciler` starts null because its construction needs
// `startRecording` + `_getAudioWsState` + …, which are imported above
// but the createReconciler call itself happens below the
// MeetingsManager constructor (which itself needs `meetingsMgr` as a
// lazy reference). The let-binding + lazy-getter pattern in every
// configure() call keeps the boot-order constraint expressible
// without circular imports.

const roomSetup = new RoomSetup({
  startRecording: (isResume) => startRecording(isResume),
  getReconciler: () => reconciler,
});

import { createReconciler } from '../../meeting-reconcile.js';

let reconciler = null;

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

// Hand the segment-store-subscribers bootstrap the strip-refresh
// helper. Done here (post-definition) so the bootstrap — which fires
// later via the import below — can invoke it without circular
// imports back into this orchestrator.
configureSegmentStoreSubscribers({
  refreshDetectedSpeakersStrip: _refreshDetectedSpeakersStrip,
});



// Hand the view-only WS module the strip-refresh hook it needs for
// its speaker_rename branch.
configureViewOnlyWs({
  refreshDetectedSpeakersStrip: _refreshDetectedSpeakersStrip,
});

// Recorder-ownership token storage. Popout windows are deliberate
// read-only surfaces — using `localStorage` there would let a popout
// silently claim the recorder role across page reloads. Admin SPA
// flips to `localStorage` so takeover survives reloads + restarts;
// popout stays on `sessionStorage` so ownership scopes to the
// window's session.
const _reconcilerStorage = POPOUT_MODE ? window.sessionStorage : window.localStorage;

reconciler = createReconciler({
  doc: document,
  storage: _reconcilerStorage,
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

// MeetingsManager construction — feeds in the boot-resident deps the
// class needs (recording-lifecycle helpers, the summary-panel +
// finalization renderers, the language pair wrappers, and the three
// speaker-review wrappers that bind the shared `store` + the refresh
// hook).
const meetingsMgr = new MeetingsManager({
  exitLiveMeetingView,
  _resumeMeeting,
  _renderSummaryPanel,
  _renderFinalizationSummary,
  _updateColumnHeaders,
  showFinalizationSummaryFor,
  _defaultLanguagePair,
  findSpeakerSegments,
  showSpeakerModal,
  renderSpeakerTimeline,
});
// Hook the bg-finalize toast module up to the meetings manager so its
// terminal-success path can repaint the list. Done here (right after
// the manager is constructed) so the reference is non-null by the
// time the first WS finalize event fires.
setMeetingsMgrRef(meetingsMgr);

// Lazy getters for reconciler / meetingsMgr / roomSetup keep the dep
// lookups stable across this file's let-rebinding + post-construction
// phases.
configureRecordingLifecycle({
  renderFinalizationSummary: _renderFinalizationSummary,
  updateColumnHeaders: _updateColumnHeaders,
  refreshDetectedSpeakersStrip: _refreshDetectedSpeakersStrip,
  refreshTranscriptSpeakerLabels: _refreshTranscriptSpeakerLabels,
  getReconciler: () => reconciler,
  getMeetingsMgr: () => meetingsMgr,
  getRoomSetup: () => roomSetup,
});

// Wire + boot the meeting-controls module (btn-record, btn-takeover,
// btn-cancel-meeting). Lazy getters for reconciler + meetingsMgr
// resolve at click-time, not boot-time.
configureMeetingControls({
  getReconciler: () => reconciler,
  getMeetingsMgr: () => meetingsMgr,
});
bootMeetingControls();

// Wire the /api/status poll handler with a lazy getter for the
// reconciler binding. The poll loop lives in this orchestrator (it
// depends on POPOUT_MODE + document.body.classList) — it just calls
// the imported `checkStatus()` from features/status-poll.js.
configureStatusPoll({
  getReconciler: () => reconciler,
});

// Wire + boot the admin-navigation listeners. Skip in popout mode
// since popout windows have their own navigation surface.
configureAdminNavigation({
  getMeetingsMgr: () => meetingsMgr,
  getRoomSetup: () => roomSetup,
});
if (!POPOUT_MODE) bootAdminNavigation();

// Expose `showFinalizationSummaryFor` for inline onclick handlers in
// the meetings-panel rows.
window.showFinalizationSummaryFor = showFinalizationSummaryFor;

// Thin wrapper passed into MeetingsManager so its `_renderSummaryPanel`
// call site stays 2-arg; the real renderer lives in
// features/finalize-summary.js.
function _renderSummaryPanel(summary, meetingId) {
  return _renderSummaryPanelRaw(summary, meetingId, {
    onOpenSummary: (mid) => showFinalizationSummaryFor(mid),
  });
}

// Wire the room-editor overlay with the two hook wrappers + the
// roomSetup singleton. The bootstrap half wires the DOM event
// listeners + window._onRoomLayoutUpdate publish after this call.
configureRoomEditor({
  roomSetup,
  openSpeakerRenameModal: _openSpeakerRenameModal,
  refreshDetectedSpeakersStrip: _refreshDetectedSpeakersStrip,
});


