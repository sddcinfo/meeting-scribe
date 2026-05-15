// Cross-module shared state for the admin SPA.
//
// External contract (per docs/scribe-window-contract.md):
//   - `window.current_meeting_id` — read+write by SPA code, tests, and
//     classic sibling scripts. Backed by `state.current_meeting_id`
//     via Object.defineProperty so the two surfaces stay in lockstep
//     with a single backing storage.
//
// `store` is exported as its own named binding rather than nested
// under `state` because every consumer references it as bare
// `store.X` and renaming would churn the entire feature tree.

import { SegmentStore } from "../segment-store.js";
import { AudioPipeline } from "./features/audio-pipeline.js";
import { MeetingTimer } from "./features/meeting-timer.js";

export const store = new SegmentStore();
// Live ingestion gate: when false, every store.ingest(...) call is
// dropped. Used by the reconciler's enterLiveMeetingMode to suppress
// WS events during the journal-replay rebuild step.
store._liveEnabled = true;

// Mic-capture / ASR / live WS pipeline singleton. Lives here so the
// recording-lifecycle helpers (startRecording / stopRecording /
// _resumeMeeting / ...) can use it without a deps bag.
export const audio = new AudioPipeline();

// Meeting-clock ticker singleton — the mm:ss readout in the meeting-
// mode header. Construction takes the timer DOM node directly; this
// runs at module-eval time which is AFTER the deferred entry-point
// script kicks in, so the DOM is parsed by then.
export const timer = new MeetingTimer(document.getElementById("timer"));


export const state = {
  current_meeting_id: null,
  // Bilingual default — overwritten on first /api/status hydration.
  // Single-string codes (e.g. "en") denote monolingual meetings.
  currentLanguagePair: "en,ja",
  // code → { name, native_name, css_font_class }. Populated from the
  // first /api/languages fetch; read by transcript renderers, the
  // bilingual layout helpers, and the popout views.
  languageNames: {},
  // Wall-clock epoch (ms) for the moment recording started — or, when
  // viewing a completed meeting, `meta.created_at`. Drives the
  // HH:MM:SS-in-browser-tz formatting in lib/time-format.js. Zero
  // means "no meeting active yet"; formatTime falls back to elapsed
  // mm:ss in that case.
  meetingStartWallMs: 0,
  // Header live-stats counters. Bumped by `AudioPipeline.start` on
  // every inbound WS message + every outbound mic-chunk frame; read
  // by `updateLiveStats` to render the header stats popover. Reset
  // to zero each time recording starts in startRecording().
  wsMessageCount: 0,
  audioChunkCount: 0,
};

// current_meeting_id is BOTH internal state AND a window contract.
// Routing it through `state` via defineProperty keeps both surfaces
// single-backed.
Object.defineProperty(window, "current_meeting_id", {
  get: () => state.current_meeting_id,
  set: (v) => {
    state.current_meeting_id = v;
  },
  configurable: true,
});
