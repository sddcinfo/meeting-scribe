// Meeting Scribe — View-only WebSocket lifecycle.
//
// When a tab does NOT own the recorder but is showing a live meeting,
// the reconciler attaches a read-only WS at /api/ws/view to stream
// the same transcript events the recording tab is producing. This
// keeps observer tabs (laptop B watching laptop A's live recording)
// in lockstep without dueling for mic ownership.
//
// `getAudioWsState` is the recording-WS readyState probe the
// reconciler uses to decide when to upgrade / take over a meeting;
// kept in this module because it lives in the same conceptual
// "what's the WS doing right now" bucket.
//
// The `.view-only` body class is the single source of truth for the
// observer-tab state — every other surface reads it.

import { audio } from "../state.js";
import {
  refreshTranscriptSpeakerLabels,
  renameSpeaker,
} from "./speaker-registry.js";
import { ingestFromLiveWs } from "./live-session.js";

const WS_PROTO = location.protocol === "https:" ? "wss:" : "ws:";

let _deps = null;
let _viewOnlyWs = null;
let _viewOnlyKeepAlive = null;

export function configureViewOnlyWs(deps) {
  _deps = deps;
}

export function getAudioWsState() {
  const ws = audio && audio.ws;
  if (!ws) return "closed";
  switch (ws.readyState) {
    case WebSocket.OPEN: return "open";
    case WebSocket.CONNECTING: return "connecting";
    default: return "closed";
  }
}

export function attachViewOnlyWs(meetingId) {
  if (_viewOnlyWs && _viewOnlyWs.readyState !== WebSocket.CLOSED) return;
  document.body.classList.add("view-only");
  const viewWs = new WebSocket(`${WS_PROTO}//${location.host}/api/ws/view`);
  viewWs.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      // Server-authoritative popout layout sync. Route the event to
      // the storage module's handler so the kiosk mirror and every
      // other admin popout re-render off the same source of truth
      // without echoing back through PUT.
      if (msg.type === "popout_layout_changed" && window.PopoutLayoutStorage?.handleWsMessage) {
        window.PopoutLayoutStorage.handleWsMessage(msg);
        return;
      }
      if (msg.type === "speaker_rename" && msg.cluster_id != null) {
        renameSpeaker(msg.cluster_id, msg.display_name);
        _deps?.refreshDetectedSpeakersStrip?.();
        refreshTranscriptSpeakerLabels();
      } else if (msg.type === "bt_status") {
        window.dispatchEvent(new CustomEvent("meeting-scribe:bt-status", { detail: msg.status || msg }));
      } else if (msg.type === "mic_level") {
        const bar = document.getElementById("meter-bar");
        if (bar) bar.style.width = `${msg.peak_pct ?? 0}%`;
      } else {
        ingestFromLiveWs(msg);
      }
    } catch {}
  };
  viewWs.onopen = () => {
    document.getElementById("status-line").textContent = `Viewing ${meetingId} (read-only)`;
  };
  _viewOnlyWs = viewWs;
  _viewOnlyKeepAlive = setInterval(() => {
    if (viewWs.readyState === WebSocket.OPEN) viewWs.send("ping");
    else { clearInterval(_viewOnlyKeepAlive); _viewOnlyKeepAlive = null; }
  }, 30000);
}

export function detachViewOnlyWs() {
  if (_viewOnlyWs) { _viewOnlyWs.close(); _viewOnlyWs = null; }
  if (_viewOnlyKeepAlive) { clearInterval(_viewOnlyKeepAlive); _viewOnlyKeepAlive = null; }
  document.body.classList.remove("view-only");
}
