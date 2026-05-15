// Meeting Scribe — Live-session pure helpers.
//
// Five small utilities consumed by the admin SPA boot orchestrator
// and the reconciler-init deps bag:
//
//   * validateMic — pre-flight mic permissions + amplitude check
//     fired by ``startRecording(false)``. Pure DOM + getUserMedia.
//   * setStoreLive — toggle for the live-WS ingestion gate;
//     mutates `store._liveEnabled`. The reconciler's
//     `enterLiveMeetingMode` step 2 calls this with false to
//     suppress live events during journal-replay rebuilds.
//   * loadMeetingJournal — fetch a meeting's events array via
//     /api/meetings/{id} and ingest into `store`. Bypasses the
//     live gate because this is authoritative replay, not a live
//     stream.
//   * resetLiveStore — clear `store`, re-init the speaker
//     registry, construct a fresh `CompactGridRenderer` on the
//     transcript grid. Registered as the reconciler's
//     ``resetStore`` callback so it fires on every
//     enterLiveMeetingMode pass.
//   * showMeetingMode — DOM-only panel switcher: hide
//     landing/setup/view, show meeting + control bar. Registered
//     as the reconciler's ``showMeetingMode`` callback.

import { _enc } from "../lib/meeting-url.js";
import { formatTime } from "../lib/time-format.js";
import { state, store } from "../state.js";
import { CompactGridRenderer } from "./compact-grid.js";
import { resetSpeakerRegistry } from "./speaker-registry.js";

const API = "";

export async function validateMic() {
  const el = document.getElementById("status-line");
  el.textContent = "Checking mic...";
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext();
  if (ctx.state === "suspended") await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser();
  an.fftSize = 2048;
  src.connect(an);
  let max = 0;
  const buf = new Float32Array(an.fftSize);
  for (let i = 0; i < 10; i++) {
    await new Promise((r) => setTimeout(r, 50));
    an.getFloatTimeDomainData(buf);
    max = Math.max(
      max,
      buf.reduce((m, v) => Math.max(m, Math.abs(v)), 0),
    );
  }
  stream.getTracks().forEach((t) => t.stop());
  await ctx.close();
  console.log("Mic validation: max amplitude =", max);
  if (max === 0) {
    el.textContent =
      "No audio detected — check mic is selected and input volume is turned up";
    throw new Error("Mic silent (zero signal)");
  } else if (max < 0.0001) {
    el.textContent = `Mic too quiet (${(max * 100).toFixed(3)}%) — increase input volume in system settings`;
    throw new Error("Mic silent");
  }
  el.textContent = `Mic OK (${(max * 100).toFixed(1)}%)`;
  return true;
}

export function setStoreLive(enabled) {
  store._liveEnabled = !!enabled;
}

export async function loadMeetingJournal(meetingId) {
  const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}`);
  const data = await resp.json();
  if (data.meta?.created_at) {
    state.meetingStartWallMs = new Date(data.meta.created_at).getTime();
  }
  if (data.events) {
    // Direct store.ingest — bypasses the live-WS gate since this is
    // authoritative journal replay, not a live event stream.
    for (const ev of data.events) store.ingest(ev);
  }
  return data;
}

export function resetLiveStore() {
  store.clear();
  resetSpeakerRegistry();
  window._gridRenderer = new CompactGridRenderer(
    document.getElementById("transcript-grid"),
    null,
    formatTime,
  );
}

export function showMeetingMode() {
  document.getElementById("landing-mode").style.display = "none";
  document.getElementById("room-setup").style.display = "none";
  document.getElementById("view-mode").style.display = "none";
  document.getElementById("meeting-mode").style.display = "";
  document.getElementById("control-bar").style.display = "";
  document.body.classList.add("hide-table");
}

// Live-WS event ingestion shim. Wraps `store.ingest` with the
// browser-test bookkeeping hooks (__test_ingest_count / __test_msg_log)
// and the cross-window `scribe-ws-message` dispatch every popout +
// admin slide bar listens for. The live gate (`store._liveEnabled`)
// drops segment events while the reconciler is mid-rebuild — control-
// plane events (slide changes, etc.) still propagate via the window
// dispatch above the gate.
export function ingestFromLiveWs(event) {
  if (window.__test_msg_log) {
    window.__test_ingest_count++;
    window.__test_msg_log.push({
      seg: event?.segment_id || null,
      type: event?.type || null,
      text: (event?.text || "").slice(0, 30),
    });
  }
  try {
    window.dispatchEvent(new CustomEvent("scribe-ws-message", { detail: event }));
  } catch {}
  if (!store._liveEnabled) return;
  store.ingest(event);
}
