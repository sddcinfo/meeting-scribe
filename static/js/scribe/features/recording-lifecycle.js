// Meeting Scribe — Recording lifecycle.
//
// `resumeMeeting`, `startRecording`, `stopRecording`, and
// `showFinalizationSummaryFor` drive every transition through the
// meeting state machine (resume from rebound mic → fresh start →
// user-stop → summary review). They're co-located so they share a
// single deps bag.
//
// The admin SPA boot orchestrator calls
// `configureRecordingLifecycle({...})` once to hand over the
// reconciler / meetingsMgr / roomSetup singletons + the refresh
// helpers; the module reads them via the `_deps` indirection.
// Late-bound singletons come in as getters so the lookups always
// resolve current.

import { audio, state, store, timer } from "../state.js";
import { closeModal, showModal } from "./modal-system.js";
import { esc } from "../lib/escape.js";
import { _enc } from "../lib/meeting-url.js";
import { ingestFromLiveWs, validateMic } from "./live-session.js";
import { resetSpeakerRegistry } from "./speaker-registry.js";
import { CompactGridRenderer } from "./compact-grid.js";
import { formatTime } from "../lib/time-format.js";
import { setMetricsVisible } from "./metrics-dashboard.js";
import { startAnalytics, stopAnalytics } from "./analytics-polling.js";
import {
  _clearFinalizeEta,
  _finalizingMeetings,
  _renderBackgroundFinalizeToast,
  _setFinalizeEta,
} from "./bg-finalize-toast.js";

const API = "";

let _deps = null;

export function configureRecordingLifecycle(deps) {
  _deps = deps;
}

export async function resumeMeeting(meetingId) {
  const statusEl = document.getElementById("status-line");
  const btnResume = document.getElementById("btn-resume");
  try {
    statusEl.textContent = `Resuming ${meetingId}...`;
    if (btnResume) { btnResume.disabled = true; btnResume.textContent = "Resuming..."; }

    const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/resume`, { method: "POST" });
    const data = await resp.json();
    if (!data.resumed) {
      statusEl.textContent = data.error || "Resume failed";
      if (btnResume) { btnResume.disabled = false; btnResume.textContent = "Resume"; }
      return;
    }

    state.currentLanguagePair = (data.language_pair || ["en", "ja"]).join(",");
    _deps.updateColumnHeaders();
    document.getElementById("landing-mode").style.display = "none";
    document.getElementById("room-setup").style.display = "none";
    document.getElementById("view-mode").style.display = "none";
    document.getElementById("meeting-mode").style.display = "";
    document.getElementById("control-bar").style.display = "";
    document.body.classList.add("meeting-active");

    window._gridRenderer = new CompactGridRenderer(document.getElementById("transcript-grid"), meetingId, formatTime);
    statusEl.textContent = `Loading transcript...`;
    const meetResp = await fetch(`${API}/api/meetings/${_enc(meetingId)}`);
    const meetData = await meetResp.json();
    if (meetData.events) {
      for (const ev of meetData.events) store.ingest(ev);
    }
    statusEl.textContent = `Starting mic...`;

    _deps.getRoomSetup()._renderTableStrip();

    await startRecording(true);
    statusEl.textContent = `Recording — resumed ${meetingId}`;
  } catch (err) {
    console.error("Resume failed:", err);
    statusEl.textContent = `Resume failed: ${err.message}`;
    if (btnResume) { btnResume.disabled = false; btnResume.textContent = "Resume"; }
  }
}

export async function startRecording(isResume) {
  try {
    if (!isResume) {
      await validateMic();
      // Fresh meeting — clear every UI surface that can carry data from
      // a previous meeting. The transcript store is already cleared
      // below after the start API succeeds; the rest of these surfaces
      // are not naturally rebuilt by the start handshake (no event
      // re-emission, no re-render hook), so a stale slide thumbnail or
      // a leftover speaker chip would otherwise stick around until the
      // first event of the new meeting overwrote it.
      resetSpeakerRegistry();
      window._adminSlideResetUI?.();
      const meetingTableStrip = document.getElementById("meeting-table-strip");
      if (meetingTableStrip) meetingTableStrip.innerHTML = "";
      const meetingSummaryPanel = document.getElementById("meeting-summary-panel");
      if (meetingSummaryPanel) {
        meetingSummaryPanel.style.display = "none";
        meetingSummaryPanel.innerHTML = "";
      }
      const speakerTimeline = document.getElementById("speaker-timeline");
      if (speakerTimeline) speakerTimeline.style.display = "none";
      const transcriptGrid = document.getElementById("transcript-grid");
      if (transcriptGrid) transcriptGrid.innerHTML = "";
      // Drop any reconnect/return banner left over from a prior session.
      // The reconciler's status loop will re-raise it if the server
      // disagrees with the new local state.
      _deps.getReconciler()?.clearReconnectState?.();
      const meetingBanner = document.getElementById("meeting-banner");
      if (meetingBanner) meetingBanner.classList.remove("visible");
      // Wipe layout overlays from the previous meeting so a fresh start
      // never inherits show-only-a/b, hide-live, hide-table, compact, or
      // metrics-split.
      document.body.classList.remove(
        "show-only-a",
        "show-only-b",
        "hide-live",
        "hide-table",
        "view-only",
      );
      setMetricsVisible(false);
      // 1:1 mode is gated by a module-level flag rather than a body
      // class. Stash the prior state on the toggle button itself so
      // existing wiring (which inspects the DOM, not the module flag)
      // doesn't have to change.
      const oneOnOneContainer = document.getElementById("one-on-one-container");
      if (oneOnOneContainer && oneOnOneContainer.style.display !== "none") {
        document.getElementById("btn-one-on-one")?.click();
      }
    }
    // Atomic start payload (W4): the setup wizard hands its draft to
    // the start endpoint instead of PUTting field-by-field. This
    // eliminates the race where rapid clicks could leave the start
    // request firing before the per-field PUTs settled.
    const startPayload = { language_pair: state.currentLanguagePair };
    try {
      const draft = window._adminAudioCard?.getSetupDraft?.();
      if (draft && Object.keys(draft).length > 0) {
        startPayload.audio_config = draft;
      }
    } catch {}

    const resp = await fetch(`${API}/api/meeting/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(startPayload),
    });
    // Read as text first so a non-JSON 500 ("Internal Server Error") gives
    // us a useful error instead of "Unexpected token I in JSON".
    const rawBody = await resp.text();
    let data;
    try {
      data = rawBody ? JSON.parse(rawBody) : {};
    } catch {
      throw new Error(
        `Server returned ${resp.status} ${resp.statusText}: ${rawBody || "(empty body)"}`,
      );
    }

    // Atomic audio_config rejection: the operator explicitly chose
    // routing in the wizard and it could not be applied. Refuse to
    // start so they can fix the device before recording.
    if (resp.status === 503 && data.error === "audio_routing_failed") {
      const micMsg = data.mic && !data.mic.ok ? data.mic.error_message : "";
      const sinkMsg = data.sink && !data.sink.ok ? data.sink.error_message : "";
      const detail = [micMsg, sinkMsg].filter(Boolean).join(" · ");
      const banner = `
        <div class="backends-not-ready-header">
          <div class="backends-not-ready-icon">!</div>
          <h2>Couldn't apply audio routing</h2>
        </div>
        <div class="backends-not-ready-body">
          <p class="backends-not-ready-message">${esc(data.message || "Audio routing failed")}</p>
          <div class="backends-not-ready-list">
            ${micMsg ? `<div class="backends-not-ready-list-item"><span class="backend-name">Mic</span><span class="backend-detail">${esc(micMsg)}</span></div>` : ""}
            ${sinkMsg ? `<div class="backends-not-ready-list-item"><span class="backend-name">Sink</span><span class="backend-detail">${esc(sinkMsg)}</span></div>` : ""}
          </div>
          <p class="backends-not-ready-hint">Check the audio routing fields and click Start again.</p>
        </div>`;
      const statusLine = document.getElementById("status-line");
      if (statusLine) statusLine.textContent = `Couldn't apply audio routing (${detail})`;
      // Reuse the backends-not-ready modal infrastructure below.
      data._renderAsAudioRoutingFailure = banner;
    }

    // Server refused because backends aren't deep-healthy
    if (resp.status === 503 && !data._renderAsAudioRoutingFailure) {
      const notReady = (data.not_ready || []).map((x) => {
        return x.detail ? `${x.backend}: ${x.detail}` : x.backend;
      }).join(" · ");
      const msg = data.message || "Backends not ready";
      document.getElementById("status-line").textContent = `${msg} (${notReady})`;
      const items = (data.not_ready || []).map((x) => `
        <div class="backends-not-ready-list-item">
          <span class="backend-name">${esc(x.backend)}</span>
          <span class="backend-detail">${esc(x.detail || "not ready")}</span>
        </div>
      `).join("");
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
      showModal(banner, "backends-not-ready");
      throw new Error("Backends not ready");
    }

    if (resp.status === 503 && data._renderAsAudioRoutingFailure) {
      showModal(
        data._renderAsAudioRoutingFailure +
          '<div class="backends-not-ready-footer"><button class="modal-btn primary" onclick="closeModal()">OK</button></div>',
        "backends-not-ready",
      );
      throw new Error("audio_routing_failed");
    }

    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    if (data.language_pair) {
      state.currentLanguagePair = data.language_pair.join(",");
      _deps.updateColumnHeaders();
    }
    if (!isResume && !data.resumed) store.clear();
    state.wsMessageCount = 0; state.audioChunkCount = 0;
    document.body.classList.add("recording");
    document.getElementById("status-line").textContent = `Recording ${data.meeting_id}`;
    // Cold-load fresh-start path doesn't go through the reconciler's
    // ``resetStore`` step (that fires on resume / observer-tab entry), so
    // ``_gridRenderer`` may be null on the very first record click after
    // a page load. Without this construct-or-update, live WS events
    // ingest into ``store`` with no surface to render them on.
    if (window._gridRenderer) {
      window._gridRenderer._meetingId = data.meeting_id;
    } else {
      window._gridRenderer = new CompactGridRenderer(
        document.getElementById("transcript-grid"),
        data.meeting_id,
        formatTime,
      );
    }
    // Null-guard: `window.audioPlayer` is stamped by
    // features/audio-player.bootstrap.js. If a regression in module
    // load order (or a constructor throw during AudioPlayer init — eg
    // #player-play missing in the DOM) leaves it undefined,
    // recording-lifecycle still has to make forward progress on the
    // rest of the start sequence. Without the guard, the next line
    // throws ``Cannot set properties of undefined``, the catch
    // handler at the bottom logs "Start failed:" and rolls back to
    // room-setup — leaving the operator staring at "Initializing..."
    // with a meaningless error in the console.
    if (window.audioPlayer) {
      window.audioPlayer.meetingId = data.meeting_id;
    } else {
      console.warn("audioPlayer global missing at start — past-meeting review/playback won't have a meeting bound");
    }
    window.current_meeting_id = data.meeting_id;
    state.meetingStartWallMs = Date.now();
    timer.start();
    await audio.start(
      (event) => ingestFromLiveWs(event),
      {
        onSpeakerRegistryChanged: () => {
          _deps.refreshDetectedSpeakersStrip();
          _deps.refreshTranscriptSpeakerLabels();
        },
        onMeetingCancelled: () => {
          store.clear();
        },
      },
    );
    startAnalytics(); window.refreshWifiQR();
  } catch (err) {
    console.error("Start failed:", err);
    document.getElementById("status-line").textContent = `Error: ${err.message}`;
    // Roll back to room-setup so the user can retry. RoomSetup.startMeeting
    // owns btnStart restore via its own finally; we just clean the body
    // classes and unhide room-setup here.
    document.body.classList.remove("meeting-active");
    document.body.classList.remove("starting");
    _deps.getReconciler()?.releaseOwnership();
    document.getElementById("meeting-mode").style.display = "none";
    document.getElementById("control-bar").style.display = "none";
    document.getElementById("room-setup").style.display = "";
    throw err;
  }
}

export async function stopRecording() {
  timer.stop();
  document.getElementById("meter-bar").style.width = "0%";
  audio.workletNode?.disconnect(); audio.stream?.getTracks().forEach((t) => t.stop());
  if (audio.audioCtx) await audio.audioCtx.close();
  audio.audioCtx = null; audio.stream = null; audio.workletNode = null; audio.analyser = null;

  // Stop audio-out listener if active
  if (window.audioOutListener.enabled) {
    window.audioOutListener.stop();
    const listenBtn = document.getElementById("btn-listen");
    if (listenBtn) listenBtn.classList.remove("active");
    const listenLang = document.getElementById("listen-lang");
    if (listenLang) listenLang.style.display = "none";
  }

  // Get meeting ID before stopping
  const statusResp = await fetch(`${API}/api/status`);
  const statusData = await statusResp.json();
  const meetingId = statusData.meeting?.id;

  // Track this meeting as finalizing
  if (meetingId) _finalizingMeetings.set(meetingId, { step: 0, label: "Starting...", ws: audio.ws });

  // Show finalization modal with progress steps — matches the 6 steps
  // emitted by /api/meeting/stop on the server (which is now identical
  // to the /finalize pipeline including full-audio diarization).
  const steps = [
    { label: "Flushing speech recognition", icon: "mic" },
    { label: "Completing translations", icon: "translate" },
    { label: "Saving speaker data", icon: "waveform" },
    { label: "Running full-audio diarization", icon: "speakers" },
    { label: "Generating timeline", icon: "timeline" },
    { label: "Generating meeting summary", icon: "summary" },
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
              ${i < steps.length - 1 ? '<div class="step-connector"></div>' : ""}
            </div>
            <span class="step-label">${s.label}</span>
          </div>
        `).join("")}
      </div>
      <div class="finalize-eta" id="finalize-eta"></div>
      <div class="finalize-summary" id="finalize-summary" style="display:none"></div>
    </div>
  `, "finalize");

  let receivedFinalStep = false;

  const closeAndReload = () => {
    closeModal();
    if (meetingId) {
      setTimeout(() => _deps.getMeetingsMgr()?.viewMeeting(meetingId), 300);
    }
  };
  card.querySelector("#finalize-close-btn")?.addEventListener("click", closeAndReload);

  const _updateProgress = (step, { allDone = false } = {}) => {
    const pct = allDone ? 100 : Math.min(100, (step / 6) * 100);
    const fill = card.querySelector("#finalize-progress-fill");
    if (fill) fill.style.width = `${pct}%`;

    card.querySelectorAll(".finalize-step").forEach((el) => {
      const s = parseInt(el.dataset.step);
      el.classList.toggle("done", allDone ? true : s < step);
      el.classList.toggle("active", allDone ? false : s === step);
    });
  };

  if (audio.ws) {
    audio.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "background_finalize_progress") {
          _renderBackgroundFinalizeToast(msg);
          return;
        }
        if (msg.type === "finalize_progress") {
          // The server sends TWO step=6 messages: the first marks the START
          // of summary generation (no summary field yet), the second carries
          // the generated summary and signals real completion. Treat only
          // the summary-bearing message as the terminal event — otherwise
          // we close the WS before the summary arrives and the user sees
          // "Generating meeting summary" spinning forever.
          const isCompletion = msg.step >= 6 && (msg.summary !== undefined || msg.meeting_id);

          _updateProgress(msg.step, { allDone: isCompletion });

          if (meetingId) {
            const tracker = _finalizingMeetings.get(meetingId);
            if (tracker) { tracker.step = msg.step; tracker.label = msg.label || ""; }
          }

          const subtitle = card.querySelector("#finalize-subtitle");
          if (subtitle && msg.label) {
            subtitle.textContent = msg.label;
          }

          const etaEl = card.querySelector("#finalize-eta");
          if (etaEl) {
            if (msg.eta_seconds > 0) {
              _setFinalizeEta(etaEl, msg.eta_seconds);
              etaEl.classList.add("visible");
            } else {
              _clearFinalizeEta(etaEl);
              etaEl.classList.remove("visible");
            }
          }

          if (isCompletion) {
            receivedFinalStep = true;
            const pulse = card.querySelector(".finalize-pulse");
            if (pulse) pulse.classList.add("complete");
            card.querySelector(".finalize-modal")?.classList.add("done");
            if (subtitle) subtitle.textContent = "✓ Meeting finalized";
            const etaElDone = card.querySelector("#finalize-eta");
            if (etaElDone) {
              _clearFinalizeEta(etaElDone);
              etaElDone.classList.remove("visible");
            }

            if (msg.summary) {
              _deps.renderFinalizationSummary(msg.summary, meetingId);
            }
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
    stopAnalytics(); window.refreshWifiQR(); document.body.classList.remove("recording");
    document.body.classList.remove("meeting-active");
    document.body.classList.remove("starting");
    const reconciler = _deps.getReconciler();
    reconciler?.releaseOwnership();
    reconciler?.clearReconnectState();
    resetSpeakerRegistry();
    document.getElementById("control-bar").style.display = "none";
    document.getElementById("status-line").textContent = meetingId
      ? `Meeting complete — ${meetingId}`
      : "Stopped";
    if (meetingId) _finalizingMeetings.delete(meetingId);
    _deps.getMeetingsMgr()?.refresh();
  };

  // Fire stop — WS handler manages the progress; cleanup happens on step 6
  document.getElementById("status-line").textContent = "Finalizing...";
  fetch(`${API}/api/meeting/stop`, { method: "POST" }).then(() => {
    // HTTP response arrived — if WS already delivered step 6, nothing to do.
    // If WS died or never sent step 6, clean up after a short grace period.
    if (!receivedFinalStep) {
      setTimeout(() => {
        if (!receivedFinalStep) {
          _updateProgress(6, { allDone: true });
          const subtitle = card.querySelector("#finalize-subtitle");
          if (subtitle) subtitle.textContent = "Meeting finalized";
          const pulse = card.querySelector(".finalize-pulse");
          if (pulse) pulse.classList.add("complete");
          _finalizeCleanup();
          fetch(`${API}/api/meetings/${_enc(meetingId)}/summary`).then((r) => r.json()).then((summary) => {
            if (summary && !summary.error) _deps.renderFinalizationSummary(summary, meetingId);
          }).catch(() => {});
        }
      }, 2000);
    }
  }).catch(() => {
    document.getElementById("status-line").textContent = "Stop failed";
    const subtitle = card.querySelector("#finalize-subtitle");
    if (subtitle) { subtitle.textContent = "Finalization failed"; subtitle.classList.add("error"); }
  });
}

/**
 * Re-show the finalization summary modal for any completed meeting.
 * Useful for reviewing past meetings or regenerating summaries.
 */
export async function showFinalizationSummaryFor(meetingId) {
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
  `, "finalize");
  card.querySelector("#finalize-close-btn")?.addEventListener("click", closeModal);

  const subtitle = card.querySelector("#finalize-subtitle");
  const summaryEl = card.querySelector("#finalize-summary");
  const langToggle = card.querySelector("#finalize-lang-toggle");

  let _meetingPair = null;
  try {
    const mr = await fetch(`${API}/api/meetings/${_enc(meetingId)}`);
    if (mr.ok) {
      const md = await mr.json();
      const lp = md?.meta?.language_pair;
      if (Array.isArray(lp) && lp.length === 2) _meetingPair = lp;
    }
  } catch {}

  let _activeLang = null;
  function _renderLangToggle() {
    if (!langToggle || !_meetingPair) return;
    langToggle.style.display = "";
    langToggle.innerHTML = "";
    const opts = [{ code: null, label: "Default" }, ..._meetingPair.map((c) => ({ code: c, label: (state.languageNames[c]?.name || c.toUpperCase()) }))];
    for (const opt of opts) {
      const btn = document.createElement("button");
      btn.className = "popout-btn popout-lang-btn";
      btn.textContent = opt.label;
      btn.style.cssText = "border:none;border-right:1px solid var(--border);border-radius:0;padding:2px 10px;font-size:0.7rem;background:none;cursor:pointer";
      if ((_activeLang || null) === (opt.code || null)) {
        btn.style.background = "var(--text-primary)";
        btn.style.color = "var(--bg-surface)";
        btn.style.fontWeight = "600";
      }
      btn.addEventListener("click", () => _loadSummary(opt.code));
      langToggle.appendChild(btn);
    }
  }

  async function _loadSummary(lang) {
    _activeLang = lang;
    _renderLangToggle();
    if (subtitle) subtitle.textContent = lang ? `Translating to ${(state.languageNames[lang]?.name || lang.toUpperCase())}…` : "Loading…";
    summaryEl.innerHTML = '<p style="padding:2rem;text-align:center;color:var(--text-secondary)">Working…</p>';
    const url = `${API}/api/meetings/${_enc(meetingId)}/summary${lang ? `?lang=${encodeURIComponent(lang)}` : ""}`;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      const summary = await r.json();
      if (!summary || summary.error) throw new Error(summary?.error || "No summary");
      if (subtitle) subtitle.textContent = lang ? `Translated · ${(state.languageNames[lang]?.name || lang.toUpperCase())}` : "Meeting finalized";
      summaryEl.innerHTML = "";
      _deps.renderFinalizationSummary(summary, meetingId);
    } catch (e) {
      summaryEl.innerHTML = `<p class="finalize-error">${esc(String(e.message || e))}</p>`;
    }
  }

  try {
    const resp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/summary`);
    if (resp.ok) {
      const summary = await resp.json();
      if (summary && !summary.error) {
        if (subtitle) subtitle.textContent = "Meeting finalized";
        _renderLangToggle();
        _deps.renderFinalizationSummary(summary, meetingId);
        return;
      }
    }
    if (subtitle) subtitle.textContent = "No summary available";
    summaryEl.style.display = "";
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
    card.querySelector("#finalize-regenerate-btn")?.addEventListener("click", async () => {
      if (subtitle) subtitle.textContent = "Generating summary...";
      summaryEl.innerHTML = '<p style="padding:2rem;text-align:center;color:var(--text-secondary)">Working...</p>';
      try {
        const finalizeResp = await fetch(
          `${API}/api/meetings/${_enc(meetingId)}/finalize?force=true`,
          { method: "POST" },
        );
        if (finalizeResp.ok) {
          const sumResp = await fetch(`${API}/api/meetings/${_enc(meetingId)}/summary`);
          if (sumResp.ok) {
            const summary = await sumResp.json();
            if (summary && !summary.error) {
              if (subtitle) subtitle.textContent = "Meeting finalized";
              _deps.renderFinalizationSummary(summary, meetingId);
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
    if (subtitle) subtitle.textContent = "Failed to load summary";
    summaryEl.style.display = "";
    summaryEl.innerHTML = `<p class="finalize-error">${esc(String(e))}</p>`;
  }
}
