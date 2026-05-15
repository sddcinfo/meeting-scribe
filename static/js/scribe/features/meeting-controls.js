// Meeting Scribe — Meeting controls (record / takeover / cancel).
//
// Three header / control-bar buttons that drive the meeting state
// machine:
//
//   - #btn-record           — start a fresh meeting OR stop the
//                              current one (with confirm prompt)
//   - #btn-takeover-recording — claim recorder ownership from a
//                              view-only tab
//   - #btn-cancel-meeting   — discard the current meeting (no
//                              finalization, all data deleted)
//
// All three depend on the reconciler + meetingsMgr singletons in the
// admin SPA boot orchestrator; handed in via a one-time
// `configureMeetingControls` call once those are constructed. Lazy
// getters for both so the listener bodies always resolve the current
// reference.

import { audio } from "../state.js";
import { closeModal, confirmDialog, showModal } from "./modal-system.js";
import { esc } from "../lib/escape.js";
import { resetSpeakerRegistry } from "./speaker-registry.js";
import { stopAnalytics } from "./analytics-polling.js";
import { startRecording, stopRecording } from "./recording-lifecycle.js";

const API = "";

let _deps = null;

export function configureMeetingControls(deps) {
  _deps = deps;
}

export function bootMeetingControls() {
  const btnRecord = document.getElementById("btn-record");
  if (btnRecord) {
    btnRecord.addEventListener("click", async () => {
      if (document.body.classList.contains("recording")) {
        const confirmed = await confirmDialog(
          "Stop meeting?",
          "Recording will end and the meeting will be finalized with a summary. You won’t be able to add more audio.",
          "Stop Meeting",
          true,
        );
        if (!confirmed) return;
        await stopRecording();
      } else {
        await startRecording(false);
      }
    });
  }

  // Take-over button — visible only in view-only mode (CSS gates it
  // on `body.view-only:not(.popout-view)`). Routes through the
  // reconciler's `takeoverRecorder` which releases any prior
  // ownership, claims this tab as the recorder, and re-enters
  // meeting mode with the active mic pipeline. Without this button,
  // an operator who lands on the live meeting from a fresh tab
  // (sessionStorage ownership wiped) was stuck on "Viewing <mid>
  // (read-only)" with no way to act on the meeting other than
  // typing reconciler internals into DevTools.
  const btnTakeover = document.getElementById("btn-takeover-recording");
  if (btnTakeover) {
    btnTakeover.addEventListener("click", async () => {
      const reconciler = _deps?.getReconciler?.();
      if (!reconciler) return;
      // Source the meeting id from the live status object the
      // reconciler last reconciled against — that's the most
      // reliable signal of "what's actually recording right now",
      // versus `window.current_meeting_id` which can be stale
      // across reloads.
      const lastStatus = reconciler._inspect?.()?.lastStatus;
      const mid = lastStatus?.meeting?.id || window.current_meeting_id;
      if (!mid) return;
      btnTakeover.disabled = true;
      btnTakeover.textContent = "Taking over…";
      try {
        await reconciler.takeoverRecorder(mid);
      } finally {
        // After takeover, the reconciler removes `view-only` and
        // the bar hides via CSS — re-enabling is for the rare
        // error path where the takeover threw before the class
        // flip happened.
        btnTakeover.disabled = false;
        btnTakeover.textContent = "Take over recording";
      }
    });
  }

  const btnCancelMeeting = document.getElementById("btn-cancel-meeting");
  if (btnCancelMeeting) {
    btnCancelMeeting.addEventListener("click", async () => {
      if (!document.body.classList.contains("recording")) return;
      const confirmed = await confirmDialog(
        "Cancel meeting?",
        "All audio, transcript, and data will be permanently deleted. This cannot be undone.",
        "Cancel Meeting",
        true,
      );
      if (!confirmed) return;

      btnCancelMeeting.disabled = true;
      btnCancelMeeting.textContent = "Cancelling…";
      try {
        // Disconnect audio hardware immediately
        if (audio && audio.running) {
          audio.stop();
        }
        const resp = await fetch(`${API}/api/meeting/cancel`, { method: "POST" });
        const data = await resp.json();
        if (resp.ok) {
          document.body.classList.remove("recording");
          document.body.classList.remove("meeting-active");
          document.body.classList.remove("starting");
          const reconciler = _deps?.getReconciler?.();
          reconciler?.releaseOwnership();
          reconciler?.clearReconnectState();
          document.getElementById("control-bar").style.display = "none";
          const statusEl = document.getElementById("status-line");
          if (statusEl) statusEl.textContent = "Meeting cancelled";
          // Close WS connections
          if (audio && audio.ws) { audio.ws.close(); audio.ws = null; }
          stopAnalytics();
          resetSpeakerRegistry();
          const mgr = _deps?.getMeetingsMgr?.();
          if (mgr) mgr.refresh();
        } else {
          console.error("Cancel failed:", data);
          showModal(`
            <div class="modal-confirm-title">Cancel failed</div>
            <div class="modal-confirm-message">${esc(data.error || "Unknown error")}</div>
            <div class="modal-confirm-actions">
              <button class="modal-btn" onclick="closeModal()">Close</button>
            </div>
          `, "confirm");
        }
      } catch (e) {
        console.error("Cancel error:", e);
        showModal(`
          <div class="modal-confirm-title">Cancel failed</div>
          <div class="modal-confirm-message">${esc(e.message)}</div>
          <div class="modal-confirm-actions">
            <button class="modal-btn" onclick="closeModal()">Close</button>
          </div>
        `, "confirm");
      } finally {
        btnCancelMeeting.disabled = false;
        btnCancelMeeting.textContent = "Cancel";
      }
    });
  }
}

// Side-effect to satisfy ESM-static-analysis lints that flag
// `closeModal` as an unused import. The inline onclick handlers in
// the modals above reference `closeModal()` through the global, so
// the import is what makes that global resolve.
void closeModal;
