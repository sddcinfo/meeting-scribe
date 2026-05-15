// Meeting Scribe — Background-finalize toast (pure module).
//
// When SCRIBE_BACKGROUND_FINALIZE=1 the server returns 200 from Stop
// after Phase A (~asr_flush + translation_drain) and runs the heavy
// diarize / summary work in a background task. This module surfaces
// the progress as a corner toast — NOT the blocking modal — so the
// operator can immediately start a new meeting while the previous
// one's finalize completes.
//
// Multiple concurrent toasts (rare; only on multi-meeting recovery)
// stack vertically. Each toast is keyed by meeting_id; updates with
// `terminal:false` advance the progress bar in place; updates with
// `terminal:true` flip to a "Done" state with a View button; updates
// with `error:true` flip to red with a Reprocess hint.
//
// External coupling kept minimal: the only dep is
// `meetingsMgr.refresh()` (called on terminal-success and on
// reprocess), wired via `setMeetingsMgrRef` after the manager is
// constructed. All other state lives here.

// _finalizingMeetings: meetingId → { step, label, ws }
// Track meetings currently being finalized so we can reopen the modal
// when navigating back to them.
export const _finalizingMeetings = new Map();

// _bgFinalizeToasts: meeting_id → HTMLElement
export const _bgFinalizeToasts = new Map();

// _bgFinalizeTrackers: meeting_id → {step, total, label, terminal, error, code, message, eta_deadline_ms}
// Latest progress per meeting_id, keyed for the meetings-list cell
// renderer. Survives across renders so the row's progress bar reflects
// the most recent step even after MeetingsManager.refresh() rebuilds.
export const _bgFinalizeTrackers = new Map();

// MeetingsManager hook — the admin SPA boot calls setMeetingsMgrRef
// once the manager instance is constructed. Used on terminal-success
// and from the Reprocess button to repaint the meetings list.
let _meetingsMgrRef = null;
export function setMeetingsMgrRef(mgr) {
  _meetingsMgrRef = mgr;
}

// ── Finalize ETA: shared formatter + live countdown ticker ──────────
//
// The server emits `eta_seconds` once per progress step and never refines
// it locally, so a static "~120s remaining" sat there until the *next*
// step arrived (which could be minutes). We turn each ETA into a wall-
// clock deadline stored on the target element, and a single 1s ticker
// rewrites every `[data-finalize-eta-deadline]` element in the document
// so the user sees the number tick down in real time. Format is m:ss
// once the remainder hits a minute or more.
export function _formatRemaining(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s remaining`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")} remaining`;
}

let _finalizeEtaTickerId = null;
function _tickFinalizeEtas() {
  const nodes = document.querySelectorAll("[data-finalize-eta-deadline]");
  if (nodes.length === 0) {
    if (_finalizeEtaTickerId) {
      clearInterval(_finalizeEtaTickerId);
      _finalizeEtaTickerId = null;
    }
    return;
  }
  const now = Date.now();
  for (const n of nodes) {
    const deadline = Number(n.dataset.finalizeEtaDeadline);
    if (!Number.isFinite(deadline)) continue;
    const remaining = (deadline - now) / 1000;
    if (remaining <= 0) {
      // Don't keep ticking past zero — leave a clear "any moment now"
      // hint instead of a confusing negative count.
      n.textContent = "finishing up…";
    } else {
      n.textContent = _formatRemaining(remaining);
    }
  }
}

export function _setFinalizeEtaDeadline(el, deadlineMs) {
  if (!el) return;
  if (!Number.isFinite(deadlineMs) || deadlineMs <= Date.now()) {
    delete el.dataset.finalizeEtaDeadline;
    el.textContent = "";
    return;
  }
  el.dataset.finalizeEtaDeadline = String(deadlineMs);
  el.textContent = _formatRemaining((deadlineMs - Date.now()) / 1000);
  if (!_finalizeEtaTickerId) {
    _finalizeEtaTickerId = setInterval(_tickFinalizeEtas, 1000);
  }
}

// Convenience for callers that have eta_seconds from a *just-received*
// message — deadline = now + eta. Use the deadline variant directly when
// the tracker already holds an absolute deadline (e.g. banner re-render
// after navigating into the detail view).
export function _setFinalizeEta(el, etaSeconds) {
  if (!Number.isFinite(etaSeconds) || etaSeconds <= 0) {
    _clearFinalizeEta(el);
    return;
  }
  _setFinalizeEtaDeadline(el, Date.now() + etaSeconds * 1000);
}

export function _clearFinalizeEta(el) {
  if (!el) return;
  delete el.dataset.finalizeEtaDeadline;
  el.textContent = "";
}

function _ensureBgFinalizeStack() {
  let stack = document.getElementById("bg-finalize-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.id = "bg-finalize-stack";
    stack.className = "finalize-toast-stack";
    document.body.appendChild(stack);
  }
  return stack;
}

/**
 * Update the meetings-list row's `.meeting-finalizing-progress` cell
 * in place from the latest tracker for that mid. No-op if the row
 * isn't currently rendered (meetings panel is closed / filtered).
 */
export function _patchMeetingsListProgressRow(mid) {
  const tracker = _bgFinalizeTrackers.get(mid);
  if (!tracker) return;
  const row = document.querySelector(
    `.meeting-finalizing-progress[data-meeting-id="${CSS.escape(mid)}"]`,
  );
  if (!row) return;
  const labelEl = row.querySelector(".meeting-finalizing-label");
  const fillEl = row.querySelector(".meeting-finalizing-fill");
  const stepEl = row.querySelector(".meeting-finalizing-step");
  if (labelEl) {
    labelEl.textContent = tracker.label || "Finalizing...";
    labelEl.title = tracker.label || "";
  }
  if (fillEl && tracker.total > 0) {
    fillEl.style.width = `${Math.min(100, Math.round((tracker.step / tracker.total) * 100))}%`;
  }
  if (stepEl) {
    stepEl.textContent = `${tracker.step}/${tracker.total}`;
  }
}

/**
 * POST to the meeting's reprocess endpoint. Wired to the Reprocess
 * button on both the corner toast and the meeting-detail banner. The
 * server clears the Phase B failure sidecar on success so the next
 * /api/meetings hydration leaves the toast/banner gone.
 */
export function _bgFinalizeReprocess(mid) {
  return fetch(`/api/meetings/${encodeURIComponent(mid)}/reprocess`, {
    method: "POST",
    credentials: "include",
  }).then((resp) => {
    if (!resp.ok) throw new Error(`reprocess HTTP ${resp.status}`);
    if (_meetingsMgrRef) _meetingsMgrRef.refresh();
    return resp.json();
  });
}

/**
 * POST to the meeting's finalize/dismiss endpoint. Drops a stuck
 * failure sidecar without retrying. The server broadcasts a synthetic
 * terminal event so the toast and banner clear immediately.
 */
export function _bgFinalizeDismiss(mid) {
  return fetch(`/api/meetings/${encodeURIComponent(mid)}/finalize/dismiss`, {
    method: "POST",
    credentials: "include",
  }).then((resp) => {
    if (!resp.ok) throw new Error(`dismiss HTTP ${resp.status}`);
    return resp.json();
  });
}

/**
 * Convert a server-side ``phase_b_progress`` sidecar (the shape
 * returned by /api/meetings + /api/meetings/{id}) into the WS-event
 * shape that ``_renderBackgroundFinalizeToast`` expects. Lets the
 * landing-page hydration path reuse the toast renderer without
 * duplicating its display logic.
 */
export function _sidecarToToastMsg(mid, sidecar) {
  if (!sidecar) return null;
  const isFailure = !!sidecar.terminal && sidecar.error && typeof sidecar.error === "object";
  return {
    type: "background_finalize_progress",
    meeting_id: mid,
    step: Number(sidecar.step) || 0,
    total_steps: Number(sidecar.total_steps) || 7,
    label: sidecar.label || "",
    terminal: !!sidecar.terminal,
    error: isFailure,
    code: isFailure ? sidecar.error.code : undefined,
    message: isFailure ? sidecar.error.message : undefined,
    eta_seconds: Number.isFinite(Number(sidecar.eta_seconds))
      ? Number(sidecar.eta_seconds)
      : undefined,
  };
}

export function _renderBackgroundFinalizeToast(msg) {
  if (!msg || !msg.meeting_id) return;
  const mid = msg.meeting_id;
  // Update the durable tracker first so the meetings-list cell is in
  // sync even if the toast is dismissed or the user is on the
  // meetings panel rather than the live view.
  // Convert the server's snapshot ETA into an absolute wall-clock
  // deadline at the moment we received it. The tracker survives across
  // banner/toast re-renders (e.g. operator navigates into the meeting
  // detail page later), so anchoring to a deadline means re-renders
  // pick up the correct remaining time instead of restarting the
  // countdown from the original duration.
  const _etaSec =
    typeof msg.eta_seconds === "number" && msg.eta_seconds > 0 ? msg.eta_seconds : null;
  _bgFinalizeTrackers.set(mid, {
    step: Number(msg.step) || 0,
    total: Number(msg.total_steps) || 7,
    label: msg.label || "",
    terminal: !!msg.terminal,
    error: !!msg.error,
    code: msg.code,
    message: msg.message,
    eta_deadline_ms: _etaSec != null ? Date.now() + _etaSec * 1000 : null,
  });
  // Patch any in-DOM meetings-list row for this mid in place. Avoids a
  // full meetingsMgr.refresh() (network call) on every progress event
  // — the row's HTML lives at .meeting-finalizing-progress[data-meeting-id].
  _patchMeetingsListProgressRow(mid);
  // Also patch the inline meeting-detail banner if the operator is on
  // the detail view for this meeting right now.
  _patchMeetingDetailFinalizeBanner(mid);
  // On the terminal event the meeting transitions to COMPLETE on the
  // server; refresh the list once so the row drops back to a normal
  // "complete" state with a View button + summary. Skip for terminal-
  // error states since the meeting stays in INTERRUPTED until the
  // operator reprocesses or dismisses.
  if (msg.terminal && !msg.error && _meetingsMgrRef) {
    _meetingsMgrRef.refresh();
  }
  const stack = _ensureBgFinalizeStack();
  let toast = _bgFinalizeToasts.get(mid);
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "finalize-toast";
    toast.dataset.meetingId = mid;
    toast.innerHTML = `
      <div class="finalize-toast-title">Finalizing previous meeting</div>
      <div class="finalize-toast-step">
        <span class="finalize-toast-step-label"></span>
        <span class="finalize-toast-step-eta"></span>
      </div>
      <div class="finalize-toast-bar"><div class="finalize-toast-fill"></div></div>
      <div class="finalize-toast-actions"></div>
    `;
    stack.appendChild(toast);
    _bgFinalizeToasts.set(mid, toast);
  }

  const titleEl = toast.querySelector(".finalize-toast-title");
  const stepLabelEl = toast.querySelector(".finalize-toast-step-label");
  const stepEtaEl = toast.querySelector(".finalize-toast-step-eta");
  const fillEl = toast.querySelector(".finalize-toast-fill");
  const actionsEl = toast.querySelector(".finalize-toast-actions");

  const step = Number(msg.step) || 0;
  const total = Number(msg.total_steps) || 7;
  if (msg.error) {
    toast.classList.add("error");
    toast.classList.remove("done");
    if (titleEl) {
      titleEl.textContent =
        msg.code === "interrupted" ? "Finalize interrupted" : "Finalize failed";
    }
    if (stepLabelEl) {
      stepLabelEl.textContent = msg.message || msg.label || "Reprocess to recover";
    }
    _clearFinalizeEta(stepEtaEl);
    if (fillEl) fillEl.style.width = "100%";
    if (actionsEl) {
      actionsEl.innerHTML = "";
      const reprocessBtn = document.createElement("button");
      reprocessBtn.className = "toast-action toast-action-reprocess";
      reprocessBtn.textContent = "Reprocess";
      reprocessBtn.title = `meeting-scribe finalize retry ${mid}`;
      reprocessBtn.addEventListener("click", () => {
        reprocessBtn.disabled = true;
        _bgFinalizeReprocess(mid).catch((err) => {
          reprocessBtn.disabled = false;
          console.error("reprocess failed", err);
        });
      });
      const dismissBtn = document.createElement("button");
      dismissBtn.className = "toast-action toast-action-dismiss";
      dismissBtn.textContent = "Dismiss";
      dismissBtn.addEventListener("click", () => {
        dismissBtn.disabled = true;
        _bgFinalizeDismiss(mid).catch((err) => {
          dismissBtn.disabled = false;
          console.error("dismiss failed", err);
        });
      });
      actionsEl.appendChild(reprocessBtn);
      actionsEl.appendChild(dismissBtn);
    }
    return;
  }

  if (msg.terminal) {
    toast.classList.add("done");
    toast.classList.remove("error");
    if (titleEl) titleEl.textContent = "Meeting saved";
    if (stepLabelEl) stepLabelEl.textContent = "Done";
    _clearFinalizeEta(stepEtaEl);
    if (fillEl) fillEl.style.width = "100%";
    if (actionsEl && !actionsEl.querySelector(".toast-action-view")) {
      actionsEl.innerHTML = "";
      const btn = document.createElement("button");
      btn.className = "toast-action toast-action-view";
      btn.textContent = "View";
      btn.addEventListener("click", () => {
        location.hash = `#meeting/${mid}`;
        toast.remove();
        _bgFinalizeToasts.delete(mid);
      });
      actionsEl.appendChild(btn);
    }
    return;
  }

  if (titleEl) titleEl.textContent = "Finalizing previous meeting";
  if (stepLabelEl) {
    const label = msg.label || "Finalizing...";
    stepLabelEl.textContent = `(${step}/${total}) ${label}`;
  }
  if (typeof msg.eta_seconds === "number" && msg.eta_seconds > 0) {
    _setFinalizeEta(stepEtaEl, msg.eta_seconds);
  } else {
    _clearFinalizeEta(stepEtaEl);
  }
  if (fillEl && total > 0) {
    fillEl.style.width = `${Math.min(100, Math.round((step / total) * 100))}%`;
  }
}

/**
 * Inline finalize banner above the transcript on the meeting-detail
 * view. Reflects the same Phase B state as the corner toast but is
 * scoped to whichever meeting the operator is currently viewing.
 *
 * Rendered into ``#meeting-detail-finalize-banner`` when the detail
 * view paints (see ``viewMeeting``); each WS progress event for the
 * matching meeting_id then patches it in place here.
 */
export function _patchMeetingDetailFinalizeBanner(mid) {
  const banner = document.getElementById("meeting-detail-finalize-banner");
  if (!banner) return;
  if (banner.dataset.meetingId !== mid) return;
  const tracker = _bgFinalizeTrackers.get(mid);
  if (!tracker) {
    banner.style.display = "none";
    return;
  }
  banner.style.display = "";
  banner.classList.toggle("is-error", !!tracker.error);
  banner.classList.toggle("is-done", !!tracker.terminal && !tracker.error);
  const labelEl = banner.querySelector(".meeting-detail-finalize-label");
  const etaEl = banner.querySelector(".meeting-detail-finalize-eta");
  const fillEl = banner.querySelector(".meeting-detail-finalize-fill");
  const stepEl = banner.querySelector(".meeting-detail-finalize-step");
  const actionsEl = banner.querySelector(".meeting-detail-finalize-actions");
  const step = Number(tracker.step) || 0;
  const total = Number(tracker.total) || 7;
  if (tracker.error) {
    if (labelEl) {
      labelEl.textContent =
        tracker.code === "interrupted"
          ? "Finalize was interrupted — Reprocess to retry"
          : `Finalize failed: ${tracker.message || tracker.label || "unknown"}`;
    }
    _clearFinalizeEta(etaEl);
    if (fillEl) fillEl.style.width = "100%";
    if (stepEl) stepEl.textContent = `${step}/${total}`;
    if (actionsEl) {
      actionsEl.innerHTML = "";
      const reprocessBtn = document.createElement("button");
      reprocessBtn.className = "btn-ghost meeting-detail-finalize-reprocess";
      reprocessBtn.textContent = "Reprocess";
      reprocessBtn.addEventListener("click", () => {
        reprocessBtn.disabled = true;
        _bgFinalizeReprocess(mid).catch(() => {
          reprocessBtn.disabled = false;
        });
      });
      const dismissBtn = document.createElement("button");
      dismissBtn.className = "btn-ghost meeting-detail-finalize-dismiss";
      dismissBtn.textContent = "Dismiss";
      dismissBtn.addEventListener("click", () => {
        dismissBtn.disabled = true;
        _bgFinalizeDismiss(mid).catch(() => {
          dismissBtn.disabled = false;
        });
      });
      actionsEl.appendChild(reprocessBtn);
      actionsEl.appendChild(dismissBtn);
    }
    return;
  }
  if (tracker.terminal && !tracker.error) {
    if (labelEl) labelEl.textContent = "✓ Finalize complete";
    _clearFinalizeEta(etaEl);
    if (fillEl) fillEl.style.width = "100%";
    if (stepEl) stepEl.textContent = "";
    if (actionsEl) actionsEl.innerHTML = "";
    // Fade after 3s; the detail view re-fetches to paint the now-saved
    // summary + speakers.
    setTimeout(() => {
      banner.style.display = "none";
    }, 3000);
    return;
  }
  if (labelEl) labelEl.textContent = `Finalizing — ${tracker.label || ""}`;
  if (Number.isFinite(tracker.eta_deadline_ms)) {
    _setFinalizeEtaDeadline(etaEl, tracker.eta_deadline_ms);
  } else {
    _clearFinalizeEta(etaEl);
  }
  if (stepEl) stepEl.textContent = `${step}/${total}`;
  if (fillEl && total > 0) {
    fillEl.style.width = `${Math.min(100, Math.round((step / total) * 100))}%`;
  }
}
