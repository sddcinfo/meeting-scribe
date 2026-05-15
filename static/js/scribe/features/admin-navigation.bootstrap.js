// Meeting Scribe — Admin SPA navigation bootstrap.
//
// Wires the three top-of-page navigation entry points (Home icon,
// landing-page Start, landing-page Quick-English) + the hash-based
// router (#home / #setup / #meeting/<id>) + the two
// MeetingsManager prototype patches that keep the URL hash in sync
// with viewMeeting / showSetup transitions.
//
// Pure DOM-level event wiring + a thin router function — no shared
// module state besides the `meetingsMgr` and `roomSetup` singletons,
// both of which arrive via the one-time configure() call below
// (lazy getters so the listener bodies always see the current
// binding).
//
// Skipped entirely under `?popout=view` since popout windows have
// their own navigation surface.

import { state } from "../state.js";
import { MeetingsManager } from "./meetings-manager.js";
import { updateColumnHeaders } from "./column-headers.js";
import {
  exitLiveMeetingView,
  hideLanding,
  showLanding,
} from "./navigation.js";

let _deps = null;

export function configureAdminNavigation(deps) {
  _deps = deps;
}

export function bootAdminNavigation() {
  // Home icon — always navigates to the landing page. Navigation is
  // always allowed even while recording. The "Meeting in progress"
  // banner is the return affordance; exitLiveMeetingView keeps the
  // live pipeline alive while the user is off the meeting view, and
  // detaches rendering so events don't corrupt other views.
  document.getElementById("btn-home")?.addEventListener("click", (e) => {
    e.preventDefault();
    if (document.body.classList.contains("recording")
     || document.body.classList.contains("meeting-active")) {
      exitLiveMeetingView();
    }
    history.pushState(null, "", "#home");
    showLanding();
  });

  document.getElementById("landing-start-btn")?.addEventListener("click", () => {
    hideLanding();
    history.pushState(null, "", "#setup");
    _deps?.getMeetingsMgr?.()?.showSetup();
  });

  // Quick-start English: skip the setup screen entirely and fire a
  // monolingual English meeting. Sets `state.currentLanguagePair`
  // to "en" before calling `roomSetup.startMeeting()` — that method
  // owns the UI transition + kicks off the actual recording
  // pipeline. The server's strict parser accepts the single code
  // and creates the meeting in monolingual mode.
  document.getElementById("landing-quick-english-btn")?.addEventListener("click", (e) => {
    const btn = e.currentTarget;
    if (btn.disabled) return;
    btn.disabled = true;
    const origText = btn.textContent;
    btn.textContent = "Starting…";
    try {
      state.currentLanguagePair = "en";
      updateColumnHeaders(state);

      hideLanding();
      history.pushState(null, "", "#meeting");
      _deps?.getRoomSetup?.()?.startMeeting();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = origText;
      throw err;
    }
  });

  // Hash-based routing: #home, #setup, #meeting/{id}. Navigation
  // is always honored — the live pipeline keeps streaming to the
  // server regardless of which panel is visible, and the "Meeting
  // in progress" banner drives the user back when needed.
  function handleHashRoute() {
    const hash = location.hash || "#home";

    if (hash === "#home" || hash === "#" || hash === "") {
      if (document.body.classList.contains("recording")
       || document.body.classList.contains("meeting-active")) {
        exitLiveMeetingView();
      }
      showLanding();
      return;
    }

    if (hash === "#setup") {
      if (document.body.classList.contains("recording")
       || document.body.classList.contains("meeting-active")) {
        exitLiveMeetingView();
      }
      hideLanding();
      _deps?.getMeetingsMgr?.()?.showSetup();
      return;
    }

    const match = hash.match(/^#meeting\/(.+)/);
    if (match && match[1]) {
      const meetingId = match[1];
      setTimeout(() => _deps?.getMeetingsMgr?.()?.viewMeeting(meetingId), 500);
    }
  }

  // MeetingsManager prototype patches: keep the URL hash in sync
  // with viewMeeting / showSetup transitions so back/forward + a
  // reload both restore the operator to where they were.
  const _origViewMeeting = MeetingsManager.prototype.viewMeeting;
  MeetingsManager.prototype.viewMeeting = async function (meetingId) {
    history.pushState(null, "", `#meeting/${meetingId}`);
    return _origViewMeeting.call(this, meetingId);
  };

  const _origShowSetup = MeetingsManager.prototype.showSetup;
  MeetingsManager.prototype.showSetup = function () {
    history.pushState(null, "", "#setup");
    return _origShowSetup.call(this);
  };

  // Initial hash on page load + back/forward.
  handleHashRoute();
  window.addEventListener("hashchange", handleHashRoute);
}

// No auto-boot — the admin SPA boot orchestrator calls
// `bootAdminNavigation()` itself after `configureAdminNavigation({...})`
// populates the lazy getters. ESM static-import hoisting means an
// auto-boot here would fire before the orchestrator's body runs,
// which is before deps exist. Explicit order avoids the race.
