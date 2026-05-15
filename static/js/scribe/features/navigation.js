// Meeting Scribe — Landing-page navigation.
//
// Three view-switcher helpers. Each is a short DOM-only function —
// no shared module-scope state.
//
// `showLanding` deliberately doesn't reset the status line —
// `checkStatus` polls the backend every 2s and owns the text, so
// resetting here would create a 0–2s wrong-state flash on the
// landing page when the user clicks Home.

import { micWarmup } from "./mic-warmup.js";
import { setStoreLive } from "./live-session.js";

export function showLanding() {
  document.getElementById("landing-mode").style.display = "";
  document.getElementById("room-setup").style.display = "none";
  document.getElementById("meeting-mode").style.display = "none";
  document.getElementById("view-mode").style.display = "none";
  document.getElementById("speaker-timeline").style.display = "none";
  document.getElementById("meeting-summary-panel").style.display = "none";
  document.getElementById("player-bar").style.display = "none";
  // Release the warmed-up mic so the browser indicator turns off when the
  // user backs out of setup without starting a meeting.
  micWarmup.release();
  // Don't clobber the status line here — checkStatus() polls backend state
  // every 2s and owns the text so it stays consistent whether the page was
  // loaded at `/` or navigated to via a click on the Home button (`/#home`).
}

export function hideLanding() {
  document.getElementById("landing-mode").style.display = "none";
}

// Leave the live meeting view while the meeting is still running on
// the server. Suppresses live ingestion so landing / review / setup
// can't be corrupted by incoming WS events, but does NOT touch the
// AudioPipeline / audio.ws — the meeting keeps recording. The
// reconciler's return banner will drive the user back when they want
// to resume the live view; enterLiveMeetingMode re-enables ingestion
// and re-fetches the journal so the rebuild is clean.
export function exitLiveMeetingView() {
  setStoreLive(false);
  document.body.classList.remove("meeting-active");
  document.body.classList.add("off-meeting-view");
  document.getElementById("meeting-mode").style.display = "none";
}
