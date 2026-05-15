// Meeting Scribe — Hamburger meetings panel (boot).
//
// Wires the hamburger button + the backdrop click → close handler.
// Runs after the admin SPA boot so MeetingsManager has had its chance
// to construct the slide-over content before the user can trigger
// the open.

import { toggleMeetingsPanel } from "./meetings-panel.js";

document.getElementById("btn-hamburger")?.addEventListener("click", toggleMeetingsPanel);
document.getElementById("meetings-backdrop")?.addEventListener("click", toggleMeetingsPanel);
