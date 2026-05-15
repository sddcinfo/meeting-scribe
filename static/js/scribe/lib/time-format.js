// Meeting-clock time formatter — pure helper shared by every transcript
// surface (compact grid, speaker timeline, speaker modal, segment-list
// rows, popout view). Reads `state.meetingStartWallMs` so the same
// `formatTime(offsetMs)` call yields wall-clock (HH:MM:SS in the browser
// timezone) once recording has started, and elapsed (mm:ss) before that.
//
// Single-backed: the wall-clock start lives in `state.meetingStartWallMs`,
// written by the recording lifecycle module (start-recording handler), the
// view-only attach path, and viewMeeting on completed-meeting load. This
// module is read-only.

import { state } from "../state.js";

export function formatTime(offsetMs) {
  if (state.meetingStartWallMs > 0) {
    const wallMs = state.meetingStartWallMs + (offsetMs || 0);
    const d = new Date(wallMs);
    const h = String(d.getHours()).padStart(2, "0");
    const m = String(d.getMinutes()).padStart(2, "0");
    const s = String(d.getSeconds()).padStart(2, "0");
    return `${h}:${m}:${s}`;
  }
  const total = Math.floor((offsetMs || 0) / 1000);
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}
