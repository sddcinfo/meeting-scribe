// Meeting Scribe — Room Editor bootstrap.
//
// Wires the three DOM listeners + the `window._onRoomLayoutUpdate`
// publish. Runs after the admin SPA boot orchestrator calls
// `configureRoomEditor(...)` to inject `roomSetup` + the
// speaker-strip refresh hook, so those deps are populated by the
// time these handlers register.

import {
  applyRoomLayoutUpdate,
  closeRoomEditor,
  openRoomEditor,
} from "./room-editor.js";

document.getElementById("room-editor-close")?.addEventListener("click", closeRoomEditor);

document.addEventListener("keydown", (e) => {
  if (
    e.key === "Escape" &&
    document.getElementById("room-editor-overlay")?.style.display === "flex"
  ) {
    closeRoomEditor();
  }
});

document.getElementById("meeting-table-strip")?.addEventListener("click", () => {
  if (!document.body.classList.contains("meeting-active")) return;
  // Don't open for past-meeting viewing — only for the active live meeting.
  if (document.body.classList.contains("view-only")) return;
  const mid = window.current_meeting_id || null;
  if (mid) openRoomEditor("live", mid);
});

window._onRoomLayoutUpdate = applyRoomLayoutUpdate;
