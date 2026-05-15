// Meeting Scribe — Hamburger meetings panel toggle (pure).
//
// The slide-over list of meetings opens via the hamburger button or
// any of the four ``toggleMeetingsPanel()`` callsites inside
// MeetingsManager.

export function toggleMeetingsPanel() {
  const panel = document.getElementById("meetings-panel");
  const backdrop = document.getElementById("meetings-backdrop");
  if (!panel || !backdrop) return false;
  const open = panel.classList.toggle("open");
  backdrop.classList.toggle("open", open);
  return open;
}

export function isMeetingsPanelOpen() {
  const panel = document.getElementById("meetings-panel");
  return !!panel && panel.classList.contains("open");
}
