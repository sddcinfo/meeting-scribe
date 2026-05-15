// Meeting Scribe — Audio routing popover open/close wiring.
//
// The popover (#audio-routing-popover) contains the TTS / device /
// mute controls; every input keeps its class + id so the event
// handlers in the audio-out feature and admin-audio-card.js drive
// state. This module owns ONLY the popover open/close + outside-click
// + Escape-to-close behaviour. No state, no model — purely
// presentational.

// The popover element is also #meeting-interpretation-controls (same
// node, two ids/classes worth of identity) so admin-audio-card.js can
// keep doing getElementById('meeting-interpretation-controls') without
// finding null when we lift the node out of the control bar.
const POP_ID = "meeting-interpretation-controls";
const TRIGGER_ID = "btn-audio-routing";

// Relocate the popover to <body> so position:fixed isn't trapped by
// any ancestor's transform / containment / overflow context inside
// .control-bar. The popover's DOM children (inputs/selects) keep
// their classes + ids, so the event handlers wired by class/id still
// find them after the move.
(function _liftPopoverToBody() {
  const pop = document.getElementById(POP_ID);
  if (pop && pop.parentElement !== document.body) {
    document.body.appendChild(pop);
  }
})();

function _pop() {
  return document.getElementById(POP_ID);
}

function _trigger() {
  return document.getElementById(TRIGGER_ID);
}

function _open() {
  const pop = _pop();
  const btn = _trigger();
  if (!pop || !btn) return;
  pop.hidden = false;
  btn.setAttribute("aria-expanded", "true");
  // Defer outside-click handler binding by one frame so the click
  // that opened the popover doesn't immediately close it.
  requestAnimationFrame(() => {
    document.addEventListener("click", _onOutsideClick, true);
  });
  document.addEventListener("keydown", _onKeyDown);
}

function _close() {
  const pop = _pop();
  const btn = _trigger();
  if (!pop) return;
  pop.hidden = true;
  if (btn) btn.setAttribute("aria-expanded", "false");
  document.removeEventListener("click", _onOutsideClick, true);
  document.removeEventListener("keydown", _onKeyDown);
}

function _onOutsideClick(e) {
  const pop = _pop();
  const btn = _trigger();
  if (!pop) return _close();
  // Click inside the popover or on its trigger → keep open.
  if (pop.contains(e.target)) return;
  if (btn && btn.contains(e.target)) return;
  _close();
}

function _onKeyDown(e) {
  if (e.key === "Escape") _close();
}

// ──────────────────────────────────────────────────────────────────
// Bootstrap
// ──────────────────────────────────────────────────────────────────

const trigger = _trigger();
if (trigger) {
  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    const pop = _pop();
    if (!pop) return;
    if (pop.hidden) _open();
    else _close();
  });
}

const closeBtn = document.getElementById("audio-pop-close");
if (closeBtn) {
  closeBtn.addEventListener("click", _close);
}
