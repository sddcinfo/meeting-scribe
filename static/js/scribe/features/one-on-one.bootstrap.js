// Meeting Scribe — 1:1 conversation mode (boot).
//
// Wires the #btn-one-on-one toggle and the #oo-swap swap button.
// Runs after the admin SPA boot so the segment-store subscriber wired
// in admin-misc.bootstrap.js can call ``renderSegment`` for live
// events.

import { isActive, renderSegment, setActive, toggleSwap } from "./one-on-one.js";

document.getElementById("btn-one-on-one")?.addEventListener("click", (e) => {
  const next = !isActive();
  setActive(next);
  e.target.classList.toggle("active-toggle", next);
  const container = document.getElementById("one-on-one-container");
  if (container) container.style.display = next ? "" : "none";
  const headers = document.querySelector(".transcript-col-headers");
  if (headers) headers.style.display = next ? "none" : "";
  const grid = document.getElementById("transcript-grid");
  if (grid) grid.style.display = next ? "none" : "";
});

document.getElementById("oo-swap")?.addEventListener("click", toggleSwap);

// Expose for the segment-store subscriber bridge in
// admin-misc.bootstrap.js (kept on window so the bridge resolves them
// regardless of evaluation order).
window._oneOnOneRender = renderSegment;
window._oneOnOneActive = isActive;
