// Meeting Scribe — Header stats popover (boot).
//
// Toggle behaviour for #btn-stats and the click-outside dismiss on
// the wrapping #header-stats. Pure DOM — no shared module state.

document.getElementById("btn-stats")?.addEventListener("click", (e) => {
  e.stopPropagation();
  const popover = document.getElementById("stats-popover");
  const btn = document.getElementById("btn-stats");
  if (!popover || !btn) return;
  const open = popover.hasAttribute("hidden");
  popover.toggleAttribute("hidden", !open);
  btn.setAttribute("aria-expanded", open ? "true" : "false");
});

document.addEventListener("click", (e) => {
  const root = document.getElementById("header-stats");
  const popover = document.getElementById("stats-popover");
  const btn = document.getElementById("btn-stats");
  if (!root || !popover || popover.hasAttribute("hidden")) return;
  if (root.contains(e.target)) return;
  popover.setAttribute("hidden", "");
  if (btn) btn.setAttribute("aria-expanded", "false");
});
