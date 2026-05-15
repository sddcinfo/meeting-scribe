// Meeting Scribe — meeting banner (pure module).
//
// The sticky banner at the top of the meeting view that says
// "Meeting in progress · Return" (when navigated away) or
// "Reconnecting…" (during transient WS drops). Visual layer; the
// reconciler drives state through `showBanner({mode, label, button,
// onClick})` and clears it with `showBanner(null)`.
//
// Styling lives in static/css/src/components/_meeting-banner.css.

export function ensureBannerEl() {
  let el = document.getElementById("meeting-banner");
  if (el) return el;
  el = document.createElement("div");
  el.id = "meeting-banner";
  el.className = "meeting-banner";
  el.setAttribute("role", "status");
  el.setAttribute("aria-live", "polite");
  el.innerHTML = `
    <span class="meeting-banner-dot" aria-hidden="true"></span>
    <span class="meeting-banner-label"></span>
    <button class="meeting-banner-btn" type="button"></button>
  `;
  document.body.appendChild(el);
  return el;
}

export function showBanner(state) {
  const el = ensureBannerEl();
  if (!state) {
    el.classList.remove("visible", "return", "reconnecting");
    const btn = el.querySelector(".meeting-banner-btn");
    if (btn) btn.onclick = null;
    return;
  }
  el.classList.add("visible");
  el.classList.toggle("return", state.mode === "return");
  el.classList.toggle("reconnecting", state.mode === "reconnecting");
  el.querySelector(".meeting-banner-label").textContent = state.label || "";
  const btn = el.querySelector(".meeting-banner-btn");
  if (state.button) {
    btn.textContent = state.button;
    btn.style.display = "";
    btn.setAttribute("aria-hidden", "false");
    btn.onclick = state.onClick || null;
  } else {
    btn.style.display = "none";
    btn.setAttribute("aria-hidden", "true");
    btn.onclick = null;
  }
}
