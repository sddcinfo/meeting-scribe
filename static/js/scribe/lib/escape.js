// Meeting Scribe — HTML-escape helper.
//
// Tiny utility consumed by every feature module that builds DOM via
// innerHTML interpolation (refreshWifiQR, modal system, transcript
// rendering, slide viewer, etc.).

export function esc(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}
