// Meeting Scribe — 1:1 conversation mode (pure module).
//
// Two-pane side-by-side transcript for 1:1 meetings — first detected
// speaker takes the left pane, second takes the right, anything
// further falls back to the left. Hangs off the standard segment
// stream — `renderSegment(event)` is called by the segment-store
// subscriber bootstrap when the mode is active.
//
// State is module-private (active flag + left/right cluster ids +
// swap toggle). The bootstrap half wires the toggle + swap buttons.

import { esc } from "../lib/escape.js";
import { getLangA, getLangB } from "../lib/lang-helpers.js";
import { state } from "../state.js";
import { getSpeakerDisplayName } from "./speaker-registry.js";

const _speakers = { left: null, right: null }; // cluster_id → side
let _active = false;
let _swapped = false;

export function isActive() {
  return _active;
}

export function setActive(active) {
  _active = !!active;
  if (_active) {
    _speakers.left = null;
    _speakers.right = null;
    const leftEl = document.getElementById("oo-left");
    const rightEl = document.getElementById("oo-right");
    if (leftEl) leftEl.innerHTML = "";
    if (rightEl) rightEl.innerHTML = "";
  }
}

export function toggleSwap() {
  _swapped = !_swapped;
  const leftLabel = document.getElementById("oo-speaker-left");
  const rightLabel = document.getElementById("oo-speaker-right");
  if (leftLabel && rightLabel) {
    const tmp = leftLabel.textContent;
    leftLabel.textContent = rightLabel.textContent;
    rightLabel.textContent = tmp;
  }
  const tmpId = _speakers.left;
  _speakers.left = _speakers.right;
  _speakers.right = tmpId;
}

export function renderSegment(event) {
  if (!_active || !event.is_final || !event.text) return;

  const clusterId = event.speakers?.[0]?.cluster_id ?? 0;
  const speakerName = getSpeakerDisplayName(
    clusterId,
    event.speakers?.[0]?.identity || event.speakers?.[0]?.display_name,
  );

  // Assign speaker to side on first appearance.
  if (_speakers.left === null) {
    _speakers.left = clusterId;
    const el = document.getElementById("oo-speaker-left");
    if (el) el.textContent = speakerName;
  } else if (
    _speakers.right === null &&
    clusterId !== _speakers.left
  ) {
    _speakers.right = clusterId;
    const el = document.getElementById("oo-speaker-right");
    if (el) el.textContent = speakerName;
  }

  let paneId;
  if (clusterId === _speakers.left) paneId = "oo-left";
  else if (clusterId === _speakers.right) paneId = "oo-right";
  else paneId = "oo-left"; // Fallback for 3rd+ speakers

  const pane = document.getElementById(paneId);
  if (!pane) return;
  const tr = event.translation?.text || "";
  const langA = getLangA();
  const langB = getLangB();
  const cssClass =
    event.language === langA
      ? state.languageNames[langA]?.css_font_class || ""
      : state.languageNames[langB]?.css_font_class || "";
  const trCssClass =
    event.language === langA
      ? state.languageNames[langB]?.css_font_class || ""
      : state.languageNames[langA]?.css_font_class || "";
  // Prefer server-rendered ruby for both source and translation.
  const srcBody = event.furigana_html || esc(event.text);
  const trBody = event.translation?.furigana_html || (tr ? esc(tr) : "");
  const html = `<div class="oo-original ${cssClass}">${srcBody}</div>${tr ? `<div class="oo-translation ${trCssClass}">${trBody}</div>` : ""}`;

  // Dedup: server broadcasts the same segment_id multiple times.
  const safeId =
    window.CSS && CSS.escape ? CSS.escape(event.segment_id) : event.segment_id;
  const selector = `[data-segment-id="${safeId}"]`;
  const existing = pane.querySelector(selector);
  if (existing) {
    existing.innerHTML = html;
    return;
  }

  const block = document.createElement("div");
  block.className = "oo-block";
  block.dataset.segmentId = event.segment_id;
  block.innerHTML = html;
  pane.appendChild(block);
  pane.scrollTop = pane.scrollHeight;
}
