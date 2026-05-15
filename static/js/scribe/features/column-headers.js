// Meeting Scribe — bilingual transcript column-header renderer.
//
// Drives the two `.col-header` cells above the transcript grid AND
// the `#btn-col-a` / `#btn-col-b` column-filter buttons in the
// control bar. Reads the active language pair from the shared
// `state` object plus the local lang helpers — no module-scope
// state of its own.

import { getLangA, getLangB } from "../lib/lang-helpers.js";

export function defaultLanguagePair(cache) {
  // Guarantee two distinct codes. If the server ever returned a mono
  // default (legacy config), we still promote it to bilingual here —
  // the setup screen never shows mono.
  const pair = (cache || ["en", "ja"]).slice(0, 2);
  if (pair.length < 2) pair.push(pair[0] === "en" ? "ja" : "en");
  if (pair[0] === pair[1]) pair[1] = pair[0] === "en" ? "ja" : "en";
  return pair;
}

export function updateColumnHeaders(state) {
  const langA = getLangA();
  const langB = getLangB();
  const langNames = state.languageNames;
  const nameA =
    langNames[langA]?.native_name ||
    langNames[langA]?.name ||
    langA.toUpperCase();
  const codeA = langA.toUpperCase();
  const el = (id) => document.getElementById(id);

  // Reflect monolingual on <body> so transcript-grid CSS collapses
  // the second column and the A/B toggle buttons hide.
  document.body.classList.toggle("monolingual", !langB);

  if (!langB) {
    if (el("col-lang-a-label")) el("col-lang-a-label").textContent = nameA;
    if (el("col-lang-a-hint"))
      el("col-lang-a-hint").textContent = `${codeA} original`;
    const btnA = el("btn-col-a");
    const btnB = el("btn-col-b");
    if (btnA) {
      btnA.textContent = langNames[langA]?.name || codeA;
      btnA.style.display = "";
    }
    if (btnB) btnB.style.display = "none";
    return;
  }

  const nameB =
    langNames[langB]?.native_name ||
    langNames[langB]?.name ||
    langB.toUpperCase();
  const codeB = langB.toUpperCase();

  if (el("col-lang-a-label")) el("col-lang-a-label").textContent = nameA;
  if (el("col-lang-a-hint"))
    el("col-lang-a-hint").textContent =
      `${codeA} original + ${codeB}→${codeA}`;
  if (el("col-lang-b-label")) el("col-lang-b-label").textContent = nameB;
  if (el("col-lang-b-hint"))
    el("col-lang-b-hint").textContent =
      `${codeB} original + ${codeA}→${codeB}`;

  const shortA = langNames[langA]?.name || codeA;
  const shortB = langNames[langB]?.name || codeB;
  const btnA = el("btn-col-a");
  const btnB = el("btn-col-b");
  if (btnA) {
    btnA.textContent = shortA;
    btnA.style.display = "";
  }
  if (btnB) {
    btnB.textContent = shortB;
    btnB.style.display = "";
  }
}
