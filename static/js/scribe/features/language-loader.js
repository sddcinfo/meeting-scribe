// Meeting Scribe — language registry loader.
//
// Hydrates the two Language A/B dropdowns on the setup screen from
// /api/languages, with a hardcoded in-process fallback so the
// dropdowns are never blank even if the server is down or returns a
// stale shape. Also caches the server's default pair so the
// MeetingsManager can ask "what pair should new sessions default to?"
// at any time after boot.
//
// Why labels show "English — native" instead of just the native name:
// before this rewrite users routinely misread "Deutsch ↔ English" as
// "Dutch ↔ English" and ended up in a German meeting when they meant
// Dutch. Showing the English name first makes the false cognate
// impossible.
//
// External surface:
//   bootLanguageLoader()    — fires the IIFE that wires the
//                              selA/selB dropdowns + change listeners
//   getDefaultLanguagePair() — what MeetingsManager calls into for the
//                              "next meeting" default
//
// Dependency surface (named imports):
//   state.js          — `state` singleton (languageNames + currentLanguagePair)
//   column-headers.js — `defaultLanguagePair` + `updateColumnHeaders`

import { state } from "../state.js";
import {
  defaultLanguagePair as _defaultLanguagePairRaw,
  updateColumnHeaders as _updateColumnHeadersRaw,
} from "./column-headers.js";

const API = "";

// Server's default pair, cached so MeetingsManager can ask later
// without re-fetching. Seeded with the en↔ja default that ships in
// the Python registry; updated to whatever the server returns once
// /api/languages responds.
let _defaultLanguagePairCache = ["en", "ja"];

export function getDefaultLanguagePair() {
  return _defaultLanguagePairRaw(_defaultLanguagePairCache);
}

// Hard-coded fallback so the dropdowns never render empty even if the API
// is down or returns a stale shape. Kept in sync with the Python registry.
const _LANG_FALLBACK = [
  { code: "en", name: "English", native_name: "English", tts_supported: true },
  { code: "zh", name: "Chinese", native_name: "中文", tts_supported: true },
  { code: "ja", name: "Japanese", native_name: "日本語", tts_supported: true },
  { code: "ko", name: "Korean", native_name: "한국어", tts_supported: true },
  { code: "fr", name: "French", native_name: "Français", tts_supported: true },
  { code: "de", name: "German", native_name: "Deutsch", tts_supported: true },
  { code: "es", name: "Spanish", native_name: "Español", tts_supported: true },
  { code: "it", name: "Italian", native_name: "Italiano", tts_supported: true },
  { code: "pt", name: "Portuguese", native_name: "Português", tts_supported: true },
  { code: "ru", name: "Russian", native_name: "Русский", tts_supported: true },
  { code: "nl", name: "Dutch", native_name: "Nederlands", tts_supported: false },
  { code: "ar", name: "Arabic", native_name: "العربية", tts_supported: false },
  { code: "th", name: "Thai", native_name: "ไทย", tts_supported: false },
  { code: "vi", name: "Vietnamese", native_name: "Tiếng Việt", tts_supported: false },
  { code: "id", name: "Indonesian", native_name: "Bahasa Indonesia", tts_supported: false },
  { code: "ms", name: "Malay", native_name: "Bahasa Melayu", tts_supported: false },
  { code: "hi", name: "Hindi", native_name: "हिन्दी", tts_supported: false },
  { code: "tr", name: "Turkish", native_name: "Türkçe", tts_supported: false },
  { code: "pl", name: "Polish", native_name: "Polski", tts_supported: false },
  { code: "uk", name: "Ukrainian", native_name: "Українська", tts_supported: false },
];

function _updateColumnHeaders() {
  return _updateColumnHeadersRaw(state);
}

export function bootLanguageLoader() {
  (async function loadLanguages() {
    const selA = document.getElementById("lang-a-select");
    const selB = document.getElementById("lang-b-select");

    const buildOption = (lang) => {
      const opt = document.createElement("option");
      opt.value = lang.code;
      // Show English name first so "Deutsch" can't be misread as "Dutch".
      let label = lang.native_name && lang.native_name !== lang.name
        ? `${lang.name} — ${lang.native_name}`
        : lang.name;
      if (lang.tts_supported === false) label += " (text only)";
      opt.textContent = label;
      return opt;
    };

    const populate = (select, langs, selectedCode) => {
      select.innerHTML = "";
      for (const lang of langs) select.appendChild(buildOption(lang));
      if (selectedCode && [...select.options].some((o) => o.value === selectedCode)) {
        select.value = selectedCode;
      }
    };

    // The right dropdown disables whichever language is currently on the
    // left, so the user can never pick the same language on both sides.
    // There is no __none__ sentinel on the setup screen — mono lives on
    // the landing page's quick-start path.
    const syncDisabled = () => {
      if (!selA || !selB) return;
      for (const opt of selA.options) opt.disabled = false;
      for (const opt of selB.options) {
        opt.disabled = (opt.value === selA.value);
      }
    };

    // If A and B collide, swap B to the next available language (NOT mono).
    const pickDifferent = (avoid, langs) => {
      return langs.find((l) => l.code !== avoid)?.code || avoid;
    };

    let listenersAttached = false;
    let _langsCache = [];
    const wireUp = (langs, defaultPair) => {
      if (!selA || !selB || langs.length < 2) return;
      _langsCache = langs;
      for (const lang of langs) state.languageNames[lang.code] = lang;

      // Promote the server default to bilingual for the setup screen. The
      // setup screen is always bilingual; mono-only defaults still get
      // paired here so the dropdowns are usable.
      const [defaultA, defaultB] = defaultPair.length >= 2
        ? defaultPair
        : [defaultPair[0], pickDifferent(defaultPair[0], langs)];
      _defaultLanguagePairCache = [defaultA, defaultB];

      // state.currentLanguagePair may be "en" (monolingual — user is returning
      // from a quick-start session) or "ja,en" (bilingual). Either way, the
      // setup screen renders a bilingual pair.
      const curParts = (state.currentLanguagePair || `${defaultA},${defaultB}`).split(",");
      const has = (code) => langs.some((l) => l.code === code);
      const pickA = has(curParts[0]) ? curParts[0] : defaultA;
      let pickB;
      if (curParts.length >= 2 && has(curParts[1]) && curParts[1] !== pickA) {
        pickB = curParts[1];
      } else if (has(defaultB) && defaultB !== pickA) {
        pickB = defaultB;
      } else {
        pickB = pickDifferent(pickA, langs);
      }

      populate(selA, langs, pickA);
      populate(selB, langs, pickB);
      state.currentLanguagePair = `${selA.value},${selB.value}`;
      syncDisabled();
      // Setup screen is always bilingual — make sure the mono CSS never
      // sticks around from a previous session.
      const selector = document.getElementById("language-selector");
      if (selector) selector.classList.remove("mono");

      if (listenersAttached) return;
      listenersAttached = true;
      const onChangeA = () => {
        // Collision with B → swap B to a different language (NOT mono).
        if (selA.value === selB.value) {
          selB.value = pickDifferent(selA.value, _langsCache);
        }
        state.currentLanguagePair = `${selA.value},${selB.value}`;
        syncDisabled();
        _updateColumnHeaders();
      };
      const onChangeB = () => {
        // Collision with A (shouldn't happen due to disabled options) →
        // swap B to a different language.
        if (selB.value === selA.value) {
          selB.value = pickDifferent(selA.value, _langsCache);
        }
        state.currentLanguagePair = `${selA.value},${selB.value}`;
        syncDisabled();
        _updateColumnHeaders();
      };
      selA.addEventListener("change", onChangeA);
      selB.addEventListener("change", onChangeB);
    };

    // Render the fallback synchronously so the dropdowns are never blank, even
    // for a split second while the fetch is in flight or if the server is down.
    wireUp(_LANG_FALLBACK, ["en", "ja"]);

    try {
      const resp = await fetch(`${API}/api/languages`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const langs = (data.languages && data.languages.length >= 2) ? data.languages : _LANG_FALLBACK;
      const defaultPair = (
        Array.isArray(data.default_pair)
        && data.default_pair.length >= 1
        && data.default_pair.length <= 2
      ) ? data.default_pair : ["en", "ja"];
      wireUp(langs, defaultPair);
    } catch (e) {
      console.warn("Failed to load /api/languages, using built-in fallback list:", e);
    }
    _updateColumnHeaders();
  })();
}
