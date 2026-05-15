// Meeting Scribe — Language helpers.
//
// Pure functions used by every transcript-rendering feature. The
// "A vs B" split is the bilingual column convention: A is the
// operator-chosen primary language, B is the partner language (or
// null in monolingual meetings).
//
// All helpers read ``state.currentLanguagePair`` so a runtime
// language change is observed everywhere.

import { state } from "../state.js";

const CJK_CODES = new Set(["ja", "zh", "ko"]);

export function getLangA() {
  return state.currentLanguagePair.split(",")[0];
}

export function getLangB() {
  return state.currentLanguagePair.split(",")[1];
}

export function isMonolingual() {
  return !getLangB();
}

/**
 * Script-based language routing. ASR occasionally mislabels segments
 * (Japanese tagged "en", English tagged "ja"), which leaks text into
 * the wrong column. The script is authoritative:
 *   - Kana → ja if in pair
 *   - Hangul → ko if in pair
 *   - Other CJK (Han) → ja, zh, ko in order if in pair
 *   - otherwise → non-CJK lang in pair (or ASR label if non-CJK
 *     and in-pair; finally langA as fallback).
 */
export function routeLangByScript(text, asrLang, langA, langB) {
  const t = text || "";
  // Kanji / Hiragana / Katakana / CJK Compatibility + fullwidth punct
  const hasJaKana = /[぀-ゟ゠-ヿ]/.test(t);
  const hasCJK = /[㐀-鿿豈-﫿]/.test(t);
  const hasHangul = /[가-힯ᄀ-ᇿ]/.test(t);
  const pair = [langA, langB];
  const findInPair = (code) => (pair.includes(code) ? code : null);

  if (hasHangul) {
    const ko = findInPair("ko");
    if (ko) return ko;
  }
  if (hasJaKana) {
    const ja = findInPair("ja");
    if (ja) return ja;
  }
  if (hasCJK) {
    for (const c of ["ja", "zh", "ko"]) {
      const m = findInPair(c);
      if (m) return m;
    }
  }
  // No CJK at all → must NOT land in a CJK column.
  if (asrLang && pair.includes(asrLang) && !CJK_CODES.has(asrLang)) {
    return asrLang;
  }
  const nonCjk = pair.find((l) => !CJK_CODES.has(l));
  return nonCjk || langA;
}
