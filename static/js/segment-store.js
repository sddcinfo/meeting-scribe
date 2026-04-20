// Per-meeting segment store. Receives transcript events from the
// meeting-scribe websocket and converges them into a single Map keyed by
// segment_id, where each entry is the **union** of every dimension the
// server has told us about that segment.
//
// Why union and not "highest revision wins":
//
// The server pushes updates for the same segment_id from THREE independent
// background loops, none of which coordinate revision numbers:
//
//   1. ASR finals          — bumps revision as text is refined
//   2. Furigana annotation — bumps revision (server.py: _annotate_and_broadcast)
//   3. Speaker catch-up    — bumps revision (server.py: _speaker_catchup_loop)
//   4. Translation worker  — does NOT bump revision; rebroadcasts at the
//                            same revision as the original event via
//                            `event.with_translation()` (models.py:92)
//
// A naive "drop if existing.rev > new.rev" gate therefore silently drops
// every translation that races a faster furigana / speaker update. On
// Japanese segments furigana ALWAYS wins (sync pykakasi, ~5 ms vs. ~300 ms
// vLLM round-trip), so every Japanese translation gets dropped.
//
// We instead split the event fields into two groups:
//
//   - **Primary** (text, is_final, revision, timing, language) — highest
//     revision wins, replaces the existing values.
//   - **Additive** (translation, furigana_html, speakers) — the latest
//     non-empty value lands regardless of revision and is preserved across
//     subsequent updates that don't carry it.
//
// That converges to "everything the server has ever told us about this
// segment", in any arrival order.

export class SegmentStore {
  constructor() {
    this.segments = new Map();
    this.order = [];
    this._listeners = new Set();
  }

  subscribe(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  ingest(event) {
    const { segment_id, revision } = event;
    const existing = this.segments.get(segment_id);

    let merged = existing ? { ...existing } : null;
    let changed = false;

    // Additive: translation. Under multi-target fan-out the server
    // sends one event per (segment_id, target_lang), each carrying a
    // single `translation` state. We accumulate them into a
    // `translations` map keyed by target_language, and keep the flat
    // `translation` slot populated with the most-complete copy so
    // legacy read sites (scribe-app render paths) still work without
    // a display-language picker.
    //
    // Lower-revision arrivals are still allowed to land their
    // translation onto a higher-revision existing record — furigana
    // or speaker catch-up may have bumped the revision first.
    const newTr = event.translation;
    const oldTr = existing?.translation;
    if (newTr) {
      const newHasText = !!newTr.text;
      const oldHasText = !!oldTr?.text;
      const statusChanged = newTr.status && newTr.status !== oldTr?.status;

      const newLang = newTr.target_language || '';
      const existingMap = existing?.translations || {};
      merged = merged || {};
      merged.translations = { ...existingMap };
      if (newLang) {
        const priorSameLang = existingMap[newLang];
        const priorHasText = !!priorSameLang?.text;
        if (newHasText || !priorHasText) {
          merged.translations[newLang] = newTr;
          changed = true;
        }
      }

      if (newHasText || (!oldHasText && statusChanged) || !oldTr) {
        merged.translation = newTr;
        changed = true;
      }
    }

    // Additive: furigana on the source text.
    if (event.furigana_html && event.furigana_html !== existing?.furigana_html) {
      merged = merged || {};
      merged.furigana_html = event.furigana_html;
      changed = true;
    }

    // Additive: furigana on the translated text.
    if (
      event.translation?.furigana_html
      && event.translation.furigana_html !== existing?.translation?.furigana_html
    ) {
      merged = merged || {};
      merged.translation = {
        ...(merged.translation || existing?.translation || {}),
        furigana_html: event.translation.furigana_html,
      };
      changed = true;
    }

    // Additive: speakers. Latest non-empty wins.
    if ((event.speakers?.length || 0) > (existing?.speakers?.length || 0)) {
      merged = merged || {};
      merged.speakers = event.speakers;
      changed = true;
    }

    // Primary: text + is_final + revision. Higher revision wins and
    // replaces the text/final/revision/timing fields. Same-revision arrivals
    // can still flip is_final from false → true.
    const isHigherRev = !existing || revision > existing.revision;
    const isSameRevNewFinal =
      existing && revision === existing.revision && !existing.is_final && event.is_final;
    if (isHigherRev || isSameRevNewFinal) {
      merged = merged || { ...event };
      merged.text = event.text;
      merged.is_final = event.is_final;
      merged.revision = Math.max(revision, existing?.revision || 0);
      merged.start_ms = event.start_ms;
      merged.end_ms = event.end_ms;
      merged.language = event.language;
      merged.segment_id = event.segment_id;
      changed = true;
    }

    if (!changed) return;
    if (!merged) return;

    // Carry over additive dimensions that the new event omitted (e.g. a
    // higher-rev update that didn't include the previously-attached
    // translation/furigana/speakers).
    if (existing) {
      if (!merged.translation && existing.translation) merged.translation = existing.translation;
      if (!merged.translations && existing.translations) merged.translations = existing.translations;
      if (!merged.furigana_html && existing.furigana_html) merged.furigana_html = existing.furigana_html;
      if ((!merged.speakers || merged.speakers.length === 0) && existing.speakers?.length) {
        merged.speakers = existing.speakers;
      }
    }

    const isNew = !existing;
    this.segments.set(segment_id, merged);
    if (isNew) this.order.push(segment_id);
    for (const fn of this._listeners) fn(segment_id, merged, isNew);
  }

  clear() {
    this.segments.clear();
    this.order = [];
    for (const fn of this._listeners) fn(null, null, false);
  }

  get count() {
    return this.segments.size;
  }
}
