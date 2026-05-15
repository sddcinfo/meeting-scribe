// Meeting Scribe — Compact grid renderer (pure module).
//
// The sole active transcript renderer for the admin SPA. Groups
// consecutive segments by speaker into single blocks; each block
// shows the speaker name once and accumulates text. New block on:
// speaker change, >5 s gap, language direction change, > 45 s of
// accumulated audio, or > 12 merged segments (readability cap).
//
// Window-contract surface: ``window._gridRenderer`` (instance).
// Constructed + published by the bootstrap module so cross-feature
// consumers (audio-player, popout receiver, audio-pipeline
// message-handler reset path) see the same instance.

import { esc } from "../lib/escape.js";
import { getLangA, getLangB, routeLangByScript } from "../lib/lang-helpers.js";
import { getSpeakerColor } from "../lib/speaker-palette.js";
import { state } from "../state.js";
import { SEEK_LEADIN_MS } from "./audio-player.js";
import { getSpeakerDisplayName } from "./speaker-registry.js";

export class CompactGridRenderer {
  /**
   * @param {HTMLElement} gridEl — container.
   * @param {string|null} meetingId — meeting uuid (null for live before
   *   the first start) — set later via setMeetingId().
   * @param {(offsetMs: number) => string} formatTime — wall-clock
   *   formatter. Caller owns this because the meeting-start wall
   *   clock is tracked in the recording lifecycle module.
   */
  constructor(gridEl, meetingId, formatTime) {
    this.gridEl = gridEl;
    this._meetingId = meetingId || null;
    this._formatTime = formatTime || ((ms) => `${Math.floor(ms / 1000)}s`);
    this._autoScroll = true;
    this._newestFirst = true; // newest at top (reverse scroll)
    this._currentBlock = null;
    this._segmentMap = new Map(); // segmentId → block reference
    this.gridEl.addEventListener("scroll", () => {
      if (this._newestFirst) {
        this._autoScroll = this.gridEl.scrollTop < 60;
      } else {
        const { scrollTop, scrollHeight, clientHeight } = this.gridEl;
        this._autoScroll = scrollHeight - scrollTop - clientHeight < 60;
      }
    });
  }

  toggleDirection() {
    this._newestFirst = !this._newestFirst;
    const children = [...this.gridEl.children];
    this.gridEl.innerHTML = "";
    children.reverse().forEach((c) => this.gridEl.appendChild(c));
    this._autoScroll = true;

    // After reversing, re-anchor scroll to the block corresponding
    // to the audio player's *current* position. Resolve live from
    // audio.currentTime + the timeline, NOT from ap.currentSegmentId
    // (last-highlighted id is typically one tick stale).
    const ap = window.audioPlayer;
    if (!ap) return this._newestFirst;

    let liveId = null;
    if (ap._timeline && ap.audio && !isNaN(ap.audio.currentTime)) {
      const currentMs = ap.audio.currentTime * 1000 + (ap._segmentOffset || 0);
      const liveSeg = ap._timeline.find(
        (s) => currentMs >= s.start_ms && currentMs <= s.end_ms,
      );
      if (liveSeg) liveId = liveSeg.segment_id;
    }
    if (!liveId) liveId = ap.currentSegmentId;

    const anchorEl = liveId
      ? ap._findSegmentRow(liveId)
      : this.gridEl.querySelector(".compact-block.playing");

    if (anchorEl) {
      if (liveId) {
        ap._clearHighlight();
        anchorEl.classList.add("playing");
        ap.currentSegmentId = liveId;
      }
      anchorEl.scrollIntoView({ behavior: "auto", block: "center" });
    } else {
      this.gridEl.scrollTop = this._newestFirst ? 0 : this.gridEl.scrollHeight;
    }
    return this._newestFirst;
  }

  update(segmentId, event) {
    if (!segmentId) {
      this._clear();
      return;
    }
    if (!event.text) return;

    let lang = event.language;
    const tr = event.translation?.text || "";
    const langA = getLangA();
    const langB = getLangB();
    lang = routeLangByScript(event.text, lang, langA, langB);

    const speakerKey = this._getSpeakerKey(event);
    const startMs = event.start_ms || 0;
    const endMs = event.end_ms || startMs;

    const segRecord = {
      text: event.text,
      tr,
      lang,
      isFinal: event.is_final,
      speakers: event.speakers || [],
      ts: Date.now(),
      furiganaHtml: event.furigana_html || null,
      trFuriganaHtml: event.translation?.furigana_html || null,
    };

    if (this._segmentMap.has(segmentId)) {
      const existingBlock = this._segmentMap.get(segmentId);
      existingBlock.segments.set(segmentId, segRecord);
      existingBlock.endMs = Math.max(existingBlock.endMs, endMs);

      if (existingBlock.speakerKey !== speakerKey) {
        existingBlock.speakerKey = speakerKey;
        const clusterId = event.speakers?.[0]?.cluster_id;
        if (clusterId != null) {
          const color = getSpeakerColor(clusterId);
          existingBlock.row.style.setProperty("--speaker-color", color);
          existingBlock.row.dataset.clusterId = String(clusterId);
        }
      }
      this._rebuildBlock(existingBlock);
      return;
    }

    const blockDurationMs = this._currentBlock
      ? this._currentBlock.endMs - this._currentBlock.startMs
      : 0;
    const blockSegmentCount = this._currentBlock
      ? this._currentBlock.segments.size
      : 0;
    const shouldMerge =
      this._currentBlock &&
      this._currentBlock.speakerKey === speakerKey &&
      this._currentBlock.lang === lang &&
      startMs - this._currentBlock.endMs < 5000 &&
      blockDurationMs < 45000 &&
      blockSegmentCount < 12;

    if (shouldMerge) {
      this._currentBlock.segments.set(segmentId, segRecord);
      this._segmentMap.set(segmentId, this._currentBlock);
      this._currentBlock.endMs = Math.max(this._currentBlock.endMs, endMs);
      this._rebuildBlock(this._currentBlock);
    } else {
      const block = this._createBlock(speakerKey, lang, startMs, event);
      block.segments.set(segmentId, segRecord);
      this._segmentMap.set(segmentId, block);
      this._currentBlock = block;
      this._rebuildBlock(block);
    }

    if (this._autoScroll) {
      requestAnimationFrame(() => {
        this.gridEl.scrollTop = this._newestFirst ? 0 : this.gridEl.scrollHeight;
      });
    }
  }

  _getSpeakerKey(event) {
    if (!event.speakers?.length) return "unknown";
    const s = event.speakers[0];
    return s.identity || s.display_name || `speaker-${s.cluster_id || 0}`;
  }

  _createBlock(speakerKey, lang, startMs, event) {
    const row = document.createElement("div");
    row.className = "compact-block";

    const clusterId = event.speakers?.[0]?.cluster_id;
    const color = getSpeakerColor(clusterId);
    row.style.setProperty("--speaker-color", color);
    row.dataset.clusterId = String(clusterId ?? "");

    const primary = event.speakers?.[0];
    const speakerName = primary?.identity || primary?.display_name || speakerKey;
    const overlapChips = (event.speakers || [])
      .slice(1)
      .map((s) => {
        const cid = s.cluster_id;
        const c = getSpeakerColor(cid);
        const n =
          s.identity || s.display_name || getSpeakerDisplayName(cid, null);
        return `<span class="compact-speaker-overlap" style="--c:${c}" title="Overlapping: ${esc(n)}">+ ${esc(n)}</span>`;
      })
      .join("");
    const timeStr = this._formatTime(startMs);

    row.innerHTML = `
      <div class="compact-block-header">
        <span class="compact-speaker" style="color:${color}">${esc(speakerName)}</span>${overlapChips}
        <span class="compact-time">${timeStr}</span>
      </div>
      <div class="compact-columns">
        <div class="compact-col compact-col-a"></div>
        <div class="compact-col compact-col-b"></div>
      </div>
    `;

    if (this._newestFirst) {
      this.gridEl.prepend(row);
    } else {
      this.gridEl.appendChild(row);
    }

    const block = {
      speakerKey,
      lang,
      row,
      colA: row.querySelector(".compact-col-a"),
      colB: row.querySelector(".compact-col-b"),
      speakerEl: row.querySelector(".compact-speaker"),
      startMs,
      endMs: startMs,
      segments: new Map(),
    };

    // Click-to-seek is REVIEW-ONLY. During a live meeting we don't
    // let the UI play any audio out — the operator is typically in
    // the room with the speaker and any audio from the admin
    // interface creates a feedback loop / confuses the meeting.
    const isLiveNow = () => document.body.classList.contains("recording");
    row.addEventListener("click", (ev) => {
      if (isLiveNow()) return;
      if (
        ev.target.closest(
          ".compact-speaker, .compact-speaker-overlap, .compact-time, button, a, input",
        )
      )
        return;
      if (!this._meetingId) return;
      const seekStart = Math.max(0, block.startMs - SEEK_LEADIN_MS);
      const eMs = Math.max(block.endMs, block.startMs + 500);
      const firstSeg = block.segments.keys().next().value || null;
      if (window.audioPlayer._fullMeetingSrc) {
        window.audioPlayer.seekFullMeeting(seekStart, firstSeg);
      } else {
        window.audioPlayer.playSegment(this._meetingId, seekStart, eMs, firstSeg);
      }
    });
    row.dataset.clickToSeek = "1";

    return block;
  }

  _rebuildBlock(block) {
    const langA = getLangA();
    const langB = getLangB();
    const cssA = state.languageNames[langA]?.css_font_class || "";
    const cssB = state.languageNames[langB]?.css_font_class || "";

    // Monolingual: route everything into column A.
    if (!langB) {
      const textOnly = [];
      let anyRuby = false;
      for (const [, seg] of block.segments) {
        if (seg.furiganaHtml) {
          textOnly.push(seg.furiganaHtml);
          anyRuby = true;
        } else textOnly.push(esc(seg.text));
      }
      const newA = textOnly.join(" ");
      if (block.colA.innerHTML !== newA) block.colA.innerHTML = newA;
      if (block.colB.innerHTML !== "") block.colB.innerHTML = "";
      block.colA.classList.toggle("has-ruby", anyRuby);
      if (cssA) block.colA.classList.add(cssA);
      return;
    }

    let textA = [];
    let textB = [];

    const matchesColumn = (text, col) => {
      const routed = routeLangByScript(text, col, langA, langB);
      return routed === col;
    };
    let anyRubyA = false;
    let anyRubyB = false;
    for (const [, seg] of block.segments) {
      const srcCol = seg.lang === langA ? "A" : seg.lang === langB ? "B" : null;
      if (!srcCol) continue;
      if (srcCol === "A" && matchesColumn(seg.text, langA)) {
        if (seg.furiganaHtml) {
          textA.push(seg.furiganaHtml);
          anyRubyA = true;
        } else textA.push(esc(seg.text));
        if (seg.tr && matchesColumn(seg.tr, langB)) {
          if (seg.trFuriganaHtml) {
            textB.push(seg.trFuriganaHtml);
            anyRubyB = true;
          } else textB.push(esc(seg.tr));
        } else if (!seg.isFinal) textB.push("...");
      } else if (srcCol === "B" && matchesColumn(seg.text, langB)) {
        if (seg.furiganaHtml) {
          textB.push(seg.furiganaHtml);
          anyRubyB = true;
        } else textB.push(esc(seg.text));
        if (seg.tr && matchesColumn(seg.tr, langA)) {
          if (seg.trFuriganaHtml) {
            textA.push(seg.trFuriganaHtml);
            anyRubyA = true;
          } else textA.push(esc(seg.tr));
        } else if (!seg.isFinal) textA.push("...");
      }
    }

    const classA = `compact-col compact-col-a ${cssA}${anyRubyA ? " has-ruby" : ""}`;
    const classB = `compact-col compact-col-b ${cssB}${anyRubyB ? " has-ruby" : ""}`;
    if (block.colA.className !== classA) block.colA.className = classA;
    if (block.colB.className !== classB) block.colB.className = classB;
    const newA = textA.join(" ");
    const newB = textB.join(" ");
    if (block.colA.innerHTML !== newA) block.colA.innerHTML = newA;
    if (block.colB.innerHTML !== newB) block.colB.innerHTML = newB;

    if (block.speakerEl) {
      let latestSpeakers = null;
      let latestTs = 0;
      for (const [, seg] of block.segments) {
        if (seg.speakers?.length && (seg.ts || 0) >= latestTs) {
          latestSpeakers = seg.speakers;
          latestTs = seg.ts || 0;
        }
      }
      if (latestSpeakers?.length) {
        const s = latestSpeakers[0];
        const name = getSpeakerDisplayName(s.cluster_id, s.identity || s.display_name);
        if (block.speakerEl.textContent !== name) {
          block.speakerEl.textContent = name;
          const color = getSpeakerColor(s.cluster_id);
          block.speakerEl.style.color = color;
          block.row.style.setProperty("--speaker-color", color);
          block.row.dataset.clusterId = String(s.cluster_id ?? "");
        }
      }
    }

    const segIds = [...block.segments.keys()];
    block.row.dataset.segmentId = segIds[0] || "";
    block.row.dataset.segmentIds = segIds.join(",");

    const hasPartial = [...block.segments.values()].some((s) => !s.isFinal);
    block.row.classList.toggle("partial", hasPartial);
  }

  _clear(force = false) {
    if (!force && document.body.classList.contains("recording")) return;
    this._currentBlock = null;
    this._segmentMap.clear();
    this.gridEl.innerHTML = "";
  }
}
