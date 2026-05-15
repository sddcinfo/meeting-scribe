// Meeting Scribe — Audio player (pure module).
//
// AudioPlayer instance drives the persistent "player bar" at the
// bottom of the admin SPA when reviewing a past meeting. Hooks the
// HTML audio element to:
//   * Click-to-jump on transcript / diarize-lane / compact-block.
//   * Full-meeting podcast playback with speaker-colored scrub bar.
//   * Live segment / speaker / seat highlighting during playback.
//
// Window-contract surface: ``window.audioPlayer`` (object). The
// instance is published by the bootstrap half so consumers
// (slide-viewer.js, showSpeakerModal, etc.) share one singleton.

import { getSpeakerColor } from "../lib/speaker-palette.js";

const API = "";
const _enc = encodeURIComponent;

// Seek lead-in for click-to-jump playback. ASR alignment locks
// ``start_ms`` to the strongest spectral cue of the first word,
// usually 50-200 ms after the breath / soft-consonant onset; without
// a lead-in the leading partial syllable is clipped. 150 ms covers
// the typical alignment slop without bleeding noticeably into the
// previous segment. NOT applied to "play all sequentially" (would
// compound) or to scrub-bar / empty-track seeks.
export const SEEK_LEADIN_MS = 150;

export class AudioPlayer {
  constructor() {
    this.audio = new Audio();
    this.audio.preload = "none";
    this.meetingId = null;
    this.currentSegmentId = null;
    this._fullMeetingSrc = null; // track full-meeting audio URL
    // Setting `src` triggers an async load; calling play() immediately
    // races it. If a second `src` lands before the first play()
    // promise settles, the browser rejects with AbortError ("The
    // play() request was interrupted by a new load request").
    // _setSrcAndPlay swallows that specific rejection and surfaces
    // real failures via the error/stalled handlers below.
    this._pendingSeekMs = null; // queued currentTime for next loadedmetadata
    this._bar = document.getElementById("player-bar");
    this._playBtn = document.getElementById("player-play");
    this._scrub = document.getElementById("player-scrub");
    this._current = document.getElementById("player-current");
    this._durationEl = document.getElementById("player-duration");
    this._speed = document.getElementById("player-speed");

    this._playBtn.addEventListener("click", () => this.togglePlay());
    this._scrub.addEventListener("input", () => {
      if (this.audio.duration)
        this.audio.currentTime = (this._scrub.value / 100) * this.audio.duration;
    });
    this._speed.addEventListener("change", () => {
      this.audio.playbackRate = parseFloat(this._speed.value);
    });
    this.audio.addEventListener("timeupdate", () => this._onTimeUpdate());
    this.audio.addEventListener("ended", () => this._onEnded());
    this.audio.addEventListener("loadedmetadata", () => this._onLoadedMetadata());
    this.audio.addEventListener("error", () => this._onAudioError());
    this.audio.addEventListener("stalled", () => this._onAudioStalled("stalled"));
    this.audio.addEventListener("waiting", () => this._onAudioStalled("waiting"));
  }

  /** Set src (if changed) and start playback, swallowing the AbortError
   *  that fires when a subsequent src change interrupts the previous
   *  play() promise. Real failures land in _onAudioError. */
  _setSrcAndPlay(src) {
    if (this.audio.src !== src) {
      this.audio.src = src;
    }
    this.audio.playbackRate = parseFloat(this._speed.value);
    this._safePlay();
  }

  /** Call audio.play() without surfacing the standard interruption rejection. */
  _safePlay() {
    const p = this.audio.play();
    if (p && typeof p.catch === "function") {
      p.catch((err) => {
        if (err && err.name !== "AbortError" && err.name !== "NotAllowedError") {
          console.warn("audio play failed:", err);
        }
      });
    }
  }

  /**
   * Find the transcript row for a segment id. Compact blocks merge
   * multiple segments into one row, so we have to check both the
   * single-segment ``data-segment-id`` attribute and the merged
   * ``data-segment-ids`` list. The plain
   * ``querySelector([data-segment-id=X])`` path misses any segment
   * that isn't the *first* one in its merged block — that's the root
   * cause of the "text/audio mapping breaks after direction toggle"
   * bug, because the primary id of a merged block no longer always
   * matches the playhead.
   */
  _findSegmentRow(segmentId) {
    if (!segmentId) return null;
    let row = document.querySelector(
      `.compact-block[data-segment-id="${segmentId}"]`,
    );
    if (row) return row;
    for (const el of document.querySelectorAll("[data-segment-ids]")) {
      const ids = el.dataset.segmentIds;
      if (ids && ids.split(",").includes(segmentId)) return el;
    }
    return null;
  }

  /** Play a specific segment from a meeting */
  playSegment(meetingId, startMs, endMs, segmentId) {
    this.meetingId = meetingId;
    this._segmentOffset = startMs; // absolute offset for timeline sync
    this._pendingSeekMs = null;    // segment src has its own start_ms — don't seek
    this._setSrcAndPlay(
      `${API}/api/meetings/${_enc(meetingId)}/audio?start_ms=${startMs}&end_ms=${endMs}`,
    );
    this._highlightRow(segmentId);
    this._playBtn.textContent = "⏸";
    this._playBtn.classList.add("playing");

    // Immediately scroll transcript to the segment
    if (segmentId && window._gridRenderer) {
      const row = this._findSegmentRow(segmentId);
      if (row) row.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  /** Load full meeting audio for podcast-style playback */
  loadMeeting(meetingId, durationMs, timeline) {
    this.meetingId = meetingId;
    this._timeline = timeline || null;
    this._segmentOffset = 0; // full meeting — no offset
    this._fullMeetingSrc = `${API}/api/meetings/${_enc(meetingId)}/audio`;
    this.audio.src = this._fullMeetingSrc;
    this._durationEl.textContent = this._fmt(durationMs);
    this._scrub.value = 0;
    this._current.textContent = "00:00";
    this._bar.style.display = "";
    document.body.classList.add("has-player");

    // Generate speaker color bands on scrub bar.
    if (timeline && timeline.length && durationMs > 0) {
      // Use canonical cluster_id-based coloring so scrub bar matches
      // the speaker timeline lanes and transcript blocks.
      const stops = timeline.map((seg) => {
        const color = getSpeakerColor(seg.speaker_id);
        const s = ((seg.start_ms / durationMs) * 100).toFixed(2);
        const e = ((seg.end_ms / durationMs) * 100).toFixed(2);
        return `${color} ${s}%, ${color} ${e}%`;
      });
      this._scrub.style.setProperty(
        "--speaker-gradient",
        `linear-gradient(to right, var(--bg-raised) 0%, ${stops.join(", ")}, var(--bg-raised) 100%)`,
      );
    } else {
      this._scrub.style.removeProperty("--speaker-gradient");
    }
  }

  /** Seek within the full meeting audio, restoring it if playSegment
   *  changed the src. */
  seekFullMeeting(ms, segmentId) {
    this._segmentOffset = 0;
    if (this._fullMeetingSrc && this.audio.src !== this._fullMeetingSrc) {
      // Setting currentTime now would be lost when the new src
      // finishes loading. Defer the seek to loadedmetadata.
      this._pendingSeekMs = ms;
      this._setSrcAndPlay(this._fullMeetingSrc);
    } else {
      this._pendingSeekMs = null;
      this.audio.currentTime = ms / 1000;
      this.audio.playbackRate = parseFloat(this._speed.value);
      this._safePlay();
    }
    this._highlightRow(segmentId);
    this._playBtn.textContent = "⏸";
    this._playBtn.classList.add("playing");
    if (segmentId) {
      const row = this._findSegmentRow(segmentId);
      if (row) row.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  togglePlay() {
    if (this.audio.paused) {
      this._safePlay();
      this._playBtn.textContent = "⏸";
      this._playBtn.classList.add("playing");
    } else {
      this.audio.pause();
      this._playBtn.textContent = "▶";
      this._playBtn.classList.remove("playing");
    }
  }

  hide() {
    this.audio.pause();
    this._bar.style.display = "none";
    document.body.classList.remove("has-player");
    this._clearHighlight();
  }

  _onTimeUpdate() {
    if (!this.audio.duration) return;
    const pct = (this.audio.currentTime / this.audio.duration) * 100;
    this._scrub.value = pct;
    this._current.textContent = this._fmt(this.audio.currentTime * 1000);

    // Scroll transcript to current playback position, highlight active
    // block, and sync the active timeline block so all three views stay
    // in lockstep.
    if (this._timeline && window._gridRenderer) {
      const currentMs = this.audio.currentTime * 1000 + (this._segmentOffset || 0);
      const seg = this._timeline.find(
        (s) => currentMs >= s.start_ms && currentMs <= s.end_ms,
      );
      if (seg && seg.segment_id) {
        const row = this._findSegmentRow(seg.segment_id);
        // Track the *segment id* we last highlighted, not a DOM class,
        // so that toggling transcript direction (which leaves
        // ``.playing`` on the same element but visually repositions it)
        // re-triggers scrolling.
        if (row && this.currentSegmentId !== seg.segment_id) {
          this._clearHighlight();
          row.classList.add("playing");
          row.scrollIntoView({ behavior: "smooth", block: "center" });
          this.currentSegmentId = seg.segment_id;
        }
        this._highlightTimelineBlock(seg.segment_id);
        this._highlightSpeakingSeat(seg.speaker_id);
      } else {
        this._clearSpeakingSeats();
      }
    }
  }

  _onEnded() {
    this._playBtn.textContent = "▶";
    this._playBtn.classList.remove("playing");
    this._clearHighlight();
  }

  _onLoadedMetadata() {
    if (this._pendingSeekMs != null && this.audio.duration > 0) {
      this.audio.currentTime = this._pendingSeekMs / 1000;
      this._pendingSeekMs = null;
    }
  }

  _onAudioError() {
    const e = this.audio.error;
    console.warn(
      "audio element error:",
      e && e.code,
      e && e.message,
      "src=",
      this.audio.currentSrc,
    );
    this._playBtn.textContent = "▶";
    this._playBtn.classList.remove("playing");
  }

  _onAudioStalled(kind) {
    // Lightweight diagnostic — surfaces stalled/waiting in devtools
    // without thrashing the UI. Real failures escalate via
    // _onAudioError.
    console.debug(`audio ${kind} at ${this.audio.currentTime.toFixed(2)}s`);
  }

  _highlightRow(segmentId) {
    this._clearHighlight();
    if (segmentId) {
      this.currentSegmentId = segmentId;
      const row = this._findSegmentRow(segmentId);
      if (row) row.classList.add("playing");
    }
  }

  _highlightTimelineBlock(segmentId) {
    // Clear previously active timeline block.
    document
      .querySelectorAll(".speaker-timeline-block.playing")
      .forEach((el) => {
        el.classList.remove("playing");
      });
    if (!segmentId) return;
    const blockEl = document.querySelector(
      `.speaker-timeline-block[data-segment-id="${segmentId}"]`,
    );
    if (blockEl) blockEl.classList.add("playing");
  }

  _highlightSpeakingSeat(speakerId) {
    this._clearSpeakingSeats();
    if (speakerId == null) return;
    // Match by cluster_id (canonical) — seats are tagged with
    // data-cluster-id by the meeting loader. Fall back to legacy
    // data-speaker-id for live meetings.
    const target = String(speakerId);
    document.querySelectorAll(".strip-seat").forEach((seat) => {
      if (
        seat.dataset.clusterId === target ||
        seat.dataset.speakerId === target
      ) {
        seat.classList.add("speaking");
      }
    });
  }

  _clearSpeakingSeats() {
    document
      .querySelectorAll(".strip-seat.speaking")
      .forEach((el) => el.classList.remove("speaking"));
  }

  _clearHighlight() {
    // Scoped to transcript + timeline so we don't clobber
    // .player-play.playing.
    document
      .querySelectorAll(
        ".compact-block.playing, .speaker-timeline-block.playing",
      )
      .forEach((r) => r.classList.remove("playing"));
    this.currentSegmentId = null;
  }

  _fmt(ms) {
    const s = Math.floor((ms || 0) / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }
}

// Singleton instance is created + published by audio-player.bootstrap.js
// after the admin SPA boots. The constructor binds event listeners to
// DOM elements (#player-bar, #player-play, …) so it's a top-level
// side effect — exactly what the bootstrap-module split exists for.
