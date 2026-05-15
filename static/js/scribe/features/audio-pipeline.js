// Meeting Scribe — audio pipeline (pure module).
//
// Owns the recorder + WebSocket + AudioWorklet stack that drives
// ASR/translate during a live meeting. The flow is:
//
//   AudioWorklet → WebSocket (binary) → Server ASR/Translate
//                                          ↓ JSON
//                                  WebSocket message handler
//                                          ↓
//                       SegmentStore.ingest via `onEvent` callback
//
// Plus the side-channel WS messages the server emits during a
// meeting (tts_audio, seat_update, speaker_pulse, room_layout_update,
// speaker_rename, speaker_remap, summary_regenerated, meeting_*).
//
// Every dep this module needs is either passed in (hooks), imported
// from another pure module (micWarmup, speaker-registry primitives),
// or read off `window.*` / `state.*` contracts.

import { micWarmup } from "./mic-warmup.js";
import { renameSpeaker, _speakerRegistry } from "./speaker-registry.js";
import { state } from "../state.js";

const _WS_PROTO = typeof location !== "undefined" && location.protocol === "https:"
  ? "wss:"
  : "ws:";
const _WS_URL = typeof location !== "undefined"
  ? `${_WS_PROTO}//${location.host}/api/ws`
  : "";
const _API = "";

export class AudioPipeline {
  constructor() {
    this.ws = null;
    this.audioCtx = null;
    this.workletNode = null;
    this.stream = null;
    this.analyser = null;
    this.running = false;
  }

  async start(onEvent, hooks = {}) {
    const onSpeakerRegistryChanged = hooks.onSpeakerRegistryChanged || (() => {});
    const onMeetingCancelled = hooks.onMeetingCancelled || (() => {});

    // Ownership check: when the server owns the mic (PipeWire pw-record
    // against the configured Poly / SP325 / line-in), the browser MUST
    // NOT also open getUserMedia. Two reasons:
    //   1. The frames the browser would send are silently dropped server-
    //      side (state.server_mic_active gate in ws/audio_input.py), so
    //      the device-acquire is pure overhead.
    //   2. The laptop mic the browser would grab is whatever the OS
    //      defaults to — often the operator's headset or built-in array.
    //      During an SP325-driven room meeting that's room cross-talk we
    //      don't want anywhere near ASR even if the gate lets a frame
    //      through during a brief server_mic_active flicker.
    // We fetch /api/status once at start. If the gate is on, we open the
    // WS for control + TTS reception only — no mic, no worklet.
    this.serverMicOwnsInput = false;
    try {
      const r = await fetch("/api/status", { credentials: "same-origin" });
      if (r.ok) {
        const s = await r.json();
        this.serverMicOwnsInput = !!s?.audio?.route?.server_mic_active_live;
      }
    } catch {
      /* status unreachable — fall back to opening the local mic. */
    }

    let source;
    if (this.serverMicOwnsInput) {
      // Skip every capture-side step. We still need an AudioContext for
      // TTS playback the WS will deliver, but it has no source attached.
      this.stream = null;
      this.audioCtx = new AudioContext();
      this.analyser = null;
      this.workletNode = null;
    } else {
      // If the setup-panel mic warm-up is still primed, consume it so the
      // recorder starts capturing audio with zero device-init latency.
      const warm = micWarmup.consume();
      if (warm) {
        this.stream = warm.stream;
        this.audioCtx = warm.ac;
        source = warm.source;
      } else {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
        });
        this.audioCtx = new AudioContext();
      }
      if (this.audioCtx.state === "suspended") await this.audioCtx.resume();
      const sampleRate = this.audioCtx.sampleRate;
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      await this.audioCtx.audioWorklet.addModule("/static/js/audio-worklet.js");
      this.workletNode = new AudioWorkletNode(
        this.audioCtx,
        "scribe-audio-processor",
        { processorOptions: { sampleRate } },
      );
      if (!source) source = this.audioCtx.createMediaStreamSource(this.stream);
      source.connect(this.analyser);
      source.connect(this.workletNode);
    }
    this.ws = new WebSocket(_WS_URL);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (evt) => {
      state.wsMessageCount++;
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "tts_audio" && msg.audio_url) {
          // Auto-play TTS audio (interpretation-audio track)
          if (!document.getElementById("tts-mute")?.checked) {
            window.audioPlayer?._setSrcAndPlay(`${_API}${msg.audio_url}`);
          }
        } else if (msg.type === "seat_update") {
          // Update specific seat name in the table strip
          const seats = document.querySelectorAll(".strip-seat");
          let updated = false;
          seats.forEach((s) => {
            const nameSpan = s.querySelector(".strip-seat-name");
            if (nameSpan && !nameSpan.textContent.trim() && !updated) {
              nameSpan.textContent = msg.speaker_name;
              s.classList.add("enrolled");
              updated = true;
            }
          });
        } else if (msg.type === "speaker_pulse") {
          // Update speaker pulse indicators on seats
          const seats = document.querySelectorAll(".strip-seat");
          const activeNames = new Set(
            (msg.active_speakers || []).map((s) => s.name).filter(Boolean),
          );
          seats.forEach((s) => {
            const nameSpan = s.querySelector(".strip-seat-name");
            const name = nameSpan?.textContent?.trim();
            if (name && activeNames.has(name)) {
              s.classList.add("speaking");
            } else {
              s.classList.remove("speaking");
            }
          });
        } else if (msg.type === "room_layout_update") {
          // Mid-meeting layout change from another client — re-render
          window._onRoomLayoutUpdate?.(msg.layout);
        } else if (msg.type === "speaker_rename") {
          // Another client renamed a speaker — sync the registry and
          // refresh all UI surfaces so names stay consistent.
          if (msg.cluster_id != null && msg.display_name) {
            renameSpeaker(msg.cluster_id, msg.display_name);
            onSpeakerRegistryChanged();
          }
        } else if (msg.type === "speaker_remap") {
          // Backend collapsed raw cluster_ids when diarize centroids merged
          // (fix 4). The surviving id takes over the retired id's label and
          // color so "Speaker 41" no longer leaks into the live view after
          // a consolidation pass. See _speaker_catchup_loop.
          if (msg.renames && typeof msg.renames === "object") {
            for (const [retiredStr, survivor] of Object.entries(msg.renames)) {
              const retired = parseInt(retiredStr, 10);
              if (!Number.isFinite(retired)) continue;
              const survivorEntry = _speakerRegistry.clusters.get(survivor);
              const retiredEntry = _speakerRegistry.clusters.get(retired);
              if (retiredEntry && survivorEntry) {
                // Carry over any human-set name from retired → survivor.
                if (retiredEntry.displayName && !survivorEntry.displayName) {
                  survivorEntry.displayName = retiredEntry.displayName;
                }
              }
              _speakerRegistry.clusters.delete(retired);
            }
            onSpeakerRegistryChanged();
          }
        } else if (msg.type === "summary_regenerated") {
          // Server rebuilt summary.json (e.g. after a rename). Poke any
          // review view open on this meeting so the topics / action items
          // refresh.
          window._onSummaryRegenerated?.(msg.meeting_id);
        } else if (msg.type === "meeting_warning") {
          // Silence watchdog (fix 3): audio hasn't landed in 10s+.
          console.warn("meeting_warning", msg);
          const s = document.getElementById("status-line");
          if (s) s.textContent = `Warning: ${msg.reason} (${msg.age_s}s)`;
        } else if (msg.type === "meeting_warning_cleared") {
          // Reset banner if we set one.
          // (no-op beyond clearing — the next normal status update overwrites)
        } else if (msg.type === "interpretation_status") {
          window.dispatchEvent(
            new CustomEvent("meeting-scribe:interpretation-status", { detail: msg }),
          );
        } else if (msg.type === "bt_status") {
          window.dispatchEvent(
            new CustomEvent("meeting-scribe:bt-status", { detail: msg.status || msg }),
          );
        } else if (msg.type === "mic_level") {
          // Server-broadcast mic level (peak amplitude 0-100). Drives
          // the header meter bar. Same handler services browser-mic
          // frames AND SP325/server-mic frames since both funnel
          // through ws/audio_input._handle_audio on the server.
          const bar = document.getElementById("meter-bar");
          if (bar) bar.style.width = `${msg.peak_pct ?? 0}%`;
        } else if (msg.type === "meeting_stopped") {
          console.info("meeting_stopped", msg);
        } else if (msg.type === "meeting_cancelled") {
          console.info("meeting_cancelled", msg);
          // Server cancelled the meeting — clean up UI
          document.body.classList.remove("recording");
          document.body.classList.remove("meeting-active");
          const _cb = document.getElementById("control-bar");
          if (_cb) _cb.style.display = "none";
          const _sl = document.getElementById("status-line");
          if (_sl) _sl.textContent = "Meeting cancelled";
          onMeetingCancelled();
        } else {
          onEvent(msg);
        }
      } catch (e) {
        console.warn("WS parse:", e);
      }
    };
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });
    this.workletNode.port.onmessage = (evt) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        state.audioChunkCount++;
        this.ws.send(evt.data);
      }
    };
    this.running = true;
  }

  async stop() {
    this.running = false;
    this.workletNode?.disconnect();
    if (this.audioCtx) await this.audioCtx.close();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ws?.close();
    this.workletNode = null;
    this.audioCtx = null;
    this.stream = null;
    this.ws = null;
    this.analyser = null;
  }

  getLevel() {
    if (!this.analyser) return 0;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    return Math.min(Math.sqrt(sum / data.length) * 8, 1);
  }
}
