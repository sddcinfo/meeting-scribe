// Meeting Scribe — Mic warmup (pure module).
//
// Singleton service that opens a getUserMedia stream + AudioContext
// eagerly on the room-setup view so the meeting-start click handler
// doesn't have to wait for permission negotiation. The recorder
// pipeline can later .consume() the warmed-up stream and take
// ownership (the warm-up forgets it so there's no double-close).
//
// Released when the meeting starts (recorder takes over) or when
// the operator navigates away from the room-setup view.
//
// Pure module: no top-level side effects, no window publish — the
// singleton is consumed by direct named imports.

export const micWarmup = {
  ac: null,
  stream: null,
  source: null,
  primed: false,
  primingPromise: null,

  async prime() {
    if (this.primed) return true;
    if (this.primingPromise) return this.primingPromise;
    this.primingPromise = (async () => {
      try {
        // Use the same constraints as the meeting-recording pipeline so
        // the warmed-up stream can be handed over to AudioPipeline
        // without re-prompting / re-negotiating the device.
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
        });
        this.ac = new AudioContext();
        if (this.ac.state === "suspended") {
          // Some browsers won't resume an AudioContext until the next
          // user gesture. That's fine — _enrollSeat re-resumes inside
          // the click handler. Either way the device + permission are
          // already warm.
          try {
            await this.ac.resume();
          } catch {}
        }
        this.source = this.ac.createMediaStreamSource(this.stream);
        this.primed = true;
        return true;
      } catch (e) {
        console.warn("Mic warm-up failed (will retry on first chair tap):", e);
        this.release();
        return false;
      } finally {
        this.primingPromise = null;
      }
    })();
    return this.primingPromise;
  },

  release() {
    try {
      this.source?.disconnect();
    } catch {}
    if (this.stream) {
      try {
        this.stream.getTracks().forEach((t) => t.stop());
      } catch {}
    }
    if (this.ac && this.ac.state !== "closed") {
      try {
        this.ac.close();
      } catch {}
    }
    this.ac = null;
    this.stream = null;
    this.source = null;
    this.primed = false;
  },

  // Hand the warmed-up stream + AudioContext to a long-lived owner
  // (e.g. the meeting recorder). The caller becomes responsible for
  // tearing them down; the warm-up forgets them so it doesn't
  // double-close.
  consume() {
    if (!this.primed) return null;
    const handover = { ac: this.ac, stream: this.stream, source: this.source };
    this.ac = null;
    this.stream = null;
    this.source = null;
    this.primed = false;
    return handover;
  },
};
