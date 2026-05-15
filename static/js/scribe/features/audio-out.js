// Meeting Scribe — Audio-out listener (pure module).
//
// AudioOutListener owns the admin SPA's real-time interpretation
// audio pipeline: opens /api/ws/audio-out, queues incoming WAV
// blobs, decodes them with the same AudioContext that the click
// handler primed (so the user's gesture stays "hot" through the
// async setup), and plays them back through the default destination.
//
// Reconnect behaviour: an exponential-backoff retry loop heals
// transient WS drops (server event-loop hiccup, keepalive miss) in
// under 2 s without the user re-tapping Listen.
//
// Bootstrap half: constructs the singleton + starts the 2-second
// /api/diag/listener telemetry poll + stamps ``window.audioOutListener``.

const API = "";

export class AudioOutListener {
  constructor() {
    this.ws = null;
    this.audioCtx = null;
    this.enabled = false;
    this.preferredLanguage = "";
    this.mode = "translation"; // 'translation' or 'full'
    this._queue = [];
    this._playing = false;
    // Auto-reconnect state. The server's event loop sometimes blocks
    // long enough that the WebSocket keepalive can't fire and the
    // browser kills the connection from under us. Without
    // auto-reconnect that means "audio worked then went silent forever
    // until you re-tap Listen". With auto-reconnect, the same scenario
    // heals itself in <2 s and the user never notices.
    this._reconnectAttempts = 0;
    this._intentionalStop = false;
    this._reconnectTimer = null;
  }

  /**
   * Create the AudioContext SYNCHRONOUSLY inside the click handler
   * that triggered Listen, so the browser counts the resume as a
   * user-gesture action. After the first ``await`` in the click
   * handler the gesture context is gone and a deferred resume may
   * silently no-op, leaving the context permanently suspended —
   * every decoded WAV plays into a muted destination and the user
   * hears nothing despite the server delivering audio successfully.
   * Idempotent: safe to call multiple times.
   */
  primeAudioContext() {
    if (this.audioCtx && this.audioCtx.state !== "closed") return;
    try {
      this.audioCtx = new AudioContext();
      // Synchronous resume() returns a Promise but starts the work
      // immediately while the gesture is still hot. We do NOT await
      // it here — that would push the work past the gesture boundary.
      if (this.audioCtx.state === "suspended") {
        this.audioCtx.resume().catch(() => {});
      }
    } catch (e) {
      console.warn("Failed to prime audio-out context:", e);
    }
  }

  async start(language, mode) {
    this.preferredLanguage = language || "";
    this.mode = mode || "translation";
    this._intentionalStop = false;
    this._reconnectAttempts = 0;
    // Reuse the context primed by primeAudioContext() in the click
    // handler if it's still live. Only build a fresh one if priming
    // was skipped.
    if (!this.audioCtx || this.audioCtx.state === "closed") {
      this.audioCtx = new AudioContext();
    }
    if (this.audioCtx.state === "suspended") {
      try {
        await this.audioCtx.resume();
      } catch {}
    }
    this._connect();
  }

  _connect() {
    if (this._intentionalStop) return;
    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    try {
      this.ws = new WebSocket(`${wsProto}//${location.host}/api/ws/audio-out`);
    } catch (e) {
      this._lastErr = `ws ctor: ${e.message}`;
      this._scheduleReconnect();
      return;
    }
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      this.enabled = true;
      this._reconnectAttempts = 0; // reset backoff on a successful open
      if (this.preferredLanguage) {
        this.ws.send(
          JSON.stringify({
            type: "set_language",
            language: this.preferredLanguage,
          }),
        );
      }
      this.ws.send(JSON.stringify({ type: "set_mode", mode: this.mode }));
    };

    this.ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        this._bytesIn = (this._bytesIn || 0) + evt.data.byteLength;
        this._blobsIn = (this._blobsIn || 0) + 1;
        this._queue.push(evt.data);
        if (!this._playing) this._playNext();
      } else if (evt.data instanceof Blob) {
        this._lastErr = "blob arrival converted";
        evt.data.arrayBuffer().then((ab) => {
          this._bytesIn = (this._bytesIn || 0) + ab.byteLength;
          this._blobsIn = (this._blobsIn || 0) + 1;
          this._queue.push(ab);
          if (!this._playing) this._playNext();
        });
      }
    };

    this.ws.onerror = () => {
      this._lastErr = "ws error";
    };
    this.ws.onclose = (e) => {
      this.enabled = false;
      this._lastErr = `ws close ${e?.code || ""}`;
      this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    if (this._intentionalStop) return;
    if (this._reconnectTimer) return; // already pending
    // Exponential backoff capped at 5 s, with the first retry after
    // ~500 ms so a momentary server hiccup heals nearly instantly.
    // The user never re-taps Listen unless they explicitly stop it.
    const delay = Math.min(5000, 500 * Math.pow(2, this._reconnectAttempts));
    this._reconnectAttempts++;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._connect();
    }, delay);
  }

  setLanguage(lang) {
    this.preferredLanguage = lang;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "set_language", language: lang }));
    }
  }

  setMode(mode) {
    this.mode = mode;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "set_mode", mode: mode }));
    }
  }

  async _playNext() {
    if (this._queue.length === 0) {
      this._playing = false;
      return;
    }
    this._playing = true;
    const wavData = this._queue.shift();

    try {
      if (!this.audioCtx) {
        this._lastErr = "play: ctx null";
        return;
      }
      if (this.audioCtx.state === "suspended") {
        try {
          await this.audioCtx.resume();
        } catch (e) {
          this._lastErr = `resume: ${e.message}`;
        }
      }
      const audioBuffer = await this.audioCtx.decodeAudioData(wavData.slice(0));
      if (!audioBuffer || audioBuffer.length === 0) {
        this._lastErr = "decoded empty";
        this._playNext();
        return;
      }
      this._decoded = (this._decoded || 0) + 1;
      const source = this.audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioCtx.destination);
      source.onended = () => {
        this._played = (this._played || 0) + 1;
        this._playNext();
      };
      source.start();
    } catch (e) {
      this._decodeErr = (this._decodeErr || 0) + 1;
      this._lastErr = `decode: ${e.message || e.name}`;
      console.warn("Audio-out playback error:", e);
      this._playNext();
    }
  }

  stop() {
    this._intentionalStop = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this.enabled = false;
    this._queue = [];
    this._playing = false;
    this._bytesIn = 0;
    this._blobsIn = 0;
    this._decoded = 0;
    this._decodeErr = 0;
    this._played = 0;
    this._lastErr = "";
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {});
      this.audioCtx = null;
    }
  }
}

// Telemetry helpers — exported so the bootstrap can drive the 2 s
// /api/diag/listener poll. Pure functions; the per-tab identity is
// resolved at module load.
const ADMIN_CLIENT_ID = (() => {
  let id = sessionStorage.getItem("scribe_client_id");
  if (!id) {
    id = crypto.randomUUID
      ? crypto.randomUUID()
      : `c-${Math.random().toString(36).slice(2)}`;
    sessionStorage.setItem("scribe_client_id", id);
  }
  return id;
})();

const ADMIN_UA_SHORT = (() => {
  const ua = navigator.userAgent || "";
  if (/iPhone|iPad|iPod/.test(ua)) return "iOS";
  if (/Android/.test(ua)) return "Android";
  if (/Macintosh/.test(ua)) return "Mac";
  if (/Windows/.test(ua)) return "Win";
  return "Other";
})();

export function adminAudioDiagSnapshot(listener) {
  const ctx = listener.audioCtx;
  const ws = listener.ws;
  return {
    client_id: ADMIN_CLIENT_ID,
    page: "admin",
    ua_short: ADMIN_UA_SHORT,
    ctx_state: ctx ? ctx.state : "null",
    ctx_rate: ctx ? ctx.sampleRate : 0,
    ws_state: ws
      ? ["CONNECTING", "OPEN", "CLOSING", "CLOSED"][ws.readyState]
      : "NULL",
    primed: !!(ctx && ctx.state === "running"),
    queue: listener._queue?.length || 0,
    bytes_in: listener._bytesIn || 0,
    blobs_in: listener._blobsIn || 0,
    decoded: listener._decoded || 0,
    decode_err: listener._decodeErr || 0,
    played: listener._played || 0,
    last_err: listener._lastErr || "",
  };
}

export function postListenerSnapshot(listener) {
  try {
    fetch(`${API}/api/diag/listener`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(adminAudioDiagSnapshot(listener)),
      keepalive: true,
    }).catch(() => {});
  } catch {}
}
