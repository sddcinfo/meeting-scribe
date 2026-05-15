/**
 * AudioWorklet processor — captures mic at native device rate, sends s16le PCM.
 *
 * Quality strategy (see research in git history):
 *   Browser-side linear-interpolation resampling causes aliasing (frequencies
 *   above Nyquist fold back as static). Instead of implementing a proper
 *   polyphase FIR filter in JS, we send PCM at the device's native rate and
 *   let the server resample with torchaudio's Kaiser-windowed sinc interpolator
 *   (effectively transparent quality, runs on GPU).
 *
 * Benefits vs browser-side downsampling:
 *   - Zero resampling CPU on the client (best battery life)
 *   - No aliasing — server uses a proper anti-aliased kernel
 *   - Single authoritative resample path regardless of browser
 *   - 3× more bandwidth at 48kHz (~1.5 Mbps), trivial on local WiFi/hotspot
 *
 * Wire format (per chunk):
 *   [4 bytes] uint32 little-endian sample_rate
 *   [N bytes] Int16 PCM little-endian payload
 *
 * Chunks are ~250ms at the device rate (typically 12000 samples at 48kHz).
 * The server reads the header, resamples to 16kHz on GPU, then feeds ASR.
 */
class ScribeAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Native device rate — typically 48000, sometimes 44100.
    // `sampleRate` is a global provided by AudioWorkletGlobalScope.
    this.deviceRate = options.processorOptions?.sampleRate || sampleRate;

    // Accumulate ~250ms of device-rate audio before sending.
    // Smaller chunks reduce latency but increase WebSocket overhead;
    // larger chunks increase latency but are more network-efficient.
    // 250ms is a good balance for real-time translation.
    this.chunkSize = Math.floor(this.deviceRate * 0.25);
    this.buffer = [];
    this.bufferSamples = 0;

    // Pre-allocate the 4-byte sample rate header so we can prepend it
    // to every chunk without re-creating it.
    this.sampleRateBytes = new Uint8Array(4);
    new DataView(this.sampleRateBytes.buffer).setUint32(0, this.deviceRate, true);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // mono (or first channel of multi-ch)

    // Keep a copy — AudioWorklet reuses input arrays across process() calls
    this.buffer.push(new Float32Array(channelData));
    this.bufferSamples += channelData.length;

    if (this.bufferSamples >= this.chunkSize) {
      this._flush();
    }

    return true;
  }

  _flush() {
    if (this.bufferSamples === 0) return;

    // Concatenate accumulated float32 samples
    const audio = new Float32Array(this.bufferSamples);
    let offset = 0;
    for (const chunk of this.buffer) {
      audio.set(chunk, offset);
      offset += chunk.length;
    }

    // Convert float32 [-1, 1] to s16le (little-endian native on ARM64/x86-64)
    const s16 = new Int16Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
      const s = Math.max(-1, Math.min(1, audio[i]));
      s16[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
    }

    // Build framed message: [4B sample_rate LE][N*2 bytes s16le PCM]
    const payload = new Uint8Array(4 + s16.byteLength);
    payload.set(this.sampleRateBytes, 0);
    payload.set(new Uint8Array(s16.buffer), 4);

    // Transfer (zero-copy) the underlying buffer
    this.port.postMessage(payload.buffer, [payload.buffer]);

    this.buffer = [];
    this.bufferSamples = 0;
  }
}

registerProcessor('scribe-audio-processor', ScribeAudioProcessor);
