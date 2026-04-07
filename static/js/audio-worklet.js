/**
 * AudioWorklet processor — captures mic, downsamples to 16kHz, sends s16le PCM.
 *
 * Wire format (per chunk, every ~250ms at 16kHz = ~4000 samples):
 *   [Int16 PCM payload, little-endian]
 *
 * No header — the server knows the format is 16kHz s16le mono.
 * Downsampling from device rate (typically 48kHz) to 16kHz happens here
 * to minimize data over WebSocket and remove server-side resampling.
 */
class ScribeAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.deviceRate = options.processorOptions?.sampleRate || sampleRate;
    this.targetRate = 16000;
    this.ratio = this.deviceRate / this.targetRate;

    // Accumulate ~250ms of 16kHz audio before sending
    this.chunkSize = Math.floor(this.targetRate * 0.25); // 4000 samples
    this.buffer = [];
    this.bufferSamples = 0;

    // Resampling state — linear interpolation
    this.resamplePos = 0;
    this.lastSample = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // mono

    // Downsample device rate → 16kHz using linear interpolation
    const resampled = this._downsample(channelData);
    if (resampled.length > 0) {
      this.buffer.push(resampled);
      this.bufferSamples += resampled.length;
    }

    if (this.bufferSamples >= this.chunkSize) {
      this._flush();
    }

    return true;
  }

  _downsample(input) {
    if (this.ratio <= 1.0) {
      // No downsampling needed (device is already <= 16kHz)
      return new Float32Array(input);
    }

    const outputLen = Math.floor(input.length / this.ratio);
    if (outputLen === 0) return new Float32Array(0);

    const output = new Float32Array(outputLen);
    for (let i = 0; i < outputLen; i++) {
      const srcPos = i * this.ratio;
      const srcIdx = Math.floor(srcPos);
      const frac = srcPos - srcIdx;

      const s0 = srcIdx < input.length ? input[srcIdx] : this.lastSample;
      const s1 = srcIdx + 1 < input.length ? input[srcIdx + 1] : s0;
      output[i] = s0 + frac * (s1 - s0);
    }

    this.lastSample = input[input.length - 1];
    return output;
  }

  _flush() {
    if (this.bufferSamples === 0) return;

    // Concatenate accumulated 16kHz float32 samples
    const audio = new Float32Array(this.bufferSamples);
    let offset = 0;
    for (const chunk of this.buffer) {
      audio.set(chunk, offset);
      offset += chunk.length;
    }

    // Convert float32 [-1, 1] to s16le
    const s16 = new Int16Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
      const s = Math.max(-1, Math.min(1, audio[i]));
      s16[i] = s < 0 ? s * 32768 : s * 32767;
    }

    // Send raw s16le bytes (no header)
    this.port.postMessage(s16.buffer, [s16.buffer]);

    this.buffer = [];
    this.bufferSamples = 0;
  }
}

registerProcessor('scribe-audio-processor', ScribeAudioProcessor);
