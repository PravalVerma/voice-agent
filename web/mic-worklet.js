// Mic capture worklet: resamples the context rate (usually 48 kHz) down to 16 kHz
// with an incremental linear interpolator, converts to Int16, and posts fixed
// 512-sample (1024-byte) frames — Silero VAD's exact window size — to the main thread.

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    this.frameSamples = opts.frameSamples || 512;
    this.step = sampleRate / this.targetRate; // source samples per output sample
    this.pos = 0;                             // fractional read position in `pending`
    this.pending = new Float32Array(0);
    this.out = new Int16Array(this.frameSamples);
    this.outLen = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;

    const merged = new Float32Array(this.pending.length + ch.length);
    merged.set(this.pending, 0);
    merged.set(ch, this.pending.length);

    let pos = this.pos;
    while (pos + 1 < merged.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const s = merged[i] * (1 - frac) + merged[i + 1] * frac;
      const v = Math.max(-1, Math.min(1, s));
      this.out[this.outLen++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      if (this.outLen === this.frameSamples) {
        const copy = this.out.slice();
        this.port.postMessage(copy.buffer, [copy.buffer]);
        this.outLen = 0;
      }
      pos += this.step;
    }
    const consumed = Math.floor(pos);
    this.pending = merged.slice(consumed);
    this.pos = pos - consumed;
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
