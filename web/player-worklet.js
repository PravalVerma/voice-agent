// PCM player worklet: ring buffer of Float32 samples at the context rate.
// Push Int16 PCM with {cmd:"push", pcm:ArrayBuffer}; clear instantly with
// {cmd:"flush"} (barge-in). Primes with ~80 ms before starting so network
// jitter doesn't crackle.

class PcmPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capacity = 12 * sampleRate; // ~12 s
    this.buf = new Float32Array(this.capacity);
    this.readIdx = 0;
    this.writeIdx = 0;
    this.available = 0;
    this.primeSamples = Math.floor(0.08 * sampleRate);
    this.playing = false;

    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg.cmd === "flush") {
        this.readIdx = this.writeIdx = this.available = 0;
        this.playing = false;
        this.warnedOverflow = false;
      } else if (msg.cmd === "push") {
        const int16 = new Int16Array(msg.pcm);
        for (let i = 0; i < int16.length; i++) {
          if (this.available >= this.capacity) {
            // overflow: the browser has fallen more than 12s behind real-time
            // (e.g. the tab was backgrounded/throttled) - drop the OLDEST
            // buffered sample to make room, not the newest. Dropping the
            // newest would silently chop off the tail of whatever the agent
            // is currently saying while still playing out stale backlog
            // first; dropping the oldest instead "fast-forwards" through the
            // stale part and keeps the freshest (currently relevant) audio.
            this.readIdx = (this.readIdx + 1) % this.capacity;
            this.available--;
            if (!this.warnedOverflow) {
              this.warnedOverflow = true;
              this.port.postMessage({ event: "overflow" });
            }
          }
          this.buf[this.writeIdx] = int16[i] / 32768;
          this.writeIdx = (this.writeIdx + 1) % this.capacity;
          this.available++;
        }
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!this.playing && this.available >= this.primeSamples) this.playing = true;
    if (!this.playing) {
      out.fill(0);
      return true;
    }
    for (let i = 0; i < out.length; i++) {
      if (this.available > 0) {
        out[i] = this.buf[this.readIdx];
        this.readIdx = (this.readIdx + 1) % this.capacity;
        this.available--;
      } else {
        out[i] = 0;
      }
    }
    if (this.available === 0) this.playing = false; // drained: re-prime
    return true;
  }
}

registerProcessor("pcm-player", PcmPlayerProcessor);
