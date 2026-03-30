class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.port.onmessage = (e) => {
      if (e.data.type === 'audio') {
        // Preserve the original PCM view so chunk headers or buffer padding are never re-read.
        const int16 = e.data.samples instanceof Int16Array
          ? e.data.samples
          : new Int16Array(e.data.samples);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768.0;
        }
        const newBuffer = new Float32Array(this.buffer.length + float32.length);
        newBuffer.set(this.buffer);
        newBuffer.set(float32, this.buffer.length);
        this.buffer = newBuffer;
        // Report buffer depth
        this.port.postMessage({
          type: 'bufferDepth',
          samples: this.buffer.length,
        });
      } else if (e.data.type === 'clear') {
        this.buffer = new Float32Array(0);
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0]; // mono, 128 samples per quantum
    if (!output) return true;

    const needed = output.length;
    if (this.buffer.length >= needed) {
      output.set(this.buffer.subarray(0, needed));
      this.buffer = this.buffer.slice(needed); // slice() copies, avoiding memory retention
    } else if (this.buffer.length > 0) {
      // Partial fill + silence
      output.set(this.buffer);
      for (let i = this.buffer.length; i < needed; i++) {
        output[i] = 0;
      }
      this.buffer = new Float32Array(0);
    } else {
      // Silence
      output.fill(0);
    }
    return true;
  }
}

registerProcessor('pcm-player-processor', PCMPlayerProcessor);
