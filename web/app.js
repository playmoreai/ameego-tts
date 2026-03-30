const HEADER_SIZE = 16;
const HEADER_MAGIC = 0x47454d41; // "AMEG" little-endian
const SAMPLE_RATE = 24000;
const BUFFER_SIZE = 4096;

function uuid() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

class AmeegoTTSClient {
  constructor() {
    this.ws = null;
    this.audioCtx = null;
    this.audioNode = null; // AudioWorkletNode or ScriptProcessorNode
    this.useWorklet = false;
    this.pcmBuffer = []; // ring buffer for ScriptProcessor fallback
    this.isConnected = false;
    this.voiceClonePromptId = null;
    this.voiceClonePromptModel = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.currentRequestId = null;
    this.synthesizing = false;
    this.recordedChunks = [];

    // UI elements
    this.statusDot = document.getElementById('status-dot');
    this.statusText = document.getElementById('status-text');
    this.connectBtn = document.getElementById('connect-btn');
    this.textInput = document.getElementById('text-input');
    this.modelSelect = document.getElementById('model-select');
    this.languageSelect = document.getElementById('language-select');
    this.chunkSizeSlider = document.getElementById('chunk-size');
    this.chunkSizeValue = document.getElementById('chunk-size-value');
    this.speakBtn = document.getElementById('speak-btn');
    this.stopBtn = document.getElementById('stop-btn');
    this.voiceModeRadios = document.querySelectorAll('input[name="voice-mode"]');
    this.clonePanel = document.getElementById('clone-panel');
    this.refAudioInput = document.getElementById('ref-audio');
    this.refTextInput = document.getElementById('ref-text');
    this.createVoiceBtn = document.getElementById('create-voice-btn');
    this.cloneStatus = document.getElementById('clone-status');
    this.downloadBtn = document.getElementById('download-btn');
    this.synthesisError = document.getElementById('synthesis-error');
    this.waveformCanvas = document.getElementById('waveform');
    this.waveformCtx = this.waveformCanvas.getContext('2d');

    // Metrics
    this.metricTTFA = document.getElementById('metric-ttfa');
    this.metricRTF = document.getElementById('metric-rtf');
    this.metricDuration = document.getElementById('metric-duration');
    this.metricChunks = document.getElementById('metric-chunks');
    this.metricBuffer = document.getElementById('metric-buffer');
    this.metricModel = document.getElementById('metric-model');

    this.bindEvents();
  }

  bindEvents() {
    this.connectBtn.addEventListener('click', () => this.toggleConnection());
    this.speakBtn.addEventListener('click', () => this.speak());
    this.stopBtn.addEventListener('click', () => this.stop());
    this.chunkSizeSlider.addEventListener('input', () => {
      this.chunkSizeValue.textContent = this.chunkSizeSlider.value;
    });
    this.voiceModeRadios.forEach((r) =>
      r.addEventListener('change', () => {
        this.clonePanel.style.display =
          document.querySelector('input[name="voice-mode"]:checked').value === 'clone'
            ? 'block'
            : 'none';
      })
    );
    this.modelSelect.addEventListener('change', () => {
      if (this.voiceClonePromptId && this.voiceClonePromptModel !== this.modelSelect.value) {
        this.voiceClonePromptId = null;
        this.voiceClonePromptModel = null;
        this.cloneStatus.textContent = 'Voice clone must be re-created for this model.';
        this.cloneStatus.className = 'clone-status';
      }
    });
    this.createVoiceBtn.addEventListener('click', () => this.uploadRefAudio());
    this.downloadBtn.addEventListener('click', () => this.downloadWav());

    // Resize canvas
    const resizeCanvas = () => {
      this.waveformCanvas.width = this.waveformCanvas.offsetWidth * devicePixelRatio;
      this.waveformCanvas.height = this.waveformCanvas.offsetHeight * devicePixelRatio;
      this.waveformCtx.scale(devicePixelRatio, devicePixelRatio);
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  }

  getWsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/tts`;
  }

  toggleConnection() {
    if (this.isConnected) {
      this.disconnect();
    } else {
      this.connect();
    }
  }

  async initAudio() {
    if (this.audioCtx) return;

    this.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });

    // Try AudioWorklet (requires secure context: HTTPS or localhost)
    if (this.audioCtx.audioWorklet) {
      try {
        await this.audioCtx.audioWorklet.addModule('/audio-worklet-processor.js');
        this.audioNode = new AudioWorkletNode(this.audioCtx, 'pcm-player-processor');
        this.audioNode.port.onmessage = (e) => {
          if (e.data.type === 'bufferDepth') {
            const ms = ((e.data.samples / SAMPLE_RATE) * 1000) | 0;
            this.metricBuffer.textContent = `${ms}ms`;
          }
        };
        this.audioNode.connect(this.audioCtx.destination);
        this.useWorklet = true;
        console.log('Audio: using AudioWorklet');
        return;
      } catch (e) {
        console.warn('AudioWorklet failed, falling back to ScriptProcessor:', e);
      }
    }

    // Fallback: ScriptProcessorNode (works on HTTP)
    this.pcmBuffer = [];
    this.audioNode = this.audioCtx.createScriptProcessor(BUFFER_SIZE, 0, 1);
    this.audioNode.onaudioprocess = (e) => {
      const output = e.outputBuffer.getChannelData(0);
      let written = 0;
      while (written < output.length && this.pcmBuffer.length > 0) {
        const chunk = this.pcmBuffer[0];
        const available = chunk.length;
        const needed = output.length - written;
        if (available <= needed) {
          output.set(chunk, written);
          written += available;
          this.pcmBuffer.shift();
        } else {
          output.set(chunk.subarray(0, needed), written);
          this.pcmBuffer[0] = chunk.subarray(needed);
          written = output.length;
        }
      }
      // Fill remaining with silence
      for (let i = written; i < output.length; i++) output[i] = 0;

      // Report buffer depth
      const totalSamples = this.pcmBuffer.reduce((s, c) => s + c.length, 0);
      const ms = ((totalSamples / SAMPLE_RATE) * 1000) | 0;
      this.metricBuffer.textContent = `${ms}ms`;
    };
    this.audioNode.connect(this.audioCtx.destination);
    this.useWorklet = false;
    console.log('Audio: using ScriptProcessorNode (HTTP fallback)');
  }

  feedAudio(int16Array) {
    if (this.useWorklet) {
      this.audioNode.port.postMessage({ type: 'audio', samples: int16Array });
    } else {
      // Convert Int16 to Float32 for ScriptProcessor
      const float32 = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32[i] = int16Array[i] / 32768.0;
      }
      this.pcmBuffer.push(float32);
    }
  }

  clearAudio() {
    if (this.useWorklet) {
      this.audioNode.port.postMessage({ type: 'clear' });
    } else {
      this.pcmBuffer.length = 0;
    }
  }

  async fetchAvailableModels() {
    const MODEL_LABELS = {
      '0.6B': 'Standard (0.6B)',
      '1.7B': 'High Quality (1.7B)',
    };
    try {
      const res = await fetch('/health');
      const data = await res.json();
      const models = data.available_models || [];
      const defaultModel = data.default_model || models[0];
      this.modelSelect.innerHTML = '';
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = MODEL_LABELS[m] || m;
        if (m === defaultModel) opt.selected = true;
        this.modelSelect.appendChild(opt);
      }
    } catch (e) {
      console.warn('Failed to fetch available models:', e);
      if (this.modelSelect.options.length === 0) {
        const opt = document.createElement('option');
        opt.value = '0.6B';
        opt.textContent = 'Standard (0.6B)';
        this.modelSelect.appendChild(opt);
      }
    }
  }

  async connect() {
    if (this.ws) return;
    this.setStatus('connecting');

    try {
      await this.initAudio();
      await this.fetchAvailableModels();

      this.ws = new WebSocket(this.getWsUrl());
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.isConnected = true;
        this.reconnectDelay = 1000;
        this.setStatus('connected');
        this.speakBtn.disabled = false;
      };

      this.ws.onmessage = (e) => this.handleMessage(e);

      this.ws.onclose = (e) => {
        this.isConnected = false;
        this.ws = null;
        this.speakBtn.disabled = true;
        this.setStatus('disconnected');
        if (e.code !== 1000) {
          setTimeout(() => this.connect(), this.reconnectDelay);
          this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
        }
      };

      this.ws.onerror = () => {
        this.setStatus('error');
      };
    } catch (err) {
      console.error('Connection error:', err);
      this.setStatus('error');
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000);
      this.ws = null;
    }
    this.isConnected = false;
    this.setStatus('disconnected');
    this.speakBtn.disabled = true;
  }

  setStatus(state) {
    const states = {
      disconnected: { color: '#888', text: 'Disconnected' },
      connecting: { color: '#f0ad4e', text: 'Connecting...' },
      connected: { color: '#5cb85c', text: 'Connected' },
      error: { color: '#d9534f', text: 'Error' },
    };
    const s = states[state] || states.disconnected;
    this.statusDot.style.background = s.color;
    this.statusText.textContent = s.text;
    this.connectBtn.textContent = state === 'connected' ? 'Disconnect' : 'Connect';
  }

  handleMessage(e) {
    if (e.data instanceof ArrayBuffer) {
      this.handleBinaryFrame(e.data);
    } else {
      this.handleJsonMessage(JSON.parse(e.data));
    }
  }

  handleBinaryFrame(buffer) {
    if (buffer.byteLength < HEADER_SIZE) return;

    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== HEADER_MAGIC) return;

    // Extract PCM data after header
    const pcmData = new Int16Array(buffer, HEADER_SIZE);
    this.recordedChunks.push(new Int16Array(pcmData));

    // Feed to audio player
    this.feedAudio(pcmData);

    // Draw waveform
    this.drawWaveform(pcmData);
  }

  handleJsonMessage(msg) {
    switch (msg.type) {
      case 'synthesis_start':
        this.synthesizing = true;
        this.stopBtn.disabled = false;
        this.speakBtn.disabled = true;
        if (msg.model) this.metricModel.textContent = msg.model;
        break;

      case 'synthesis_end':
        this.synthesizing = false;
        this.stopBtn.disabled = true;
        this.speakBtn.disabled = false;
        this.downloadBtn.disabled = false;
        this.metricTTFA.textContent = `${msg.ttfa_ms}ms`;
        this.metricRTF.textContent = `${msg.rtf}x`;
        this.metricDuration.textContent = `${(msg.duration_ms / 1000).toFixed(1)}s`;
        this.metricChunks.textContent = msg.total_chunks;
        if (msg.model) this.metricModel.textContent = msg.model;
        break;

      case 'synthesis_cancelled':
        this.synthesizing = false;
        this.stopBtn.disabled = true;
        this.speakBtn.disabled = false;
        if (msg.chunks_sent > 0) this.downloadBtn.disabled = false;
        break;

      case 'voice_clone_prompt_ready':
        this.voiceClonePromptId = msg.prompt_id;
        this.voiceClonePromptModel = msg.model;
        this.cloneStatus.textContent = `Voice ready — ${msg.model} (${msg.processing_ms}ms)`;
        this.cloneStatus.className = 'clone-status success';
        this.createVoiceBtn.disabled = false;
        break;

      case 'error':
        console.error('Server error:', msg.code, msg.message);
        if (['INVALID_AUDIO', 'VOICE_CLONE_ERROR'].includes(msg.code)) {
          this.cloneStatus.textContent = `Error: ${msg.message}`;
          this.cloneStatus.className = 'clone-status error';
          this.createVoiceBtn.disabled = false;
        } else {
          this.synthesisError.textContent = msg.message;
        }
        this.synthesizing = false;
        this.stopBtn.disabled = true;
        this.speakBtn.disabled = false;
        break;

      case 'pong':
        break;
    }
  }

  speak() {
    if (!this.isConnected || this.synthesizing) return;

    const text = this.textInput.value.trim();
    if (!text) return;

    // Resume audio context (browser policy)
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }

    // Clear previous state
    this.clearAudio();
    this.recordedChunks = [];
    this.clearWaveform();
    this.downloadBtn.disabled = true;
    this.synthesisError.textContent = '';

    // Reset metrics
    this.metricTTFA.textContent = '-';
    this.metricRTF.textContent = '-';
    this.metricDuration.textContent = '-';
    this.metricChunks.textContent = '-';
    this.metricBuffer.textContent = '-';

    const voiceMode = document.querySelector('input[name="voice-mode"]:checked').value;

    this.currentRequestId = uuid();
    this.ws.send(
      JSON.stringify({
        type: 'synthesize',
        request_id: this.currentRequestId,
        text: text,
        model: this.modelSelect.value,
        language: this.languageSelect.value,
        voice_clone_prompt_id: voiceMode === 'clone' ? this.voiceClonePromptId : null,
        chunk_size: parseInt(this.chunkSizeSlider.value),
      })
    );
  }

  stop() {
    // Send cancel to server to stop streaming
    if (this.ws && this.isConnected && this.currentRequestId) {
      try {
        this.ws.send(
          JSON.stringify({
            type: 'cancel',
            request_id: this.currentRequestId,
          })
        );
      } catch (e) {
        console.warn('Failed to send cancel:', e);
      }
    }
    this.clearAudio();
    this.synthesizing = false;
    this.stopBtn.disabled = true;
    this.speakBtn.disabled = false;
  }

  async uploadRefAudio() {
    const file = this.refAudioInput.files[0];
    if (!file) {
      this.cloneStatus.textContent = 'Please select an audio file.';
      this.cloneStatus.className = 'clone-status error';
      return;
    }

    const refText = this.refTextInput.value.trim();
    if (!refText) {
      this.cloneStatus.textContent = 'Please enter the reference text.';
      this.cloneStatus.className = 'clone-status error';
      return;
    }

    this.createVoiceBtn.disabled = true;
    this.cloneStatus.textContent = 'Processing...';
    this.cloneStatus.className = 'clone-status';

    const buffer = await file.arrayBuffer();
    const base64 = btoa(
      new Uint8Array(buffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
    );

    const ext = file.name.split('.').pop()?.toLowerCase() || 'wav';

    this.ws.send(
      JSON.stringify({
        type: 'upload_ref_audio',
        request_id: uuid(),
        audio_base64: base64,
        ref_text: refText,
        audio_format: ext,
        model: this.modelSelect.value,
      })
    );
  }

  drawWaveform(pcmData) {
    const canvas = this.waveformCanvas;
    const ctx = this.waveformCtx;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    // Shift existing waveform left
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const shift = Math.max(1, (pcmData.length / SAMPLE_RATE) * 200);
    ctx.putImageData(imageData, -shift, 0);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(w - shift, 0, shift, h);

    // Draw new samples
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(pcmData.length / shift));
    for (let i = 0; i < shift && i * step < pcmData.length; i++) {
      const sample = pcmData[i * step] / 32768;
      const y = (1 - sample) * (h / 2);
      const x = w - shift + i;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  clearWaveform() {
    const ctx = this.waveformCtx;
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, this.waveformCanvas.offsetWidth, this.waveformCanvas.offsetHeight);
  }

  downloadWav() {
    if (this.recordedChunks.length === 0) return;

    // Concatenate all chunks
    const totalLen = this.recordedChunks.reduce((sum, c) => sum + c.length, 0);
    const allSamples = new Int16Array(totalLen);
    let offset = 0;
    for (const chunk of this.recordedChunks) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    // Build WAV file
    const wavBuffer = new ArrayBuffer(44 + allSamples.byteLength);
    const view = new DataView(wavBuffer);

    // RIFF header
    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + allSamples.byteLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // chunk size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, allSamples.byteLength, true);
    new Int16Array(wavBuffer, 44).set(allSamples);

    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ameego-tts-${Date.now()}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

// Initialize when DOM is ready and auto-connect
document.addEventListener('DOMContentLoaded', () => {
  window.ttsClient = new AmeegoTTSClient();
  window.ttsClient.connect();
});
