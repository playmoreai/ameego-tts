const HEADER_SIZE = 16;
const HEADER_MAGIC = 0x47454d41; // "AMEG" little-endian
const SAMPLE_RATE = 24000;
const BUFFER_SIZE = 4096;
const SWITCH_POLL_INTERVAL_MS = 1000;
const SWITCH_TIMEOUT_MS = 120000;

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
    this.connecting = false;
    this.userInitiatedDisconnect = false;
    this.audioCtx = null;
    this.audioNode = null;
    this.useWorklet = false;
    this.pcmBuffer = [];
    this.isConnected = false;
    this.synthesizing = false;
    this.recordedChunks = [];
    this.voiceClonePromptId = null;
    this.voiceClonePromptModel = null;
    this.voiceClonePromptGeneration = null;
    this.currentRequestId = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.runtimeStatus = 'loading';
    this.runtimeGeneration = null;
    this.activeMode = 'voice_clone';
    this.activeCloneModelSize = '0.6B';
    this.voiceDesignEnabled = false;
    this.switchStartedAt = 0;
    this.currentSwitchTarget = null;
    this.lastSelectedVoiceMode = this.selectedVoiceMode();
    this.lastSelectedCloneModel = null;
    this.selectionHydrated = false;
    this.lastHealth = null;

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
    this.voiceDesignRadio = document.querySelector('input[name="voice-mode"][value="design"]');
    this.clonePanel = document.getElementById('clone-panel');
    this.voiceDesignPanel = document.getElementById('voice-design-panel');
    this.voiceDesignInput = document.getElementById('voice-design-input');
    this.refAudioInput = document.getElementById('ref-audio');
    this.createVoiceBtn = document.getElementById('create-voice-btn');
    this.cloneStatus = document.getElementById('clone-status');
    this.downloadBtn = document.getElementById('download-btn');
    this.synthesisError = document.getElementById('synthesis-error');
    this.waveformCanvas = document.getElementById('waveform');
    this.waveformCtx = this.waveformCanvas.getContext('2d');
    this.switchOverlay = document.getElementById('switch-overlay');
    this.switchSubtitle = document.getElementById('switch-subtitle');
    this.switchElapsed = document.getElementById('switch-elapsed');
    this.switchTarget = document.getElementById('switch-target');

    this.metricTTFA = document.getElementById('metric-ttfa');
    this.metricRTF = document.getElementById('metric-rtf');
    this.metricDuration = document.getElementById('metric-duration');
    this.metricChunks = document.getElementById('metric-chunks');
    this.metricBuffer = document.getElementById('metric-buffer');
    this.metricModel = document.getElementById('metric-model');

    this.bindEvents();
    this.applyVoiceModeUI();
    this.updateControls();
  }

  clearClonePrompt(message = '', className = 'clone-status') {
    this.voiceClonePromptId = null;
    this.voiceClonePromptModel = null;
    this.voiceClonePromptGeneration = null;
    this.cloneStatus.textContent = message;
    this.cloneStatus.className = className;
  }

  bindEvents() {
    this.connectBtn.addEventListener('click', () => this.toggleConnection());
    this.speakBtn.addEventListener('click', () => this.speak());
    this.stopBtn.addEventListener('click', () => this.stop());
    this.chunkSizeSlider.addEventListener('input', () => {
      this.chunkSizeValue.textContent = this.chunkSizeSlider.value;
    });
    this.voiceModeRadios.forEach((radio) =>
      radio.addEventListener('change', () => this.handleVoiceModeChange())
    );
    this.modelSelect.addEventListener('change', () => this.handleCloneModelChange());
    this.createVoiceBtn.addEventListener('click', () => this.uploadRefAudio());
    this.downloadBtn.addEventListener('click', () => this.downloadWav());
    this.voiceDesignInput.addEventListener('input', () => this.updateControls());

    const resizeCanvas = () => {
      this.waveformCanvas.width = this.waveformCanvas.offsetWidth * devicePixelRatio;
      this.waveformCanvas.height = this.waveformCanvas.offsetHeight * devicePixelRatio;
      this.waveformCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      this.clearWaveform();
    };
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  }

  selectedVoiceMode() {
    return document.querySelector('input[name="voice-mode"]:checked')?.value || 'default';
  }

  desiredRuntimeMode() {
    return this.selectedVoiceMode() === 'design' ? 'voice_design' : 'voice_clone';
  }

  getWsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/tts`;
  }

  async fetchHealth() {
    const res = await fetch('/health');
    if (!res.ok) {
      throw new Error(`Health request failed: ${res.status}`);
    }
    const data = await res.json();
    const previousGeneration = this.runtimeGeneration;
    this.runtimeStatus = data.status || 'loading';
    this.runtimeGeneration = data.runtime_generation ?? this.runtimeGeneration;
    this.activeMode = data.active_mode || 'voice_clone';
    this.activeCloneModelSize = data.active_clone_model_size || this.activeCloneModelSize;
    this.voiceDesignEnabled = Boolean(data.voice_design_enabled);
    this.lastHealth = data;

    const modelLabels = {
      '0.6B': 'Standard (0.6B)',
      '1.7B': 'High Quality (1.7B)',
    };
    const models = data.available_clone_models || data.available_models || [];
    const defaultModel = data.default_clone_model_size || data.default_model || models[0];
    const previousValue = this.modelSelect.value;
    this.modelSelect.innerHTML = '';
    for (const model of models) {
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = modelLabels[model] || model;
      if (model === previousValue || (!previousValue && model === defaultModel)) {
        opt.selected = true;
      }
      this.modelSelect.appendChild(opt);
    }
    if (!this.modelSelect.value && defaultModel) {
      this.modelSelect.value = defaultModel;
    }

    if (!this.selectionHydrated) {
      if (data.active_mode === 'voice_design' || data.switch_target_mode === 'voice_design') {
        this.setVoiceMode('design');
        this.lastSelectedVoiceMode = 'design';
      } else {
        this.modelSelect.value =
          data.switch_target_clone_model_size ||
          data.active_clone_model_size ||
          this.modelSelect.value ||
          defaultModel;
        this.lastSelectedVoiceMode = this.selectedVoiceMode();
      }
      this.lastSelectedCloneModel =
        data.switch_target_clone_model_size ||
        data.active_clone_model_size ||
        this.modelSelect.value ||
        defaultModel;
      this.selectionHydrated = true;
    } else if (!this.lastSelectedCloneModel) {
      this.lastSelectedCloneModel = this.modelSelect.value || defaultModel;
    }

    if (
      previousGeneration !== null &&
      this.runtimeGeneration !== null &&
      previousGeneration !== this.runtimeGeneration &&
      this.voiceClonePromptId
    ) {
      this.clearClonePrompt(
        'Voice clone must be re-created after the model runtime changed.',
        'clone-status'
      );
    }

    if (!this.voiceDesignEnabled && this.selectedVoiceMode() === 'design') {
      this.setVoiceMode('default');
      this.lastSelectedVoiceMode = 'default';
    }
    this.voiceDesignRadio.disabled = !this.voiceDesignEnabled;
    this.applyVoiceModeUI();
    this.updateControls();
    return data;
  }

  toggleConnection() {
    if (this.isConnected || this.connecting) {
      this.disconnect();
    } else {
      this.connect();
    }
  }

  async initAudio() {
    if (this.audioCtx) return;
    this.audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });

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
        return;
      } catch (e) {
        console.warn('AudioWorklet failed, falling back to ScriptProcessor:', e);
      }
    }

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
      for (let i = written; i < output.length; i++) output[i] = 0;
      const totalSamples = this.pcmBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
      const ms = ((totalSamples / SAMPLE_RATE) * 1000) | 0;
      this.metricBuffer.textContent = `${ms}ms`;
    };
    this.audioNode.connect(this.audioCtx.destination);
    this.useWorklet = false;
  }

  async connect() {
    if (this.ws || this.connecting) return;
    this.connecting = true;
    this.userInitiatedDisconnect = false;
    this.setStatus('connecting', 'Connecting...');

    try {
      await this.initAudio();
      await this.fetchHealth();
      if (this.userInitiatedDisconnect) {
        this.connecting = false;
        this.setStatus('disconnected', 'Disconnected');
        this.updateControls();
        return;
      }

      this.ws = new WebSocket(this.getWsUrl());
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = async () => {
        if (this.userInitiatedDisconnect) {
          this.ws.close(1000);
          return;
        }
        this.connecting = false;
        this.isConnected = true;
        this.reconnectDelay = 1000;
        this.syncStatusFromRuntime();
        this.updateControls();
        try {
          await this.ensureModeForSelection();
        } catch (e) {
          console.warn('Initial mode sync failed:', e);
        }
      };

      this.ws.onmessage = (e) => this.handleMessage(e);

      this.ws.onclose = (e) => {
        this.connecting = false;
        this.isConnected = false;
        this.ws = null;
        this.hideSwitchOverlay();
        this.setStatus('disconnected', 'Disconnected');
        this.updateControls();
        if (e.code !== 1000 && !this.userInitiatedDisconnect) {
          setTimeout(() => this.connect(), this.reconnectDelay);
          this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
        }
      };

      this.ws.onerror = () => {
        this.connecting = false;
        this.setStatus('error', 'Connection error');
      };
    } catch (err) {
      this.connecting = false;
      console.error('Connection error:', err);
      this.setStatus('error', 'Connection error');
      this.updateControls();
    }
  }

  disconnect() {
    this.userInitiatedDisconnect = true;
    this.connecting = false;
    if (this.ws) {
      this.ws.close(1000);
      this.ws = null;
    }
    this.isConnected = false;
    this.hideSwitchOverlay();
    this.setStatus('disconnected', 'Disconnected');
    this.updateControls();
  }

  setStatus(state, text = null) {
    const states = {
      disconnected: { color: '#888', text: 'Disconnected' },
      connecting: { color: '#f0ad4e', text: 'Connecting...' },
      connected: { color: '#5cb85c', text: 'Connected' },
      busy: { color: '#f0ad4e', text: 'Switching mode...' },
      error: { color: '#d9534f', text: 'Error' },
    };
    const s = states[state] || states.disconnected;
    this.statusDot.style.background = s.color;
    this.statusText.textContent = text || s.text;
    this.connectBtn.textContent = this.isConnected ? 'Disconnect' : 'Connect';
  }

  syncStatusFromRuntime() {
    if (!this.isConnected) {
      this.setStatus('disconnected', 'Disconnected');
      return;
    }
    if (this.runtimeStatus === 'ready') {
      this.setStatus('connected', 'Connected');
    } else if (this.runtimeStatus === 'error') {
      this.setStatus('error', this.lastHealth?.message || 'Runtime error');
    } else {
      this.setStatus('busy', 'Switching mode...');
    }
  }

  applyVoiceModeUI() {
    const mode = this.selectedVoiceMode();
    this.clonePanel.style.display = mode === 'clone' ? 'block' : 'none';
    this.voiceDesignPanel.style.display = mode === 'design' ? 'block' : 'none';
  }

  updateControls() {
    const selectedMode = this.selectedVoiceMode();
    const isDesign = selectedMode === 'design';
    const runtimeReady = this.runtimeStatus === 'ready';
    const switching = this.switchOverlay.classList.contains('visible') || this.runtimeStatus === 'switching';
    const canSpeak =
      this.isConnected &&
      runtimeReady &&
      !switching &&
      !this.synthesizing &&
      (!isDesign || Boolean(this.voiceDesignInput.value.trim()));

    this.speakBtn.disabled = !canSpeak;
    this.stopBtn.disabled = !this.synthesizing;
    this.createVoiceBtn.disabled = !this.isConnected || !runtimeReady || switching || isDesign;
    this.modelSelect.disabled = !this.isConnected || switching || this.synthesizing || isDesign;
    this.textInput.disabled = switching;
    this.languageSelect.disabled = switching;
    this.chunkSizeSlider.disabled = switching;
    this.refAudioInput.disabled = switching || isDesign;
    this.voiceDesignInput.disabled = switching || !isDesign;
    this.voiceModeRadios.forEach((radio) => {
      radio.disabled = switching || this.synthesizing || (!this.voiceDesignEnabled && radio.value === 'design');
    });
  }

  setVoiceMode(mode) {
    const radio = document.querySelector(`input[name="voice-mode"][value="${mode}"]`);
    if (radio) {
      radio.checked = true;
      this.applyVoiceModeUI();
      this.updateControls();
    }
  }

  async handleVoiceModeChange() {
    const nextMode = this.selectedVoiceMode();
    const previousMode = this.lastSelectedVoiceMode;
    this.applyVoiceModeUI();

    if (!this.voiceDesignEnabled && nextMode === 'design') {
      this.synthesisError.textContent = 'Voice Design is not enabled on this deployment.';
      this.setVoiceMode(previousMode);
      return;
    }

    if (!this.isConnected) {
      this.lastSelectedVoiceMode = nextMode;
      this.updateControls();
      return;
    }

    const switched = await this.ensureModeForSelection();
    if (!switched) {
      this.setVoiceMode(previousMode);
      return;
    }

    this.lastSelectedVoiceMode = nextMode;
    this.updateControls();
  }

  async handleCloneModelChange() {
    const nextModel = this.modelSelect.value;
    const previousModel = this.lastSelectedCloneModel || nextModel;

    if (!this.isConnected || this.desiredRuntimeMode() !== 'voice_clone') {
      this.lastSelectedCloneModel = nextModel;
      if (this.voiceClonePromptId && this.voiceClonePromptModel !== nextModel) {
        this.clearClonePrompt('Voice clone must be re-created for this model.', 'clone-status');
      }
      this.updateControls();
      return;
    }

    const switched = await this.ensureModeForSelection();
    if (!switched) {
      this.modelSelect.value = previousModel;
      return;
    }

    this.lastSelectedCloneModel = nextModel;
    if (this.voiceClonePromptId && this.voiceClonePromptModel !== nextModel) {
      this.clearClonePrompt('Voice clone must be re-created for this model.', 'clone-status');
    }
    this.updateControls();
  }

  async ensureModeForSelection() {
    const targetMode = this.desiredRuntimeMode();
    const cloneModel = targetMode === 'voice_clone' ? this.modelSelect.value : null;

    if (
      this.runtimeStatus === 'ready' &&
      this.activeMode === targetMode &&
      (targetMode === 'voice_design' || !cloneModel || cloneModel === this.activeCloneModelSize)
    ) {
      this.setStatus('connected');
      return true;
    }

    if (this.runtimeStatus === 'switching') {
      const switchingToTarget =
        this.lastHealth?.switch_target_mode === targetMode &&
        (
          targetMode === 'voice_design' ||
          !cloneModel ||
          this.lastHealth?.switch_target_clone_model_size === cloneModel
        );
      if (switchingToTarget) {
        const targetLabel = targetMode === 'voice_design'
          ? 'Voice Design'
          : this.modelSelect.options[this.modelSelect.selectedIndex]?.textContent || cloneModel || 'Voice Clone';
        this.showSwitchOverlay(targetLabel, targetMode);
        this.setStatus('busy', 'Switching mode...');
        try {
          await this.pollUntilReady(targetMode, cloneModel);
          this.hideSwitchOverlay();
          this.syncStatusFromRuntime();
          return true;
        } catch (e) {
          this.handleSwitchFailure(e);
          return false;
        }
      }
      this.synthesisError.textContent = this.lastHealth?.message || 'Another model switch is already in progress.';
      return false;
    }

    return await this.requestModeSwitch(targetMode, cloneModel);
  }

  async requestModeSwitch(targetMode, cloneModel) {
    const targetLabel = targetMode === 'voice_design'
      ? 'Voice Design'
      : this.modelSelect.options[this.modelSelect.selectedIndex]?.textContent || cloneModel || 'Voice Clone';

    this.showSwitchOverlay(targetLabel, targetMode);
    this.setStatus('busy', 'Switching mode...');
    this.updateControls();

    try {
      const res = await fetch('/mode/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: targetMode, model: cloneModel }),
      });
      const data = await res.json();

      if (data.status === 'unchanged') {
        await this.fetchHealth();
        this.hideSwitchOverlay();
        this.syncStatusFromRuntime();
        return true;
      }

      if (data.status === 'switching') {
        await this.pollUntilReady(targetMode, cloneModel);
        this.hideSwitchOverlay();
        this.syncStatusFromRuntime();
        return true;
      }

      if (data.status === 'busy') {
        await this.handleSwitchRejected(data.message || 'The server is busy.');
        return false;
      }

      if (data.status === 'error') {
        await this.handleSwitchRejected(data.message || 'Mode switch failed.');
        return false;
      }

      this.handleSwitchFailure(new Error(data.message || 'Mode switch failed.'));
      return false;
    } catch (e) {
      this.handleSwitchFailure(e);
      return false;
    }
  }

  async pollUntilReady(targetMode, cloneModel) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < SWITCH_TIMEOUT_MS) {
      let data;
      try {
        data = await this.fetchHealth();
      } catch (e) {
        console.warn('Health polling failed during switch:', e);
        data = null;
      }

      if (data?.status === 'error') {
        throw new Error(data.message || 'Mode switch failed.');
      }

      const ready =
        data?.status === 'ready' &&
        data.active_mode === targetMode &&
        (targetMode === 'voice_design' || !cloneModel || data.active_clone_model_size === cloneModel);
      if (ready) return;

      if (!data) {
        console.warn('Retrying health polling during switch after a transient failure.');
      }

      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      this.updateSwitchOverlay(elapsed, targetMode);
      this.setStatus('busy', `Switching mode... (${elapsed}s)`);
      await new Promise((resolve) => setTimeout(resolve, SWITCH_POLL_INTERVAL_MS));
    }
    throw new Error('Mode switch timeout');
  }

  showSwitchOverlay(targetLabel, targetMode) {
    this.switchStartedAt = Date.now();
    this.currentSwitchTarget = targetMode;
    this.switchTarget.textContent = targetLabel;
    this.switchElapsed.textContent = '0s';
    this.switchSubtitle.textContent =
      targetMode === 'voice_design'
        ? 'Preparing Voice Design model. Audio will resume when it is ready.'
        : 'Preparing Voice Clone model. Audio will resume when it is ready.';
    this.switchOverlay.classList.add('visible');
    this.switchOverlay.setAttribute('aria-hidden', 'false');
  }

  updateSwitchOverlay(elapsedSec = null, targetMode = this.currentSwitchTarget) {
    const seconds = elapsedSec ?? Math.round((Date.now() - this.switchStartedAt) / 1000);
    this.switchElapsed.textContent = `${seconds}s`;
    this.switchSubtitle.textContent =
      targetMode === 'voice_design'
        ? 'Preparing Voice Design model. This usually takes 15-30 seconds.'
        : 'Preparing Voice Clone model. This usually takes 15-30 seconds.';
  }

  hideSwitchOverlay() {
    this.switchOverlay.classList.remove('visible');
    this.switchOverlay.setAttribute('aria-hidden', 'true');
    this.currentSwitchTarget = null;
    this.updateControls();
  }

  handleSwitchFailure(error) {
    this.hideSwitchOverlay();
    const msg = error?.message === 'Mode switch timeout'
      ? 'Mode switch timeout.'
      : (error?.message || 'Mode switch failed.');
    this.setStatus('error', msg);
    this.synthesisError.textContent = msg;
    this.updateControls();
  }

  async handleSwitchRejected(message) {
    this.hideSwitchOverlay();
    try {
      await this.fetchHealth();
    } catch (e) {
      console.warn('Failed to refresh health after switch rejection:', e);
    }
    this.syncStatusFromRuntime();
    this.synthesisError.textContent = message || 'Mode switch rejected.';
    this.updateControls();
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
    const pcmData = new Int16Array(buffer, HEADER_SIZE);
    this.recordedChunks.push(new Int16Array(pcmData));
    this.feedAudio(pcmData);
    this.drawWaveform(pcmData);
  }

  handleJsonMessage(msg) {
    switch (msg.type) {
      case 'synthesis_start':
        this.synthesizing = true;
        this.synthesisError.textContent = '';
        this.metricModel.textContent = msg.model || '-';
        this.updateControls();
        break;

      case 'synthesis_end':
        this.synthesizing = false;
        this.downloadBtn.disabled = false;
        this.metricTTFA.textContent = `${msg.ttfa_ms}ms`;
        this.metricRTF.textContent = `${msg.rtf}x`;
        this.metricDuration.textContent = `${(msg.duration_ms / 1000).toFixed(1)}s`;
        this.metricChunks.textContent = msg.total_chunks;
        this.metricModel.textContent = msg.model || '-';
        this.updateControls();
        break;

      case 'synthesis_cancelled':
        this.synthesizing = false;
        if (msg.chunks_sent > 0) this.downloadBtn.disabled = false;
        this.updateControls();
        break;

      case 'voice_clone_prompt_ready':
        this.voiceClonePromptId = msg.prompt_id;
        this.voiceClonePromptModel = msg.model === 'Voice Design' ? null : this.modelSelect.value;
        this.voiceClonePromptGeneration = msg.runtime_generation ?? this.runtimeGeneration;
        this.cloneStatus.textContent = `Voice ready — ${msg.model} (${msg.processing_ms}ms)`;
        this.cloneStatus.className = 'clone-status success';
        this.updateControls();
        break;

      case 'error':
        console.error('Server error:', msg.code, msg.message);
        if (['INVALID_AUDIO', 'VOICE_CLONE_ERROR'].includes(msg.code)) {
          this.cloneStatus.textContent = `Error: ${msg.message}`;
          this.cloneStatus.className = 'clone-status error';
        } else {
          this.synthesisError.textContent = msg.message;
        }
        this.synthesizing = false;
        this.updateControls();
        break;

      case 'pong':
        break;
    }
  }

  async speak() {
    if (!this.isConnected || this.synthesizing) return;

    const text = this.textInput.value.trim();
    if (!text) return;

    const runtimeReady = await this.ensureModeForSelection();
    if (!runtimeReady || this.runtimeStatus !== 'ready') {
      this.synthesisError.textContent = 'The requested mode is not ready yet.';
      return;
    }

    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }

    this.clearAudio();
    this.recordedChunks = [];
    this.clearWaveform();
    this.downloadBtn.disabled = true;
    this.synthesisError.textContent = '';
    this.metricTTFA.textContent = '-';
    this.metricRTF.textContent = '-';
    this.metricDuration.textContent = '-';
    this.metricChunks.textContent = '-';
    this.metricBuffer.textContent = '-';

    const voiceMode = this.selectedVoiceMode();
    const mode = voiceMode === 'design' ? 'voice_design' : 'voice_clone';
    const instruct = mode === 'voice_design' ? this.voiceDesignInput.value.trim() : null;

    if (
      voiceMode === 'clone' &&
      this.voiceClonePromptId &&
      this.voiceClonePromptGeneration !== null &&
      this.runtimeGeneration !== null &&
      this.voiceClonePromptGeneration !== this.runtimeGeneration
    ) {
      this.clearClonePrompt(
        'Voice clone must be re-created after the model runtime changed.',
        'clone-status error'
      );
      return;
    }

    this.currentRequestId = uuid();
    this.ws.send(
      JSON.stringify({
        type: 'synthesize',
        request_id: this.currentRequestId,
        text,
        mode,
        model: mode === 'voice_clone' ? this.modelSelect.value : null,
        language: this.languageSelect.value,
        instruct,
        voice_clone_prompt_id: voiceMode === 'clone' ? this.voiceClonePromptId : null,
        chunk_size: parseInt(this.chunkSizeSlider.value, 10),
      })
    );
  }

  stop() {
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
    if (this.recordedChunks.length > 0) this.downloadBtn.disabled = false;
    this.updateControls();
  }

  async uploadRefAudio() {
    if (this.desiredRuntimeMode() !== 'voice_clone') {
      this.cloneStatus.textContent = 'Reference audio is only available in Voice Clone mode.';
      this.cloneStatus.className = 'clone-status error';
      return;
    }

    const runtimeReady = await this.ensureModeForSelection();
    if (!runtimeReady) return;

    const file = this.refAudioInput.files[0];
    if (!file) {
      this.cloneStatus.textContent = 'Please select an audio file.';
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
        audio_format: ext,
        model: this.modelSelect.value,
      })
    );
  }

  feedAudio(int16Array) {
    if (this.useWorklet) {
      this.audioNode.port.postMessage({ type: 'audio', samples: int16Array });
      return;
    }

    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32[i] = int16Array[i] / 32768.0;
    }
    this.pcmBuffer.push(float32);
  }

  clearAudio() {
    if (this.useWorklet) {
      this.audioNode.port.postMessage({ type: 'clear' });
    } else {
      this.pcmBuffer.length = 0;
    }
  }

  drawWaveform(pcmData) {
    const canvas = this.waveformCanvas;
    const ctx = this.waveformCtx;
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const shift = Math.max(1, (pcmData.length / SAMPLE_RATE) * 200);
    ctx.putImageData(imageData, -shift, 0);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(width - shift, 0, shift, height);

    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(pcmData.length / shift));
    for (let i = 0; i < shift && i * step < pcmData.length; i++) {
      const sample = pcmData[i * step] / 32768;
      const y = (1 - sample) * (height / 2);
      const x = width - shift + i;
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

    const totalLen = this.recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const allSamples = new Int16Array(totalLen);
    let offset = 0;
    for (const chunk of this.recordedChunks) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }

    const wavBuffer = new ArrayBuffer(44 + allSamples.byteLength);
    const view = new DataView(wavBuffer);
    const writeString = (off, str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(off + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + allSamples.byteLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, allSamples.byteLength, true);

    let sampleOffset = 44;
    for (let i = 0; i < allSamples.length; i++, sampleOffset += 2) {
      view.setInt16(sampleOffset, allSamples[i], true);
    }

    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ameego-tts-${Date.now()}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.ameegoTTS = new AmeegoTTSClient();
  window.ameegoTTS.connect().catch((err) => {
    console.error('Auto-connect failed:', err);
  });
});
