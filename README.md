# Ameego TTS

Real-time streaming TTS server powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), deployed on GCE with L4 GPU.

- **Low-latency streaming** with ~150ms TTFA on the tested L4 setup
- **Voice cloning** with 3 seconds of reference audio
- **10 languages** — English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **One-command** deploy and destroy
- **Web UI** for instant testing
- **Vendored `faster-qwen3-tts`** for direct control over streaming inference changes

## Architecture

```
┌─────────────┐   WebSocket (PCM16)   ┌─────────────────────────────┐
│   Browser    │◄─────────────────────►│  FastAPI Server (GCE VM)    │
│  AudioWorklet│                       │                             │
│  + Web UI    │                       │  faster-qwen3-tts           │
└─────────────┘                       │  NVIDIA L4 GPU (24GB)       │
                                      └─────────────────────────────┘
```

**Audio pipeline:** Text → Qwen3-TTS streaming inference → PCM16 binary frames over WebSocket → AudioWorklet ring buffer → speakers

## Quick Start

### Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) logged in
- GCP project with billing enabled and GPU quota

### Deploy

```bash
# Default (loads both models, defaults to 0.6B, asia-northeast3-a)
./deploy.sh up

# 1.7B model
./deploy.sh up --model 1.7B

# Spot instance (cheaper, may be preempted)
./deploy.sh up --model 0.6B --spot

# Specific zone
./deploy.sh up --zone us-west2-b
```

The script builds via Cloud Build, creates a GCE VM with GPU, and waits for health check. By default it preserves the original behavior of building and loading both logical model sizes, while `--model` only changes the default selection. If you want a smaller single-model deployment, set `MODEL_SIZES=0.6B` or `MODEL_SIZES=1.7B` in the shell before running `./deploy.sh up`. When ready, it prints the server URL.

### Manage

```bash
./deploy.sh status    # Show instance status and health
./deploy.sh ssh       # SSH into the VM
./deploy.sh logs      # View container logs
./deploy.sh url       # Print server URL
```

### Destroy

```bash
./deploy.sh down
```

## Configuration

### Deploy options

| Flag | Default | Description |
|---|---|---|
| `--model` | `0.6B` | Model size: `0.6B` or `1.7B` |
| `--zone` | `asia-northeast3-a` | GCE zone |
| `--spot` | off | Use spot instance (cheaper) |

### Server env vars

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZES` | `0.6B,1.7B` | Comma-separated model sizes to load |
| `DEFAULT_MODEL_SIZE` | `0.6B` | Default model when client doesn't specify |
| `CHUNK_SIZE` | `2` | Streaming chunk size (codec steps per audio chunk, 1=~83ms, 2=~167ms) |
| `MAX_CONNECTIONS` | `4` | Max concurrent WebSocket connections |
| `MAX_TEXT_LENGTH` | `5000` | Max input text length (characters) |
| `CLONE_PROMPT_CACHE_SIZE` | `32` | In-memory LRU size for reference audio and prompt cache |

Additional model-loading env vars:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID_0_6B` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Hugging Face repo used for logical `0.6B` |
| `MODEL_ID_1_7B` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Hugging Face repo used for logical `1.7B` |
| `MODEL_DEVICE` | `cuda` | CUDA device passed into model load |
| `MODEL_DTYPE` | `bfloat16` | Activation dtype passed into model load |
| `ATTN_IMPLEMENTATION` | `sdpa` | Attention backend used during model load |
| `CUDA_GRAPH_MAX_SEQ_LEN` | `2048` | Static cache size used for CUDA graph capture |

Notes:
- The server now vendors `faster-qwen3-tts` under [vendor/faster-qwen3-tts](/Users/jin/Workspace/ameego-tts/vendor/faster-qwen3-tts).
- `./deploy.sh up` picks up these env vars from the current shell, so you can deploy a custom Hugging Face repo without editing the script.
- Quantization was evaluated separately and intentionally not carried into production. See [docs/quantization-evaluation-2026-03-30.md](/Users/jin/Workspace/ameego-tts/docs/quantization-evaluation-2026-03-30.md).

### Model Comparison

| | 0.6B | 1.7B |
|---|---|---|
| VRAM | ~8 GB | ~12-16 GB |
| Image size | ~15 GB | ~20 GB |
| Quality | Good | Best |
| Speed | Faster | Slower |

Both fit on a single L4 GPU (24 GB).

## WebSocket Protocol

Connect to `ws://<server-ip>:8080/ws/tts`.

### Client → Server

**Synthesize:**
```json
{
  "type": "synthesize",
  "request_id": "uuid",
  "text": "Hello, world!",
  "language": "English",
  "model": "0.6B",
  "voice_clone_prompt_id": null,
  "chunk_size": 2
}
```

**Upload reference audio (voice cloning):**
```json
{
  "type": "upload_ref_audio",
  "request_id": "uuid",
  "audio_base64": "<base64-encoded-audio>",
  "audio_format": "wav",
  "model": "0.6B"
}
```

**Cancel synthesis:**
```json
{
  "type": "cancel",
  "request_id": "uuid"
}
```

### Server → Client

**Audio chunks** — binary WebSocket frames:
```
Bytes 0-3:   Magic "AMEG" (0x41 0x4D 0x45 0x47)
Bytes 4-7:   request_id hash (uint32 LE)
Bytes 8-11:  chunk_index (uint32 LE)
Bytes 12-15: sample_rate (uint32 LE, 24000)
Bytes 16+:   Raw PCM16 audio (int16 LE, mono)
```

**Control messages** — JSON text frames:
- `synthesis_start` — synthesis began, includes `model`
- `synthesis_end` — includes `model`, `ttfa_ms`, `rtf`, `total_chunks`, `duration_ms`
- `synthesis_cancelled` — synthesis was stopped by client
- `voice_clone_prompt_ready` — voice clone prompt cached, returns `prompt_id` and `model`
- `error` — includes `code` and `message`

## Project Structure

```
ameego-tts/
├── Dockerfile              # Multi-stage: deps → model download → runtime
├── deploy.sh               # up / down / status / ssh / logs / url
├── cloudbuild.yaml         # Cloud Build config
├── docker-entrypoint.sh    # Container entrypoint
├── docs/
│   └── quantization-evaluation-2026-03-30.md  # Quantization experiment summary
├── requirements.txt
├── vendor/
│   └── faster-qwen3-tts/   # Vendored streaming inference library
├── server/
│   ├── main.py             # FastAPI app, lifespan, health endpoint
│   ├── config.py           # Env-based settings
│   ├── models.py           # Pydantic message schemas
│   ├── tts_engine.py       # faster-qwen3-tts wrapper, async streaming
│   └── ws_handler.py       # WebSocket protocol handler
├── web/
│   ├── index.html          # Test UI
│   ├── app.js              # WebSocket client
│   └── audio-worklet-processor.js  # PCM playback on audio thread
└── scripts/
    └── download_models.py             # HuggingFace model pre-download
```

## Local Development

```bash
# Install dependencies (requires CUDA GPU)
pip install -r requirements.txt
pip install -e ./vendor/faster-qwen3-tts

# Run server
MODEL_SIZES=0.6B python -m uvicorn server.main:app --host 0.0.0.0 --port 8080

# Open http://localhost:8080
```

## Cost

GCE `g2-standard-4` (L4 GPU): **~$0.83/hour** on-demand, **~$0.33/hour** spot.

Spot 인스턴스는 preempt될 수 있지만, 개발/테스트 용도로는 충분합니다.

## License

MIT
