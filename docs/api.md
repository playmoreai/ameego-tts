# Ameego TTS API

This document describes the production-facing API for integrating with Ameego TTS.

Base URL examples:
- HTTP: `http://<host>:8080`
- WebSocket: `ws://<host>:8080/ws/tts`

`api` deployments expose:
- `GET /health`
- `POST /voices`
- `GET /voices/{voice_id}`
- `WS /ws/tts`

`/mode/switch` is not available in `api` profile.

## Quick Flow

1. Create a reusable voice with `POST /voices`
2. Store the returned `voice_id`
3. Open `ws://<host>:8080/ws/tts`
4. Send `synthesize` with `voice_id`
5. Play PCM16 binary frames until `synthesis_end`

## Constraints

- Supported languages:
  - `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`
- Max text length: `5000` characters
- Supported reference-audio formats:
  - `wav`, `mp3`, `flac`, `ogg`, `m4a`, `webm`, `opus`
- Max reference-audio payload: `10 MB`
- Streaming sample format:
  - `24000 Hz`, mono, signed `int16`, little-endian

## Health

### `GET /health`

Use this to verify readiness and current runtime capacity.

Example response:
```json
{
  "status": "ready",
  "active_mode": "voice_clone",
  "active_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "active_clone_model_size": "1.7B",
  "active_profile": "api",
  "mode_switch_enabled": false,
  "active_replica_count": 1,
  "busy_replica_count": 0,
  "available_synth_capacity": 1,
  "waiting_synth_requests": 0,
  "max_waiting_synth_requests": 1,
  "max_connections": 8,
  "active_connections": 0
}
```

Important fields:
- `status`: `ready`, `loading`, `switching`, `error`
- `available_synth_capacity`: synth slots available right now
- `waiting_synth_requests`: queued synth requests waiting for capacity
- `max_waiting_synth_requests`: max queued synth requests allowed before `SERVER_BUSY`
- `max_connections`: max open WebSocket connections

## Voices

### `POST /voices`

Creates a durable `voice_id` from reference audio.

Request:
```http
POST /voices
Content-Type: application/json
```

```json
{
  "audio_base64": "<base64-encoded-audio>",
  "audio_format": "wav",
  "display_name": "Support agent"
}
```

Response:
```json
{
  "voice_id": "6c95fb35-7a20-4293-836e-cfd85ed1e091",
  "audio_format": "wav",
  "duration_ms": 1200.0,
  "created_at": "2026-03-31T00:35:29.968778+00:00",
  "display_name": "Support agent"
}
```

Notes:
- `audio_base64` may contain whitespace; it is stripped before decoding
- audio is normalized to WAV internally

### `GET /voices/{voice_id}`

Returns metadata for an existing voice.

Response:
```json
{
  "voice_id": "6c95fb35-7a20-4293-836e-cfd85ed1e091",
  "audio_format": "wav",
  "duration_ms": 1200.0,
  "created_at": "2026-03-31T00:35:29.968778+00:00",
  "display_name": "Support agent"
}
```

## WebSocket

### Connect

```text
ws://<host>:8080/ws/tts
```

The server sends:
- JSON text frames for control messages
- binary frames for PCM audio chunks

### Client Messages

#### `synthesize`

Voice clone with durable `voice_id`:

```json
{
  "type": "synthesize",
  "request_id": "req-123",
  "text": "Hello, world!",
  "mode": "voice_clone",
  "language": "English",
  "model": "1.7B",
  "voice_id": "6c95fb35-7a20-4293-836e-cfd85ed1e091",
  "chunk_size": 2
}
```

Fields:
- `request_id`: caller-generated unique id, max `128` chars
- `text`: required, non-empty
- `mode`: `voice_clone` or `voice_design`
- `language`: one of the supported languages
- `model`: `0.6B` or `1.7B` for clone mode
- `voice_id`: required for production voice-clone usage
- `chunk_size`: optional, `1..24`, default server config

Rules:
- `voice_id` and `voice_clone_prompt_id` cannot be used together
- `voice_id` is not allowed in `voice_design`
- `voice_design` requires non-empty `instruct`

Voice design:

```json
{
  "type": "synthesize",
  "request_id": "req-456",
  "text": "Welcome to the show.",
  "mode": "voice_design",
  "language": "English",
  "instruct": "Warm, confident narrator with a calm broadcast tone",
  "chunk_size": 2
}
```

#### `cancel`

```json
{
  "type": "cancel",
  "request_id": "req-123"
}
```

If the request is still queued or already streaming, the server completes it with `synthesis_cancelled`.

#### `ping`

```json
{
  "type": "ping"
}
```

### Server Messages

#### `synthesis_start`

```json
{
  "type": "synthesis_start",
  "request_id": "req-123",
  "mode": "voice_clone",
  "model": "1.7B",
  "sample_rate": 24000,
  "sample_width": 2,
  "channels": 1
}
```

#### Binary audio frames

Frame layout:

```text
Bytes 0-3:   "AMEG"
Bytes 4-7:   request_id hash (uint32 LE)
Bytes 8-11:  chunk_index (uint32 LE)
Bytes 12-15: sample_rate (uint32 LE)
Bytes 16+:   PCM16 mono audio (int16 LE)
```

#### `synthesis_end`

```json
{
  "type": "synthesis_end",
  "request_id": "req-123",
  "mode": "voice_clone",
  "model": "1.7B",
  "total_chunks": 18,
  "total_samples": 69120,
  "duration_ms": 2880.0,
  "ttfa_ms": 159.0,
  "rtf": 0.677
}
```

#### `synthesis_cancelled`

```json
{
  "type": "synthesis_cancelled",
  "request_id": "req-123",
  "chunks_sent": 5
}
```

#### `error`

```json
{
  "type": "error",
  "request_id": "req-123",
  "code": "SERVER_BUSY",
  "message": "No idle capacity is available for synthesis. Try again shortly."
}
```

## Common Error Codes

HTTP:
- `INVALID_AUDIO`
- `VOICE_NOT_FOUND`
- `VOICE_CREATE_ERROR`

WebSocket:
- `INVALID_REQUEST`
- `INVALID_TEXT`
- `INVALID_LANGUAGE`
- `INVALID_INSTRUCT`
- `VOICE_NOT_FOUND`
- `VOICE_DESIGN_DISABLED`
- `MODE_NOT_READY`
- `MODE_SWITCH_IN_PROGRESS`
- `SERVER_BUSY`
- `PROMPT_NOT_FOUND`
- `SYNTHESIS_ERROR`
- `INTERNAL_ERROR`

## Integration Notes

- Treat `voice_id` as the stable application-level identifier
- Do not depend on `voice_clone_prompt_id` for production integrations
- Expect `SERVER_BUSY` during load and retry with backoff
- Check `/health` before opening large bursts of synth requests
- Current `api` deployment allows `1` active synth request and `1` waiting synth request
- Additional synth requests beyond current active and waiting capacity fail immediately with `SERVER_BUSY`
- Voice assets are durable for the deployment's configured `VOICE_STORAGE_DIR`

## Minimal Example

1. Create voice:

```bash
curl -X POST http://<host>:8080/voices \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_base64": "<base64-audio>",
    "audio_format": "wav",
    "display_name": "Support agent"
  }'
```

2. Open WebSocket and send:

```json
{
  "type": "synthesize",
  "request_id": "req-123",
  "text": "Hello, thank you for calling.",
  "mode": "voice_clone",
  "language": "English",
  "model": "1.7B",
  "voice_id": "<voice_id>",
  "chunk_size": 2
}
```
