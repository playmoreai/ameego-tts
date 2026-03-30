# Ameego TTS

Streaming TTS server based on Qwen3-TTS and `faster-qwen3-tts`.

Default deployment target:
- model: `1.7B`
- GPU: `NVIDIA L4`

Main docs:
- API: [docs/api.md](/Users/jin/Workspace/ameego-tts/docs/api.md)
- Deploy: [docs/deploy.md](/Users/jin/Workspace/ameego-tts/docs/deploy.md)

## Quick Start

Default deploy:

```bash
./deploy.sh up
```

API-only deploy:

```bash
./deploy.sh up --profile api
```

Common commands:

```bash
./deploy.sh status
./deploy.sh logs
./deploy.sh ssh
./deploy.sh down
```

## Defaults

- default app profile: `test`
- default model: `1.7B`
- default test build: `full`
- default api build: `fast`
- default WebSocket connection limit: `8`
- default extra waiting synth requests: `1`

## Notes

- `test` serves the bundled web UI and supports runtime mode switching
- `api` disables the web UI and `/mode/switch`
- durable cloned voices are stored via `voice_id` in `VOICE_STORAGE_DIR`
