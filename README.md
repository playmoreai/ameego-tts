# Ameego TTS

Streaming TTS server based on Qwen3-TTS and `faster-qwen3-tts`.

- API: [docs/api.md](docs/api.md)
- Deploy: [docs/deploy.md](docs/deploy.md)

## Architecture

```text
Client ──HTTP / WebSocket──> FastAPI ──> runtime pool ──> Qwen3-TTS models (GPU)
```

## Quick Start

```bash
./deploy.sh up --profile web
./deploy.sh up --profile api
```

## Profiles

- `web`: bundled browser UI, manual testing, runtime mode switching
- `api`: internal-only gateway backend, no bundled web UI

## Operations

```bash
./deploy.sh status --profile api
./deploy.sh logs --profile api
./deploy.sh ssh --profile api
./deploy.sh url --profile api
./deploy.sh down --profile api
```

## Configuration

- model: `0.6B` or `1.7B`
- build: `full` or `fast`
- optional multi-model preload and voice-design enablement
- single active profile at a time

## Local Development

- runtime still uses an internal `test` app profile for the web variant
- the external operating contract is `web|api`

## Notes

- In production, `tts --profile api` is intended to be consumed only through `ameego-gateway`.
- The shared deployment contract lives in the `ameego-gateway/OPERATIONS.md` workspace doc.
