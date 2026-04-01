# Deployment Guide

This guide follows the shared engine deployment contract defined in the `ameego-gateway/OPERATIONS.md` workspace doc.

## Prerequisites

- `gcloud` installed
- `gcloud auth login`
- `gcloud config set project <PROJECT_ID>`
- active GCP project with billing and GPU quota

## Web Deploy

```bash
./deploy.sh up --profile web
```

Default web behavior:

- bundled web UI enabled
- default model: `0.6B`
- default build: `full`

## API Deploy

```bash
./deploy.sh up --profile api
```

Default api behavior:

- bundled web UI disabled
- default model: `0.6B`
- default build: `fast`

## Common Overrides

```bash
./deploy.sh up --profile web --model 1.7B
./deploy.sh up --profile api --build full
./deploy.sh up --profile web --zone us-central1-c --spot
VOICE_DESIGN_ENABLED=true MODEL_SIZES=0.6B,1.7B ./deploy.sh up --profile web
```

## Day-2 Operations

```bash
./deploy.sh status --profile api
./deploy.sh ssh --profile api
./deploy.sh logs --profile api
./deploy.sh url --profile api
./deploy.sh down --profile api
```

## Integration Check

- `web` should expose the bundled browser UI
- `api` should expose `/health`, `/voices`, and `/ws/tts`
- use the API docs in [`api.md`](api.md) for protocol-level verification
