# Deployment Guide

This guide covers the standard deployment paths for Ameego TTS.

Defaults:
- default model: `1.7B`
- default profile: `test`
- default build profile:
  - `test -> full`
  - `api -> fast`

## Prerequisites

- `gcloud` installed and authenticated
- active GCP project with billing and GPU quota

Set the project if needed:

```bash
gcloud config set project <PROJECT_ID>
gcloud auth login
```

## Standard Deploy

Default deploy:

```bash
./deploy.sh up
```

This deploys:
- profile: `test`
- model: `1.7B`
- build: `full`

If you want test mode to switch between clone sizes:

```bash
MODEL_SIZES=0.6B,1.7B ./deploy.sh up
```

If you want Voice Design in test mode:

```bash
VOICE_DESIGN_ENABLED=true MODEL_SIZES=0.6B,1.7B ./deploy.sh up
```

## API Deploy

API-only deploy:

```bash
./deploy.sh up --profile api
```

This deploys:
- profile: `api`
- model: `1.7B`
- build: `fast`

Notes:
- no bundled web UI
- `/mode/switch` disabled
- `/health`, `/voices`, `/ws/tts` available

## Common Variants

Use `0.6B`:

```bash
./deploy.sh up --model 0.6B
```

Use API profile with explicit `1.7B`:

```bash
./deploy.sh up --profile api --model 1.7B
```

Force build type:

```bash
./deploy.sh up --build full
./deploy.sh up --profile api --build fast
```

Choose a zone:

```bash
./deploy.sh up --zone us-central1-c
```

Use spot:

```bash
./deploy.sh up --spot
```

## Profiles

### `test`

Purpose:
- bundled web UI
- manual testing
- runtime switching

Default build:
- `full`

### `api`

Purpose:
- external API usage
- fixed runtime
- faster redeploys

Default build:
- `fast`

## Useful Commands

```bash
./deploy.sh status
./deploy.sh ssh
./deploy.sh logs
./deploy.sh url
./deploy.sh down
```

## Important Environment Variables

Most deployments do not need overrides. Common ones:

```bash
APP_PROFILE=test|api
MODEL_SIZES=1.7B
DEFAULT_MODEL_SIZE=1.7B
INITIAL_CLONE_MODEL_SIZE=1.7B
VOICE_DESIGN_ENABLED=false
MAX_CONNECTIONS=8
MAX_WAITING_SYNTH_REQUESTS=1
VOICE_STORAGE_DIR=/data/voices
```

Example:

```bash
VOICE_DESIGN_ENABLED=true ./deploy.sh up
```

## Fast vs Full

### `full`

- bakes model weights into the image
- slower build
- more predictable startup
- best for `test`

### `fast`

- image does not include model weights
- uses VM host Hugging Face cache
- faster rebuild and redeploy
- best for `api`

## Troubleshooting

If health stays unavailable:

```bash
./deploy.sh status
./deploy.sh logs
```

If the selected zone has no L4 capacity, try another zone:

```bash
./deploy.sh up --zone us-central1-c
```
