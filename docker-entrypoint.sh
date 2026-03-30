#!/bin/bash
set -e

MODEL_SIZES="${MODEL_SIZES:-0.6B,1.7B}"
DEFAULT_MODEL_SIZE="${DEFAULT_MODEL_SIZE:-0.6B}"
SERVER_PORT="${SERVER_PORT:-8080}"

echo "========================================="
echo " Ameego TTS Server"
echo " Models: ${MODEL_SIZES}"
echo " Default: ${DEFAULT_MODEL_SIZE}"
echo " Port:   ${SERVER_PORT}"
echo "========================================="

exec python3 -m uvicorn server.main:app \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --ws-max-size 16777216 \
    --log-level info
