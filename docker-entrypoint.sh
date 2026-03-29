#!/bin/bash
set -e

MODEL_SIZE="${MODEL_SIZE:-0.6B}"
SERVER_PORT="${SERVER_PORT:-8080}"

echo "========================================="
echo " Ameego TTS Server"
echo " Model: Qwen3-TTS-${MODEL_SIZE}"
echo " Port:  ${SERVER_PORT}"
echo "========================================="

exec python3 -m uvicorn server.main:app \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --ws-max-size 16777216 \
    --log-level info
