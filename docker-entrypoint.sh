#!/bin/bash
set -e

MODEL_SIZES="${MODEL_SIZES:-0.6B,1.7B}"
DEFAULT_MODEL_SIZE="${DEFAULT_MODEL_SIZE:-0.6B}"
INITIAL_MODE="${INITIAL_MODE:-voice_clone}"
INITIAL_CLONE_MODEL_SIZE="${INITIAL_CLONE_MODEL_SIZE:-${DEFAULT_MODEL_SIZE}}"
export INITIAL_CLONE_MODEL_SIZE
VOICE_DESIGN_ENABLED="${VOICE_DESIGN_ENABLED:-false}"
SERVER_PORT="${SERVER_PORT:-8080}"
MODEL_ID_0_6B="${MODEL_ID_0_6B:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
MODEL_ID_1_7B="${MODEL_ID_1_7B:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
VOICE_DESIGN_MODEL_ID="${VOICE_DESIGN_MODEL_ID:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
HF_CACHE_ROOT="${HF_HOME:-/root/.cache/huggingface}"

repair_qwen_tokenizer_cache() {
    local tokenizer_repo_dir="${HF_CACHE_ROOT}/models--Qwen--Qwen3-TTS-Tokenizer-12Hz"
    local tokenizer_snapshot
    local model_id
    local model_cache_dir
    local snapshot_dir
    local speech_dir
    local filename
    local -a model_ids=("${MODEL_ID_0_6B}" "${MODEL_ID_1_7B}")
    local -a tokenizer_files=("config.json" "configuration.json" "preprocessor_config.json" "model.safetensors")

    if [ "${VOICE_DESIGN_ENABLED}" = "true" ]; then
        model_ids+=("${VOICE_DESIGN_MODEL_ID}")
    fi

    tokenizer_snapshot="$(find "${tokenizer_repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
    if [ -z "${tokenizer_snapshot}" ]; then
        echo "WARNING: tokenizer snapshot not found at ${tokenizer_repo_dir}; skipping speech_tokenizer repair"
        return
    fi

    for model_id in "${model_ids[@]}"; do
        model_cache_dir="${HF_CACHE_ROOT}/models--${model_id//\//--}"
        if [ ! -d "${model_cache_dir}/snapshots" ]; then
            continue
        fi

        while IFS= read -r snapshot_dir; do
            speech_dir="${snapshot_dir}/speech_tokenizer"
            [ -d "${speech_dir}" ] || continue

            for filename in "${tokenizer_files[@]}"; do
                if [ ! -e "${speech_dir}/${filename}" ] && [ -e "${tokenizer_snapshot}/${filename}" ]; then
                    ln -sf "${tokenizer_snapshot}/${filename}" "${speech_dir}/${filename}"
                fi
            done
        done < <(find "${model_cache_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d)
    done
}

echo "========================================="
echo " Ameego TTS Server"
echo " Models: ${MODEL_SIZES}"
echo " Default: ${DEFAULT_MODEL_SIZE}"
echo " Initial Mode: ${INITIAL_MODE}"
echo " Initial Clone: ${INITIAL_CLONE_MODEL_SIZE}"
echo " Voice Design: ${VOICE_DESIGN_ENABLED}"
echo " Port:   ${SERVER_PORT}"
echo "========================================="

repair_qwen_tokenizer_cache

exec python3 -m uvicorn server.main:app \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --ws-max-size 16777216 \
    --log-level info
