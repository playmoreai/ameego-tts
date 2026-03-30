#!/bin/bash
set -euo pipefail

# ============================================================
# Ameego TTS — GCE GPU Deploy Script
# Usage:
#   ./deploy.sh up [--model 0.6B|1.7B] [--profile test|api] [--build full|fast] [--spot] [--zone ZONE]
#   ./deploy.sh down
#   ./deploy.sh status
#   ./deploy.sh ssh
#   ./deploy.sh logs
#   ./deploy.sh url
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ENV="${SCRIPT_DIR}/.deploy.env"

# Defaults
INSTANCE_NAME="ameego-tts"
DEFAULT_ZONE="asia-northeast3-a"
DEFAULT_MODEL="1.7B"
DEFAULT_APP_PROFILE="test"
DEFAULT_BUILD_PROFILE_TEST="full"
DEFAULT_BUILD_PROFILE_API="fast"
DEFAULT_MODEL_SIZES="1.7B"
DEFAULT_MODEL_ID_0_6B="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_MODEL_ID_1_7B="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_CLONE_0_6B_REPLICAS="2"
DEFAULT_CLONE_1_7B_REPLICAS="1"
DEFAULT_VOICE_DESIGN_ENABLED="false"
DEFAULT_VOICE_DESIGN_MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_VOICE_DESIGN_REPLICAS="1"
DEFAULT_ALLOWED_ORIGINS=""
DEFAULT_MAX_CONNECTIONS="8"
DEFAULT_MAX_WAITING_SYNTH_REQUESTS="1"
MACHINE_TYPE="g2-standard-4"
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="common-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
FIREWALL_RULE="ameego-tts-allow-http"
SERVER_PORT="8080"
HOST_HF_CACHE_DIR="/var/lib/ameego-tts/hf-cache"
HOST_VOICE_STORE_DIR="/var/lib/ameego-tts/voice-store"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*" >&2; }

# ---- Helpers ------------------------------------------------

load_deploy_env() {
    if [ -f "$DEPLOY_ENV" ]; then
        source "$DEPLOY_ENV"
    fi
}

save_deploy_env() {
    cat > "$DEPLOY_ENV" <<EOF
APP_PROFILE=${APP_PROFILE:-$DEFAULT_APP_PROFILE}
BUILD_PROFILE=${BUILD_PROFILE:-}
ZONE=${ZONE}
MODEL_SIZES=${MODEL_SIZES}
DEFAULT_MODEL_SIZE=${DEFAULT_MODEL_SIZE}
INITIAL_CLONE_MODEL_SIZE=${INITIAL_CLONE_MODEL_SIZE}
CLONE_0_6B_REPLICAS=${CLONE_0_6B_REPLICAS:-$DEFAULT_CLONE_0_6B_REPLICAS}
CLONE_1_7B_REPLICAS=${CLONE_1_7B_REPLICAS:-$DEFAULT_CLONE_1_7B_REPLICAS}
VOICE_DESIGN_REPLICAS=${VOICE_DESIGN_REPLICAS:-$DEFAULT_VOICE_DESIGN_REPLICAS}
VOICE_DESIGN_ENABLED=${VOICE_DESIGN_ENABLED:-false}
MODEL_ID_0_6B=${MODEL_ID_0_6B:-$DEFAULT_MODEL_ID_0_6B}
MODEL_ID_1_7B=${MODEL_ID_1_7B:-$DEFAULT_MODEL_ID_1_7B}
VOICE_DESIGN_MODEL_ID=${VOICE_DESIGN_MODEL_ID:-$DEFAULT_VOICE_DESIGN_MODEL_ID}
ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-$DEFAULT_ALLOWED_ORIGINS}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-$DEFAULT_MAX_CONNECTIONS}
MAX_WAITING_SYNTH_REQUESTS=${MAX_WAITING_SYNTH_REQUESTS:-$DEFAULT_MAX_WAITING_SYNTH_REQUESTS}
VOICE_STORAGE_DIR=${VOICE_STORAGE_DIR:-/data/voices}
MODEL_DEVICE=${MODEL_DEVICE:-cuda}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
CUDA_GRAPH_MAX_SEQ_LEN=${CUDA_GRAPH_MAX_SEQ_LEN:-2048}
CHUNK_SIZE=${CHUNK_SIZE:-2}
MAX_TEXT_LENGTH=${MAX_TEXT_LENGTH:-5000}
CLONE_PROMPT_CACHE_SIZE=${CLONE_PROMPT_CACHE_SIZE:-32}
PROJECT_ID=${PROJECT_ID}
INSTANCE_NAME=${INSTANCE_NAME}
SERVER_PORT=${SERVER_PORT}
EOF
}

get_project_id() {
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        err "No GCP project set. Run: gcloud config set project <PROJECT_ID>"
        exit 1
    fi
}

get_external_ip() {
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null
}

build_docker_env_args() {
    local key value
    local args=()

    for key in "$@"; do
        if [ "${!key+x}" = "x" ] && [ -n "${!key}" ]; then
            value="${!key}"
            args+=("-e" "${key}=${value}")
        fi
    done

    printf '%q ' "${args[@]}"
}

validate_model_sizes() {
    local sizes_csv="$1"
    local size
    IFS=',' read -r -a sizes <<< "$sizes_csv"
    if [ "${#sizes[@]}" -eq 0 ]; then
        err "MODEL_SIZES cannot be empty"
        exit 1
    fi
    for size in "${sizes[@]}"; do
        size="${size//[[:space:]]/}"
        case "$size" in
            0.6B|1.7B) ;;
            *)
                err "Unsupported model size in MODEL_SIZES: ${size}. Allowed: 0.6B,1.7B"
                exit 1
                ;;
        esac
    done
}

normalize_model_sizes_csv() {
    local sizes_csv="$1"
    local normalized=()
    local size
    IFS=',' read -r -a sizes <<< "$sizes_csv"
    for size in "${sizes[@]}"; do
        size="${size//[[:space:]]/}"
        if [ -n "$size" ]; then
            normalized+=("$size")
        fi
    done
    local joined
    joined="$(IFS=,; echo "${normalized[*]}")"
    validate_model_sizes "$joined"
    echo "$joined"
}

validate_model_id() {
    local name="$1"
    local value="$2"
    case "$value" in
        *","*|*$'\n'*|*$'\r'*)
            err "${name} cannot contain commas or newlines: ${value}"
            exit 1
            ;;
    esac
}

normalize_bool() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) echo "true" ;;
        0|false|FALSE|no|NO|off|OFF|"") echo "false" ;;
        *)
            err "Invalid boolean value: $1"
            exit 1
            ;;
    esac
}

normalize_profile() {
    case "${1:-}" in
        test|api) echo "$1" ;;
        "")
            echo "$DEFAULT_APP_PROFILE"
            ;;
        *)
            err "Invalid profile value: $1. Allowed: test, api"
            exit 1
            ;;
    esac
}

default_build_profile_for_app_profile() {
    case "$1" in
        test) echo "$DEFAULT_BUILD_PROFILE_TEST" ;;
        api) echo "$DEFAULT_BUILD_PROFILE_API" ;;
        *)
            err "Unsupported app profile for build default: $1"
            exit 1
            ;;
    esac
}

normalize_build_profile() {
    case "${1:-}" in
        full|fast) echo "$1" ;;
        "")
            err "Build profile cannot be empty"
            exit 1
            ;;
        *)
            err "Invalid build profile value: $1. Allowed: full, fast"
            exit 1
            ;;
    esac
}

validate_positive_int() {
    local name="$1"
    local value="$2"
    case "$value" in
        ''|*[!0-9]*)
            err "${name} must be a positive integer: ${value}"
            exit 1
            ;;
    esac
    if [ "$value" -lt 1 ]; then
        err "${name} must be >= 1: ${value}"
        exit 1
    fi
}

compute_image_tag() {
    local registry="$1"
    local default_model_size="$2"
    local build_profile="$3"
    local model_sizes="$4"
    local model_id_0_6b="$5"
    local model_id_1_7b="$6"
    local voice_design_enabled="$7"
    local voice_design_model_id="$8"
    local digest
    digest="$(
        printf '%s' "${build_profile}|${model_sizes}|${model_id_0_6b}|${model_id_1_7b}|${voice_design_enabled}|${voice_design_model_id}" \
        | shasum \
        | awk '{print substr($1,1,12)}'
    )"
    echo "${registry}/ameego-tts:${default_model_size}-${build_profile}-${digest}"
}

check_prerequisites() {
    if ! command -v gcloud &>/dev/null; then
        err "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! gcloud auth print-access-token &>/dev/null; then
        err "Not authenticated. Run: gcloud auth login"
        exit 1
    fi

    get_project_id
    log "Project: ${PROJECT_ID}"
}

# ---- Commands -----------------------------------------------

cmd_up() {
    local MODEL_SIZE="${MODEL_SIZE:-}"
    local APP_PROFILE_TO_USE="${APP_PROFILE:-$DEFAULT_APP_PROFILE}"
    local BUILD_PROFILE_TO_USE="${BUILD_PROFILE:-}"
    local ZONE="$DEFAULT_ZONE"
    local SPOT_FLAG=""
    local SKIP_BUILD_FLAG
    SKIP_BUILD_FLAG="$(normalize_bool "${SKIP_BUILD:-false}")"

    # Parse args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)  MODEL_SIZE="$2"; shift 2 ;;
            --profile) APP_PROFILE_TO_USE="$2"; shift 2 ;;
            --build) BUILD_PROFILE_TO_USE="$2"; shift 2 ;;
            --spot)   SPOT_FLAG="--provisioning-model=SPOT"; shift ;;
            --zone)   ZONE="$2"; shift 2 ;;
            *)        err "Unknown option: $1"; exit 1 ;;
        esac
    done

    check_prerequisites
    APP_PROFILE_TO_USE="$(normalize_profile "$APP_PROFILE_TO_USE")"
    if [ -z "$MODEL_SIZE" ]; then
        MODEL_SIZE="$DEFAULT_MODEL"
    fi
    if [ -z "$BUILD_PROFILE_TO_USE" ]; then
        BUILD_PROFILE_TO_USE="$(default_build_profile_for_app_profile "$APP_PROFILE_TO_USE")"
    fi
    BUILD_PROFILE_TO_USE="$(normalize_build_profile "$BUILD_PROFILE_TO_USE")"

    local REGION="${ZONE%-*}"
    local REPO_NAME="ameego-tts"
    local REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
    local MODEL_SIZES_TO_LOAD
    if [ "${MODEL_SIZES+x}" = "x" ] && [ -n "${MODEL_SIZES}" ]; then
        MODEL_SIZES_TO_LOAD="${MODEL_SIZES}"
    elif [ "$APP_PROFILE_TO_USE" = "api" ]; then
        MODEL_SIZES_TO_LOAD="${MODEL_SIZE}"
    else
        MODEL_SIZES_TO_LOAD="${DEFAULT_MODEL_SIZES}"
    fi
    local DEFAULT_MODEL_SIZE_TO_USE="${DEFAULT_MODEL_SIZE:-$MODEL_SIZE}"
    local INITIAL_CLONE_MODEL_SIZE_TO_USE="${INITIAL_CLONE_MODEL_SIZE:-$DEFAULT_MODEL_SIZE_TO_USE}"
    local MODEL_ID_0_6B_TO_USE="${MODEL_ID_0_6B:-$DEFAULT_MODEL_ID_0_6B}"
    local MODEL_ID_1_7B_TO_USE="${MODEL_ID_1_7B:-$DEFAULT_MODEL_ID_1_7B}"
    local CLONE_0_6B_REPLICAS_TO_USE="${CLONE_0_6B_REPLICAS:-$DEFAULT_CLONE_0_6B_REPLICAS}"
    local CLONE_1_7B_REPLICAS_TO_USE="${CLONE_1_7B_REPLICAS:-$DEFAULT_CLONE_1_7B_REPLICAS}"
    local VOICE_DESIGN_ENABLED_TO_USE
    VOICE_DESIGN_ENABLED_TO_USE="$(normalize_bool "${VOICE_DESIGN_ENABLED:-$DEFAULT_VOICE_DESIGN_ENABLED}")"
    local VOICE_DESIGN_MODEL_ID_TO_USE="${VOICE_DESIGN_MODEL_ID:-$DEFAULT_VOICE_DESIGN_MODEL_ID}"
    local VOICE_DESIGN_REPLICAS_TO_USE="${VOICE_DESIGN_REPLICAS:-$DEFAULT_VOICE_DESIGN_REPLICAS}"
    local ALLOWED_ORIGINS_TO_USE="${ALLOWED_ORIGINS:-$DEFAULT_ALLOWED_ORIGINS}"
    local MAX_CONNECTIONS_TO_USE="${MAX_CONNECTIONS:-$DEFAULT_MAX_CONNECTIONS}"
    local MAX_WAITING_SYNTH_REQUESTS_TO_USE="${MAX_WAITING_SYNTH_REQUESTS:-$DEFAULT_MAX_WAITING_SYNTH_REQUESTS}"
    local VOICE_STORAGE_DIR_TO_USE="${VOICE_STORAGE_DIR:-/data/voices}"
    MODEL_SIZES_TO_LOAD="$(normalize_model_sizes_csv "$MODEL_SIZES_TO_LOAD")"
    validate_model_id "MODEL_ID_0_6B" "$MODEL_ID_0_6B_TO_USE"
    validate_model_id "MODEL_ID_1_7B" "$MODEL_ID_1_7B_TO_USE"
    validate_model_id "VOICE_DESIGN_MODEL_ID" "$VOICE_DESIGN_MODEL_ID_TO_USE"
    validate_positive_int "CLONE_0_6B_REPLICAS" "$CLONE_0_6B_REPLICAS_TO_USE"
    validate_positive_int "CLONE_1_7B_REPLICAS" "$CLONE_1_7B_REPLICAS_TO_USE"
    validate_positive_int "VOICE_DESIGN_REPLICAS" "$VOICE_DESIGN_REPLICAS_TO_USE"
    validate_positive_int "MAX_CONNECTIONS" "$MAX_CONNECTIONS_TO_USE"
    case "$MAX_WAITING_SYNTH_REQUESTS_TO_USE" in
        ''|*[!0-9]*)
            err "MAX_WAITING_SYNTH_REQUESTS must be >= 0: ${MAX_WAITING_SYNTH_REQUESTS_TO_USE}"
            exit 1
            ;;
    esac
    local IMAGE_TAG
    IMAGE_TAG="$(compute_image_tag \
        "$REGISTRY" \
        "$DEFAULT_MODEL_SIZE_TO_USE" \
        "$BUILD_PROFILE_TO_USE" \
        "$MODEL_SIZES_TO_LOAD" \
        "$MODEL_ID_0_6B_TO_USE" \
        "$MODEL_ID_1_7B_TO_USE" \
        "$VOICE_DESIGN_ENABLED_TO_USE" \
        "$VOICE_DESIGN_MODEL_ID_TO_USE")"

    case ",${MODEL_SIZES_TO_LOAD}," in
        *",${DEFAULT_MODEL_SIZE_TO_USE},"*) ;;
        *)
            err "DEFAULT_MODEL_SIZE=${DEFAULT_MODEL_SIZE_TO_USE} must be included in MODEL_SIZES=${MODEL_SIZES_TO_LOAD}"
            exit 1
            ;;
    esac
    case ",${MODEL_SIZES_TO_LOAD}," in
        *",${INITIAL_CLONE_MODEL_SIZE_TO_USE},"*) ;;
        *)
            err "INITIAL_CLONE_MODEL_SIZE=${INITIAL_CLONE_MODEL_SIZE_TO_USE} must be included in MODEL_SIZES=${MODEL_SIZES_TO_LOAD}"
            exit 1
            ;;
    esac

    MODEL_SIZES="${MODEL_SIZES_TO_LOAD}"
    APP_PROFILE="${APP_PROFILE_TO_USE}"
    BUILD_PROFILE="${BUILD_PROFILE_TO_USE}"
    DEFAULT_MODEL_SIZE="${DEFAULT_MODEL_SIZE_TO_USE}"
    INITIAL_CLONE_MODEL_SIZE="${INITIAL_CLONE_MODEL_SIZE_TO_USE}"
    MODEL_ID_0_6B="${MODEL_ID_0_6B_TO_USE}"
    MODEL_ID_1_7B="${MODEL_ID_1_7B_TO_USE}"
    CLONE_0_6B_REPLICAS="${CLONE_0_6B_REPLICAS_TO_USE}"
    CLONE_1_7B_REPLICAS="${CLONE_1_7B_REPLICAS_TO_USE}"
    VOICE_DESIGN_ENABLED="${VOICE_DESIGN_ENABLED_TO_USE}"
    VOICE_DESIGN_MODEL_ID="${VOICE_DESIGN_MODEL_ID_TO_USE}"
    VOICE_DESIGN_REPLICAS="${VOICE_DESIGN_REPLICAS_TO_USE}"
    ALLOWED_ORIGINS="${ALLOWED_ORIGINS_TO_USE}"
    MAX_CONNECTIONS="${MAX_CONNECTIONS_TO_USE}"
    MAX_WAITING_SYNTH_REQUESTS="${MAX_WAITING_SYNTH_REQUESTS_TO_USE}"
    VOICE_STORAGE_DIR="${VOICE_STORAGE_DIR_TO_USE}"

    local DOCKER_ENV_ARGS
    DOCKER_ENV_ARGS="$(
        build_docker_env_args \
            APP_PROFILE \
            MODEL_SIZES \
            DEFAULT_MODEL_SIZE \
            INITIAL_MODE \
            INITIAL_CLONE_MODEL_SIZE \
            CLONE_0_6B_REPLICAS \
            CLONE_1_7B_REPLICAS \
            VOICE_DESIGN_REPLICAS \
            VOICE_DESIGN_ENABLED \
            VOICE_DESIGN_MODEL_ID \
            ALLOWED_ORIGINS \
            VOICE_STORAGE_DIR \
            SERVER_PORT \
            MODEL_ID_0_6B \
            MODEL_ID_1_7B \
            MODEL_DEVICE \
            MODEL_DTYPE \
            ATTN_IMPLEMENTATION \
            CUDA_GRAPH_MAX_SEQ_LEN \
            CHUNK_SIZE \
            MAX_CONNECTIONS \
            MAX_WAITING_SYNTH_REQUESTS \
            MAX_TEXT_LENGTH \
            CLONE_PROMPT_CACHE_SIZE
    )"

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       Ameego TTS — Deploying         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo "  Profile:  ${APP_PROFILE_TO_USE}"
    echo "  Build:    ${BUILD_PROFILE_TO_USE}"
    echo "  Default:  Qwen3-TTS-${DEFAULT_MODEL_SIZE_TO_USE}"
    echo "  Initial:  Qwen3-TTS-${INITIAL_CLONE_MODEL_SIZE_TO_USE}"
    echo "  Load:     ${MODEL_SIZES_TO_LOAD}"
    echo "  Voice Design: ${VOICE_DESIGN_ENABLED_TO_USE}"
    echo "  Machine:  ${MACHINE_TYPE}"
    echo "  Zone:     ${ZONE}"
    echo "  Spot:     ${SPOT_FLAG:-no}"
    echo ""

    # 1. Create Artifact Registry repo
    log "Ensuring Artifact Registry repository..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --quiet 2>/dev/null || true

    # 2. Build and push Docker image via Cloud Build
    if [ "$SKIP_BUILD_FLAG" = "true" ]; then
        log "Skipping image build and reusing: ${IMAGE_TAG}"
    else
        log "Building Docker image via Cloud Build (this may take 15-30 minutes)..."
        gcloud builds submit "$SCRIPT_DIR" \
            --config="${SCRIPT_DIR}/cloudbuild.yaml" \
            --substitutions="_BUILD_PROFILE=${BUILD_PROFILE_TO_USE},_MODEL_SIZES=${MODEL_SIZES_TO_LOAD//,/%2C},_MODEL_ID_0_6B=${MODEL_ID_0_6B_TO_USE},_MODEL_ID_1_7B=${MODEL_ID_1_7B_TO_USE},_VOICE_DESIGN_ENABLED=${VOICE_DESIGN_ENABLED_TO_USE},_VOICE_DESIGN_MODEL_ID=${VOICE_DESIGN_MODEL_ID_TO_USE},_IMAGE_TAG=${IMAGE_TAG}" \
            --quiet

        log "Image pushed: ${IMAGE_TAG}"
    fi

    # 3. Create firewall rule
    log "Ensuring firewall rule..."
    if gcloud compute firewall-rules describe "$FIREWALL_RULE" --quiet >/dev/null 2>&1; then
        gcloud compute firewall-rules update "$FIREWALL_RULE" \
            --allow=tcp:${SERVER_PORT} \
            --target-tags="$INSTANCE_NAME" \
            --quiet
    else
        gcloud compute firewall-rules create "$FIREWALL_RULE" \
            --allow=tcp:${SERVER_PORT} \
            --target-tags="$INSTANCE_NAME" \
            --description="Allow HTTP access to Ameego TTS" \
            --quiet
    fi

    # 4. Create GCE VM with GPU
    log "Creating GCE instance: ${INSTANCE_NAME} in ${ZONE}..."

    local CACHE_VOLUME_ARG=""
    if [ "$BUILD_PROFILE_TO_USE" = "fast" ]; then
        CACHE_VOLUME_ARG="-v ${HOST_HF_CACHE_DIR}:/root/.cache/huggingface"
    fi
    local VOICE_STORE_VOLUME_ARG="-v ${HOST_VOICE_STORE_DIR}:${VOICE_STORAGE_DIR_TO_USE}"

    local STARTUP_SCRIPT="#!/bin/bash
set -e
exec > /var/log/startup-script.log 2>&1

# Wait for NVIDIA drivers
echo 'Waiting for NVIDIA drivers...'
for i in \$(seq 1 60); do
    if nvidia-smi &>/dev/null; then break; fi
    sleep 5
done
nvidia-smi || { echo 'NVIDIA drivers not ready'; exit 1; }

# Install Docker
if ! command -v docker &>/dev/null; then
    echo 'Installing Docker...'
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo 'Docker installed'
fi

# Install NVIDIA Container Toolkit
if ! command -v nvidia-ctk &>/dev/null; then
    echo 'Installing NVIDIA Container Toolkit...'
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo 'NVIDIA Container Toolkit installed'
fi

# Auth to Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null || true

# Pull and run
echo 'Pulling image...'
docker pull ${IMAGE_TAG}
docker rm -f ameego-tts 2>/dev/null || true
mkdir -p ${HOST_HF_CACHE_DIR}
mkdir -p ${HOST_VOICE_STORE_DIR}
echo 'Starting container...'
docker run -d \
    --name ameego-tts \
    --gpus all \
    --restart unless-stopped \
    -p ${SERVER_PORT}:${SERVER_PORT} \
    ${CACHE_VOLUME_ARG} \
    ${VOICE_STORE_VOLUME_ARG} \
    ${DOCKER_ENV_ARGS} \
    ${IMAGE_TAG}

echo 'Ameego TTS container started'
"

    local STARTUP_SCRIPT_FILE
    STARTUP_SCRIPT_FILE="$(mktemp)"
    printf '%s\n' "$STARTUP_SCRIPT" > "$STARTUP_SCRIPT_FILE"

    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type=pd-ssd \
        --tags="$INSTANCE_NAME" \
        --scopes=cloud-platform \
        --maintenance-policy=TERMINATE \
        --metadata-from-file="startup-script=${STARTUP_SCRIPT_FILE}" \
        $SPOT_FLAG \
        --quiet

    rm -f "$STARTUP_SCRIPT_FILE"

    # Save state
    save_deploy_env

    # 5. Wait for health check
    log "Waiting for instance to be ready..."
    sleep 10

    local IP=""
    for i in $(seq 1 12); do
        IP=$(get_external_ip)
        if [ -n "$IP" ]; then break; fi
        sleep 5
    done

    if [ -z "$IP" ]; then
        err "Could not get external IP"
        exit 1
    fi

    log "External IP: ${IP}"
    log "Waiting for server to be healthy (model loading may take a few minutes)..."

    local HEALTHY=false
    for i in $(seq 1 120); do
        if curl -sf "http://${IP}:${SERVER_PORT}/health" &>/dev/null; then
            HEALTHY=true
            break
        fi
        printf "."
        sleep 10
    done
    echo ""

    if [ "$HEALTHY" = true ]; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║       Ameego TTS — Ready!            ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
        echo ""
        if [ "$APP_PROFILE_TO_USE" = "test" ]; then
            echo -e "  ${CYAN}Web UI:${NC}     http://${IP}:${SERVER_PORT}"
        fi
        echo -e "  ${CYAN}WebSocket:${NC}  ws://${IP}:${SERVER_PORT}/ws/tts"
        echo -e "  ${CYAN}Health:${NC}     http://${IP}:${SERVER_PORT}/health"
        echo ""
        echo -e "  ${CYAN}SSH:${NC}        ./deploy.sh ssh"
        echo -e "  ${CYAN}Logs:${NC}       ./deploy.sh logs"
        echo -e "  ${CYAN}Destroy:${NC}    ./deploy.sh down"
        echo ""
    else
        warn "Server not yet healthy. It may still be loading the model."
        echo "  Check status:  ./deploy.sh status"
        echo "  View logs:     ./deploy.sh logs"
        if [ "$APP_PROFILE_TO_USE" = "test" ]; then
            echo "  URL (when ready): http://${IP}:${SERVER_PORT}"
        else
            echo "  Health (when ready): http://${IP}:${SERVER_PORT}/health"
        fi
    fi
}

cmd_down() {
    load_deploy_env
    check_prerequisites

    ZONE="${ZONE:-$DEFAULT_ZONE}"

    echo ""
    echo -e "${YELLOW}╔══════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║       Ameego TTS — Destroying        ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════╝${NC}"
    echo ""

    log "Deleting instance: ${INSTANCE_NAME} in ${ZONE}..."
    gcloud compute instances delete "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --quiet 2>/dev/null || warn "Instance not found or already deleted"

    log "Deleting firewall rule: ${FIREWALL_RULE}..."
    gcloud compute firewall-rules delete "$FIREWALL_RULE" \
        --quiet 2>/dev/null || warn "Firewall rule not found"

    rm -f "$DEPLOY_ENV"
    log "Cleanup complete"
}

cmd_status() {
    load_deploy_env
    check_prerequisites

    ZONE="${ZONE:-$DEFAULT_ZONE}"

    echo ""
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format="table(name, status, networkInterfaces[0].accessConfigs[0].natIP, machineType.basename())" 2>/dev/null || {
        warn "Instance not found"
        return 1
    }

    local IP=$(get_external_ip)
    if [ -n "$IP" ]; then
        echo ""
        echo "  Profile:   ${APP_PROFILE:-$DEFAULT_APP_PROFILE}"
        echo "  Build:     ${BUILD_PROFILE:-$(default_build_profile_for_app_profile "${APP_PROFILE:-$DEFAULT_APP_PROFILE}")}"
        if curl -sf "http://${IP}:${SERVER_PORT}/health" 2>/dev/null; then
            echo ""
            log "Health: ${GREEN}OK${NC}"
        else
            warn "Health: NOT READY (model may still be loading)"
        fi
        echo ""
        if [ "${APP_PROFILE:-$DEFAULT_APP_PROFILE}" = "test" ]; then
            echo "  Web UI:    http://${IP}:${SERVER_PORT}"
        fi
        echo "  WebSocket: ws://${IP}:${SERVER_PORT}/ws/tts"
    fi
}

cmd_ssh() {
    load_deploy_env
    check_prerequisites

    ZONE="${ZONE:-$DEFAULT_ZONE}"
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE"
}

cmd_logs() {
    load_deploy_env
    check_prerequisites

    ZONE="${ZONE:-$DEFAULT_ZONE}"
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
        --command="docker logs --tail 100 -f ameego-tts 2>&1"
}

cmd_url() {
    load_deploy_env
    check_prerequisites
    ZONE="${ZONE:-$DEFAULT_ZONE}"

    local IP=$(get_external_ip)
    if [ -n "$IP" ]; then
        echo "http://${IP}:${SERVER_PORT}"
    else
        err "Could not get external IP. Is the instance running?"
        exit 1
    fi
}

# ---- Main ---------------------------------------------------

case "${1:-help}" in
    up)     shift; cmd_up "$@" ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    ssh)    cmd_ssh ;;
    logs)   cmd_logs ;;
    url)    cmd_url ;;
    *)
        echo "Usage: $0 {up|down|status|ssh|logs|url}"
        echo ""
        echo "Commands:"
        echo "  up [--model 0.6B|1.7B] [--profile test|api] [--build full|fast] [--spot] [--zone ZONE]  Deploy TTS server"
        echo "  down                                             Destroy server"
        echo "  status                                           Show server status"
        echo "  ssh                                              SSH into server"
        echo "  logs                                             View server logs"
        echo "  url                                              Print server URL"
        echo ""
        echo "Examples:"
        echo "  $0 up                                            Deploy test profile using default full image"
        echo "  $0 up --profile api                             Deploy API-only profile using default fast image"
        echo "  $0 up --profile api --build full                Deploy API-only profile with baked-in models"
        echo "  $0 up --model 1.7B --spot                       Deploy test profile defaulting to 1.7B on spot"
        echo "  MODEL_SIZES=0.6B $0 up                          Deploy only the 0.6B model"
        echo "  $0 up --zone us-west2-b                         Deploy in specific zone"
        exit 1
        ;;
esac
