#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ENV="${SCRIPT_DIR}/.deploy.env"

ENGINE_NAME="ameego-tts"
CONTAINER_NAME="ameego-tts"
REPO_NAME="ameego-tts"
DEFAULT_PROFILE="web"
DEFAULT_ZONE="asia-northeast3-a"
DEFAULT_MODEL="1.7B"
DEFAULT_BUILD_PROFILE_WEB="full"
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
SERVER_PORT="8080"
HOST_HF_CACHE_DIR="/var/lib/ameego-tts/hf-cache"
HOST_VOICE_STORE_DIR="/var/lib/ameego-tts/voice-store"
GATEWAY_TAG="ameego-gateway"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*" >&2; }

PROFILE="$DEFAULT_PROFILE"
PROFILE_TITLE="Web"
RUNTIME_APP_PROFILE="test"
INSTANCE_NAME="${ENGINE_NAME}-web"
FIREWALL_RULE="${ENGINE_NAME}-web-allow-http"
PROJECT_ID=""
ZONE="$DEFAULT_ZONE"

usage() {
    cat <<EOF
Usage: $0 {up|down|status|ssh|logs|url} [options]

Commands:
  up                       Deploy the selected profile
  down                     Destroy the selected profile
  status                   Show status for the selected profile
  ssh                      SSH into the selected profile VM
  logs                     View logs for the selected profile
  url                      Print the selected profile URL

Profile selection:
  --profile web|api        Select the deployment profile

Options for 'up':
  --model 0.6B|1.7B
  --build full|fast
  --spot
  --zone ZONE

Examples:
  $0 up --profile web
  $0 up --profile api --build fast
  $0 status --profile api
EOF
}

normalize_profile() {
    case "${1:-}" in
        web|api) echo "$1" ;;
        *)
            err "Invalid profile value: ${1:-}. Allowed: web, api"
            exit 1
            ;;
    esac
}

skip_health_check_enabled() {
    case "${SKIP_HEALTH_CHECK:-false}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

set_profile_context() {
    PROFILE="$(normalize_profile "${1:-$DEFAULT_PROFILE}")"
    case "$PROFILE" in
        web)
            PROFILE_TITLE="Web"
            RUNTIME_APP_PROFILE="test"
            INSTANCE_NAME="${ENGINE_NAME}-web"
            FIREWALL_RULE="${ENGINE_NAME}-web-allow-http"
            ;;
        api)
            PROFILE_TITLE="API"
            RUNTIME_APP_PROFILE="api"
            INSTANCE_NAME="${ENGINE_NAME}-api"
            FIREWALL_RULE="${ENGINE_NAME}-api-allow-http"
            ;;
    esac
}

load_deploy_env() {
    if [ -f "$DEPLOY_ENV" ]; then
        local line=""
        local key=""
        local value=""
        while IFS= read -r line || [ -n "$line" ]; do
            [ -z "$line" ] && continue
            case "$line" in
                \#*) continue ;;
            esac
            key="${line%%=*}"
            value="${line#*=}"
            case "$key" in
                DEPLOY_PROFILE|DEPLOY_ZONE|DEPLOY_INSTANCE_NAME|DEPLOY_FIREWALL_RULE|DEPLOY_SERVER_PORT|DEPLOY_MODEL_SIZE|DEPLOY_BUILD_PROFILE|DEPLOY_MODEL_SIZES|DEPLOY_DEFAULT_MODEL_SIZE|DEPLOY_INITIAL_MODEL_SIZE|DEPLOY_VOICE_DESIGN_ENABLED|DEPLOY_INTERNAL_IP|DEPLOY_INTERNAL_BASE_URL|DEPLOY_SCHEME|DEPLOY_WS_SCHEME)
                    printf -v "$key" '%s' "$value"
                    ;;
            esac
        done < "$DEPLOY_ENV"
    fi
}

write_deploy_env_value() {
    local key="$1"
    local value="$2"
    case "$value" in
        *$'\n'*|*$'\r'*)
            err "${key} cannot contain newlines"
            exit 1
            ;;
    esac
    printf '%s=%s\n' "$key" "$value"
}

save_deploy_env() {
    {
        write_deploy_env_value "DEPLOY_PROFILE" "${PROFILE}"
        write_deploy_env_value "DEPLOY_ZONE" "${ZONE}"
        write_deploy_env_value "DEPLOY_INSTANCE_NAME" "${INSTANCE_NAME}"
        write_deploy_env_value "DEPLOY_FIREWALL_RULE" "${FIREWALL_RULE}"
        write_deploy_env_value "DEPLOY_SERVER_PORT" "${SERVER_PORT}"
        write_deploy_env_value "DEPLOY_INTERNAL_IP" "${DEPLOY_INTERNAL_IP}"
        write_deploy_env_value "DEPLOY_INTERNAL_BASE_URL" "http://${DEPLOY_INTERNAL_IP}:${SERVER_PORT}"
        write_deploy_env_value "DEPLOY_SCHEME" "http"
        write_deploy_env_value "DEPLOY_WS_SCHEME" "ws"
        write_deploy_env_value "DEPLOY_MODEL_SIZE" "${DEPLOY_MODEL_SIZE}"
        write_deploy_env_value "DEPLOY_BUILD_PROFILE" "${DEPLOY_BUILD_PROFILE}"
        write_deploy_env_value "DEPLOY_MODEL_SIZES" "${MODEL_SIZES}"
        write_deploy_env_value "DEPLOY_DEFAULT_MODEL_SIZE" "${DEFAULT_MODEL_SIZE}"
        write_deploy_env_value "DEPLOY_INITIAL_MODEL_SIZE" "${INITIAL_CLONE_MODEL_SIZE}"
        write_deploy_env_value "DEPLOY_VOICE_DESIGN_ENABLED" "${VOICE_DESIGN_ENABLED}"
    } > "$DEPLOY_ENV"
    chmod 600 "$DEPLOY_ENV"
}

resolve_profile() {
    local requested_profile="${1:-}"
    load_deploy_env
    if [ -n "$requested_profile" ]; then
        set_profile_context "$requested_profile"
        return
    fi
    if [ -n "${DEPLOY_PROFILE:-}" ]; then
        set_profile_context "$DEPLOY_PROFILE"
        return
    fi
    set_profile_context "$DEFAULT_PROFILE"
}

get_project_id() {
    PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
    if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
        err "No GCP project set. Run: gcloud config set project <PROJECT_ID>"
        exit 1
    fi
}

get_external_ip() {
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null
}

get_internal_ip() {
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format='get(networkInterfaces[0].networkIP)' 2>/dev/null
}

find_live_zone() {
    gcloud compute instances list \
        --filter="name=${INSTANCE_NAME}" \
        --format='value(zone)' 2>/dev/null \
    | head -n1
}

prepare_instance_context() {
    local requested_profile="${1:-}"
    resolve_profile "$requested_profile"
    check_prerequisites

    if [ -n "${DEPLOY_ZONE:-}" ] && [ "${DEPLOY_PROFILE:-}" = "$PROFILE" ]; then
        ZONE="${DEPLOY_ZONE}"
    else
        ZONE="$(find_live_zone || true)"
        ZONE="${ZONE:-$DEFAULT_ZONE}"
    fi
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

default_build_profile_for_profile() {
    case "$1" in
        web) echo "$DEFAULT_BUILD_PROFILE_WEB" ;;
        api) echo "$DEFAULT_BUILD_PROFILE_API" ;;
        *)
            err "Unsupported profile for build default: $1"
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
    log "Project: ${PROJECT_ID}" >&2
}

ensure_private_egress() {
    local region="$1"
    local router="ameego-private-egress-${region}"
    local nat="${router}-nat"
    local attempt=""
    local updated="false"

    log "Ensuring private egress in ${region}..." >&2
    for attempt in 1 2 3; do
        if gcloud compute networks subnets update default \
            --region="$region" \
            --enable-private-ip-google-access \
            --quiet >/dev/null 2>&1; then
            updated="true"
            break
        fi
        sleep 5
    done
    if [ "$updated" != "true" ]; then
        err "Failed to enable Private Google Access for region ${region}"
        exit 1
    fi

    if ! gcloud compute routers describe "$router" --region="$region" >/dev/null 2>&1; then
        gcloud compute routers create "$router" \
            --network=default \
            --region="$region" \
            --quiet >/dev/null
    fi

    if ! gcloud compute routers nats describe "$nat" --router="$router" --region="$region" >/dev/null 2>&1; then
        gcloud compute routers nats create "$nat" \
            --router="$router" \
            --region="$region" \
            --nat-all-subnet-ip-ranges \
            --auto-allocate-nat-external-ips \
            --quiet >/dev/null
    fi
}

other_profile_instance_exists() {
    local other=""
    other="$(
        gcloud compute instances list \
            --filter='name~"^ameego-tts-(web|api)$"' \
            --format='value(name)' 2>/dev/null \
        | awk -v current="$INSTANCE_NAME" '$0 != current { print; exit }'
    )"
    [ -n "$other" ]
}

ssh_to_instance() {
    local remote_command="${1:-}"
    local -a args=("$INSTANCE_NAME" "--zone=$ZONE")
    if [ "$PROFILE" = "api" ]; then
        args+=("--tunnel-through-iap")
    fi
    if [ -n "$remote_command" ]; then
        args+=("--command=$remote_command")
    fi
    gcloud compute ssh "${args[@]}"
}

wait_for_profile_ip() {
    local ip=""
    local fetch_cmd="get_external_ip"
    if [ "$PROFILE" = "api" ]; then
        fetch_cmd="get_internal_ip"
    fi
    for _ in $(seq 1 12); do
        ip="$($fetch_cmd)"
        if [ -n "$ip" ]; then
            printf '%s\n' "$ip"
            return 0
        fi
        sleep 5
    done
    return 1
}

wait_for_health() {
    local ip="$1"
    for _ in $(seq 1 120); do
        if [ "$PROFILE" = "api" ]; then
            if ssh_to_instance "curl -fsS http://127.0.0.1:${SERVER_PORT}/health >/dev/null" >/dev/null 2>&1; then
                return 0
            fi
        else
            if curl -sf "http://${ip}:${SERVER_PORT}/health" >/dev/null 2>&1; then
                return 0
            fi
        fi
        printf "."
        sleep 10
    done
    echo ""
    return 1
}

maybe_register_gateway_backend() {
    local internal_ip="$1"
    [ "$PROFILE" = "api" ] || return 0
    [ -n "${GATEWAY_ADMIN_URL:-}" ] || return 0
    [ -n "${GATEWAY_ADMIN_TOKEN:-}" ] || return 0
    local payload
    payload=$(printf '{"service":"tts","profile":"api","internal_ip":"%s","port":%s,"scheme":"http","ws_scheme":"ws"}' "$internal_ip" "$SERVER_PORT")
    local -a curl_cmd=(curl -fsS --connect-timeout 3 --max-time 10)
    if [ -n "${GATEWAY_ADMIN_DOMAIN:-}" ] && [ -n "${GATEWAY_ADMIN_IP:-}" ]; then
        curl_cmd+=(--resolve "${GATEWAY_ADMIN_DOMAIN}:443:${GATEWAY_ADMIN_IP}")
    fi
    "${curl_cmd[@]}" -X POST "${GATEWAY_ADMIN_URL%/}/admin/backends/register" \
        -H "Authorization: Bearer ${GATEWAY_ADMIN_TOKEN}" \
        -H "content-type: application/json" \
        -d "$payload" >/dev/null 2>&1 || warn "Gateway registration failed for tts"
}

maybe_deregister_gateway_backend() {
    [ "$PROFILE" = "api" ] || return 0
    [ -n "${GATEWAY_ADMIN_URL:-}" ] || return 0
    [ -n "${GATEWAY_ADMIN_TOKEN:-}" ] || return 0
    local -a curl_cmd=(curl -fsS --connect-timeout 3 --max-time 10)
    if [ -n "${GATEWAY_ADMIN_DOMAIN:-}" ] && [ -n "${GATEWAY_ADMIN_IP:-}" ]; then
        curl_cmd+=(--resolve "${GATEWAY_ADMIN_DOMAIN}:443:${GATEWAY_ADMIN_IP}")
    fi
    "${curl_cmd[@]}" -X POST "${GATEWAY_ADMIN_URL%/}/admin/backends/deregister" \
        -H "Authorization: Bearer ${GATEWAY_ADMIN_TOKEN}" \
        -H "content-type: application/json" \
        -d '{"service":"tts","profile":"api"}' >/dev/null 2>&1 || warn "Gateway deregistration failed for tts"
}

print_access_info() {
    local ip="$1"
    if [ "$PROFILE" = "web" ]; then
        echo -e "  ${CYAN}Web UI:${NC}     http://${ip}:${SERVER_PORT}"
    else
        echo -e "  ${CYAN}API Base:${NC}   http://${ip}:${SERVER_PORT} (internal)"
    fi
    echo -e "  ${CYAN}WebSocket:${NC}  ws://${ip}:${SERVER_PORT}/ws/tts"
    echo -e "  ${CYAN}Health:${NC}     http://${ip}:${SERVER_PORT}/health"
}

cmd_up() {
    local requested_profile=""
    local model_size="${MODEL_SIZE:-}"
    local build_profile_to_use="${BUILD_PROFILE:-}"
    local requested_zone="$DEFAULT_ZONE"
    local spot_flag=""
    local skip_build_flag
    skip_build_flag="$(normalize_bool "${SKIP_BUILD:-false}")"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) requested_profile="$2"; shift 2 ;;
            --model) model_size="$2"; shift 2 ;;
            --build) build_profile_to_use="$2"; shift 2 ;;
            --spot) spot_flag="--provisioning-model=SPOT"; shift ;;
            --zone) requested_zone="$2"; shift 2 ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [ -z "$requested_profile" ]; then
        err "'up' requires --profile web|api"
        exit 1
    fi

    set_profile_context "$requested_profile"
    ZONE="$requested_zone"
    check_prerequisites

    local region="${ZONE%-*}"
    if [ "$PROFILE" = "api" ]; then
        ensure_private_egress "$region"
    fi

    if other_profile_instance_exists; then
        err "Another TTS profile is already deployed. Run './deploy.sh down --profile web' or './deploy.sh down --profile api' first."
        exit 1
    fi

    if [ -z "$model_size" ]; then
        model_size="$DEFAULT_MODEL"
    fi
    if [ -z "$build_profile_to_use" ]; then
        build_profile_to_use="$(default_build_profile_for_profile "$PROFILE")"
    fi
    build_profile_to_use="$(normalize_build_profile "$build_profile_to_use")"

    local registry="${region}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
    local model_sizes_to_load=""
    if [ "${MODEL_SIZES+x}" = "x" ] && [ -n "${MODEL_SIZES}" ]; then
        model_sizes_to_load="${MODEL_SIZES}"
    elif [ "$PROFILE" = "api" ]; then
        model_sizes_to_load="${model_size}"
    else
        model_sizes_to_load="${DEFAULT_MODEL_SIZES}"
    fi
    local default_model_size_to_use="${DEFAULT_MODEL_SIZE:-$model_size}"
    local initial_clone_model_size_to_use="${INITIAL_CLONE_MODEL_SIZE:-$default_model_size_to_use}"
    local model_id_0_6b_to_use="${MODEL_ID_0_6B:-$DEFAULT_MODEL_ID_0_6B}"
    local model_id_1_7b_to_use="${MODEL_ID_1_7B:-$DEFAULT_MODEL_ID_1_7B}"
    local clone_0_6b_replicas_to_use="${CLONE_0_6B_REPLICAS:-$DEFAULT_CLONE_0_6B_REPLICAS}"
    local clone_1_7b_replicas_to_use="${CLONE_1_7B_REPLICAS:-$DEFAULT_CLONE_1_7B_REPLICAS}"
    local voice_design_enabled_to_use
    voice_design_enabled_to_use="$(normalize_bool "${VOICE_DESIGN_ENABLED:-$DEFAULT_VOICE_DESIGN_ENABLED}")"
    local voice_design_model_id_to_use="${VOICE_DESIGN_MODEL_ID:-$DEFAULT_VOICE_DESIGN_MODEL_ID}"
    local voice_design_replicas_to_use="${VOICE_DESIGN_REPLICAS:-$DEFAULT_VOICE_DESIGN_REPLICAS}"
    local allowed_origins_to_use="${ALLOWED_ORIGINS:-$DEFAULT_ALLOWED_ORIGINS}"
    local max_connections_to_use="${MAX_CONNECTIONS:-$DEFAULT_MAX_CONNECTIONS}"
    local max_waiting_synth_requests_to_use="${MAX_WAITING_SYNTH_REQUESTS:-$DEFAULT_MAX_WAITING_SYNTH_REQUESTS}"
    local voice_storage_dir_to_use="${VOICE_STORAGE_DIR:-/data/voices}"
    model_sizes_to_load="$(normalize_model_sizes_csv "$model_sizes_to_load")"
    validate_model_id "MODEL_ID_0_6B" "$model_id_0_6b_to_use"
    validate_model_id "MODEL_ID_1_7B" "$model_id_1_7b_to_use"
    validate_model_id "VOICE_DESIGN_MODEL_ID" "$voice_design_model_id_to_use"
    validate_positive_int "CLONE_0_6B_REPLICAS" "$clone_0_6b_replicas_to_use"
    validate_positive_int "CLONE_1_7B_REPLICAS" "$clone_1_7b_replicas_to_use"
    validate_positive_int "VOICE_DESIGN_REPLICAS" "$voice_design_replicas_to_use"
    validate_positive_int "MAX_CONNECTIONS" "$max_connections_to_use"
    case "$max_waiting_synth_requests_to_use" in
        ''|*[!0-9]*)
            err "MAX_WAITING_SYNTH_REQUESTS must be >= 0: ${max_waiting_synth_requests_to_use}"
            exit 1
            ;;
    esac
    local image_tag
    image_tag="$(compute_image_tag \
        "$registry" \
        "$default_model_size_to_use" \
        "$build_profile_to_use" \
        "$model_sizes_to_load" \
        "$model_id_0_6b_to_use" \
        "$model_id_1_7b_to_use" \
        "$voice_design_enabled_to_use" \
        "$voice_design_model_id_to_use")"

    case ",${model_sizes_to_load}," in
        *",${default_model_size_to_use},"*) ;;
        *)
            err "DEFAULT_MODEL_SIZE=${default_model_size_to_use} must be included in MODEL_SIZES=${model_sizes_to_load}"
            exit 1
            ;;
    esac
    case ",${model_sizes_to_load}," in
        *",${initial_clone_model_size_to_use},"*) ;;
        *)
            err "INITIAL_CLONE_MODEL_SIZE=${initial_clone_model_size_to_use} must be included in MODEL_SIZES=${model_sizes_to_load}"
            exit 1
            ;;
    esac

    MODEL_SIZES="${model_sizes_to_load}"
    APP_PROFILE="${RUNTIME_APP_PROFILE}"
    BUILD_PROFILE="${build_profile_to_use}"
    DEFAULT_MODEL_SIZE="${default_model_size_to_use}"
    INITIAL_CLONE_MODEL_SIZE="${initial_clone_model_size_to_use}"
    MODEL_ID_0_6B="${model_id_0_6b_to_use}"
    MODEL_ID_1_7B="${model_id_1_7b_to_use}"
    CLONE_0_6B_REPLICAS="${clone_0_6b_replicas_to_use}"
    CLONE_1_7B_REPLICAS="${clone_1_7b_replicas_to_use}"
    VOICE_DESIGN_ENABLED="${voice_design_enabled_to_use}"
    VOICE_DESIGN_MODEL_ID="${voice_design_model_id_to_use}"
    VOICE_DESIGN_REPLICAS="${voice_design_replicas_to_use}"
    ALLOWED_ORIGINS="${allowed_origins_to_use}"
    MAX_CONNECTIONS="${max_connections_to_use}"
    MAX_WAITING_SYNTH_REQUESTS="${max_waiting_synth_requests_to_use}"
    VOICE_STORAGE_DIR="${voice_storage_dir_to_use}"
    DEPLOY_MODEL_SIZE="${model_size}"
    DEPLOY_BUILD_PROFILE="${build_profile_to_use}"

    local docker_env_args
    docker_env_args="$(
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
    echo "  Profile:  ${PROFILE}"
    echo "  Build:    ${build_profile_to_use}"
    echo "  Default:  Qwen3-TTS-${default_model_size_to_use}"
    echo "  Initial:  Qwen3-TTS-${initial_clone_model_size_to_use}"
    echo "  Load:     ${model_sizes_to_load}"
    echo "  Voice Design: ${voice_design_enabled_to_use}"
    echo "  Machine:  ${MACHINE_TYPE}"
    echo "  Zone:     ${ZONE}"
    echo "  Spot:     ${spot_flag:-no}"
    echo ""

    log "Ensuring Artifact Registry repository..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$region" \
        --quiet 2>/dev/null || true

    if [ "$skip_build_flag" = "true" ]; then
        log "Skipping image build and reusing: ${image_tag}"
    else
        log "Building Docker image via Cloud Build (this may take 15-30 minutes)..."
        gcloud builds submit "$SCRIPT_DIR" \
            --config="${SCRIPT_DIR}/cloudbuild.yaml" \
            --substitutions="_BUILD_PROFILE=${build_profile_to_use},_MODEL_SIZES=${model_sizes_to_load//,/%2C},_MODEL_ID_0_6B=${model_id_0_6b_to_use},_MODEL_ID_1_7B=${model_id_1_7b_to_use},_VOICE_DESIGN_ENABLED=${voice_design_enabled_to_use},_VOICE_DESIGN_MODEL_ID=${voice_design_model_id_to_use},_IMAGE_TAG=${image_tag}" \
            --quiet
        log "Image pushed: ${image_tag}"
    fi

    log "Ensuring firewall rule..."
    gcloud compute firewall-rules delete "$FIREWALL_RULE" --quiet >/dev/null 2>&1 || true
    if [ "$PROFILE" = "api" ]; then
        gcloud compute firewall-rules create "$FIREWALL_RULE" \
            --allow=tcp:${SERVER_PORT} \
            --target-tags="$INSTANCE_NAME" \
            --source-tags="$GATEWAY_TAG" \
            --description="Allow internal gateway access to Ameego TTS (${PROFILE})" \
            --quiet
    else
        gcloud compute firewall-rules create "$FIREWALL_RULE" \
            --allow=tcp:${SERVER_PORT} \
            --target-tags="$INSTANCE_NAME" \
            --description="Allow HTTP access to Ameego TTS (${PROFILE})" \
            --quiet
    fi

    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
        err "Instance '${INSTANCE_NAME}' already exists in ${ZONE}. Run './deploy.sh down --profile ${PROFILE}' first."
        exit 1
    fi

    log "Creating GCE instance: ${INSTANCE_NAME} in ${ZONE}..."

    local cache_volume_arg=""
    if [ "$build_profile_to_use" = "fast" ]; then
        cache_volume_arg="-v ${HOST_HF_CACHE_DIR}:/root/.cache/huggingface"
    fi
    local voice_store_volume_arg="-v ${HOST_VOICE_STORE_DIR}:${voice_storage_dir_to_use}"

    local startup_script="#!/bin/bash
set -e
exec > /var/log/startup-script.log 2>&1

echo 'Waiting for NVIDIA drivers...'
for i in \$(seq 1 60); do
    if nvidia-smi &>/dev/null; then break; fi
    sleep 5
done
nvidia-smi || { echo 'NVIDIA drivers not ready'; exit 1; }

if ! command -v docker &>/dev/null; then
    echo 'Installing Docker...'
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    echo 'Docker installed'
fi

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

ACCESS_TOKEN=\$(curl -fsSL -H \"Metadata-Flavor: Google\" \
    \"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token\" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)['\''access_token'\''])')
docker login -u oauth2accesstoken -p "\${ACCESS_TOKEN}" https://${region}-docker.pkg.dev

echo 'Pulling image...'
docker pull ${image_tag}
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
mkdir -p ${HOST_HF_CACHE_DIR}
mkdir -p ${HOST_VOICE_STORE_DIR}
echo 'Starting container...'
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --restart unless-stopped \
    -p ${SERVER_PORT}:${SERVER_PORT} \
    ${cache_volume_arg} \
    ${voice_store_volume_arg} \
    ${docker_env_args} \
    ${image_tag}

echo 'Ameego TTS container started'
"

    local startup_script_file
    startup_script_file="$(mktemp)"
    printf '%s\n' "$startup_script" > "$startup_script_file"

    local -a create_args=(
        --zone="$ZONE"
        --machine-type="$MACHINE_TYPE"
        --image-family="$IMAGE_FAMILY"
        --image-project="$IMAGE_PROJECT"
        --boot-disk-size="$BOOT_DISK_SIZE"
        --boot-disk-type=pd-ssd
        --tags="$INSTANCE_NAME"
        --scopes=cloud-platform
        --maintenance-policy=TERMINATE
        --metadata-from-file="startup-script=${startup_script_file}"
        --quiet
    )
    if [ "$PROFILE" = "api" ]; then
        create_args+=(--no-address)
    fi
    if [ -n "$spot_flag" ]; then
        create_args+=("$spot_flag")
    fi
    if ! gcloud compute instances create "$INSTANCE_NAME" "${create_args[@]}"; then
        rm -f "$startup_script_file"
        exit 1
    fi

    rm -f "$startup_script_file"

    log "Waiting for instance to be ready..."
    sleep 10

    local ip=""
    ip="$(wait_for_profile_ip || true)"
    DEPLOY_INTERNAL_IP="$(get_internal_ip || true)"
    if [ -z "$DEPLOY_INTERNAL_IP" ]; then
        DEPLOY_INTERNAL_IP="$ip"
    fi
    save_deploy_env

    if [ -z "$ip" ]; then
        err "Could not determine instance IP"
        exit 1
    fi

    maybe_register_gateway_backend "$DEPLOY_INTERNAL_IP"

    if [ "$PROFILE" = "api" ]; then
        log "Internal IP: ${ip}"
    else
        log "External IP: ${ip}"
    fi
    if skip_health_check_enabled; then
        warn "Skipping health wait because SKIP_HEALTH_CHECK=true"
        print_access_info "$ip"
        return 0
    fi
    log "Waiting for server to be healthy (model loading may take a few minutes)..."

    if wait_for_health "$ip"; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║       Ameego TTS — Ready!            ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
        echo ""
        print_access_info "$ip"
        echo ""
        echo -e "  ${CYAN}SSH:${NC}        ./deploy.sh ssh --profile ${PROFILE}"
        echo -e "  ${CYAN}Logs:${NC}       ./deploy.sh logs --profile ${PROFILE}"
        echo -e "  ${CYAN}Destroy:${NC}    ./deploy.sh down --profile ${PROFILE}"
        echo ""
    else
        warn "Server not yet healthy. It may still be loading the model."
        echo "  Check status:  ./deploy.sh status --profile ${PROFILE}"
        echo "  View logs:     ./deploy.sh logs --profile ${PROFILE}"
        if [ "$PROFILE" = "web" ]; then
            echo "  URL (when ready): http://${ip}:${SERVER_PORT}"
        else
            echo "  Health (when ready): http://${ip}:${SERVER_PORT}/health"
        fi
    fi
}

cmd_down() {
    local requested_profile=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) requested_profile="$2"; shift 2 ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    prepare_instance_context "$requested_profile"
    maybe_deregister_gateway_backend

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

    if [ "${DEPLOY_PROFILE:-}" = "$PROFILE" ]; then
        rm -f "$DEPLOY_ENV"
    fi
    log "Cleanup complete"
}

cmd_status() {
    local requested_profile=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) requested_profile="$2"; shift 2 ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    prepare_instance_context "$requested_profile"

    echo ""
    gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --format="table(name, status, networkInterfaces[0].networkIP, networkInterfaces[0].accessConfigs[0].natIP, machineType.basename())" 2>/dev/null || {
        warn "Instance not found"
        return 1
    }

    local ip=""
    if [ "$PROFILE" = "api" ]; then
        ip="${DEPLOY_INTERNAL_IP:-}"
        [ -n "$ip" ] || ip="$(get_internal_ip)"
    else
        ip="$(get_external_ip)"
    fi
    if [ -n "$ip" ]; then
        echo ""
        echo "  Profile:   ${PROFILE}"
        echo "  Build:     ${DEPLOY_BUILD_PROFILE:-$(default_build_profile_for_profile "$PROFILE")}"
        if wait_for_health "$ip" >/dev/null 2>&1; then
            echo ""
            log "Health: ${GREEN}OK${NC}"
        else
            warn "Health: NOT READY (model may still be loading)"
        fi
        echo ""
        print_access_info "$ip"
    fi
}

cmd_ssh() {
    local requested_profile=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) requested_profile="$2"; shift 2 ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    prepare_instance_context "$requested_profile"
    ssh_to_instance
}

cmd_logs() {
    local requested_profile=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) requested_profile="$2"; shift 2 ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    prepare_instance_context "$requested_profile"
    ssh_to_instance "docker logs --tail 100 -f ${CONTAINER_NAME} 2>&1"
}

cmd_url() {
    local requested_profile=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) requested_profile="$2"; shift 2 ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    prepare_instance_context "$requested_profile"

    local ip=""
    if [ "$PROFILE" = "api" ]; then
        ip="${DEPLOY_INTERNAL_IP:-}"
        [ -n "$ip" ] || ip="$(get_internal_ip)"
    else
        ip="$(get_external_ip)"
    fi
    if [ -n "$ip" ]; then
        echo "http://${ip}:${SERVER_PORT}"
    else
        err "Could not get external IP. Is the instance running?"
        exit 1
    fi
}

case "${1:-help}" in
    up)
        shift
        cmd_up "$@"
        ;;
    down)
        shift
        cmd_down "$@"
        ;;
    status)
        shift
        cmd_status "$@"
        ;;
    ssh)
        shift
        cmd_ssh "$@"
        ;;
    logs)
        shift
        cmd_logs "$@"
        ;;
    url)
        shift
        cmd_url "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac
