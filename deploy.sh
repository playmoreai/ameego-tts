#!/bin/bash
set -euo pipefail

# ============================================================
# Ameego TTS — GCE GPU Deploy Script
# Usage:
#   ./deploy.sh up [--model 0.6B|1.7B] [--spot] [--zone ZONE]
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
DEFAULT_MODEL="0.6B"
MACHINE_TYPE="g2-standard-4"
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="common-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
FIREWALL_RULE="ameego-tts-allow-http"
SERVER_PORT="8080"

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
ZONE=${ZONE}
MODEL_SIZE=${MODEL_SIZE}
PROJECT_ID=${PROJECT_ID}
INSTANCE_NAME=${INSTANCE_NAME}
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
    local MODEL_SIZE="$DEFAULT_MODEL"
    local ZONE="$DEFAULT_ZONE"
    local SPOT_FLAG=""

    # Parse args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)  MODEL_SIZE="$2"; shift 2 ;;
            --spot)   SPOT_FLAG="--provisioning-model=SPOT"; shift ;;
            --zone)   ZONE="$2"; shift 2 ;;
            *)        err "Unknown option: $1"; exit 1 ;;
        esac
    done

    check_prerequisites

    local REGION="${ZONE%-*}"
    local REPO_NAME="ameego-tts"
    local REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
    local IMAGE_TAG="${REGISTRY}/ameego-tts:${MODEL_SIZE}"

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       Ameego TTS — Deploying         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo "  Model:    Qwen3-TTS-${MODEL_SIZE}"
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
    log "Building Docker image via Cloud Build (this may take 15-30 minutes)..."
    gcloud builds submit "$SCRIPT_DIR" \
        --config="${SCRIPT_DIR}/cloudbuild.yaml" \
        --substitutions="_MODEL_SIZE=${MODEL_SIZE},_IMAGE_TAG=${IMAGE_TAG}" \
        --quiet

    log "Image pushed: ${IMAGE_TAG}"

    # 3. Create firewall rule
    log "Ensuring firewall rule..."
    gcloud compute firewall-rules create "$FIREWALL_RULE" \
        --allow=tcp:${SERVER_PORT} \
        --target-tags="$INSTANCE_NAME" \
        --description="Allow HTTP access to Ameego TTS" \
        --quiet 2>/dev/null || true

    # 4. Create GCE VM with GPU
    log "Creating GCE instance: ${INSTANCE_NAME} in ${ZONE}..."

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
echo 'Starting container...'
docker run -d \
    --name ameego-tts \
    --gpus all \
    --restart unless-stopped \
    -p ${SERVER_PORT}:${SERVER_PORT} \
    -e MODEL_SIZE=${MODEL_SIZE} \
    -e SERVER_PORT=${SERVER_PORT} \
    -e GPU_MEMORY_UTIL=0.85 \
    ${IMAGE_TAG}

echo 'Ameego TTS container started'
"

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
        --metadata="startup-script=${STARTUP_SCRIPT}" \
        $SPOT_FLAG \
        --quiet

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
        echo -e "  ${CYAN}Web UI:${NC}     http://${IP}:${SERVER_PORT}"
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
        echo "  URL (when ready): http://${IP}:${SERVER_PORT}"
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
        if curl -sf "http://${IP}:${SERVER_PORT}/health" 2>/dev/null; then
            echo ""
            log "Health: ${GREEN}OK${NC}"
        else
            warn "Health: NOT READY (model may still be loading)"
        fi
        echo ""
        echo "  Web UI:    http://${IP}:${SERVER_PORT}"
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
        --command="docker logs -f ameego-tts 2>&1 | tail -100"
}

cmd_url() {
    load_deploy_env
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
        echo "  up [--model 0.6B|1.7B] [--spot] [--zone ZONE]  Deploy TTS server"
        echo "  down                                             Destroy server"
        echo "  status                                           Show server status"
        echo "  ssh                                              SSH into server"
        echo "  logs                                             View server logs"
        echo "  url                                              Print server URL"
        echo ""
        echo "Examples:"
        echo "  $0 up                                            Deploy 0.6B model"
        echo "  $0 up --model 1.7B --spot                       Deploy 1.7B model on spot"
        echo "  $0 up --model 0.6B --zone us-west2-b            Deploy in specific zone"
        exit 1
        ;;
esac
