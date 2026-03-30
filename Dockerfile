# Stage 1: Install Python dependencies
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    build-essential cython3 \
    sox libsox-fmt-all git wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
COPY vendor/faster-qwen3-tts /tmp/vendor/faster-qwen3-tts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.11.0 torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir /tmp/vendor/faster-qwen3-tts

# Stage 2: Download model weights
FROM builder AS model-downloader

ARG MODEL_SIZES=0.6B,1.7B
ARG MODEL_ID_0_6B=Qwen/Qwen3-TTS-12Hz-0.6B-Base
ARG MODEL_ID_1_7B=Qwen/Qwen3-TTS-12Hz-1.7B-Base

COPY scripts/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py \
    --model-sizes ${MODEL_SIZES} \
    --model-id-0-6b ${MODEL_ID_0_6B} \
    --model-id-1-7b ${MODEL_ID_1_7B} \
    --cache-dir /models

# Stage 3: Runtime
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-venv sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy model weights
COPY --from=model-downloader /models /root/.cache/huggingface
ENV HF_HOME="/root/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/root/.cache/huggingface"

# Copy application code
WORKDIR /app
COPY server/ /app/server/
COPY web/ /app/web/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
