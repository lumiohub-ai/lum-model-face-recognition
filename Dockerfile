# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Person Tracking / Face Recognition Service

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

# Optional: override apt mirror when port-80 is blocked on the build network.
# Example: --build-arg APT_MIRROR=https://mirror.xtom.com.hk
ARG APT_MIRROR=""

WORKDIR /app

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://archive.ubuntu.com|${APT_MIRROR}|g; s|http://security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi

# Install Python + build tools (build-time only)
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.2
# Wheels are pre-downloaded on the host to avoid large unstable downloads inside Docker
COPY wheels/ ./wheels/
RUN if ls wheels/torch*.whl 1>/dev/null 2>&1; then \
        pip install --no-cache-dir --no-index --find-links=./wheels/ torch==2.4.0+cu121 torchvision==0.19.0+cu121; \
    else \
        pip install --no-cache-dir \
            torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
            --extra-index-url https://download.pytorch.org/whl/cu121; \
    fi && \
    rm -rf ./wheels/

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install local modules
COPY modules ./modules
RUN pip install --no-cache-dir ./modules/insightface && \
    pip install --no-cache-dir ./modules/yolo_tracking

# Force headless OpenCV — ultralytics pulls in opencv-python (full), replace it
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless~=4.12.0.88

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app/src:/app"

ARG APT_MIRROR=""

WORKDIR /app

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|http://archive.ubuntu.com|${APT_MIRROR}|g; s|http://security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi

# Install only runtime system libraries (no build tools)
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-setuptools \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy entrypoint scripts
COPY scripts/docker/*.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

# Copy config and source
COPY configs ./configs
COPY src ./src

ENTRYPOINT ["docker-entrypoint.sh"]
