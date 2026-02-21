# syntax=docker/dockerfile:1
# Simple single-stage Dockerfile for Person Tracking / Face Recognition Service

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

WORKDIR /app

# Install system dependencies and Python
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
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.2 (specific version for stability)
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install local modules
COPY modules ./modules
RUN pip install ./modules/insightface
RUN pip install ./modules/yolo_tracking

# Copy entrypoint scripts
COPY scripts/docker/*.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

# Copy config and source
COPY configs ./configs
COPY src ./src

# Set PYTHONPATH
ENV PYTHONPATH="/app/src:/app"

# Download ReID model weights
RUN mkdir -p volumes/models/weights && \
    curl -fSL --retry 3 -o volumes/models/weights/osnet_x0_25_msmt17.pt \
    https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt

ENTRYPOINT ["docker-entrypoint.sh"]
