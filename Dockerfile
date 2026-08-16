# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Person Tracking / Face Recognition Service

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

WORKDIR /app

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
    openssh-client \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# github.com must be a known host before pip can clone lum-model-vision over
# SSH, or the clone hangs on an interactive host-key prompt.
RUN mkdir -p -m 0700 /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.2
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install requirements. This pulls lum-model-vision from its own private repo
# (git+ssh, see requirements.txt) — build with `docker build --ssh default .`
# so an SSH agent with access to that repo is forwarded into this step. It
# brings boxmot from git and insightface from PyPI; no vendored forks needed.
COPY requirements.txt ./
RUN --mount=type=ssh pip install --no-cache-dir -r requirements.txt

# Force headless OpenCV — ultralytics pulls in opencv-python (full), replace it.
# Pinned to 4.11 deliberately: 4.12+ requires numpy>=2, which the boxmot fork's
# numpy==1.24.4 pin rules out. Keep in step with lum-model-vision's pyproject.
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless~=4.11.0

# Swap the CPU onnxruntime that lum-model-vision depends on for the GPU build. They
# share the same `onnxruntime/` install directory, so this is a replace, not an
# addition — which is why lum-model-vision has no `[gpu]` extra.
#
# Also upgrade cuDNN: onnxruntime-gpu 1.21 (CUDA 12.x) needs cuDNN >= ~9.7, but
# torch 2.4 pins nvidia-cudnn-cu12 9.1.0.70 — too old, so ORT's cuDNN-frontend
# Conv execute fails at runtime (CUDNN_BACKEND_API_FAILED). torch tolerates 9.8.
# See lum-model-vision/requirements/requirements.gpu.txt.
RUN pip uninstall -y onnxruntime 2>/dev/null || true && \
    pip install --no-cache-dir onnxruntime-gpu~=1.21.0 "nvidia-cudnn-cu12>=9.8,<10"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app/src:/app"

# onnxruntime-gpu loads cuDNN 9 from the pip nvidia-cudnn-cu12 package, whose lib
# dir is NOT on the default loader path — so ORT would otherwise fall back to the
# base image's system cuDNN 8 (or fail to resolve libcudnn.so.9). Prepend the pip
# cuDNN + cuBLAS dirs so ORT's CUDA EP finds cuDNN 9.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

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

# Copy config and source. lum-model-vision ships as a normal installed
# package now (see the builder stage) — nothing repo-local to copy for it.
COPY configs ./configs
COPY src ./src

ENTRYPOINT ["docker-entrypoint.sh"]
