# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Person Tracking / Face Recognition Service

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

# Optional apt mirror, for hosts whose network corrupts sustained container
# downloads (apt reports it as "Hash Sum mismatch"). Point it at a local
# caching proxy to build there; unset = stock Ubuntu repos, so CI is unchanged.
#   --build-arg APT_MIRROR=http://172.17.0.1:8899
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
      printf 'deb %s/ubuntu jammy main restricted universe multiverse\ndeb %s/ubuntu jammy-updates main restricted universe multiverse\ndeb %s/ubuntu-security jammy-security main restricted universe multiverse\n' \
        "$APT_MIRROR" "$APT_MIRROR" "$APT_MIRROR" > /etc/apt/sources.list && \
      rm -f /etc/apt/sources.list.d/*.list; \
    fi

# Same idea for pip. ARG alone is not enough — pip reads these from the
# ENVIRONMENT, so promote them here. Empty = pip's own defaults.
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

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

# Install PyTorch with CUDA 12.2.
#
# vendor/wheels is an OPTIONAL local cache (gitignored, normally empty): drop
# the torch/torchvision wheels there and pip installs them instead of
# downloading ~770MB. Needed on hosts where pip's TLS dies mid-download on
# wheels this large (BAD_RECORD_MAC); curl the wheels first, then build.
# With the directory empty this is a normal download, so CI is unaffected.
#
# torch pulls a dozen nvidia-*-cu12 wheels plus triton as transitive deps
# (cublas ~410MB, cudnn ~665MB, triton ~210MB) — any one of them can hit the
# same BAD_RECORD_MAC mid-download on a host where this happens, even with
# the torch/torchvision wheels themselves vendored, since pip still resolves
# and fetches torch's own dependencies from the network. `--find-links`
# alone does NOT fix this: pip's resolver does not prefer a local match over
# an index candidate, so it still redownloaded cublas/curand/etc. from
# pypi.nvidia.com/download.pytorch.org even with them sitting in /wheels
# (verified directly — this is why the vendored branch below uses
# --no-index, not just --find-links). That means the full dependency closure
# must be vendored, not just the two largest files: every nvidia-*-cu12 pin
# from torch's own METADATA, triton, and the small pure-Python deps
# (sympy/jinja2/networkx/fsspec/filelock/typing-extensions/markupsafe/mpmath)
# that transitively need somewhere to resolve from once the index is off.
COPY vendor/wheels/ /wheels/
# TORCH_INDEX_URL: same mirror workaround as APT_MIRROR/PIP_INDEX_URL above —
# this line hits download.pytorch.org directly via --extra-index-url, which
# PIP_INDEX_URL does not override. Empty = pytorch.org, unaffected for CI.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN if ls /wheels/torch-*.whl >/dev/null 2>&1; then \
        echo "installing torch from vendor/wheels (no-index, full dep closure)" && \
        pip install --no-cache-dir --no-index --find-links /wheels \
            torch==2.4.0+cu121 torchvision==0.19.0+cu121; \
    else \
        pip install --no-cache-dir \
            torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
            --extra-index-url "$TORCH_INDEX_URL"; \
    fi

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
#
# Same vendor/wheels fallback as the torch install above, for the same
# BAD_RECORD_MAC reason — these are the next-largest downloads in this stage
# (onnxruntime-gpu ~280MB, nvidia-cudnn-cu12 ~750MB). Pinned to exact versions
# here (not the `~=`/`>=,<` ranges pip would otherwise resolve) so a vendored
# build and a from-PyPI build can't silently diverge on which version lands.
#
# The cudnn wheel is named explicitly, not globbed (nvidia_cudnn_cu12-*.whl)
# — the torch install step above also vendors nvidia-cudnn-cu12, pinned to
# torch's OWN older 9.1.0.70 requirement, so /wheels holds two different
# cudnn versions side by side. A glob here would hand pip both files at
# once for one `pip install` invocation, which is not what "install this one
# exact version" means.
RUN pip uninstall -y onnxruntime 2>/dev/null || true && \
    if ls /wheels/onnxruntime_gpu-*.whl >/dev/null 2>&1; then \
        echo "installing onnxruntime-gpu + nvidia-cudnn-cu12 from vendor/wheels" && \
        pip install --no-cache-dir \
            /wheels/onnxruntime_gpu-*.whl \
            /wheels/nvidia_cudnn_cu12-9.25.1.1-*.whl; \
    else \
        pip install --no-cache-dir onnxruntime-gpu==1.21.1 "nvidia-cudnn-cu12==9.25.1.1"; \
    fi

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# See the builder stage: same optional mirror, redeclared because ARGs do not
# cross stage boundaries.
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
      printf 'deb %s/ubuntu jammy main restricted universe multiverse\ndeb %s/ubuntu jammy-updates main restricted universe multiverse\ndeb %s/ubuntu-security jammy-security main restricted universe multiverse\n' \
        "$APT_MIRROR" "$APT_MIRROR" "$APT_MIRROR" > /etc/apt/sources.list && \
      rm -f /etc/apt/sources.list.d/*.list; \
    fi

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
