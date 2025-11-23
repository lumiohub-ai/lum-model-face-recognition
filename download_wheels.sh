#!/bin/bash
# Script to download PyTorch wheels offline for Docker build

set -e

WHEELS_DIR="./wheels"
PYTHON_VERSION="310"  # Python 3.10
CUDA_VERSION="cu121"  # CUDA 12.1 (compatible with CUDA 12.2)
PLATFORM="linux_x86_64"

echo "Downloading PyTorch wheels to ${WHEELS_DIR}..."
echo "Note: Using CUDA 12.1 builds (compatible with CUDA 12.2.2)"

# Download torch (latest available: 2.5.1) with all dependencies
pip download torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION} \
    --dest ${WHEELS_DIR}

# Download torchvision (compatible with torch 2.5.1) with all dependencies
pip download torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION} \
    --dest ${WHEELS_DIR}

echo "Download complete! Wheels saved to ${WHEELS_DIR}"
ls -lh ${WHEELS_DIR}
