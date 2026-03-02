#!/bin/bash
# Build face embeddings inside Docker without rebuilding the image.
# Mounts local volumes/src, scripts, and src so code changes are picked up immediately.
#
# Usage:
#   ./scripts/tools/docker_build_embeddings.sh [--input <dir>] [--output <path>] [--gpu <id>]
#
# Examples:
#   ./scripts/tools/docker_build_embeddings.sh
#   ./scripts/tools/docker_build_embeddings.sh --input volumes/src/images/chockpoint
#   ./scripts/tools/docker_build_embeddings.sh --input volumes/src/images/folder_1 --gpu -1

set -euo pipefail

INPUT="volumes/src/images/folder_1"
OUTPUT="volumes/src/embeddings/main.pkl"
GPU="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)  INPUT="$2";  shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --gpu)    GPU="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

docker run --rm --gpus all \
  --entrypoint python \
  -w /app/face-recognition \
  -v "$(pwd)/volumes/src:/app/face-recognition/volumes/src" \
  -v "$(pwd)/scripts:/app/face-recognition/scripts" \
  -v "$(pwd)/src:/app/face-recognition/src" \
  humblebeeintel/face-recognition \
  scripts/tools/build_embeddings.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --gpu "$GPU"
