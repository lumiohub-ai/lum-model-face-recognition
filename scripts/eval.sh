#!/bin/bash
# Face Recognition Evaluation Script
# Usage: ./eval.sh [options]

# Set default values
ALG_NAME="fps-calculation"
BENCHMARK="ilhan"
VIDEOS="videoa1-1 videoa1-2 videoa1-3 videoa1-4 videoa1-5"
VIDEO_DIR="data/tests/ilhan_videos"
DB_PATH="data/embeddings/ilhan.pkl"
OUTPUT_DIR="results/"
MATCH_THRESHOLD=0.3
SHOW_FLAG=""  # Empty by default, will add --show if needed

# Build the command with only recognized arguments
PYTHON_CMD="python3 tests/eval.py \
--alg_name \"$ALG_NAME\" \
--benchmark \"$BENCHMARK\" \
--videos $VIDEOS \
--video_dir \"$VIDEO_DIR\" \
--db_path \"$DB_PATH\" \
--output \"$OUTPUT_DIR\" \
--match_threshold $MATCH_THRESHOLD \
$SHOW_FLAG"

# Print and execute the command
echo "Running: $PYTHON_CMD"
eval $PYTHON_CMD