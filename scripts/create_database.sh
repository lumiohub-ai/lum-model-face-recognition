#!/bin/bash

# 🔧 Configuration
INPUT_DIR="data/images/hb-facecrops"
OUTPUT_FILE="data/embeddings/hb-kor-camera.pkl"
DEVICE="gpu"  # Change to "cpu" if needed

# 🚀 Run the script
echo "Generating face database..."
python tools/generate_face_database.py \
  --input_dir "$INPUT_DIR" \
  --output_file "$OUTPUT_FILE" \
  --device "$DEVICE"
