#!/bin/bash
# Script to download InsightFace models offline

set -e

MODELS_DIR="./insightface_models"
MODEL_NAME="buffalo_l"

mkdir -p ${MODELS_DIR}

echo "Downloading InsightFace ${MODEL_NAME} model..."
echo "Note: This will download to ~/.insightface/models/"

# Download using Python
python3 -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from insightface.app import FaceAnalysis
import shutil
import os

# Download model
print('Initializing FaceAnalysis to download models...')
app = FaceAnalysis(name='${MODEL_NAME}', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)
print('Models downloaded successfully!')

# Copy to local directory
import pathlib
home = pathlib.Path.home()
model_src = home / '.insightface' / 'models' / '${MODEL_NAME}'
model_dst = '${MODELS_DIR}/${MODEL_NAME}'

if model_src.exists():
    print(f'Copying models from {model_src} to {model_dst}')
    if os.path.exists(model_dst):
        shutil.rmtree(model_dst)
    shutil.copytree(model_src, model_dst)
    print(f'Models copied to {model_dst}')
else:
    print(f'Warning: Model directory not found at {model_src}')
"

echo "Download complete! Models saved to ${MODELS_DIR}"
ls -lh ${MODELS_DIR}/${MODEL_NAME}/
