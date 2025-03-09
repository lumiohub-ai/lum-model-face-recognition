#!/bin/bash

# Set arrays for multiple configurations
VIDEO_PATHS=(
    "videos/output_10_processed_fps.mp4"
    "videos/output_11_processed_fps.mp4"
    # Add more video paths as needed
)

VIDEO_NAMES=(
    "video10"
    "video11"
    # Add more video names as needed
)

MODEL_ARCHS=(
    "models/yolov8m-face.pt"
    "models/yolov8l-face.pt"
    # Add more model architectures as needed
)

CONF_THRESHOLDS=(
    0.25
    0.4
    # Add more confidence thresholds as needed
)

IMG_SIZES=(
    960
    1280
    # Add more image sizes as needed
)

# Fixed parameters
ALG_START=3  # Start with alg3
PERSIST=True
TRACKER="botsort.yaml"
MATCH_THRESHOLD=0.7

# Create a results directory and log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="batch_results_${TIMESTAMP}"
mkdir -p "$BATCH_DIR"
LOG_FILE="logs.txt"
> "$LOG_FILE"  # Initialize empty log file

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Generate configurations
CONFIGS=()
ALG_INDEX=$ALG_START

for MODEL_ARCH in "${MODEL_ARCHS[@]}"; do
    for CONF in "${CONF_THRESHOLDS[@]}"; do
        for IMG_SIZE in "${IMG_SIZES[@]}"; do
            ALG_NAME="alg$ALG_INDEX"
            CONFIG="MODEL: $MODEL_ARCH, CONF: $CONF, IMG_SIZE: $IMG_SIZE, ALG_NAME: $ALG_NAME"
            CONFIGS+=("$CONFIG")
            
            # Create mapping for algorithm parameters
            declare "ALG_${ALG_INDEX}_MODEL=$MODEL_ARCH"
            declare "ALG_${ALG_INDEX}_CONF=$CONF"
            declare "ALG_${ALG_INDEX}_IMGSIZE=$IMG_SIZE"
            
            ALG_INDEX=$((ALG_INDEX + 1))
        done
    done
done

log_message "======== STARTING BATCH PROCESSING ========"
log_message "Total configurations: $((ALG_INDEX - ALG_START))"
for i in "${!CONFIGS[@]}"; do
    log_message "Config $((i + 1)): ${CONFIGS[$i]}"
done
log_message "========================================"

# PHASE 1: Prediction phase for all videos with all configurations
log_message "\n======== PHASE 1: PREDICTION ========="

for ALG_NUM in $(seq $ALG_START $((ALG_INDEX - 1))); do
    ALG_NAME="alg$ALG_NUM"
    MODEL_ARCH=$(eval echo \$ALG_${ALG_NUM}_MODEL)
    CONF=$(eval echo \$ALG_${ALG_NUM}_CONF)
    IMG_SIZE=$(eval echo \$ALG_${ALG_NUM}_IMGSIZE)
    
    log_message "\n=== Processing configuration: $ALG_NAME ==="
    log_message "Model: $MODEL_ARCH"
    log_message "Confidence: $CONF"
    log_message "Image Size: $IMG_SIZE"
    
    # Process all videos for this configuration
    for i in "${!VIDEO_PATHS[@]}"; do
        VIDEO_NAME="${VIDEO_NAMES[$i]}"
        VIDEO_PATH="${VIDEO_PATHS[$i]}"
        
        log_message "\nPredicting $VIDEO_NAME with $ALG_NAME..."
        
        # Run prediction script and capture output
        PRED_OUTPUT=$(python src/face_recognition/evaluation/predict_mot.py \
            -v "$VIDEO_NAME" \
            -a "$ALG_NAME" \
            -p "$VIDEO_PATH" \
            -m "$MODEL_ARCH" \
            -c "$CONF" \
            -i "$IMG_SIZE" \
            -t "$TRACKER" 2>&1)
        
        # Log the output
        log_message "$PRED_OUTPUT"
    done
done

# PHASE 2: Evaluation phase for all configurations
log_message "\n======== PHASE 2: MOT EVALUATION ========="

for ALG_NUM in $(seq $ALG_START $((ALG_INDEX - 1))); do
    ALG_NAME="alg$ALG_NUM"
    
    log_message "\n=== Evaluating $ALG_NAME ==="
    
    # Run evaluation script and capture output
    EVAL_OUTPUT=$(python TrackEval/scripts/run_mot_challenge.py \
        --BENCHMARK hbface \
        --SPLIT_TO_EVAL train \
        --TRACKERS_TO_EVAL "$ALG_NAME" \
        --METRICS HOTA CLEAR Identity VACE \
        --USE_PARALLEL False \
        --NUM_PARALLEL_CORES 1 2>&1)
    
    # Log the output
    log_message "$EVAL_OUTPUT"
done

# PHASE 3: Recognition accuracy evaluation for all videos with all configurations
log_message "\n======== PHASE 3: RECOGNITION ACCURACY EVALUATION ========="

for ALG_NUM in $(seq $ALG_START $((ALG_INDEX - 1))); do
    ALG_NAME="alg$ALG_NUM"
    
    # Process all videos for this configuration
    for i in "${!VIDEO_PATHS[@]}"; do
        VIDEO_NAME="${VIDEO_NAMES[$i]}"
        
        log_message "\nEvaluating recognition accuracy for $VIDEO_NAME with $ALG_NAME..."
        
        # Run recognition accuracy evaluation script and capture output
        RECA_OUTPUT=$(python src/face_recognition/evaluation/reca.py \
            -v "$VIDEO_NAME" \
            -a "$ALG_NAME" 2>&1)
        
        # Log the output
        log_message "$RECA_OUTPUT"
    done
done

# Copy logs to batch directory
cp "$LOG_FILE" "$BATCH_DIR/"

# Print completion message
log_message "\n======== BATCH EVALUATION COMPLETED ========="
log_message "Results are available in the results and TrackEval directories."
log_message "Logs saved to $LOG_FILE"
log_message "Configuration summary:"

for ALG_NUM in $(seq $ALG_START $((ALG_INDEX - 1))); do
    ALG_NAME="alg$ALG_NUM"
    MODEL_ARCH=$(eval echo \$ALG_${ALG_NUM}_MODEL)
    CONF=$(eval echo \$ALG_${ALG_NUM}_CONF)
    IMG_SIZE=$(eval echo \$ALG_${ALG_NUM}_IMGSIZE)
    
    log_message "$ALG_NAME: Model=$MODEL_ARCH, Conf=$CONF, ImgSize=$IMG_SIZE"
done