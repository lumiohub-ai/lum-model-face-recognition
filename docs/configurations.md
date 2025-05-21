 # Face Recognition Configuration Guide

This document provides a detailed explanation of all configuration parameters available in the face recognition system.

[**`configs/config.yaml`**](https://github.com/humblebeeintel/face-recognition/blob/main/configs/config.yaml):

```yaml
# Device settings
gpu_id: 0  # Use GPU (CUDA) or switch to "cpu" if GPU is not available

# Database settings
db_path: "data/embeddings/hb-kor.pkl" # Must be in pkl format
video_path: "data/videos/sample.mp4"  # Path to the input video file or RTSP stream

# Face recognition settings
match_threshold: 0.5  # Threshold for face similarity matching

# Other settings
show: True  # Show the output video with annotations
save_video: True  # Save the output video with annotations
record_always: True  # Always record the video

# Evaluation settings
eval: False  # Enable evaluation mode
txt_path: "results/results.txt"  # Path to save the recorded 

# ROI settings
roi: null  # Enable region of interest (ROI) mode
line_points: null # Define the ROI line points (e.g., [(0, 0), (1280, 720)])

# Other settings
timezone: "Asia/Seoul"  # Timezone for tracking times
debug: True  # Enable debug mode for detailed logging
log_file: "data/logs/debug.log"  # Path to save the debug log file              
```

## Device Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_id` | integer | 0 | Specifies which GPU to use for processing. Must have names and embeddings inside it with a format of lists. |

## Database and Input Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | "data/embeddings/hb-kor.pkl" | Path to the face embeddings database file (must be in PKL format) |
| `video_path` | string | "data/videos/sample.mp4" | Path to the input video file or RTSP stream URL. If you want to use multi camera, you can give it like lists. For example `video_path=[in_camera, out_camera],` but you must define cam_types and set ``multi_camera=True``  |

## Face Recognition Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `match_threshold` | float | 0.5 | Threshold for face similarity matching (0.0 to 1.0). Larger values are more strict. |

## Video Processing Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show` | boolean | True | Enable/disable real-time video display with annotations |
| `save_video` | boolean | True | Enable/disable saving the processed video with annotations |
| `record_always` | boolean | True | Enable/disable continuous video recording |

## Evaluation Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval` | boolean | False | Enable evaluation mode for testing and validation |
| `txt_path` | string | "results/results.txt" | Path to save evaluation results |

## Region of Interest (ROI) Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roi` | null/boolean | null | Enable/disable region of interest tracking |
| `line_points` | null/list | null | Define ROI line points for tracking (e.g., [(0, 0), (1280, 720)]) |

## System Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timezone` | string | "Asia/Seoul" | System timezone for timestamp tracking |
| `debug` | boolean | True | Enable detailed logging for debugging purposes |
| `log_file` | string | "data/logs/debug.log" | Path to the debug log file |

## Production Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multi_camera` | boolean | False | Enable support for multiple camera inputs |
| `production` | boolean | False | Enable production mode with server integration. If you enable `production=True`, It will not send in/out data for the api. |

## Usage Notes

1. **GPU Configuration**: If you don't have a GPU
2. **Database**: Ensure your face embeddings database is in the correct PKL format
3. **Video Input**: Supports both local video files and RTSP streams
4. **ROI Tracking**: When enabled, only tracks faces within the defined region
5. **Production Mode**: When enabled, the system will send data to the server

## Best Practices

1. Set `debug` to False in production environments
2. Adjust `match_threshold` based on your accuracy requirements
3. Use appropriate video paths for your environment
4. Configure timezone according to your location
5. Enable `production` mode only when server integration is required