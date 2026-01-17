# Smart Office - Person Tracking & Face Recognition

A unified system for real-time person tracking and face recognition, designed for smart office attendance and monitoring.

## Features

- **Person Tracking**: YOLOv8 + BoT-SORT for stable person detection and tracking
- **Face Recognition**: InsightFace (buffalo_l) for accurate face identification
- **Cross-Camera Tracking**: Global track IDs across multiple cameras (MCMOT ReID)
- **Attendance Logging**: Automatic IN/OUT status tracking with API integration
- **pgvector Storage**: Scalable face embedding storage with PostgreSQL

## Quick Start

### 1. Clone the Repository

```bash
git clone --recursive  https://github.com/humblebeeai/so.model-face-recognition.git
cd so.model-face-recognition
```

### 2. Checkout to MCMOT branch

```bash
git checkout MCMOT
```

### 3. Ensure all submodules are clonned correctlly

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### 4. Configure Environment

```bash
# Copy example configs
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required settings in `.env`:**

```bash
# API credentials
SA_EMAIL=your_admin_email
SA_PASSWORD=your_admin_password
HB_CLIENTSLUG=your_organization_slug
API_HOST=http://your-backend-api:7091

# Camera streams (if not using API)
HB_IN=rtsp://camera_in_stream
HB_OUT=rtsp://camera_out_stream
```

### 3. Build and Run

```bash
# Build Docker containers
./compose.sh build

# Start the application (with logs)
./compose.sh start -l

# Or start in background
./compose.sh start
```

### 4. View Logs

```bash
./compose.sh logs
```

### 5. Stop

```bash
./compose.sh stop
```

## Configuration

### Feature Flags (`configs/config.yaml`)

```yaml
# Feature flags
enable_global_tracking: true   # Cross-camera person tracking
enable_attendance_logging: true
use_pgvector: true             # Face embedding storage
use_api_for_cameras: true      # Load cameras from API

# Recognition settings
match_threshold: 0.3           # Face similarity threshold
person_detection_threshold: 0.45
face_detection_padding: 20.0   # % padding around faces
```

### Camera Configuration

Cameras can be configured via:

1. **Backend API** (recommended): Set `use_api_for_cameras: true`
2. **Environment variables**: Set `HB_IN`, `HB_OUT` in `.env`

## compose.sh Commands

| Command | Description |
|---------|-------------|
| `./compose.sh build` | Build Docker images |
| `./compose.sh start` | Start containers |
| `./compose.sh start -l` | Start with live logs |
| `./compose.sh stop` | Stop containers |
| `./compose.sh restart` | Restart containers |
| `./compose.sh logs` | View logs |
| `./compose.sh enter` | Enter container shell |
| `./compose.sh ps` | List running containers |
| `./compose.sh clean` | Remove containers and images |

## Development Setup

For development, copy the dev compose override template:

```bash
cp templates/compose/compose.override.dev.yml compose.override.yml
```

This enables:
- Source code mounting (hot reload - just restart to apply changes)
- Debug logging (`LOG_LEVEL: DEBUG`)
- Interactive terminal

Docker Compose automatically uses `compose.override.yml` when present.

## Architecture

```text
src/
├── main.py                    # Application entry point
├── face_recognition/          # Face detection & recognition
│   ├── core/                  # Detector, recognizer
│   ├── services/              # Embedding sync, image fetcher
│   ├── storage/               # pgvector store
│   └── smart_office_engine.py # Main orchestrator
└── person_tracking/           # Person detection & tracking
    ├── core/                  # Person detector, tracker, ReID
    └── video/                 # Frame annotation
```

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA Driver 525+
- CUDA 12.2 compatible GPU

## Troubleshooting

**Container won't start:**

```bash
# Check logs
./compose.sh logs

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu20.04 nvidia-smi
```

**Camera connection issues:**

- Verify RTSP URL is accessible
- Check firewall settings
- Ensure camera credentials are correct

## Video Recording (ffmpeg)

### Simple recording

```bash
ffmpeg -rtsp_transport tcp -i "rtsp://<camera-link>" -c copy output.mp4
```

### Scheduled recording

```bash
# Record for 3000 seconds starting at 17:45
echo 'ffmpeg -rtsp_transport tcp -i "rtsp://<camera-link>" -t 3000 -c copy output.mp4' | at 17:45

# Check scheduled jobs
atq

# Remove a scheduled job
atrm <job-number>
```

## References

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 detection
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Multi-object tracking
