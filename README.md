# Face Recognition System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/)

A production-ready face recognition system with real-time tracking and pgvector database integration. Designed for smart office applications, attendance tracking, and security monitoring.

## Features

### Core Capabilities
- **Real-time Face Detection** - InsightFace-based detection with GPU acceleration
- **Face Tracking** - DeepOCSORT multi-object tracking for continuous identity tracking
- **Face Recognition** - 512-dimensional embeddings with cosine similarity matching
- **pgvector Integration** - Scalable PostgreSQL database with vector similarity search
- **REST API** - FastAPI-based endpoints for integration with backend systems
- **Redis Pub/Sub** - Real-time event broadcasting for distributed systems

### Advanced Features
- **Multi-camera Support** - Process multiple RTSP streams simultaneously
- **Entry/Exit Tracking** - Automatic detection of people crossing counting lines
- **Attendance Logging** - Integration with backend API for attendance records
- **Face Quality Validation** - Frontality checks and landmark verification
- **Cloud Storage** - Google Cloud Storage integration for face images
- **Auto-sync** - Automatic embedding synchronization from backend
- **CSV Logging** - Local logging with rotating files
- **Health Monitoring** - Comprehensive health checks and metrics


## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA 12.2+ (recommended for production)
- **CPU**: Multi-core processor (minimum 4 cores)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 20GB+ for models and data

### Software
- **OS**: Ubuntu 20.04+ or compatible Linux distribution
- **Docker**: 20.10+ with Docker Compose
- **NVIDIA Docker**: nvidia-docker2 (for GPU support)
- **Python**: 3.10+ (if running without Docker)

## Quick Start

### 1. Clone Repository

```bash
git clone --recursive https://github.com/your-org/so.model-face-recognition.git
cd so.model-face-recognition
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Required Configuration:**
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_USER=face_recognition
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=face_embeddings

# Backend API
SO_BACKEND_API_URL=http://localhost:7091
HB_CLIENTSLUG='your_organization_slug'
SA_EMAIL='admin@example.com' # Super Admin Email
SA_PASSWORD='your_admin_password' #Super Admin Password
```

### 3. Start Services

#### Using Docker Helper Script (Recommended)

```bash
# Build services
./compose.sh build

# Start with logs
./compose.sh test -l

# Stop services
./compose.sh stop
```
#### Using Docker 

```bash
# Build and start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f face-recognition
```
## Configuration

### Environment Variables

See [.env.example](.env.example) for all available options.

#### Core Settings
```bash
# Face Recognition
MATCH_THRESHOLD=0.3              # Recognition confidence threshold (0-1)
EMBEDDING_DIMENSION=512          # Face embedding size
FACE_DETECTION_PADDING=20.0      # Detection box padding (%)
USE_PGVECTOR=true                # Enable pgvector database

# Service Ports
FR_PORT=5000                     # Main service port
```

### Configuration Files

- **configs/config.yaml** - Main application configuration
- **compose.yml** - Docker services configuration
- **compose.override.yml** - Local overrides

### User Face Images (Required)

**IMPORTANT**: User face images must be uploaded through the **SmartOffice Dashboard > Users** before the system can recognize individuals. The face recognition system calculates embeddings from these images for matching.

**Requirements:**
- **Face Image Upload** - Each user must have a clear frontal face photo uploaded
- **Image Quality** - High-quality images with good lighting and face visibility
- **User Profile** - Complete user profile with full name

**How to Add Users:**
1. Log in to SmartOffice Dashboard
2. Navigate to **Users**
3. Add new user or edit existing user
4. Upload a clear frontal face photo
5. Save user profile

The system automatically:
- Fetches active users and their images from the API on startup
- Calculates 512-dimensional face embeddings from the uploaded images
- Stores embeddings in the pgvector database for fast similarity matching
- Synchronizes new users and updates automatically

### Camera Configuration (Required)

**IMPORTANT**: All camera-related configurations must be entered through the **SmartOffice Dashboard > Camera Settings**. The face recognition system will not work without proper camera configuration in the dashboard.

**Required Camera Settings:**
- **IP Address** - Camera RTSP stream URL
- **Port Number** - RTSP port (usually 554)
- **Username/Password** - Camera credentials
- **Application Type** - Must be set to `FaceRecognision`
- **ROI (Region of Interest)** - Coordinates for detection area `[[x1, y1], [x2, y2]]`
- **Virtual Line Points** - Coordinates for counting line `[[x1, y1], [x2, y2]]`
- **Recognition Threshold** - Matching confidence threshold (recommended: 0.3)
- **Camera Type** - Set to `IN` or `OUT` for entry/exit tracking

**How to Configure:**
1. Log in to SmartOffice Dashboard
2. Navigate to **top right profile icon >Camera Settings**
3. Add or edit camera configuration
4. Set **Application** field to `FaceRecognision`
5. Fill in all required fields listed above
6. Save configuration

The system automatically fetches camera configurations from the API on startup. Any changes made in the dashboard will be applied on the next service restart.

## Usage

### Running Face Recognition

#### Basic Usage
```bash
# From examples directory
python examples/clients/main.py
```

## Development

## Troubleshooting


### Camera Stream Issues

```bash
# Test RTSP stream with ffmpeg
ffmpeg -rtsp_transport tcp -i "rtsp://your_stream" -frames:v 1 test.jpg

# Check stream in VLC
vlc "rtsp://your_stream"

# Verify environment variables
docker exec face-recognition env | grep HB_IN
```



## Research References

- [InsightFace](https://github.com/deepinsight/insightface) - Face detection and recognition
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [DeepOCSORT](https://github.com/mikel-brostrom/yolo_tracking) - Multi-object tracking
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) - Tracking evaluation
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- HumbleBee AI team for development and testing
- InsightFace team for face detection models
- Ultralytics for YOLO models
- pgvector team for vector database extension

## Support

For issues, questions, or feature requests:
- Create an issue on GitHub
- Check existing documentation
- Contact the development team

---

**Version**: 2.0.0
**Last Updated**: December 2025
**Status**: Production Ready
