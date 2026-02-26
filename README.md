# Smart Office - AI Service (Face Recognition & Person Tracking)

> Real-time person tracking and face recognition system for smart office attendance, activity monitoring, and security.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [API Reference](#api-reference)

---

## Overview

The Smart Office AI Service is a production-ready, multi-tenant face recognition and person tracking system designed for:

- **Attendance Tracking**: Automatic employee check-in/check-out with face recognition
- **Activity Monitoring**: Real-time detection of activities (phone usage, sleeping, etc.)
- **Multi-Camera Support**: Track persons across multiple cameras with global IDs
- **Unrecognized Face Detection**: Alert on unknown visitors
- **Location Tracking**: Real-time user location by camera zone

### Key Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| **Face Recognition** | InsightFace (buffalo_l) | 512-dim embeddings with >99% accuracy |
| **Person Detection** | YOLOv8 / YOLO26 | Fast, accurate person bounding boxes |
| **Person Tracking** | BoT-SORT | Stable track IDs across frames |
| **Cross-Camera Tracking** | ReID (OSNet) | Global person IDs across cameras |
| **Activity Recognition** | Ollama + Gemma 3 | AI-powered activity detection |
| **Embedding Storage** | pgvector | Scalable vector similarity search |
| **Message Queue** | Redis Streams + Celery | Reliable command/event processing |
| **Multi-Tenancy** | Schema-per-tenant | Isolated data for each organization |

### System Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Backend   │────────▶│  Redis MDA   │────────▶│  AI Service │
│   (NestJS)  │         │              │         │   (Python)  │
└─────────────┘         └──────────────┘         └─────────────┘
      │                        │                        │
      │                        │                        │
      ▼                        ▼                        ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│ PostgreSQL  │         │    Celery    │         │  pgvector   │
│  (Backend)  │         │   Workers    │         │ (Embeddings)│
└─────────────┘         └──────────────┘         └─────────────┘
```

**Communication Flow:**
1. **Commands**: Backend → Redis Streams → AI Service (create/update/delete embeddings, camera config)
2. **Events**: AI Service → Redis Pub/Sub → Backend (attendance, activity, unrecognized faces)
3. **Data Storage**: AI Service owns embedding data, Backend owns business logic

---

## Installation & Setup

### Prerequisites

**Hardware Requirements:**
- NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- 8GB+ GPU VRAM (for activity recognition with Ollama)
- 16GB+ system RAM
- Ubuntu 20.04+ or compatible Linux distribution

**Software Requirements:**
- Docker 24.0+
- Docker Compose 2.20+
- NVIDIA Driver 525+ (for CUDA 12.2)
- NVIDIA Container Toolkit


### Step 1: Clone Repository

```bash
git clone --recursive https://github.com/humblebeeai/so.model-face-recognition.git
cd so.model-face-recognition

# Ensure all submodules are cloned correctly
git submodule sync --recursive
git submodule update --init --recursive
```

### Step 2: Configure Environment Variables

```bash
# Copy example config
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

**Minimum required settings:**

```bash
# Client/Tenant Configuration
SO_CLIENT_SLUG=your_organization_slug    # Unique organization identifier

# PostgreSQL (pgvector database)
SO_POSTGRES_HOST=localhost                  # Database host
SO_POSTGRES_PORT=5432                       # Database port
SO_POSTGRES_USER=postgres                   # Database user
SO_POSTGRES_PASSWORD=your_secure_password   # REQUIRED: No default for security
SO_POSTGRES_DB=smart_office                 # Database name

# Redis (Message-Driven Architecture)
SO_REDIS_HOST=localhost                     # Redis host
SO_REDIS_PORT=6379                          # Redis port

# Celery (Task Queue)
SO_CELERY_BROKER_URL=redis://${SO_REDIS_HOST}:${SO_REDIS_PORT}/0
SO_CELERY_RESULT_BACKEND=redis://${SO_REDIS_HOST}:${SO_REDIS_PORT}/1

# Google Cloud Storage
SO_GCS_CREDENTIALS_PATH=/app/credentials/gcs-service-account.json
SO_GCS_BUCKET=your-gcs-bucket

# Action Recognition (Ollama)
SO_OLLAMA_API_URL=http://localhost:11434
SO_OLLAMA_MODEL=gemma3:4b

# Application Settings
SO_LOG_LEVEL=INFO
```

### Step 3: Build and Start

```bash
# Build Docker images
./compose.sh build

# Start services with live logs
./compose.sh start -l

# Or start in background
./compose.sh start
```

### Step 4: Verify Installation

```bash
# Check container status
./compose.sh ps

# View logs
./compose.sh logs

# Expected output: "SmartOfficeEngine started"
```

---

## Configuration

### Camera Configuration

Cameras are configured via **Backend API** (recommended):

1. Set `use_api_for_cameras: true` in `config.yaml`
2. Configure cameras in Backend Admin UI
3. AI Service auto-reloads when cameras are added/updated

**Alternative: Environment Variables**

```bash
# Deprecated: Use Backend API instead
HB_IN=rtsp://user:pass@camera1/stream
HB_OUT=rtsp://user:pass@camera2/stream
```

---

## Deployment

### Scaling Workers

The system uses Celery for task processing. Scale workers based on load:

**Embedding Workers** (CPU/GPU intensive):
```bash
# In docker-compose.yml or Kubernetes
celery-embedding-worker:
  replicas: 2  # Increase for more embedding processing
```

**Detection Workers** (I/O intensive):
```bash
# In docker-compose.yml or Kubernetes
celery-detection-worker:
  replicas: 4  # Increase for more cameras/detections
```

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Office AI Service                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐  │
│  │   Engine    │──────│ Frame        │──────│  Camera    │  │
│  │ (main.py)   │      │ Processor    │      │  Engines   │  │
│  └─────────────┘      └──────────────┘      └────────────┘  │
│         │                     │                     │       │
│         │                     │                     │       │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐  │
│  │ MDA Manager │      │   Models     │      │  Tracker   │  │
│  │             │      │ (YOLO, Face) │      │  (BoT-SORT)│  │
│  └─────────────┘      └──────────────┘      └────────────┘  │
│         │                                            │      │
│         │                                            │      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Infrastructure Layer                    │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  pgvector │ Repository │ GCS │ Entry Logger          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────┐                         ┌────────────────┐
│  Redis Streams  │                         │  PostgreSQL    │
│  (Commands)     │                         │  (pgvector)    │
└─────────────────┘                         └────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────┐                         ┌────────────────┐
│ Celery Workers  │                         │ Embedding      │
│ (Embeddings,    │                         │ Storage        │
│  Detections)    │                         │                │
└─────────────────┘                         └────────────────┘
```

### Message-Driven Architecture (MDA)

**Commands** (Backend → AI Service):
- `CreateEmbedding`: Add user face embeddings
- `UpdateEmbedding`: Update user face embeddings
- `DeleteEmbedding`: Remove user embeddings
- `ConfigureCamera`: Update camera settings
- `StartCamera` / `StopCamera`: Control camera processing

**Events** (AI Service → Backend):
- `AttendanceRecorded`: User check-in/out detected
- `ActivityDetected`: Activity (phone, sleeping) detected
- `UnrecognizedFaceSaved`: Unknown person detected
- `UserLocationUpdated`: User location changed

### Data Flow

1. **Embedding Creation**:
   ```
   Backend uploads user images → Redis Stream command →
   Celery Worker downloads images → InsightFace generates embeddings →
   Store in pgvector → Publish success event → Backend notified
   ```

2. **Real-time Detection**:
   ```
   Camera stream → Person detection (YOLO) →
   Face detection (InsightFace) → Face recognition (pgvector similarity) →
   Attendance logging → Event published to Backend
   ```

---

## API Reference

### compose.sh Commands

```bash
./compose.sh build           # Build Docker images
./compose.sh start           # Start containers
./compose.sh start -l        # Start with live logs
./compose.sh stop            # Stop containers
./compose.sh restart         # Restart containers
./compose.sh logs            # View logs
./compose.sh logs -f         # Follow logs (live)
./compose.sh enter           # Enter container shell
./compose.sh ps              # List running containers
./compose.sh clean           # Remove containers and images
./compose.sh clean --volumes # Remove containers, images, and volumes
```

---

## References

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition models
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Person detection
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Multi-object tracking
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [Celery](https://docs.celeryq.dev/) - Distributed task queue
- [Redis Streams](https://redis.io/docs/data-types/streams/) - Message streaming

---

## License

Copyright © 2024 HumbleBee AI. All rights reserved.

For licensing inquiries, contact: support@humblebeeai.com
