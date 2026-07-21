# Smart Office - AI Service (Face Recognition & Person Tracking)

> Real-time person tracking and face recognition system for smart office attendance, activity monitoring, and security.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [compose.sh CLI](#composesh-cli)

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
```

> If you already cloned without `--recursive`, run `git submodule update --init --recursive` to fetch submodules.

### Step 2: Configure Environment Variables

```bash
# Copy example config
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

> `.env` is gitignored — it holds secrets (DB password, GCS path) and must never be committed.

**Place GCS credentials:**

```bash
# Drop the service-account JSON the platform team gave you here:
mkdir -p credentials
cp /path/to/gcs-service-account.json credentials/
# SO_GCS_CREDENTIALS_PATH in .env must match the in-container path: /app/credentials/<filename>.json
```

**Minimum required settings:**

```bash
# Client/Tenant Configuration
SO_CLIENT_SLUG=your_organization_slug       # Unique organization identifier

# PostgreSQL (pgvector database)
# Same machine as so.stack: host.docker.internal
# Different machine (Tailscale):  100.x.x.x
SO_POSTGRES_HOST=host.docker.internal
SO_POSTGRES_PORT=5432                       # Host-exposed port of so.stack's Postgres
SO_POSTGRES_USER=postgres
SO_POSTGRES_PASSWORD=your_secure_password   # REQUIRED: No default for security
SO_POSTGRES_DB=smart_office

# Redis (Message-Driven Architecture)
# Same machine as so.stack: host.docker.internal
# Different machine (Tailscale): 100.x.x.x
SO_REDIS_HOST=host.docker.internal
SO_REDIS_PORT=6379                          # Host-exposed port of so.stack's Redis
SO_REDIS_URL=redis://host.docker.internal:6379

# Celery (Task Queue) — must match SO_REDIS_* above, hardcode the URL (no variable interpolation in .env)
SO_CELERY_BROKER_URL=redis://host.docker.internal:6379/0
SO_CELERY_RESULT_BACKEND=redis://host.docker.internal:6379/1

# Google Cloud Storage
SO_GCS_CREDENTIALS_PATH=/app/credentials/gcs-service-account.json
SO_GCS_BUCKET=your-gcs-bucket

# Action Recognition (Ollama) — uses Docker service name, always port 11434 internally
SO_OLLAMA_API_URL=http://ollama:11434
SO_OLLAMA_MODEL=gemma3:4b
SO_OLLAMA_PORT=11434                        # Host-exposed port (change if 11434 is taken)

# Application Settings
SO_LOG_LEVEL=INFO
```

> **Note:** Variable interpolation (`${VAR}`) does not work within `.env` files. Always hardcode full URLs in `SO_REDIS_URL`, `SO_CELERY_BROKER_URL`, and `SO_CELERY_RESULT_BACKEND`.

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

## Usage

This service is one component of the Smart Office platform. End-to-end usage — onboarding users, registering cameras, viewing attendance, configuring the Backend & Frontend — is documented in the platform-level repo:

> **[lum-stack/README.md](https://github.com/humblebeeai/lum-stack/blob/main/README.md)**

Once this AI service is running and connected to the same Postgres + Redis as `lum-stack`, it picks up cameras and user embeddings automatically via the MDA channels described in [`docs/SERVICE_ARCHITECTURE.md`](docs/SERVICE_ARCHITECTURE.md#54-redis-mda-contracts).

---

## Configuration

### Camera Configuration

Cameras are managed by the Backend (`lum-stack`) and stored in Postgres. The AI service loads them from the database on startup and reloads on `ConfigureCamera` / `StartCamera` / `StopCamera` Redis Stream commands — there is no local camera config to edit. Add or update cameras through the Backend Admin UI.

### Pipeline & Recognition Tuning

Service-level tuning lives in [`configs/config.yaml`](configs/config.yaml). The most useful knobs:

| Key | Default | Effect |
|---|---|---|
| `match_threshold` | `0.3` | Cosine distance for face match |
| `person_detection_threshold` | `0.45` | YOLO confidence floor |
| `person_detection_model` | `yolo26` | `yolo26` (NMS-free) or `yolov8` |
| `enable_global_tracking` | `true` | Cross-camera ReID (MCMOT) toggle |
| `pipeline.detection_interval` | `2` | Run YOLO every N frames |
| `pipeline.recognition_interval` | `3` | Run ArcFace every N detections |
| `action_recognition.enabled` | `true` | Activity classification via Ollama |

For the full set, see the comments in `configs/config.yaml`.

---

## Deployment

### Compose Override Files

The project ships with environment-specific override files in `templates/compose/`:

| File | Purpose |
|------|---------|
| `compose.override.dev.yml` | Development extras: live source mounts, debug log level, Flower monitoring UI |
| `compose.override.prod.yml` | Production tweaks (if any) |

**Production (default):**
```bash
./compose.sh start
# equivalent to: docker compose -f compose.yml up
```

**Development (with Flower + live reload):**
```bash
docker compose -f compose.yml -f templates/compose/compose.override.dev.yml up
```

> Flower (Celery monitoring UI) only runs in dev mode. Access it at `http://localhost:5555` (or `SO_FLOWER_PORT`).

---

### Networking

All services run on a dedicated Docker bridge network (`person-tracking-network`). Containers communicate with each other via service names:

- `ollama:11434` — Ollama API (used by `SO_OLLAMA_API_URL`)

External services (PostgreSQL, Redis from `so.stack`) are reached via:
- **Same machine**: `host.docker.internal` — Docker resolves this to the host machine IP
- **Different machine**: Tailscale IP (e.g. `100.100.1.20`) — set directly in `.env`

---

### Scaling Workers

The Celery service `celery-worker` (in `compose.yml`) consumes both the `embeddings` and `detections` queues by default:

```yaml
command: celery -A workers.celery_app worker -Q embeddings,detections -l warning
```

**Scale the existing worker** (simple — both queues benefit equally):

```bash
docker compose up -d --scale celery-worker=3
```

**Split queues onto dedicated workers** (when one queue dominates load — e.g. bulk embedding ingest):
add a second service in your override file with `-Q embeddings` only, and restrict the original to `-Q detections`. Adjust replicas independently per workload.

> The main `person-tracking` container is **not** horizontally scalable as-is — there is one shared GPU inference worker per process. Scale by partitioning cameras across hosts. See [`docs/SERVICE_ARCHITECTURE.md` §3.5](docs/SERVICE_ARCHITECTURE.md#35-scaling-notes).

---

## Development

**Live source reload (dev override):**

```bash
docker compose -f compose.yml -f templates/compose/compose.override.dev.yml up
```

The dev override mounts `./src` and `./configs` into the container so code edits are picked up on restart, sets `SO_LOG_LEVEL=DEBUG`, and starts Flower at [`http://localhost:5555`](http://localhost:5555).

**Get a shell inside the running container:**

```bash
./compose.sh enter
```

**Logs:**

```bash
./compose.sh logs -f          # follow all services
docker compose logs -f person-tracking   # one service
```

**Submodules:** `modules/insightface` is a git submodule. After pulling changes that touch it, run:

```bash
git submodule update --init --recursive
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SmartOfficeEngine` never logs "started" | Cannot reach Postgres / Redis | Check `SO_POSTGRES_HOST` / `SO_REDIS_HOST` are reachable from inside the container; if same machine, ensure `host.docker.internal` resolves (provided by `extra_hosts: host-gateway`) |
| `ollama` healthcheck failing | Model still pulling on first start | Wait — `gemma3:4b` pulls on first boot (`start_period: 60s`). Watch `docker compose logs ollama` |
| GPU not available in container | NVIDIA Container Toolkit missing or driver mismatch | Verify `nvidia-smi` works on host; reinstall `nvidia-container-toolkit`; ensure driver ≥ 525 |
| GCS uploads fail with 403 | Service-account file missing or wrong path | Confirm `credentials/<file>.json` exists on host and `SO_GCS_CREDENTIALS_PATH` matches its in-container path |
| Celery tasks stuck pending | Worker not consuming the queue | `./compose.sh ps` to confirm `celery-worker` is up; check Flower (dev) at `:5555` |
| `.env` variables not applied | Variable interpolation `${VAR}` does not work inside `.env` | Hardcode full URLs in `SO_REDIS_URL`, `SO_CELERY_BROKER_URL`, `SO_CELERY_RESULT_BACKEND` |
| Cameras don't appear | Backend hasn't sent `ConfigureCamera` / `StartCamera` | Add cameras via Backend Admin UI; AI service auto-reloads |

---

## Architecture

A high-level summary lives below. The full internal architecture — components, deployment topology, data flows, and database schema with diagrams — is in **[`docs/SERVICE_ARCHITECTURE.md`](docs/SERVICE_ARCHITECTURE.md)**. For the cross-repo (Backend / Frontend / AI) view, see **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

### Components (one-liner each)

- **Pipeline** — `SmartOfficeEngine` orchestrates per-camera workers and a shared GPU inference worker.
- **Domain** — ML models: YOLO (person detection), InsightFace (face detection + ArcFace embeddings), BoT-SORT (tracking), OSNet (cross-camera ReID), Ollama (activity recognition).
- **Messaging** — Redis Streams for commands (Backend → AI), Pub/Sub for events (AI → Backend).
- **Workers** — Celery tasks for embedding ingestion and detection persistence.
- **Infrastructure** — pgvector store, Postgres repositories, GCS uploader, async logger, Prometheus metrics.

### Communication summary

- **Commands** (Backend → AI): `CreateEmbedding`, `UpdateEmbedding`, `DeleteEmbedding`, `ConfigureCamera`, `StartCamera` / `StopCamera`, `CaptureFrame`, `CalibrateCamera`.
- **Events** (AI → Backend): `AttendanceRecorded`, `ActivityDetected`, `UnrecognizedFaceSaved`, `UserLocationUpdated`, `EmbeddingCreated` / `EmbeddingFailed`, `FrameCaptured`, `SystemMetrics` / `SystemAlert`.

> Component diagrams, sequence diagrams (per-frame loop, embedding creation, attendance, ReID), database ERD, and deployment topology are in **[`docs/SERVICE_ARCHITECTURE.md`](docs/SERVICE_ARCHITECTURE.md)**.

---

## compose.sh CLI

A thin wrapper around `docker compose` for common operations:

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
