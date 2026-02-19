# Person Tracking & Phone Usage Detection

A production-ready system for tracking people, recognizing faces, and detecting phone usage in real-time video streams.

## Features

- **Person Detection & Tracking:** YOLOv8-Pose for person detection with 17 keypoints + BoT-SORT for stable tracking
- **Face Recognition:** Integration with existing InsightFace pipeline within person ROIs
- **Phone Usage Detection:** YOLOv8n phone detection + pose-based spatial/temporal association
- **Identity Locking:** Confidence-weighted temporal voting for stable identity assignment
- **Multi-Person Support:** Handles 10+ people simultaneously with correct phone attribution
- **Real-time Performance:** 10-15 FPS on GPU with balanced configuration

## Project Structure

```
src/person_tracking/
├── __init__.py
├── README.md                  # This file
├── core/                      # Core detection and tracking modules
│   ├── person_detector.py     # YOLOv8-Pose person detection
│   ├── person_tracker.py      # BoT-SORT tracking
│   ├── phone_detector.py      # YOLOv8n phone detection
│   ├── face_adapter.py        # Face recognition adapter
│   ├── identity_manager.py    # Temporal voting for identity
│   ├── phone_usage_filter.py  # Temporal filtering for phone usage
│   ├── state_manager.py       # Person state management
│   └── track_manager.py       # Track history management
├── api/                       # FastAPI endpoints
│   ├── app.py                 # FastAPI application
│   └── endpoints.py           # API routes
├── config/                    # Configuration management
│   ├── models.py              # Pydantic configuration models
│   └── manager.py             # Configuration loader
├── storage/                   # Database integration
│   └── pgvector_store.py      # PostgreSQL pgvector
├── video/                     # Video processing
│   ├── stream_handler.py      # Video stream handling
│   └── frame_annotator.py     # Visualization
├── logging/                   # Logging utilities
│   ├── csv_logger.py          # CSV event logging
│   └── setup.py               # Loguru setup
└── services/                  # Background services
    └── ...
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-person-tracking.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.person-tracking.example .env

# Edit .env with your settings
nano .env
```

### 3. Create Configuration

```bash
# Use the sample config or create your own
cp configs/person_tracking/config.yaml configs/person_tracking/my_config.yaml

# Edit configuration
nano configs/person_tracking/my_config.yaml
```

### 4. Run the System

```bash
# From project root
python -m src.person_tracking.main --config configs/person_tracking/my_config.yaml
```

## Configuration

### Camera Configuration

Each camera can be individually configured with different parameters:

```yaml
cameras:
  - camera_id: 1
    camera_name: "Office Entry"
    video_path: "rtsp://..."

    person_detection:
      model_size: "s"           # n/s/m/l/x (balanced: s)
      confidence_threshold: 0.5

    face_recognition:
      match_threshold: 0.3
      identity_lock_frames: 5   # Require 5 frames for identity lock
      identity_consensus: 0.60  # 60% consensus (3/5 frames)

    phone_usage:
      confirmation_frames: 8    # Require 8 frames for phone usage
      confirmation_consensus: 0.75  # 75% consensus (6/8 frames)
      hand_distance_threshold: 0.20  # 20cm
      head_distance_threshold: 0.25  # 25cm
```

### Environment Variables

Key environment variables (see `.env.person-tracking.example`):

- `HB_CLIENTSLUG`: Organization identifier
- `POSTGRES_HOST`, `POSTGRES_PORT`: PostgreSQL connection
- `HB_IN`, `HB_OUT`: Camera RTSP URLs
- `USE_PGVECTOR`: Enable pgvector for embeddings (recommended)

## Architecture

### Data Flow

```
Video Frame
    ↓
PersonDetector (YOLOv8-Pose)
    ├─ Bounding boxes
    └─ 17 keypoints
    ↓
PersonTracker (BoT-SORT)
    └─ Assign TrackIDs
    ↓
Parallel Processing:
    ├─ FaceRecognitionAdapter
    │   └─ Identity locking (5 frames, 60% consensus)
    └─ PhoneDetector
        └─ Phone usage (8 frames, 75% consensus)
    ↓
PersonStateManager
    └─ Events: identity_locked, phone_usage_started, etc.
    ↓
Output:
    ├─ Mock API (MVP)
    ├─ CSV logs
    └─ Annotated frames
```

### Temporal Logic

#### Identity Locking (Confidence-Weighted Voting)
- Sliding window of M frames (default: 5)
- Requires 60% consensus (3/5 frames)
- Uses similarity scores for confidence
- Minimum window duration: 333ms

#### Phone Usage Confirmation
- Sliding window of N frames (default: 8)
- Requires 75% consensus (6/8 frames)
- Minimum duration: 500ms
- Spatial checks:
  - Phone in upper body zone (keypoints-based)
  - Hand within 20cm of phone
  - Phone within 25cm of head (calling)
  - Require 2 of 3 checks to pass

## Performance

### Target Performance (Balanced Configuration)

| Metric | Target |
|--------|--------|
| FPS | 10-15 per camera |
| Latency | <100ms per frame |
| GPU Memory | <4GB per camera |
| Phone Detection Accuracy | >90% |

### Model Configurations

| Profile | Person Model | Phone Model | FPS | GPU Memory |
|---------|--------------|-------------|-----|------------|
| Fast | YOLOv8n-pose | YOLOv8n | 18-22 | 2.5 GB |
| **Balanced** | **YOLOv8s-pose** | **YOLOv8n** | **12-15** | **3.5 GB** |
| Accurate | YOLOv8m-pose | YOLOv8s | 8-10 | 5.5 GB |

## API Endpoints (Mock in MVP)

### POST /api/v1/person-tracking/events
Log person tracking events (phone usage, identity changes, etc.)

```json
{
  "track_id": 1,
  "person_name": "John Doe",
  "event_type": "phone_usage_started",
  "using_phone": true,
  "confidence": 0.95,
  "camera_id": 1,
  "timestamp": "2025-11-23T10:30:15Z"
}
```

### GET /api/v1/person-tracking/status
Get current person states

```json
{
  "persons": [
    {
      "track_id": 1,
      "identity": "John Doe",
      "identity_locked": true,
      "using_phone": true,
      "confidence": 0.95,
      "camera_id": 1
    }
  ]
}
```

### GET /health
Health check endpoint

## Development Status

### ✅ Completed (Day 1-2)
- Project structure
- Configuration management (Pydantic models)
- YAML configuration loading
- Environment variable interpolation
- Requirements and dependencies

### 🔄 In Progress
- Person detection (Day 2-3)
- Person tracking (Day 3-4)
- Face recognition adapter (Day 4-5)
- Phone detection (Day 5-7)

### 📋 Planned
- Identity manager (Day 8-9)
- Phone usage logic (Day 9-11)
- State management (Day 11-12)
- Mock API (Day 12-13)
- Complete orchestrator (Day 15-16)
- Testing & deployment (Day 17-21)

See [PERSON_TRACKING_IMPLEMENTATION_PLAN.md](../../PERSON_TRACKING_IMPLEMENTATION_PLAN.md) for full roadmap.

## Contributing

This is part of the Smart Office project. For questions or issues, contact the development team.

## License

Proprietary - Smart Office Team
