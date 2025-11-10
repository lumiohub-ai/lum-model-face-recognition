# Face Recognition System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         HBFace                                   │
│                    (Main Orchestrator)                           │
│  Entry Point: examples/clients/main.py                          │
└────────┬────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────────────────────┐
         │                                                       │
         v                                                       v
┌─────────────────────┐                            ┌────────────────────────┐
│   FaceSetup         │                            │   DashboardManager     │
│  (system_setup.py)  │                            │   (dashboard/manager)  │
│                     │                            │                        │
│ - Initialize        │                            │ - FastAPI Server       │
│   cameras           │                            │ - Live streaming       │
│ - Setup engines     │                            │ - Health checks        │
│ - Configure loggers │                            └────────────────────────┘
└─────────┬───────────┘
          │
          ├─────────────────────────────────────────────────────┐
          │                                                      │
          v                                                      v
┌──────────────────────┐                          ┌───────────────────────┐
│   FaceEngine         │                          │   EntryLogger         │
│   (core/engine.py)   │◄─────────────────────────┤ (logging/)            │
│                      │                          │                       │
│  Orchestrates:       │                          │ - Status tracking     │
│  ├─ FaceDetector     │                          │ - CSV logging         │
│  ├─ FaceTracker      │                          │ - API communication   │
│  ├─ TrackManager     │                          │ - Visualization       │
│  └─ FaceRecognizer   │                          └───────┬───────────────┘
└──────────┬───────────┘                                  │
           │                                              │
           │                                              v
           │                                   ┌────────────────────┐
           │                                   │   APIClient        │
           │                                   │   (api/client.py)  │
           │                                   │                    │
           │                                   │ - Authentication   │
           │                                   │ - User management  │
           │                                   │ - Attendance       │
           │                                   │ - Face upload      │
           │                                   └────────────────────┘
           │
           └───────┬──────────────┬─────────────┬──────────────┐
                   │              │             │              │
                   v              v             v              v
         ┌─────────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐
         │FaceDetector │  │FaceTracker│  │TrackManager│  │FaceRecognizer│
         │(core/)      │  │(core/)    │  │(core/)     │  │(core/)       │
         │             │  │           │  │            │  │              │
         │-InsightFace │  │-DeepOCSORT│  │-Track      │  │-Embedding    │
         │-Embeddings  │  │-Track IDs │  │ lifecycle  │  │ comparison   │
         │-Detection   │  │-Trajectory│  │-History    │  │-Recognition  │
         └─────────────┘  └───────────┘  └────────────┘  └──────────────┘
```

## Data Flow

### 1. Frame Processing Pipeline

```
Video Stream
    │
    v
┌───────────────────┐
│  StreamHandler    │  (video/stream_handler.py)
│  Read frame       │
└────────┬──────────┘
         │
         v
┌───────────────────┐
│  FaceDetector     │  (core/detector.py)
│  - Detect faces   │
│  - Extract        │
│    embeddings     │
└────────┬──────────┘
         │
         v
┌───────────────────┐
│  FaceTracker      │  (core/tracker.py)
│  - Update tracks  │
│  - Assign IDs     │
│  - Track movement │
└────────┬──────────┘
         │
         ├─────────────────────┐
         │                     │
         v                     v
    Active Tracks        Removed Tracks
         │                     │
         v                     v
┌───────────────────┐   ┌───────────────────┐
│  TrackManager     │   │  FaceRecognizer   │
│  - Store history  │   │  - Match faces    │
│  - Track data     │   │  - Identify       │
└───────────────────┘   │    persons        │
                        └────────┬──────────┘
                                 │
                                 v
                        ┌────────────────────┐
                        │  EntryLogger       │
                        │  - Log entry/exit  │
                        │  - Send to API     │
                        │  - Write CSV       │
                        └────────────────────┘
```

### 2. Recognition Flow

```
Removed Track
    │
    v
┌─────────────────────────┐
│  Validate Track         │
│  - Has embeddings?      │
│  - Passed counting line?│
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  Perform Recognition    │
│  - Get embeddings       │
│  - Compare with DB      │
│  - Calculate similarity │
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  Validate Result        │
│  - Check frontality     │
│  - Verify landmarks     │
└──────────┬──────────────┘
           │
           ├────────────┬──────────────┐
           │            │              │
           v            v              v
      Recognized   Unrecognized   Partial Match
           │            │              │
           v            v              v
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Log Entry   │  │ Send to API  │  │ Send to API  │
│ Create      │  │ (Unknown     │  │ (Low         │
│ Attendance  │  │  person)     │  │  confidence) │
└─────────────┘  └──────────────┘  └──────────────┘
```

## Package Responsibilities

### Core (`core/`)
**Purpose:** Face processing pipeline components

- **`detector.py`**: Face detection using InsightFace
  - Input: Frame (numpy array)
  - Output: Face bounding boxes, embeddings, landmarks

- **`tracker.py`**: Face tracking using DeepOCSORT
  - Input: Detections, frame
  - Output: Active tracks, removed tracks

- **`track_manager.py`**: Track lifecycle management
  - Stores track history (embeddings, boxes, crops, landmarks)
  - Manages track expiration
  - Provides track data retrieval

- **`recognizer.py`**: Face recognition
  - Loads face database
  - Compares embeddings using cosine similarity
  - Validates face frontality

- **`engine.py`**: Orchestrates all core components
  - Delegates to detector, tracker, recognizer
  - Manages recognition pipeline
  - Handles database updates

### API (`api/`)
**Purpose:** Backend API communication

- **`auth.py`**: Authentication service
  - Login with email/password
  - Token management
  - Session handling

- **`client.py`**: API client
  - User management (fetch, sync)
  - Attendance record creation
  - Unrecognized face submission
  - Frame upload

### Logging (`logging/`)
**Purpose:** Event logging and monitoring

- **`setup.py`**: Logging configuration
  - Structured logging with loguru
  - JSON logging support
  - Performance metrics

- **`entry_logger.py`**: Entry/exit tracking
  - Status tracking (IN/OUT)
  - Recent entries visualization
  - Coordinates with API client

- **`csv_logger.py`**: CSV file logging
  - Rotating CSV files
  - Status history

### Dashboard (`dashboard/`)
**Purpose:** Live visualization and monitoring

- **`backend.py`**: FastAPI server
  - MJPEG streaming
  - Multiple camera feeds
  - Health checks

- **`manager.py`**: Dashboard lifecycle
  - Start/stop server
  - Background thread management

- **`camera_processor.py`**: Frame streaming
  - Frame buffering
  - Stream management

- **`visualizer.py`**: Frame annotation
  - Draw bounding boxes
  - Add text overlays
  - Concatenate frames

### Storage (`storage/`)
**Purpose:** Data persistence

- **`database.py`**: Face embeddings database
  - Load/save pickle files
  - Search by name
  - Update embeddings

- **`cloud_storage.py`**: Google Cloud Storage
  - Read/write files
  - Download face images
  - Upload results

### Video (`video/`)
**Purpose:** Video stream handling

- **`stream_handler.py`**: Video stream management
  - Read frames from video/RTSP
  - Handle multiple streams
  - Frame buffering

- **`frame_processor.py`**: Post-processing
  - Add timestamps
  - Draw counting lines
  - Annotate with entry info
  - Send to dashboard

### Config (`config/`)
**Purpose:** Configuration management

- **`models.py`**: Pydantic configuration models
  - Type-safe configuration
  - Validation
  - Default values

- **`manager.py`**: Configuration bridge
  - Load from YAML
  - Environment variables
  - Backward compatibility

- **`constants.py`**: System constants
  - Default values
  - Magic numbers
  - Configuration keys

## Design Patterns Used

### 1. **Facade Pattern**
- `HBFace` provides a simple interface to the complex system
- Hides complexity of initialization and orchestration

### 2. **Strategy Pattern**
- Different recognition strategies (recognized, unrecognized, partial_match)
- Pluggable detection/tracking algorithms

### 3. **Dependency Injection** (Partial)
- Components receive dependencies rather than creating them
- Easier to test and modify

### 4. **Single Responsibility Principle**
- Each class has one reason to change
- Clear, focused responsibilities

### 5. **Separation of Concerns**
- UI (dashboard) separate from business logic (core)
- I/O (API, storage) separate from processing
- Configuration separate from implementation

## Key Improvements

### Before Refactoring
- **God objects**: `engine.py` (592 LOC), `entry_logger.py` (509 LOC)
- **Tight coupling**: Everything depended on everything
- **Hard to test**: Components couldn't be isolated
- **Difficult to understand**: Too many responsibilities per file

### After Refactoring
- **Focused modules**: Largest file is ~300 LOC
- **Loose coupling**: Clear interfaces between components
- **Testable**: Components can be tested independently
- **Clear structure**: Easy to find and understand code

## Testing Strategy

### Unit Tests
```python
# Test detector
def test_face_detector_initialization():
    detector = FaceDetector(gpu_id=0)
    assert detector.model is not None

def test_face_detection():
    detector = FaceDetector(gpu_id=0)
    frame = load_test_image()
    faces = detector.detect(frame)
    assert len(faces) > 0
```

### Integration Tests
```python
# Test engine pipeline
def test_engine_tracking():
    engine = FaceEngine(test_config)
    frame = load_test_frame()
    active, removed = engine.track(frame)
    assert isinstance(active, list)
```

### End-to-End Tests
```python
# Test full system
def test_hbface_run():
    hbface = HBFace(
        cam_types=["IN"],
        video_path="test_video.mp4"
    )
    # Run for N frames
    # Verify results
```

## Future Enhancements

1. **Async API calls**: Use `aiohttp` for non-blocking API requests
2. **Caching layer**: Add Redis for faster lookups
3. **Metrics**: Add Prometheus metrics
4. **Health checks**: Monitor component health
5. **Configuration hot-reload**: Update config without restart
6. **Plugin system**: Allow custom detection/recognition algorithms
7. **Distributed processing**: Scale across multiple machines
