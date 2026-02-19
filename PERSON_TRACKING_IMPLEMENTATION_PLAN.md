# Person Tracking & Phone Usage Detection - Implementation Plan

**Project:** Smart Office Person Tracking with Phone Usage Detection
**Timeline:** 3 weeks (MVP) + Post-MVP Production Features
**Date Created:** 2025-11-23
**Last Updated:** 2025-11-24
**Status:** ✅ Week 2 Complete - 66% Done - Integration Phase Next

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Context](#project-context)
3. [Requirements](#requirements)
4. [Architecture Overview](#architecture-overview)
5. [Technical Decisions](#technical-decisions)
6. [Implementation Plan](#implementation-plan)
7. [Configuration](#configuration)
8. [Performance Targets](#performance-targets)
9. [Success Criteria](#success-criteria)
10. [Post-MVP Features](#post-mvp-features)
11. [Risk Mitigation](#risk-mitigation)

---

## 📊 Implementation Progress

### Overall Progress: 66% Complete (Week 2/3)

| Phase | Status | Completion |
|-------|--------|------------|
| **Week 1: Core Pipeline** | ✅ **COMPLETE** | 100% |
| **Week 2: Intelligence Layer** | ✅ **COMPLETE** | 100% |
| **Week 3: Integration & Testing** | 🔄 **IN PROGRESS** | 0% |
| **Post-MVP: Production Features** | ⏳ **NOT STARTED** | 0% |

### ✅ Completed (Week 1)

#### Day 1-2: Project Setup ✅
- [x] Project structure created (`src/person_tracking/`)
- [x] Pydantic configuration models (`config/models.py`)
- [x] Configuration manager with YAML loading (`config/manager.py`)
- [x] Sample YAML configuration (`configs/person_tracking/config.yaml`)
- [x] Requirements file (`requirements-person-tracking.txt`)
- [x] Environment template (`.env.person-tracking.example`)
- [x] README documentation

#### Day 2-3: Person Detection ✅
- [x] `PersonDetector` class implemented (`core/person_detector.py`)
- [x] YOLOv8s-Pose integration
- [x] 17 COCO keypoints extraction
- [x] Bounding box + pose keypoint detection
- [x] Visualization utilities

#### Day 3-4: Person Tracking ✅
- [x] `PersonTracker` class implemented (`core/person_tracker.py`)
- [x] BoT-SORT-style tracking algorithm
- [x] Stable TrackID assignment
- [x] Track lifecycle management
- [x] `PersonTrackManager` class (`core/track_manager.py`)
- [x] Track history storage (bboxes, keypoints, trajectories)
- [x] Identity and phone usage state management

#### Day 4-5: Face Recognition Integration ✅
- [x] Face recognition adapter (`core/face_adapter.py`)
- [x] **Reuses existing `FaceDetector` and `FaceRecognition` classes**
- [x] Person ROI cropping utilities
- [x] Integration with shared pgvector database
- [x] Simple helper functions (no wrapper overhead)

#### Day 5-7: Phone Detection ✅
- [x] `PhoneDetector` class implemented (`core/phone_detector.py`)
- [x] YOLOv8n for phone detection
- [x] COCO class 67 (cell phone) filtering
- [x] Basic spatial association with person bboxes
- [x] Visualization utilities

---

### ✅ Completed (Week 2)

#### Day 8-9: Identity Manager ✅
- [x] `IdentityManager` class implemented (`core/identity_manager.py`)
- [x] Confidence-weighted temporal voting algorithm
- [x] Sliding window approach (M=5 frames)
- [x] 60% consensus requirement for identity lock
- [x] Average similarity scoring
- [x] Identity lock/unlock capabilities
- [x] Voting status tracking

#### Day 9-10: Phone Usage Spatial Logic ✅
- [x] `PhoneUsageSpatialLogic` class implemented (`core/phone_usage_logic.py`)
- [x] Pose-based phone association using COCO keypoints
- [x] Upper body zone detection (shoulders to above head)
- [x] Hand-to-phone proximity check (20cm threshold)
- [x] Phone-to-head proximity check (25cm threshold for calling)
- [x] Requires 2 of 3 checks to pass
- [x] Pixel-to-meter conversion for accurate distance calculation

#### Day 10-11: Temporal Filtering ✅
- [x] `PhoneUsageFilter` class implemented (`core/phone_usage_filter.py`)
- [x] Sliding window temporal smoothing (N=8 frames)
- [x] 75% consensus requirement (6/8 frames)
- [x] Minimum 500ms duration enforcement
- [x] State machine implementation (NOT_USING ↔ USING)
- [x] Usage duration tracking
- [x] Reduces false positives effectively

#### Day 11-12: State Management ✅
- [x] `PersonStateManager` class implemented (`core/state_manager.py`)
- [x] Complete person state tracking (identity, phone usage, timestamps)
- [x] Event system with 6 event types:
  - [x] person_entered
  - [x] person_exited
  - [x] identity_locked
  - [x] identity_changed
  - [x] phone_usage_started
  - [x] phone_usage_stopped
- [x] Event queue management
- [x] State transition logging
- [x] Statistics and reporting

#### Day 12-13: Mock API ✅
- [x] FastAPI application (`api/app.py`)
- [x] Pydantic models for request/response validation
- [x] POST `/api/v1/person-tracking/events` endpoint
- [x] GET `/api/v1/person-tracking/status` endpoint
- [x] GET `/health` health check endpoint
- [x] GET `/api/v1/person-tracking/events` (query events)
- [x] DELETE `/api/v1/person-tracking/reset` (testing)
- [x] GET `/api/v1/person-tracking/statistics`
- [x] OpenAPI/Swagger documentation
- [x] In-memory mock data store
- [x] Error handling

#### Day 13-14: Logging ✅
- [x] `CSVLogger` class implemented (`logging/csv_logger.py`)
- [x] Event logging to CSV files
- [x] Summary logging per track
- [x] Automatic file rotation
- [x] Organized by client/camera structure
- [x] Batch event logging support
- [x] `setup_logging` function (`logging/setup.py`)
- [x] Structured logging with Loguru
- [x] Console + file output
- [x] Log rotation and retention policies

---

### 🔄 In Progress

Currently at: **Week 3 - Integration & Testing**

### ⏳ Pending

#### Week 3: Integration & Testing (Days 15-21)
- [ ] Main Orchestrator (PersonTrackingEngine)
- [ ] Frame Annotation (FrameAnnotator)
- [ ] Video Stream Integration
- [ ] Integration Testing
- [ ] Configuration Management
- [ ] Docker Deployment
- [ ] Basic Documentation

#### Post-MVP
- [ ] Backend API Integration
- [ ] PostgreSQL Time-Series Storage
- [ ] Redis Pub/Sub
- [ ] Dynamic Configuration
- [ ] Performance Optimization
- [ ] Advanced Testing
- [ ] Monitoring & Observability
- [ ] Comprehensive Documentation

---

## Executive Summary

### Objective
Extend the existing Smart Office face recognition system by adding **person tracking**, **face identity association**, and **phone usage detection** capabilities in a separate microservice architecture.

### Key Features
- **Person Detection & Tracking:** YOLOv8-Pose for person detection with 17 keypoints + BoT-SORT for multi-object tracking
- **Face Recognition Integration:** Reuse existing InsightFace pipeline within person ROIs, share pgvector embedding database
- **Phone Usage Detection:** YOLOv8n for phone detection + pose-based spatial/temporal association logic
- **Identity Locking:** Confidence-weighted temporal voting (5 frames, 60% consensus)
- **Temporal Filtering:** 8-frame sliding window to reduce false positives (75% consensus, 500ms minimum)
- **Multi-Person Support:** Independent tracking per camera, handles 10+ people simultaneously
- **Real-time Performance:** 10-15 FPS on GPU (balanced configuration)

### Timeline
- **Week 1:** Core pipeline (detection, tracking, basic phone detection)
- **Week 2:** Intelligence layer (temporal logic, state management, API)
- **Week 3:** Integration, testing, deployment
- **Post-MVP:** Production features (backend API, time-series DB, monitoring)

---

## Project Context

### Current System
The Smart Office system currently has a production-ready face recognition service that:
- Detects faces using InsightFace (RetinaFace detector + ArcFace embeddings)
- Tracks attendance with IN/OUT logging
- Stores embeddings in PostgreSQL with pgvector extension
- Supports multi-camera RTSP streams
- Integrates with backend API for user management

**Entry Point:** `examples/clients/main.py`
**Main Class:** `HBFace` in `src/face_recognition/hbface.py`

### Gap
The current system tracks faces but does not:
- Maintain stable person tracking across frames (uses face-level tracking)
- Detect phone usage
- Associate phone usage with person identity
- Provide per-person state tracking

---

## Requirements

### Functional Requirements

#### FR1: Person Detection & Tracking
- Detect all persons in video frame with bounding boxes
- Extract 17 pose keypoints (COCO format): nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- Assign unique TrackID to each person
- Maintain TrackID across frames (handle occlusions, brief disappearances)
- Track trajectory and history per person

#### FR2: Face Identity Association
- Detect faces within person ROI (not full frame)
- Extract face embeddings using existing InsightFace pipeline
- Match embeddings against shared pgvector database
- Lock identity after M consecutive frames (default: 5 frames, 60% consensus)
- Handle identity changes and multiple people

#### FR3: Phone Usage Detection
- Detect phones (cell phones) in video frame
- Associate phone with person using pose keypoints:
  - Phone in upper body zone (shoulders to 30cm above head)
  - Hand within 20cm of phone
  - Phone within 25cm of head (for calling)
  - Require 2 of 3 checks to pass
- Temporal filtering: require N consecutive frames (default: 8 frames, 75% consensus, 500ms minimum)
- Report phone usage state per person: `using_phone: True/False`

#### FR4: Multi-Person Support
- Handle 10+ people in single frame
- Independent tracking per camera (no cross-camera tracking in MVP)
- Correctly attribute phone usage when multiple people present
- Handle edge cases: phone handoff, multiple phones, brief occlusions

#### FR5: State Tracking & Events
- Maintain per-person state: `{track_id, identity, identity_locked, using_phone, confidence, timestamps}`
- Emit events:
  - `identity_locked`: When person identity confirmed
  - `identity_changed`: When identity changes (rare)
  - `phone_usage_started`: When person starts using phone
  - `phone_usage_stopped`: When person stops using phone
  - `person_entered`: New person detected
  - `person_exited`: Person left frame

#### FR6: Configuration & Tunability
- Per-camera configuration via backend API
- Tunable parameters:
  - Face match threshold (0.0-1.0)
  - Identity lock frames (M)
  - Phone usage confirmation frames (N)
  - Spatial thresholds (hand distance, head distance, zone overlap)
  - Model sizes (yolov8n/s/m/l)
- Backwards compatible with existing face recognition service

### Non-Functional Requirements

#### NFR1: Performance
- Target: 10-15 FPS per camera (balanced configuration)
- Latency: <100ms per frame
- GPU Memory: <4GB per camera
- Scalability: Support 4+ cameras on single GPU

#### NFR2: Accuracy
- Face recognition: >95% accuracy (reuse

 existing pipeline)
- Person tracking: <5% ID switches per minute
- Phone usage detection: >90% precision, >85% recall (after temporal filtering)

#### NFR3: Reliability
- Graceful degradation: If face not detected, continue tracking with TrackID
- Error handling: Handle stream disconnections, model failures
- Data consistency: Shared database access must not conflict with face recognition service

#### NFR4: Maintainability
- Modular architecture (separate detector, tracker, recognizer, logic engine)
- Configuration-driven (YAML + env vars)
- Comprehensive logging (console + CSV + structured logs)
- Docker deployment

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│         PERSON TRACKING MICROSERVICE                    │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  PersonTrackingEngine (Orchestrator)           │     │
│  │  ├─ PersonDetector (YOLOv8-Pose)               │     │
│  │  ├─ PersonTracker (BoT-SORT)                   │     │
│  │  ├─ FaceRecognitionAdapter                     │     │
│  │  │   └─ Wraps existing InsightFace pipeline    │     │
│  │  ├─ PhoneDetector (YOLOv8n)                    │     │
│  │  ├─ IdentityManager (temporal voting)          │     │
│  │  ├─ PhoneUsageFilter (temporal smoothing)      │     │
│  │  └─ PersonStateManager (state tracking)        │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │  FastAPI Service (Mock in MVP)                 │     │
│  │  ├─ POST /api/v1/person-tracking/events        │     │
│  │  ├─ GET  /api/v1/person-tracking/status        │     │
│  │  └─ GET  /health                               │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
           ↓                    ↓                ↓
    ┌──────────┐        ┌──────────┐      ┌──────────┐
    │PostgreSQL│        │  Redis   │      │ Backend  │
    │ (shared) │        │(pub/sub) │      │   API    │
    │ pgvector │        │  (post)  │      │  (post)  │
    └──────────┘        └──────────┘      └──────────┘
```

### Data Flow

```
Video Frame
    ↓
┌───────────────────────────────────────────────────────┐
│ PersonDetector (YOLOv8-Pose)                          │
│   ├─ BBox: [x1, y1, x2, y2, confidence]               │
│   └─ Keypoints: 17 points (nose, eyes, wrists, etc.)  │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│ PersonTracker (BoT-SORT)                              │
│   ├─ Assign TrackID to each person                    │
│   ├─ Update trajectories                              │
│   └─ Return: (active_tracks, removed_tracks)          │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│ Parallel Processing for Each Track:                   │
│                                                        │
│ ┌─────────────────────────────────────────────────┐   │
│ │ FaceRecognitionAdapter                          │   │
│ │   1. Crop person ROI from frame                 │   │
│ │   2. Run RetinaFace detection inside ROI        │   │
│ │   3. Extract ArcFace embedding (512D)           │   │
│ │   4. Match against pgvector database            │   │
│ │   5. IdentityManager: Temporal voting           │   │
│ │      - Sliding window: M=5 frames               │   │
│ │      - Consensus: 60% (3/5 frames)              │   │
│ │      - Lock identity when conditions met        │   │
│ └─────────────────────────────────────────────────┘   │
│                                                        │
│ ┌─────────────────────────────────────────────────┐   │
│ │ PhoneDetector (YOLOv8n)                         │   │
│ │   1. Detect all phones in frame                 │   │
│ │   2. Spatial Association (pose-based):          │   │
│ │      a. Phone in upper body zone?               │   │
│ │         (using shoulders/nose keypoints)        │   │
│ │      b. Hand within 20cm of phone?              │   │
│ │         (using wrist keypoints)                 │   │
│ │      c. Phone within 25cm of head?              │   │
│ │         (using nose keypoint)                   │   │
│ │      d. Require 2 of 3 checks to pass           │   │
│ │   3. PhoneUsageFilter: Temporal smoothing       │   │
│ │      - Sliding window: N=8 frames               │   │
│ │      - Consensus: 75% (6/8 frames)              │   │
│ │      - Minimum duration: 500ms                  │   │
│ └─────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│ PersonStateManager                                     │
│   - Update state: {track_id, identity, using_phone}   │
│   - Emit events: identity_locked, phone_usage_*       │
└───────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────┐
│ Output Channels:                                       │
│   ├─ Mock API: Log events to console (MVP)            │
│   ├─ CSV Logger: Save to disk for auditing            │
│   ├─ Frame Annotator: Draw visualization              │
│   └─ (Post-MVP) Backend API, Redis, PostgreSQL        │
└───────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **PersonDetector** | Detect persons with bbox + 17 keypoints | YOLOv8s-pose |
| **PersonTracker** | Assign and maintain TrackIDs | BoT-SORT (ultralytics) |
| **FaceRecognitionAdapter** | Crop person ROI, run face detection/recognition | InsightFace (RetinaFace + ArcFace) |
| **PhoneDetector** | Detect phones in frame | YOLOv8n (COCO 'cell phone') |
| **IdentityManager** | Lock identity using temporal voting | Sliding window + consensus logic |
| **PhoneUsageFilter** | Confirm phone usage using temporal filtering | Sliding window + state machine |
| **PersonStateManager** | Track per-person state across frames | In-memory dict + event emitter |
| **PersonTrackingEngine** | Orchestrate entire pipeline | Main controller |
| **FrameAnnotator** | Visualize results (bboxes, keypoints, labels) | OpenCV drawing |
| **CSVLogger** | Log events to CSV for auditing | Python csv module |

---

## Technical Decisions

### Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Service Type** | Separate microservice | Better isolation, independent scaling, doesn't affect existing face recognition service |
| **Tracking Approach** | YOLOv8-Pose + BoT-SORT | Pose keypoints enable accurate phone attribution; BoT-SORT has low ID switches |
| **Face Recognition** | Shared pgvector database | Consistent identities with existing service; avoid duplicate embeddings |
| **Phone Logic** | Pose-based spatial association | Accurate + real-time without additional ML inference |
| **Temporal Strategy** | Confidence-weighted voting | More robust than simple frame counting; handles detector noise |
| **Multi-Camera** | Independent tracking per camera | Simpler MVP; can unify later if needed |
| **Performance Target** | 10-15 FPS (balanced) | YOLOv8s-pose + YOLOv8n-phone on GPU |
| **API Strategy** | Mock endpoints → Real API later | MVP can work standalone; integrate backend when ready |
| **Configuration** | YAML + env vars → API later | Simple MVP; add dynamic config reload post-MVP |

### Model Selection

| Purpose | Model | Size | Speed | Accuracy | Rationale |
|---------|-------|------|-------|----------|-----------|
| **Person Detection** | YOLOv8s-pose | 11 MB | 12-15 FPS | High | Balanced speed/accuracy; provides 17 keypoints |
| **Person Tracking** | BoT-SORT | - | - | High | Lower ID switches than SORT/DeepSORT; handles occlusions |
| **Face Recognition** | InsightFace (buffalo_l) | 60 MB | - | Very High | Existing pipeline; proven accuracy |
| **Phone Detection** | YOLOv8n | 6 MB | 18-22 FPS | Medium | Fast; COCO 'cell phone' class; minimal overhead |

**Alternative Configurations:**

| Profile | Person Model | Phone Model | FPS | GPU Memory | Use Case |
|---------|-------------|-------------|-----|------------|----------|
| Fast | YOLOv8n-pose | YOLOv8n | 18-22 | 2.5 GB | Real-time monitoring |
| **Balanced (MVP)** | **YOLOv8s-pose** | **YOLOv8n** | **12-15** | **3.5 GB** | **Production** |
| Accurate | YOLOv8m-pose | YOLOv8s | 8-10 | 5.5 GB | Forensic/compliance |

### Temporal Logic Design

#### Identity Locking: Confidence-Weighted Temporal Voting

**Why better than simple frame counting:**
- Handles conflicting detections (same person matched to different identities)
- Uses similarity scores for confidence
- Prevents premature locking from false positives
- Configurable per camera

**Algorithm:**
```python
class IdentityManager:
    def update_identity(self, track_id, face_match):
        """
        Maintains sliding window of identity votes
        """
        window = self.identity_votes[track_id]
        window.append({
            'name': face_match['name'],
            'similarity': face_match['similarity'],
            'timestamp': time.time()
        })

        # Keep only last M frames (default 5)
        window = window[-self.M:]

        # Count votes by identity
        votes = Counter([v['name'] for v in window])
        top_identity, count = votes.most_common(1)[0]

        # Lock if consensus reached
        if (count >= self.M * 0.60 and              # 60% consensus (3/5)
            len(window) >= self.M and
            window[-1]['timestamp'] - window[0]['timestamp'] >= 0.333):  # 333ms

            # Calculate confidence
            avg_similarity = np.mean([v['similarity']
                                     for v in window
                                     if v['name'] == top_identity])

            self.locked_identities[track_id] = {
                'name': top_identity,
                'confidence': avg_similarity,
                'locked_at': time.time()
            }
```

**Parameters:**
- `M = 5` frames (configurable)
- Consensus threshold: 60% (3/5 frames)
- Minimum window duration: 333ms (~5 frames at 15 FPS)

#### Phone Usage: Pose-Based Spatial + Temporal Filtering

**Spatial Association Logic:**

```python
def calculate_phone_usage_score(person_pose, phone_bbox, config):
    """
    Multi-signal phone usage detection using pose keypoints
    """
    keypoints = person_pose['keypoints']  # 17 keypoints from YOLOv8-Pose

    # Extract relevant keypoints (COCO format)
    nose = keypoints[0]              # Head position
    left_wrist = keypoints[9]        # Left hand
    right_wrist = keypoints[10]      # Right hand
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # Define upper body zone
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    upper_body_top = nose + np.array([0, -0.30])  # 30cm above head
    upper_body_zone = BBox(
        x1=min(left_shoulder[0], right_shoulder[0]) - 0.2,
        y1=upper_body_top[1],
        x2=max(left_shoulder[0], right_shoulder[0]) + 0.2,
        y2=shoulder_midpoint[1] + 0.3
    )

    # Check 1: Phone in upper body zone?
    zone_check = phone_bbox.intersects(upper_body_zone)

    # Check 2: Hand near phone?
    phone_center = phone_bbox.center
    left_hand_dist = np.linalg.norm(left_wrist - phone_center)
    right_hand_dist = np.linalg.norm(right_wrist - phone_center)
    hand_check = min(left_hand_dist, right_hand_dist) < config.hand_distance_threshold

    # Check 3: Phone near head? (calling)
    phone_head_dist = np.linalg.norm(phone_center - nose)
    head_check = phone_head_dist < config.head_distance_threshold

    # Require 2 of 3 checks
    checks_passed = sum([zone_check, hand_check, head_check])

    return {
        'score': checks_passed >= config.required_checks,
        'confidence': checks_passed / 3.0,
        'details': {
            'zone': zone_check,
            'hand': hand_check,
            'head': head_check
        }
    }
```

**Temporal Filtering:**

```python
class PhoneUsageFilter:
    def update(self, track_id, spatial_score):
        """
        Sliding window temporal filtering for phone usage
        """
        window = self.usage_history[track_id]
        window.append({
            'using_phone': spatial_score['score'],
            'timestamp': time.time()
        })

        # Keep only last N frames (default 8)
        window = window[-self.N:]

        # Check consensus
        usage_count = sum([1 for w in window if w['using_phone']])
        consensus = usage_count / len(window)

        # Check duration
        if len(window) >= self.N:
            duration_ms = (window[-1]['timestamp'] - window[0]['timestamp']) * 1000
        else:
            duration_ms = 0

        # Confirm usage if:
        # 1. At least N frames collected
        # 2. ≥75% frames show usage (6/8)
        # 3. Duration ≥500ms
        if (len(window) >= self.N and
            consensus >= 0.75 and
            duration_ms >= 500):
            return True
        else:
            return False
```

**Parameters:**
- `N = 8` frames (configurable)
- Consensus threshold: 75% (6/8 frames)
- Minimum duration: 500ms (~8 frames at 15 FPS)
- Required spatial checks: 2 of 3 (zone, hand, head)

**Spatial Thresholds:**
- Hand-to-phone distance: 20cm (0.20m)
- Phone-to-head distance: 25cm (0.25m)
- Upper body zone margin: 30cm above head

---

## Implementation Plan

### Timeline Overview

| Phase | Duration | Deliverables | Status |
|-------|----------|--------------|--------|
| **Week 1: Core Pipeline** | Days 1-7 | Person detection, tracking, basic phone detection | ✅ **COMPLETE** |
| **Week 2: Intelligence** | Days 8-14 | Temporal logic, state management, API | ✅ **COMPLETE** |
| **Week 3: Integration** | Days 15-21 | Orchestration, testing, deployment | 🔄 **IN PROGRESS** |
| **Post-MVP** | Ongoing | Production features, optimization | ⏳ **PENDING** |

---

### ✅ Week 1: Core Pipeline (Days 1-7) - COMPLETE

#### Day 1-2: Project Setup
**Goal:** Create project structure and Docker environment

**Tasks:**
1. Create project structure:
   ```
   src/person_tracking/
   ├── __init__.py
   ├── main.py
   ├── engine.py
   ├── core/
   ├── api/
   ├── config/
   ├── video/
   └── logging/
   ```
2. Create `Dockerfile` (multi-stage with CUDA support)
3. Create `docker-compose.yml` with services: person-tracking, postgres, redis
4. Create `requirements.txt` with dependencies:
   - ultralytics (YOLOv8)
   - torch, torchvision
   - opencv-python
   - insightface
   - psycopg2-binary
   - redis
   - fastapi, uvicorn
   - pydantic
   - numpy, scipy
   - loguru
5. Create `.env.example` with environment variables
6. Basic configuration management (Pydantic models)

**Deliverables:**
- ✅ Project structure
- ✅ Docker environment
- ✅ Dependencies installed

---

#### Day 2-3: Person Detection
**Goal:** Implement YOLOv8-Pose person detection

**Tasks:**
1. Create `src/person_tracking/core/person_detector.py`
2. Implement `PersonDetector` class:
   ```python
   class PersonDetector:
       def __init__(self, model_size='s', confidence_threshold=0.5):
           self.model = YOLO(f'yolov8{model_size}-pose.pt')

       def detect_persons(self, frame):
           """
           Returns: List[{
               'bbox': [x1, y1, x2, y2, confidence],
               'keypoints': np.array(17, 3),  # x, y, confidence
               'person_id': int
           }]
           """
   ```
3. Download and cache `yolov8s-pose.pt` model
4. Test on sample video
5. Verify keypoint extraction (17 COCO keypoints)

**Deliverables:**
- ✅ PersonDetector class
- ✅ Test script with sample video
- ✅ Visualize bboxes and keypoints

---

#### Day 3-4: Person Tracking
**Goal:** Implement BoT-SORT tracking with stable TrackIDs

**Tasks:**
1. Create `src/person_tracking/core/person_tracker.py`
2. Implement `PersonTracker` class using ultralytics tracking:
   ```python
   class PersonTracker:
       def __init__(self, tracker_type='botsort'):
           self.tracker_type = tracker_type

       def update(self, detections, frame):
           """
           Returns: (active_tracks, removed_tracks)
           active_tracks: List[{track_id, bbox, keypoints, ...}]
           removed_tracks: List[{track_id, ...}] (left frame)
           """
   ```
3. Create `src/person_tracking/core/track_manager.py`
4. Implement `PersonTrackManager`:
   ```python
   class PersonTrackManager:
       def __init__(self):
           self.track_history = {}  # {track_id: history}

       def update_track(self, track_id, bbox, keypoints, frame):
           """Store track data per frame"""

       def get_track_data(self, track_id):
           """Retrieve all data for a track"""
   ```
5. Test tracking on video with multiple people

**Deliverables:**
- ✅ PersonTracker class
- ✅ PersonTrackManager class
- ✅ Stable TrackIDs across frames
- ✅ Trajectory visualization

---

#### Day 4-5: Face Recognition Adapter
**Goal:** Integrate existing face recognition within person ROIs

**Tasks:**
1. Create `src/person_tracking/core/face_adapter.py`
2. Implement `FaceRecognitionAdapter`:
   ```python
   class FaceRecognitionAdapter:
       def __init__(self, pgvector_store, match_threshold=0.3):
           self.detector = FaceDetector()  # Existing InsightFace
           self.recognizer = FaceRecognition()  # Existing

       def recognize_person(self, frame, person_bbox):
           """
           Crop person ROI, detect face, extract embedding, match
           Returns: {
               'name': str,
               'similarity': float,
               'recognized': bool
           }
           """
   ```
3. Integrate with shared pgvector database (`org_{client_slug}.face_embeddings`)
4. Test face recognition inside person ROIs
5. Verify no conflicts with existing face recognition service

**Deliverables:**
- ✅ FaceRecognitionAdapter class
- ✅ Integration with pgvector
- ✅ Test face matching

---

#### Day 5-7: Phone Detection
**Goal:** Implement phone detection and basic spatial association

**Tasks:**
1. Create `src/person_tracking/core/phone_detector.py`
2. Implement `PhoneDetector` class:
   ```python
   class PhoneDetector:
       def __init__(self, model_size='n', confidence_threshold=0.4):
           self.model = YOLO(f'yolov8{model_size}.pt')

       def detect_phones(self, frame):
           """
           Returns: List[{
               'bbox': [x1, y1, x2, y2, confidence],
               'class': 'cell phone'
           }]
           """
   ```
3. Filter YOLO results for 'cell phone' class (class_id=67 in COCO)
4. Implement basic spatial association (phone bbox overlaps person bbox)
5. Test phone detection accuracy

**Deliverables:**
- ✅ PhoneDetector class
- ✅ Basic phone-to-person association
- ✅ Visualization of phone bboxes

---

### ✅ Week 2: Intelligence & Integration (Days 8-14) - COMPLETE

#### Day 8-9: Identity Manager
**Goal:** Implement confidence-weighted temporal voting for identity locking

**Tasks:**
1. Create `src/person_tracking/core/identity_manager.py`
2. Implement `IdentityManager`:
   ```python
   class IdentityManager:
       def __init__(self, M=5, consensus_threshold=0.60):
           self.identity_votes = defaultdict(list)
           self.locked_identities = {}

       def update_identity(self, track_id, face_match):
           """Add vote to sliding window"""

       def is_identity_locked(self, track_id):
           """Check if identity has reached consensus"""

       def get_locked_identity(self, track_id):
           """Get locked identity with confidence"""
   ```
3. Test with multi-person videos
4. Verify identity lock stability
5. Test edge cases (identity changes, conflicting detections)

**Deliverables:**
- ✅ IdentityManager class
- ✅ Identity locking with temporal voting
- ✅ Unit tests

---

#### Day 9-10: Phone Usage Spatial Logic
**Goal:** Implement pose-based spatial association for phone usage

**Tasks:**
1. Create spatial association functions in `phone_detector.py`:
   ```python
   def calculate_upper_body_zone(keypoints):
       """Define zone using shoulders and nose"""

   def calculate_hand_to_phone_distance(keypoints, phone_bbox):
       """Distance from wrists to phone center"""

   def calculate_head_to_phone_distance(keypoints, phone_bbox):
       """Distance from nose to phone center"""

   def is_phone_associated_with_person(keypoints, phone_bbox, config):
       """
       Check 3 conditions:
       1. Phone in upper body zone
       2. Hand within 20cm of phone
       3. Phone within 25cm of head

       Return True if ≥2 conditions met
       """
   ```
2. Test spatial logic with different phone usage scenarios:
   - Texting (hand to phone)
   - Calling (phone to head)
   - Browsing (phone in upper body)
3. Tune thresholds based on test results

**Deliverables:**
- ✅ Spatial association logic
- ✅ Configurable thresholds
- ✅ Test cases for different scenarios

---

#### Day 10-11: Temporal Filtering
**Goal:** Implement temporal filtering for phone usage confirmation

**Tasks:**
1. Create `src/person_tracking/core/phone_usage_filter.py`
2. Implement `PhoneUsageFilter`:
   ```python
   class PhoneUsageFilter:
       def __init__(self, N=8, consensus_threshold=0.75, min_duration_ms=500):
           self.usage_history = defaultdict(list)

       def update(self, track_id, spatial_score):
           """Add frame to sliding window"""

       def is_using_phone(self, track_id):
           """
           Check if phone usage confirmed:
           - ≥N frames collected
           - ≥75% frames show usage
           - Duration ≥500ms
           """

       def get_usage_confidence(self, track_id):
           """Return confidence score 0-1"""
   ```
3. Implement state machine:
   - `not_using` → `using` (after N frames)
   - `using` → `not_using` (after N frames without phone)
4. Test with videos showing intermittent phone usage

**Deliverables:**
- ✅ PhoneUsageFilter class
- ✅ State machine implementation
- ✅ Unit tests

---

#### Day 11-12: State Management
**Goal:** Centralized per-person state tracking and event emission

**Tasks:**
1. Create `src/person_tracking/core/state_manager.py`
2. Implement `PersonStateManager`:
   ```python
   class PersonStateManager:
       def __init__(self):
           self.person_states = {}  # {track_id: state}
           self.event_queue = []

       def update_state(self, track_id, identity, using_phone, confidence):
           """Update person state and emit events if changed"""

       def get_state(self, track_id):
           """Get current state for track"""

       def get_all_states(self):
           """Get all active person states"""

       def emit_event(self, event_type, track_id, data):
           """Add event to queue"""

       def get_events(self):
           """Retrieve and clear event queue"""
   ```
3. Define state structure:
   ```python
   {
       'track_id': int,
       'identity': str or None,
       'identity_locked': bool,
       'identity_confidence': float,
       'using_phone': bool,
       'phone_confidence': float,
       'first_seen': timestamp,
       'last_seen': timestamp,
       'total_frames': int
   }
   ```
4. Define event types:
   - `identity_locked`
   - `identity_changed`
   - `phone_usage_started`
   - `phone_usage_stopped`
   - `person_entered`
   - `person_exited`

**Deliverables:**
- ✅ PersonStateManager class
- ✅ Event system
- ✅ State tracking

---

#### Day 12-13: Mock API
**Goal:** Create FastAPI endpoints (mock implementation)

**Tasks:**
1. Create `src/person_tracking/api/app.py`
2. Implement FastAPI application:
   ```python
   from fastapi import FastAPI

   app = FastAPI(title="Person Tracking API")

   @app.post("/api/v1/person-tracking/events")
   async def create_event(event: EventCreate):
       """Log phone usage event (mock: print to console)"""

   @app.get("/api/v1/person-tracking/status")
   async def get_status():
       """Return current person states"""

   @app.get("/health")
   async def health_check():
       """Health check endpoint"""
   ```
3. Define Pydantic models:
   ```python
   class EventCreate(BaseModel):
       track_id: int
       person_name: Optional[str]
       event_type: str
       using_phone: bool
       confidence: float
       camera_id: int
       timestamp: datetime
   ```
4. Add OpenAPI/Swagger documentation
5. Test API endpoints with curl/Postman

**Deliverables:**
- ✅ FastAPI application
- ✅ Mock endpoints
- ✅ API documentation

---

#### Day 13-14: Basic Logging
**Goal:** Implement CSV and console logging

**Tasks:**
1. Create `src/person_tracking/logging/csv_logger.py`
2. Implement `CSVLogger`:
   ```python
   class CSVLogger:
       def __init__(self, output_dir, client_slug):
           self.filepath = f"{output_dir}/{client_slug}/events.csv"

       def log_event(self, track_id, person_name, phone_usage,
                     camera_id, confidence, timestamp):
           """Append event to CSV"""
   ```
3. CSV format:
   ```csv
   timestamp,track_id,person_name,phone_usage,camera_id,confidence,duration_seconds
   2025-11-23 10:30:15,1,John Doe,True,1,0.95,12.5
   ```
4. Set up structured logging with loguru:
   ```python
   from loguru import logger

   logger.add("logs/person_tracking.log",
              rotation="1 day",
              level="INFO")
   ```
5. Log key events: detection counts, FPS, identity locks, phone usage changes

**Deliverables:**
- ✅ CSVLogger class
- ✅ Structured logging with loguru
- ✅ Log files in `volumes/storage/person-tracking/logs/`

---

### 🔄 Week 3: Integration, Testing & Deployment (Days 15-21) - IN PROGRESS

#### Day 15-16: Main Orchestrator
**Goal:** Implement PersonTrackingEngine to coordinate entire pipeline

**Tasks:**
1. Create `src/person_tracking/engine.py`
2. Implement `PersonTrackingEngine`:
   ```python
   class PersonTrackingEngine:
       def __init__(self, config):
           self.person_detector = PersonDetector(...)
           self.person_tracker = PersonTracker(...)
           self.face_adapter = FaceRecognitionAdapter(...)
           self.phone_detector = PhoneDetector(...)
           self.identity_manager = IdentityManager(...)
           self.phone_filter = PhoneUsageFilter(...)
           self.state_manager = PersonStateManager(...)

       def process_frame(self, frame, frame_num):
           """
           Complete pipeline:
           1. Detect persons
           2. Update tracker
           3. For each active track:
              a. Recognize face (identity manager)
              b. Detect phones (spatial association)
              c. Update phone filter
           4. Update state manager
           5. Process events
           6. Return annotated frame
           """

       def run(self):
           """Main event loop"""
   ```
3. Implement frame processing pipeline
4. Add performance monitoring (FPS, latency)
5. Test end-to-end pipeline

**Deliverables:**
- ✅ PersonTrackingEngine class
- ✅ Complete pipeline working
- ✅ Performance metrics

---

#### Day 16-17: Frame Annotation
**Goal:** Implement visualization for debugging and monitoring

**Tasks:**
1. Create `src/person_tracking/video/frame_annotator.py`
2. Implement `FrameAnnotator`:
   ```python
   class FrameAnnotator:
       def draw_person_bbox(self, frame, bbox, track_id, identity):
           """Draw person bounding box with ID and name"""

       def draw_keypoints(self, frame, keypoints):
           """Draw 17 keypoint skeleton"""

       def draw_phone_bbox(self, frame, phone_bbox):
           """Draw phone bounding box"""

       def draw_phone_status(self, frame, track_id, using_phone):
           """Show phone usage status text"""

       def draw_trajectory(self, frame, trajectory):
           """Draw person movement path"""

       def annotate_frame(self, frame, person_states):
           """Complete frame annotation"""
   ```
3. Color coding:
   - Green bbox: Person with locked identity
   - Yellow bbox: Person with tentative identity
   - Red bbox: Unidentified person
   - Blue bbox: Phone detected
4. Test visualization with sample videos

**Deliverables:**
- ✅ FrameAnnotator class
- ✅ Visual debugging interface
- ✅ Color-coded annotations

---

#### Day 17-18: Video Stream Integration
**Goal:** Integrate video stream handling and multi-camera support

**Tasks:**
1. Reuse `StreamHandler` from existing face recognition service
2. Create `src/person_tracking/main.py`:
   ```python
   def main():
       # Load config
       config = ConfigurationManager.load_config()

       # Create engines per camera
       engines = {}
       for camera_config in config.cameras:
           stream = StreamHandler(camera_config.video_path)
           engine = PersonTrackingEngine(camera_config)
           engines[camera_config.camera_id] = (stream, engine)

       # Start processing
       while True:
           for camera_id, (stream, engine) in engines.items():
               ret, frame = stream.read()
               if ret:
                   annotated_frame = engine.process_frame(frame, frame_num)
                   # Display or save
   ```
3. Implement multi-camera support (independent tracking)
4. Add video file fallback for testing
5. Add graceful shutdown handling

**Deliverables:**
- ✅ Video stream integration
- ✅ Multi-camera support
- ✅ Entry point script

---

#### Day 18-19: Integration Testing
**Goal:** Test entire system with sample videos

**Tasks:**
1. Create test video dataset:
   - Single person with phone
   - Multiple people (2-5)
   - Phone handoff between people
   - Identity changes (person leaves, new person enters)
   - Occlusions and brief disappearances
2. Run integration tests:
   ```bash
   python src/person_tracking/main.py --config configs/test_config.yaml
   ```
3. Measure performance:
   - FPS per camera
   - Latency per frame
   - GPU memory usage
   - CPU usage
4. Verify accuracy:
   - Identity lock correctness
   - Phone usage detection precision/recall
   - Track ID stability (count ID switches)
5. Fix bugs and edge cases

**Deliverables:**
- ✅ Test video dataset
- ✅ Integration test results
- ✅ Performance benchmarks
- ✅ Bug fixes

---

#### Day 19-20: Configuration Management
**Goal:** Implement Pydantic config models and YAML loading

**Tasks:**
1. Create `src/person_tracking/config/models.py`:
   ```python
   class CameraConfig(BaseModel):
       camera_id: int
       camera_name: str
       video_path: str
       person_detection: PersonDetectionConfig
       person_tracking: PersonTrackingConfig
       face_recognition: FaceRecognitionConfig
       phone_detection: PhoneDetectionConfig
       phone_usage: PhoneUsageConfig

   class PersonTrackingConfig(BaseModel):
       cameras: List[CameraConfig]
       database: DatabaseConfig
       logging: LoggingConfig
   ```
2. Create `configs/person_tracking_config.yaml`:
   ```yaml
   cameras:
     - camera_id: 1
       camera_name: "Office Floor 1"
       video_path: "${HB_IN}"
       person_detection:
         model_size: "s"
         confidence_threshold: 0.5
       face_recognition:
         match_threshold: 0.3
         identity_lock_frames: 5
         identity_consensus: 0.60
       phone_usage:
         confirmation_frames: 8
         confirmation_consensus: 0.75
         hand_distance_threshold: 0.20
         head_distance_threshold: 0.25
   ```
3. Implement environment variable interpolation
4. Add validation and error handling

**Deliverables:**
- ✅ Pydantic config models
- ✅ YAML config file
- ✅ Configuration validation

---

#### Day 20-21: Docker Deployment
**Goal:** Build Docker image and deploy with docker-compose

**Tasks:**
1. Create `Dockerfile`:
   ```dockerfile
   # Multi-stage build
   FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

   # Install Python, dependencies
   RUN apt-get update && apt-get install -y python3.10 ...

   # Install Python packages
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Download models
   RUN python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"

   FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
   # Runtime stage...
   ```
2. Create `docker-compose.yml`:
   ```yaml
   services:
     postgres:
       image: pgvector/pgvector:pg16
       # ... (shared with face recognition)

     redis:
       image: redis:7-alpine
       # ... (shared with face recognition)

     person-tracking:
       build: .
       image: humblebeeintel/person-tracking
       volumes:
         - ./src:/app/src
         - ./configs:/app/configs
         - ./volumes/storage:/app/volumes/storage
       environment:
         - USE_PGVECTOR=true
         - POSTGRES_HOST=postgres
       depends_on:
         - postgres
         - redis
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```
3. Create `.env.example`
4. Test full Docker deployment
5. Document deployment steps

**Deliverables:**
- ✅ Dockerfile
- ✅ docker-compose.yml
- ✅ .env.example
- ✅ Working Docker deployment

---

#### Day 21: Basic Documentation
**Goal:** Document API, configuration, and setup

**Tasks:**
1. Create `README.md`:
   - Overview
   - Installation steps
   - Docker deployment
   - Configuration guide
   - Usage examples
2. Document API endpoints (OpenAPI/Swagger auto-generated)
3. Create configuration reference:
   - List all parameters
   - Describe purpose
   - Show default values
   - Tuning recommendations
4. Add sample usage:
   ```bash
   # Single camera
   docker-compose up person-tracking

   # View logs
   docker-compose logs -f person-tracking

   # Check CSV logs
   tail -f volumes/storage/person-tracking/logs/events.csv
   ```

**Deliverables:**
- ✅ README.md
- ✅ API documentation
- ✅ Configuration reference
- ✅ Usage examples

---

## Configuration

### Default Configuration Template

```yaml
# configs/person_tracking_config.yaml

# Global Settings
project_name: "person_tracking"
client_slug: "${HB_CLIENTSLUG}"

# Database (shared with face recognition)
database:
  use_pgvector: true
  postgres_host: "${POSTGRES_HOST}"
  postgres_port: 5434
  postgres_user: "${POSTGRES_USER}"
  postgres_password: "${POSTGRES_PASSWORD}"
  postgres_db: "${POSTGRES_DB}"

# Redis (for pub/sub)
redis:
  host: "${REDIS_HOST}"
  port: 6379

# Cameras
cameras:
  - camera_id: 1
    camera_name: "Office Entry"
    video_path: "${HB_IN}"

    # Person Detection
    person_detection:
      model_size: "s"                    # n/s/m/l (YOLOv8-Pose)
      confidence_threshold: 0.5
      iou_threshold: 0.45

    # Person Tracking
    person_tracking:
      tracker_type: "botsort"            # botsort/bytetrack/ocsort
      max_track_age: 120                 # seconds
      min_track_hits: 3                  # frames before confirming track
      iou_threshold: 0.3

    # Face Recognition
    face_recognition:
      match_threshold: 0.3                # cosine similarity threshold
      identity_lock_frames: 5             # M frames for identity lock
      identity_consensus: 0.60            # 60% votes needed (3/5)
      min_window_duration_ms: 333         # ~5 frames at 15 FPS

    # Phone Detection
    phone_detection:
      model_size: "n"                     # n/s/m/l (YOLOv8)
      confidence_threshold: 0.4

    # Phone Usage Logic
    phone_usage:
      confirmation_frames: 8              # N frames for usage confirmation
      confirmation_consensus: 0.75        # 75% frames (6/8)
      min_duration_ms: 500                # minimum duration

      # Spatial thresholds (meters)
      hand_distance_threshold: 0.20       # 20cm
      head_distance_threshold: 0.25       # 25cm
      upper_body_zone_margin: 0.30        # 30cm above head

      # Logic: require X of Y checks
      required_checks: 2                  # Of: zone, hand, head

    # Performance
    target_fps: 12
    resize_width: 1280                    # Downscale for performance

    # Storage
    save_annotated_frames: true
    save_phone_usage_clips: true
    output_dir: "volumes/storage/person-tracking"

# Logging
logging:
  level: "INFO"                           # DEBUG/INFO/WARNING/ERROR
  console: true
  file: true
  file_path: "logs/person_tracking.log"
  csv_logging: true
  csv_path: "volumes/storage/person-tracking/logs"

# API (mock in MVP)
api:
  host: "0.0.0.0"
  port: 5002
  backend_url: "${SO_BACKEND_API_URL}"
```

### Environment Variables

```bash
# .env.example

# PostgreSQL (shared with face recognition)
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_USER=face_recognition
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=face_embeddings

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Client
HB_CLIENTSLUG=my_organization

# Cameras
HB_IN=rtsp://admin:password@192.168.1.100:554/stream1
HB_OUT=rtsp://admin:password@192.168.1.101:554/stream1

# Backend API (for post-MVP)
SO_BACKEND_API_URL=http://localhost:7091

# Admin Credentials
SA_EMAIL=admin@example.com
SA_PASSWORD=password
```

---

## Performance Targets

### MVP Performance Targets (Balanced Configuration)

| Metric | Target | Notes |
|--------|--------|-------|
| **FPS** | 10-15 FPS | Per camera, YOLOv8s-pose + YOLOv8n |
| **Latency** | <100ms | Per frame processing |
| **GPU Memory** | <4GB | Per camera |
| **Accuracy** | >90% | Phone usage detection (post temporal filter) |
| **Track Stability** | <5% ID switches | Per minute |
| **Identity Lock Time** | 333-500ms | 5 frames at 15 FPS |
| **Phone Confirmation Time** | 500-667ms | 8 frames at 12 FPS |

### Multi-Camera Scaling

| Configuration | Cameras | FPS per Camera | Total FPS | GPU Memory |
|--------------|---------|----------------|-----------|------------|
| Single | 1 | 15 FPS | 15 | 3.5 GB |
| Multi (no batch) | 4 | 12 FPS | 48 | 10 GB |
| Multi (batched) | 8 | 8-10 FPS | 64-80 | 12 GB |

**Optimization Strategies:**
- Batch processing for multiple cameras
- Model quantization (FP16/INT8)
- Frame skip for non-critical cameras
- Adaptive resolution (downscale based on load)

---

## Success Criteria

### MVP Success Criteria

The MVP will be considered successful if:

1. **Core Functionality:**
   - ✅ Detects and tracks multiple people (10+ per frame)
   - ✅ Assigns stable TrackIDs (< 5% ID switches per minute)
   - ✅ Recognizes faces within person ROIs
   - ✅ Locks identity after 5 frames with 60% consensus
   - ✅ Detects phones and associates with correct person using pose
   - ✅ Confirms phone usage after 8 frames with 75% consensus

2. **Performance:**
   - ✅ Runs at 10-15 FPS on single camera with GPU
   - ✅ Latency < 100ms per frame
   - ✅ GPU memory usage < 4GB per camera

3. **Accuracy:**
   - ✅ Phone usage detection: >90% precision, >85% recall (after temporal filter)
   - ✅ Identity lock correctness: >95% (reuses existing face recognition)
   - ✅ Phone-to-person attribution: >90% correct in multi-person scenarios

4. **Output:**
   - ✅ Events logged to console (mock API)
   - ✅ Events saved to CSV for auditing
   - ✅ Annotated frames with visualization

5. **Deployment:**
   - ✅ Dockerized deployment with docker-compose
   - ✅ Configuration via YAML + environment variables
   - ✅ Basic documentation (README, config reference)

6. **Backwards Compatibility:**
   - ✅ Existing face recognition service continues to work
   - ✅ Shared pgvector database with no conflicts
   - ✅ Independent deployment (can run standalone)

---

## Post-MVP Features

### Production-Ready Features (Implement After MVP Validation)

#### 1. Backend API Integration
**Priority:** High
**Effort:** 2-3 days

- Replace mock API endpoints with real backend calls
- Implement authentication (JWT tokens)
- Add retry logic and error handling
- Create backend API endpoints:
  ```
  POST /api/person-tracking/events
  POST /api/person-tracking/phone-usage
  GET  /api/camera-configs/{camera_id}
  ```

#### 2. PostgreSQL Time-Series Storage
**Priority:** High
**Effort:** 2-3 days

- Create database tables:
  ```sql
  CREATE TABLE person_tracking_events (
      id SERIAL PRIMARY KEY,
      track_id INT,
      person_name VARCHAR(255),
      event_type VARCHAR(50),
      phone_usage BOOLEAN,
      camera_id INT,
      timestamp TIMESTAMP,
      metadata JSONB
  );

  CREATE TABLE tracking_sessions (
      id SERIAL PRIMARY KEY,
      track_id INT,
      camera_id INT,
      start_time TIMESTAMP,
      end_time TIMESTAMP,
      identity VARCHAR(255),
      total_phone_time_seconds INT,
      metadata JSONB
  );
  ```
- Implement data retention policies (e.g., keep 90 days)
- Add indexes for fast queries

#### 3. Redis Pub/Sub for Real-Time Events
**Priority:** Medium
**Effort:** 2 days

- Publish events to Redis channel: `person-tracking:events:{client_slug}`
- Message format:
  ```json
  {
    "event_type": "phone_usage_started",
    "track_id": 1,
    "person_name": "John Doe",
    "camera_id": 1,
    "timestamp": "2025-11-23T10:30:15Z",
    "confidence": 0.95
  }
  ```
- Add subscriber for embedding updates (similar to existing service)

#### 4. Dynamic Configuration Reload
**Priority:** Medium
**Effort:** 2-3 days

- Fetch camera configs from backend API every 60 seconds
- Support hot-reload of thresholds without service restart
- Implement ConfigReloader service
- Log configuration changes

#### 5. Performance Optimization
**Priority:** Medium
**Effort:** 3-5 days

- Batch processing for multiple cameras
- Model quantization (FP16/INT8) for faster inference
- Optimize frame preprocessing (GPU-based resize/normalize)
- Add GPU memory pooling
- Implement frame skip strategies for non-critical cameras

#### 6. Comprehensive Testing
**Priority:** Medium
**Effort:** 3-4 days

- Unit tests for all modules (pytest)
- Integration tests with ground truth data
- Performance benchmarks across different GPUs
- Create evaluation harness:
  - Annotated test videos with ground truth
  - Compute precision/recall/F1 for phone usage
  - Measure track stability (MOTA, MOTP metrics)

#### 7. Monitoring & Observability
**Priority:** Medium
**Effort:** 2-3 days

- Add Prometheus metrics:
  - `person_tracking_fps`
  - `person_tracking_detections_total`
  - `person_tracking_phone_usage_events_total`
  - `person_tracking_latency_seconds`
- Implement health checks (GPU status, model loaded, database connected)
- Add error alerting (email/Slack on critical errors)
- Dashboard with Grafana

#### 8. Advanced Features
**Priority:** Low
**Effort:** 5-7 days

- Cross-camera tracking (unified TrackID across multiple cameras)
- Zone-based configuration (cameras in same zone share tracking)
- Person re-identification (ReID embeddings for better tracking)
- Phone usage analytics (duration histograms, compliance reports)
- Alert system (notify when phone usage exceeds threshold)

#### 9. Documentation
**Priority:** Medium
**Effort:** 2-3 days

- Deployment guide (production setup, scaling strategies)
- Configuration tuning guide (how to adjust thresholds per camera)
- Troubleshooting guide (common issues and solutions)
- Architecture diagrams (system design, data flow)
- API reference (complete endpoint documentation)

---

## Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **YOLOv8-Pose FPS too low** | Medium | High | Use YOLOv8n-pose (faster) or reduce resolution; implement adaptive FPS |
| **Phone detection accuracy low** | Medium | Medium | Fine-tune confidence threshold; consider training custom model on office data |
| **ID switches during occlusions** | Medium | Medium | BoT-SORT handles this well; tune `max_track_age` parameter |
| **False positive phone usage** | Medium | High | Temporal filtering (8 frames) + strict spatial checks (2/3 conditions) |
| **pgvector database conflicts** | Low | High | Use shared database in read-only mode; coordinate with face recognition team |
| **GPU memory overflow** | Medium | High | Implement batch size limits; add memory monitoring; graceful degradation |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **RTSP stream disconnections** | High | Medium | Reuse existing StreamHandler with auto-reconnection logic |
| **Configuration errors** | Medium | Medium | Pydantic validation; provide clear error messages |
| **Deployment complexity** | Low | Medium | Docker Compose simplifies deployment; comprehensive documentation |
| **Performance degradation over time** | Low | Medium | Memory profiling; periodic model reloading; monitoring |

### Project Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Timeline delays** | Medium | Medium | Focus on MVP first; defer non-critical features to post-MVP |
| **Requirement changes** | Medium | High | Modular architecture allows easy changes; frequent stakeholder demos |
| **Integration issues** | Low | High | Test integration with existing service early; maintain backward compatibility |

---

## Next Steps

### Immediate Actions

1. **Review and Approve Plan**
   - Stakeholder review of this document
   - Confirm architecture decisions
   - Approve timeline and success criteria

2. **Prepare Sample Videos**
   - Collect test videos showing:
     - Single person with phone
     - Multiple people
     - Phone usage scenarios (texting, calling)
     - Occlusions and edge cases
   - Annotate ground truth for accuracy evaluation

3. **Set Up Development Environment**
   - Provision GPU machine (or ensure existing GPU available)
   - Set up Git repository
   - Configure Docker environment

4. **Begin Implementation**
   - Follow 3-week MVP plan
   - Daily standups to track progress
   - Weekly demos to show progress

### Post-MVP

5. **Evaluate MVP**
   - Test with real-world scenarios
   - Measure performance and accuracy
   - Gather stakeholder feedback

6. **Prioritize Post-MVP Features**
   - Based on business needs
   - Based on technical debt
   - Based on user feedback

7. **Production Deployment**
   - Deploy to production environment
   - Monitor performance and errors
   - Iterate and improve

---

## Appendix

### A. Project Structure

```
/media/SmartOffice/so.fr/so.person-tracking/
├── src/
│   └── person_tracking/
│       ├── __init__.py
│       ├── main.py                      # Entry point
│       ├── engine.py                    # PersonTrackingEngine (orchestrator)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── person_detector.py       # YOLOv8-Pose detection
│       │   ├── person_tracker.py        # BoT-SORT tracking
│       │   ├── track_manager.py         # Track history management
│       │   ├── phone_detector.py        # YOLOv8n phone detection
│       │   ├── face_adapter.py          # FaceRecognitionAdapter
│       │   ├── identity_manager.py      # Temporal voting for identity
│       │   ├── phone_usage_filter.py    # Temporal smoothing
│       │   └── state_manager.py         # PersonStateManager
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                   # FastAPI application
│       │   └── endpoints.py             # API routes
│       ├── config/
│       │   ├── __init__.py
│       │   ├── manager.py               # ConfigurationManager
│       │   └── models.py                # Pydantic config models
│       ├── storage/
│       │   ├── __init__.py
│       │   └── pgvector_store.py        # PostgreSQL integration
│       ├── video/
│       │   ├── __init__.py
│       │   ├── stream_handler.py        # Video stream handling (reused)
│       │   └── frame_annotator.py       # Visualization
│       └── logging/
│           ├── __init__.py
│           ├── csv_logger.py            # CSV logging
│           └── setup.py                 # Loguru setup
├── configs/
│   ├── person_tracking_config.yaml      # Main configuration
│   └── test_config.yaml                 # Test configuration
├── examples/
│   ├── test_tracking.py                 # Sample usage
│   └── benchmark.py                     # Performance testing
├── tests/
│   ├── unit/
│   │   ├── test_person_detector.py
│   │   ├── test_phone_detector.py
│   │   ├── test_identity_manager.py
│   │   └── test_phone_usage_filter.py
│   └── integration/
│       └── test_pipeline.py
├── volumes/
│   └── storage/
│       └── person-tracking/
│           ├── logs/                     # CSV logs
│           ├── frames/                   # Saved frames
│           └── clips/                    # Phone usage clips
├── Dockerfile                            # Multi-stage build
├── docker-compose.yml                    # Services definition
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment template
├── .gitignore
├── README.md                             # Setup and usage guide
└── PERSON_TRACKING_IMPLEMENTATION_PLAN.md  # This document
```

### B. Dependencies

```txt
# requirements.txt

# Deep Learning & Computer Vision
ultralytics>=8.0.0              # YOLOv8 (detection, tracking)
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
insightface>=0.7.3              # Face recognition (existing)

# Database & Caching
psycopg2-binary>=2.9.0          # PostgreSQL
pgvector>=0.2.0                 # Vector similarity
redis>=5.0.0                    # Pub/sub

# API & Web
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Utilities
numpy>=1.24.0
scipy>=1.10.0
Pillow>=10.0.0
python-dotenv>=1.0.0
loguru>=0.7.0

# Storage
gcsfs>=2023.0.0                 # Google Cloud Storage (existing)

# Testing (dev)
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

### C. COCO Keypoint Format (17 Points)

YOLOv8-Pose outputs 17 keypoints in COCO format:

| Index | Keypoint | Use in Phone Detection |
|-------|----------|------------------------|
| 0 | Nose | Head position for phone-to-head distance |
| 1 | Left Eye | - |
| 2 | Right Eye | - |
| 3 | Left Ear | - |
| 4 | Right Ear | - |
| 5 | Left Shoulder | Upper body zone boundary |
| 6 | Right Shoulder | Upper body zone boundary |
| 7 | Left Elbow | - |
| 8 | Right Elbow | - |
| 9 | Left Wrist | Hand position for hand-to-phone distance |
| 10 | Right Wrist | Hand position for hand-to-phone distance |
| 11 | Left Hip | - |
| 12 | Right Hip | - |
| 13 | Left Knee | - |
| 14 | Right Knee | - |
| 15 | Left Ankle | - |
| 16 | Right Ankle | - |

Each keypoint returns: `(x, y, confidence)` where confidence indicates detection quality.

---

**Document Version:** 1.2
**Last Updated:** 2025-11-24
**Status:** ✅ Week 2 Complete - 66% Done - Integration Phase

---

## 📝 Implementation Notes

### Week 1 Completion Summary (2025-11-24 Morning)

**What Was Completed:**
- ✅ Full project structure with modular architecture
- ✅ Pydantic configuration system with YAML support
- ✅ PersonDetector using YOLOv8s-Pose (17 keypoints)
- ✅ PersonTracker with BoT-SORT algorithm
- ✅ PersonTrackManager for track history
- ✅ Face recognition integration (reusing existing components)
- ✅ PhoneDetector using YOLOv8n

**Key Design Decisions:**
- **Reused existing face recognition code** instead of creating wrappers (simpler, more maintainable)
- **Shared pgvector database** with existing service (consistent identities)
- **Modular architecture** allows independent testing of each component

---

### Week 2 Completion Summary (2025-11-24 Afternoon)

**What Was Completed:**
- ✅ IdentityManager with confidence-weighted temporal voting (M=5 frames, 60% consensus)
- ✅ PhoneUsageSpatialLogic with pose-based association (3 spatial checks, requires 2/3)
- ✅ PhoneUsageFilter with temporal smoothing (N=8 frames, 75% consensus, 500ms minimum)
- ✅ PersonStateManager with complete event system (6 event types)
- ✅ Mock FastAPI application with 8 endpoints + Swagger docs
- ✅ CSVLogger for event and summary logging with auto-rotation
- ✅ Structured logging setup with Loguru

**Key Implementation Highlights:**
- **Temporal voting** prevents identity flickering and false positives
- **Pose-based phone detection** uses keypoints for accurate hand/head proximity
- **State machine** ensures smooth phone usage transitions
- **Event-driven architecture** enables real-time notifications
- **Mock API** ready for seamless backend integration

**Components Created (15 total):**
```
core/: 9 modules (detector, tracker, manager, adapter, phone, identity, logic, filter, state)
api/: 1 module (app)
logging/: 2 modules (csv_logger, setup)
config/: 2 modules (models, manager)
```

**Next Session:**
- Begin Week 3: Main orchestrator (PersonTrackingEngine)
- Frame annotation and visualization
- Video stream integration
- End-to-end testing

---

## Questions & Clarifications

If you have questions about this implementation plan, please contact the development team or create an issue in the project repository.

**Key Contacts:**
- Project Lead: [Name]
- Backend Integration: [Name]
- DevOps/Deployment: [Name]

**Repository:** `/media/SmartOffice/so.fr/so.model-face-recognition/` (integrated in same repo)
**Person Tracking Code:** `/media/SmartOffice/so.fr/so.model-face-recognition/src/person_tracking/`
**Existing Face Recognition:** `/media/SmartOffice/so.fr/so.model-face-recognition/src/face_recognition/`
