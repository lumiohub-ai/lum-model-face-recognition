# MCMOT ReID Integration - Phased Safe Migration Plan

## Executive Summary

**Goal**: Add cross-camera global track IDs to existing production system without disrupting current per-camera tracking.

**Approach**: GlobalTrackManager as a **post-processing layer** that consumes local tracks and assigns global IDs. No changes to existing PersonTracker initially.

**Philosophy**:
- Prefer duplicate global IDs over incorrect merges
- Conservative matching with hard rejection rules
- Body ReID as primary signal (face often not visible)
- Feature flag + shadow mode for safe rollout

---

## Current State (Production)

```
┌─────────────────────────────────────────────────────┐
│              SmartOfficeEngine                       │
│  10 cameras (2+ per room)                            │
└─────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ CameraEngine │        │ CameraEngine │
│   Camera 1   │        │   Camera 2   │
└──────────────┘        └──────────────┘
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│PersonTracker │        │PersonTracker │
│  (IoU only)  │        │  (IoU only)  │
│  STABLE ✓    │        │  STABLE ✓    │
└──────────────┘        └──────────────┘
        │                       │
        ▼                       ▼
  Local Track IDs          Local Track IDs
  (1, 2, 3...)             (1, 2, 3...)
        │                       │
        ▼                       ▼
   Face Recognition        Face Recognition
   (when visible)          (when visible)
        │                       │
        ▼                       ▼
     API Logs                API Logs
```

**Key Constraints**:
- System is LIVE and stable
- Faces NOT often visible (people looking away, turned around)
- Cannot break existing tracking/API integration
- Must add global IDs as separate layer

---

## Target Architecture (Phased)

```
┌─────────────────────────────────────────────────────┐
│              SmartOfficeEngine                       │
└─────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ CameraEngine │        │ CameraEngine │
│   Camera 1   │        │   Camera 2   │
└──────────────┘        └──────────────┘
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│PersonTracker │        │PersonTracker │
│  UNCHANGED   │        │  UNCHANGED   │
└──────────────┘        └──────────────┘
        │                       │
   Local Tracks            Local Tracks
        │                       │
        │    ┌──────────────────────────────┐
        └────►  GlobalTrackManager (NEW)   │◄─────┘
             │  POST-PROCESSING LAYER       │
             │  - Body ReID extraction      │
             │  - Cross-camera matching     │
             │  - Global ID assignment      │
             └──────────────────────────────┘
                        │
                        ▼
              Global Track IDs (1001, 1002...)
              Mapping: local_id → global_id
                        │
                        ▼
                  API Logs (enhanced)
```

**Key Changes**:
- PersonTracker stays UNCHANGED (stable)
- GlobalTrackManager added as separate layer
- Global IDs assigned in post-processing
- Shadow mode first, then gradual rollout

---

## Phase 0: Instrumentation + Shadow Mode (Week 1)

**Goal**: Add metrics and feature flags without changing behavior.

### Tasks:
1. Add feature flag: `ENABLE_GLOBAL_TRACKING=false` (default off)
2. Add instrumentation to log local track lifecycles:
   ```python
   logger.info(f"TRACK_CREATED | camera={cam_id} local_id={track_id} duration=0s")
   logger.info(f"TRACK_REMOVED | camera={cam_id} local_id={track_id} duration={dur}s")
   ```
3. Log when face recognition occurs (or fails):
   ```python
   logger.info(f"FACE_DETECTED | camera={cam_id} track={track_id} quality={qual}")
   logger.info(f"FACE_NOT_VISIBLE | camera={cam_id} track={track_id}")
   ```
4. Create stub GlobalTrackManager class (does nothing, just logs)
5. Measure baseline:
   - How often do tracks have faces?
   - What is average track duration?
   - How many tracks per camera per hour?

### Deliverables:
- Feature flag infrastructure
- Logging added to track lifecycle
- Baseline metrics collected
- Zero behavior change

### Success Criteria:
- Production runs normally
- Logs collected for analysis
- Data shows face visibility rate

---

## Phase 1: GlobalTrackManager v1 - Body ReID Only (Week 2-3)

**Goal**: Add cross-camera matching using body ReID. Conservative, simple.

### Architecture: GlobalTrackManager as Post-Processing Layer

```python
class GlobalTrackManager:
    """Assigns global IDs across cameras (post-processing layer)."""

    def __init__(self):
        # Global tracks
        self.global_tracks: Dict[int, GlobalTrack] = {}

        # Mapping: camera_id -> {local_track_id: global_track_id}
        self.local_to_global: Dict[int, Dict[int, int]] = defaultdict(dict)

        # ReID model (body appearance)
        self.body_reid_model = self._init_body_reid_model()

        # ID generator
        self.global_id_counter = 1000  # Start at 1000

        # Feature flag
        self.enabled = os.getenv('ENABLE_GLOBAL_TRACKING', 'false') == 'true'

        # Crop quality thresholds (from config)
        self.min_crop_height = 80
        self.min_crop_width = 40
        self.min_crop_area = 3200
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 4.0

        # Overlapping camera groups (from config)
        # Example: [[1, 2], [3, 4, 5]] means cameras 1&2 overlap, 3&4&5 overlap
        self.overlapping_camera_groups: List[List[int]] = []

class GlobalTrack:
    """Represents a person across multiple cameras."""

    def __init__(self, global_id: int):
        self.global_id = global_id

        # Embeddings: top-K + prototype
        self.body_prototype: Optional[np.ndarray] = None
        self.body_top_k: List[EmbeddingQuality] = []  # Max 5

        # Per-camera presence
        self.camera_tracks: Dict[int, CameraTrackInfo] = {}

        # Lifecycle
        self.first_seen: datetime = datetime.now()
        self.last_seen: datetime = datetime.now()
        self.active: bool = True

class EmbeddingQuality:
    """Embedding with quality score."""
    embedding: np.ndarray   # 512-d
    quality: float          # 0.0-1.0 (detection confidence)
    timestamp: datetime

class CameraTrackInfo:
    """Track info for one camera."""
    camera_id: int
    local_track_id: int
    first_seen: datetime
    last_seen: datetime
    active: bool
```

### Integration Points

```python
# In SmartOfficeEngine.__init__()
self.global_track_manager = GlobalTrackManager()

# In SmartOfficeEngine.run() processing loop
for camera_idx, frame, frame_num in frames:
    engine = self.camera_engines[camera_idx]

    # Process frame (LOCAL tracking - unchanged)
    recognized, processed = engine.process_frame(frame, frame_num)

    # NEW: Assign global IDs (post-processing)
    if self.global_track_manager.enabled:
        for person in recognized:
            global_id = self.global_track_manager.assign_global_id(
                camera_id=engine.camera_id,
                local_track_id=person['track_id'],
                person_crop=person['crop'],  # For body ReID
                face_embedding=person.get('face_embedding'),  # May be None
                detection_confidence=person['confidence']
            )

            # Expose global ID (keep local for debugging)
            person['global_track_id'] = global_id
            person['local_track_id'] = person['track_id']  # Original
```

### Matching Logic: Conservative Real-Time Assignment

```python
def assign_global_id(
    self,
    camera_id: int,
    local_track_id: int,
    person_crop: np.ndarray,
    face_embedding: Optional[np.ndarray],
    detection_confidence: float
) -> int:
    """Assign global ID to local track (called per frame for active tracks)."""

    # Check if already assigned
    if local_track_id in self.local_to_global[camera_id]:
        return self.local_to_global[camera_id][local_track_id]

    # STEP 1: Crop quality checks (before extraction)
    crop_quality_valid = self._validate_crop_quality(person_crop)

    if not crop_quality_valid:
        # Poor crop quality - skip matching, create new global ID (conservative)
        logger.debug(f"Poor crop quality for camera={camera_id} track={local_track_id}, skipping ReID")
        return self._create_new_global_track(camera_id, local_track_id, None, 0.0)

    # STEP 2: Extract body embedding
    body_embedding = self._extract_body_embedding(person_crop)
    body_quality = detection_confidence  # Use detection conf as quality

    # STEP 3: Hard rejection checks
    if body_quality < 0.5:
        # Low quality - assign new global ID (conservative)
        return self._create_new_global_track(camera_id, local_track_id, body_embedding, body_quality)

    # STEP 4: Search for candidates (temporal gating)
    candidates = self._get_candidate_tracks(
        camera_id=camera_id,
        time_window=60.0  # 60 seconds
    )

    if not candidates:
        # No candidates - create new
        return self._create_new_global_track(camera_id, local_track_id, body_embedding, body_quality)

    # STEP 5: Match against candidates (body ReID only in v1)
    best_match = None
    best_similarity = 0.0

    for candidate in candidates:
        # Hard rejection: same camera recently
        if camera_id in candidate.camera_tracks:
            last_seen = candidate.camera_tracks[camera_id].last_seen
            if (datetime.now() - last_seen).total_seconds() < 10.0:
                continue  # Skip: too recent (avoid duplicates)

        # Compute body similarity
        if candidate.body_prototype is not None:
            similarity = self._cosine_similarity(body_embedding, candidate.body_prototype)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate

    # STEP 6: Conservative decision
    THRESHOLD = 0.70  # High threshold (conservative)

    if best_match and best_similarity >= THRESHOLD:
        # MATCH: Assign existing global ID
        global_id = best_match.global_id
        self._associate_track(global_id, camera_id, local_track_id, body_embedding, body_quality)

        logger.info(f"GLOBAL_MATCH | global_id={global_id} camera={camera_id} "
                   f"local_id={local_track_id} similarity={best_similarity:.3f}")
        return global_id
    else:
        # NO MATCH: Create new global ID (prefer false negatives)
        global_id = self._create_new_global_track(camera_id, local_track_id, body_embedding, body_quality)

        logger.info(f"GLOBAL_NEW | global_id={global_id} camera={camera_id} "
                   f"local_id={local_track_id} best_sim={best_similarity:.3f}")
        return global_id
```

### Hard Rejection Rules

```python
def _get_candidate_tracks(self, camera_id: int, time_window: float) -> List[GlobalTrack]:
    """Get candidate global tracks for matching (with gating)."""

    current_time = datetime.now()
    candidates = []

    for global_id, track in self.global_tracks.items():
        # Gate 1: Temporal - only recent tracks
        time_since_last_seen = (current_time - track.last_seen).total_seconds()
        if time_since_last_seen > time_window:
            continue  # Too old

        # Gate 2: Same-camera re-entry cooldown - exclude tracks on SAME camera if very recent
        if camera_id in track.camera_tracks:
            camera_track = track.camera_tracks[camera_id]
            time_since_on_camera = (current_time - camera_track.last_seen).total_seconds()
            if time_since_on_camera < 10.0:
                continue  # Too recent on this camera (avoid duplicates)

        # Gate 3: Quality - only use tracks with good embeddings
        if track.body_prototype is None:
            continue  # No embedding yet

        candidates.append(track)

    return candidates
```

### Embedding Management: Top-K + Prototype

```python
def _associate_track(
    self,
    global_id: int,
    camera_id: int,
    local_track_id: int,
    body_embedding: np.ndarray,
    body_quality: float
):
    """Associate local track with existing global track."""

    track = self.global_tracks[global_id]

    # Update camera presence
    track.camera_tracks[camera_id] = CameraTrackInfo(
        camera_id=camera_id,
        local_track_id=local_track_id,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        active=True
    )

    # Update embeddings (top-K + prototype)
    self._add_embedding(track, body_embedding, body_quality)

    # Update mapping
    self.local_to_global[camera_id][local_track_id] = global_id

    # Update lifecycle
    track.last_seen = datetime.now()

def _add_embedding(self, track: GlobalTrack, embedding: np.ndarray, quality: float):
    """Add embedding using top-K + prototype strategy."""

    # Add to top-K list
    track.body_top_k.append(EmbeddingQuality(
        embedding=embedding,
        quality=quality,
        timestamp=datetime.now()
    ))

    # Sort by quality (descending)
    track.body_top_k.sort(key=lambda x: x.quality, reverse=True)

    # Keep only top 5
    if len(track.body_top_k) > 5:
        track.body_top_k = track.body_top_k[:5]

    # Update prototype (running mean)
    if track.body_prototype is None:
        track.body_prototype = embedding.copy()
    else:
        # Exponential moving average
        alpha = 0.2
        track.body_prototype = (1 - alpha) * track.body_prototype + alpha * embedding

        # Normalize
        norm = np.linalg.norm(track.body_prototype)
        if norm > 0:
            track.body_prototype = track.body_prototype / norm
```

### Body ReID Model Initialization

```python
def _init_body_reid_model(self):
    """Initialize OSNet ReID model."""
    from modules.yolo_tracking.boxmot.appearance.reid_auto_backend import ReidAutoBackend

    model_path = Path("weights/osnet_x0_25_msmt17.pt")

    backend = ReidAutoBackend(
        weights=model_path,
        device=torch.device("cuda:0"),
        half=False
    )

    return backend.get_backend()

def _extract_body_embedding(self, person_crop: np.ndarray) -> np.ndarray:
    """Extract body ReID embedding from person crop."""

    # Preprocess for OSNet (128x256)
    crop_resized = cv2.resize(person_crop, (128, 256))

    # Convert to tensor
    crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float()
    crop_tensor = crop_tensor.unsqueeze(0) / 255.0
    crop_tensor = crop_tensor.to(self.body_reid_model.device)

    # Extract features
    with torch.no_grad():
        embedding = self.body_reid_model(crop_tensor)

    # Normalize
    embedding = embedding.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

def _validate_crop_quality(self, person_crop: np.ndarray) -> bool:
    """Validate crop quality before ReID extraction.

    Args:
        person_crop: Person crop image (H, W, C)

    Returns:
        True if crop meets quality requirements, False otherwise
    """
    h, w = person_crop.shape[:2]

    # Check minimum dimensions
    if h < self.min_crop_height or w < self.min_crop_width:
        return False

    # Check aspect ratio (person should be roughly vertical)
    aspect_ratio = h / w if w > 0 else 0
    if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
        return False

    # Check minimum area
    area = h * w
    if area < self.min_crop_area:
        return False

    return True
```

### Configuration

```yaml
# configs/global_tracking.yaml
global_tracking:
  enabled: false  # Feature flag (default OFF)

  body_reid:
    model: "osnet_x0_25_msmt17"
    weights_path: "weights/osnet_x0_25_msmt17.pt"
    device: "cuda:0"

  matching:
    threshold: 0.70           # Conservative (prefer false negatives)
    temporal_window_sec: 60   # 1 minute
    min_reentry_gap_sec: 10   # Min gap before re-entry on same camera
    min_quality: 0.5          # Min detection confidence

  # Crop quality requirements (before ReID extraction)
  crop_quality:
    min_height: 80            # Minimum crop height in pixels
    min_width: 40             # Minimum crop width in pixels
    min_area: 3200            # Minimum crop area (80*40)
    min_aspect_ratio: 1.5     # Min height/width ratio (vertical person)
    max_aspect_ratio: 4.0     # Max height/width ratio (avoid extreme crops)

  embedding_management:
    top_k_size: 5             # Keep top-5 embeddings
    prototype_alpha: 0.2      # EMA smoothing factor
```

### Deliverables (Phase 1):
- GlobalTrackManager class implemented
- Body ReID extraction working
- Conservative matching logic
- Shadow mode (logs only, doesn't change API yet)
- Metrics: match rate, duplicate rate, false merge detection

### Success Criteria:
- No production disruption
- Global IDs assigned to local tracks
- Logs show matching behavior
- Low false merge rate (<1%)

---

## Phase 2: Body ReID Optimization (Week 4)

**Goal**: Optimize embedding extraction to reduce GPU load.

### Current Problem:
- Extracting body embedding for EVERY active track EVERY frame
- 10 cameras × ~5 tracks/camera × 30 FPS = 1500 extractions/sec
- High GPU load

### Solution: Interval-Based Extraction + Caching

```python
class GlobalTrackManager:
    def __init__(self):
        # ...existing fields...

        # Caching for embeddings
        self.embedding_cache: Dict[Tuple[int, int], CachedEmbedding] = {}
        self.extract_interval = 10  # Extract every 10 frames

class CachedEmbedding:
    """Cached embedding with metadata."""
    embedding: np.ndarray
    frame_num: int
    quality: float
    timestamp: datetime

def assign_global_id(
    self,
    camera_id: int,
    local_track_id: int,
    person_crop: np.ndarray,
    face_embedding: Optional[np.ndarray],
    detection_confidence: float,
    frame_num: int  # NEW: pass frame number
) -> int:
    """Assign global ID (optimized with caching)."""

    # Check if already assigned
    if local_track_id in self.local_to_global[camera_id]:
        global_id = self.local_to_global[camera_id][local_track_id]

        # Update cached embedding periodically
        self._maybe_update_embedding(
            camera_id, local_track_id, person_crop,
            detection_confidence, frame_num
        )

        return global_id

    # First time: extract embedding and assign
    body_embedding = self._extract_body_embedding(person_crop)

    # Cache it
    self.embedding_cache[(camera_id, local_track_id)] = CachedEmbedding(
        embedding=body_embedding,
        frame_num=frame_num,
        quality=detection_confidence,
        timestamp=datetime.now()
    )

    # ... rest of matching logic ...

def _maybe_update_embedding(
    self,
    camera_id: int,
    local_track_id: int,
    person_crop: np.ndarray,
    quality: float,
    frame_num: int
):
    """Update embedding if interval passed."""

    key = (camera_id, local_track_id)

    if key not in self.embedding_cache:
        return  # No cache yet

    cached = self.embedding_cache[key]
    frames_since_last = frame_num - cached.frame_num

    # Only extract if interval passed
    if frames_since_last >= self.extract_interval:
        new_embedding = self._extract_body_embedding(person_crop)

        # Update cache
        self.embedding_cache[key] = CachedEmbedding(
            embedding=new_embedding,
            frame_num=frame_num,
            quality=quality,
            timestamp=datetime.now()
        )

        # Update global track prototype
        global_id = self.local_to_global[camera_id][local_track_id]
        if global_id in self.global_tracks:
            self._add_embedding(self.global_tracks[global_id], new_embedding, quality)
```

### Batch Processing

```python
def batch_assign_global_ids(
    self,
    camera_id: int,
    tracks: List[Dict],  # List of track data
    frame: np.ndarray,
    frame_num: int
) -> Dict[int, int]:  # Returns {local_id: global_id}
    """Batch process tracks for efficiency."""

    # Identify tracks needing embedding extraction
    tracks_to_extract = []

    for track in tracks:
        local_id = track['track_id']
        key = (camera_id, local_id)

        # Check if new or needs refresh
        if key not in self.embedding_cache:
            tracks_to_extract.append(track)
        else:
            cached = self.embedding_cache[key]
            if (frame_num - cached.frame_num) >= self.extract_interval:
                tracks_to_extract.append(track)

    # Batch extract embeddings (GPU efficient)
    if tracks_to_extract:
        crops = [track['crop'] for track in tracks_to_extract]
        embeddings = self._batch_extract_embeddings(crops)

        # Update cache
        for track, emb in zip(tracks_to_extract, embeddings):
            key = (camera_id, track['track_id'])
            self.embedding_cache[key] = CachedEmbedding(
                embedding=emb,
                frame_num=frame_num,
                quality=track['confidence'],
                timestamp=datetime.now()
            )

    # Assign global IDs (using cached embeddings)
    result = {}
    for track in tracks:
        global_id = self.assign_global_id(
            camera_id=camera_id,
            local_track_id=track['track_id'],
            person_crop=track['crop'],
            face_embedding=track.get('face_embedding'),
            detection_confidence=track['confidence'],
            frame_num=frame_num
        )
        result[track['track_id']] = global_id

    return result

def _batch_extract_embeddings(self, crops: List[np.ndarray]) -> List[np.ndarray]:
    """Extract embeddings in batch for GPU efficiency."""

    # Preprocess all crops
    tensors = []
    for crop in crops:
        crop_resized = cv2.resize(crop, (128, 256))
        tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)

    # Stack into batch
    batch = torch.stack(tensors).to(self.body_reid_model.device)

    # Extract features in batch
    with torch.no_grad():
        embeddings = self.body_reid_model(batch)

    # Normalize
    embeddings = embeddings.cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return list(embeddings)
```

### Deliverables (Phase 2):
- Interval-based extraction (every 10 frames)
- Batch processing for GPU efficiency
- Embedding cache with TTL
- Reduced GPU load by ~10x

### Success Criteria:
- GPU usage reduced significantly
- FPS unchanged or improved
- Matching quality maintained

---

## Phase 3: Track-End (Removal-Time) Matching (Week 5)

**Goal**: Improve matching by using averaged embeddings when track is removed.

### Motivation:
- At track creation: only 1 embedding (may be noisy)
- At track removal: 5-10 embeddings (averaged, more reliable)
- Can correct initial misassignments

### Implementation:

```python
def on_track_removed(
    self,
    camera_id: int,
    local_track_id: int,
    track_history: List[Dict]  # Historical detections
):
    """Handle track removal (opportunity for better matching)."""

    # Get current global ID assignment
    current_global_id = self.local_to_global[camera_id].get(local_track_id)

    if current_global_id is None:
        return  # Track was never assigned (very brief)

    # Mark camera track as inactive
    if current_global_id in self.global_tracks:
        track = self.global_tracks[current_global_id]
        if camera_id in track.camera_tracks:
            track.camera_tracks[camera_id].active = False
            track.camera_tracks[camera_id].last_seen = datetime.now()

    # OPTIONAL: Re-match with better embeddings
    # (Only if current match seems weak)
    if len(track_history) >= 5:  # Need sufficient history
        avg_embedding = self._compute_averaged_embedding(camera_id, local_track_id)

        if avg_embedding is not None:
            # Re-evaluate match
            candidates = self._get_candidate_tracks(
                camera_id=camera_id,
                time_window=300.0,  # 5 minutes (longer for removal-time)
                include_same_camera=True  # Allow re-entry matching
            )

            best_match = None
            best_similarity = 0.0

            for candidate in candidates:
                if candidate.global_id == current_global_id:
                    continue  # Skip current assignment

                similarity = self._cosine_similarity(avg_embedding, candidate.body_prototype)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate

            # If much better match found, suggest correction
            THRESHOLD = 0.75  # Higher threshold for re-assignment
            if best_match and best_similarity >= THRESHOLD:
                logger.warning(
                    f"POTENTIAL_REMATCH | current_global={current_global_id} "
                    f"better_match={best_match.global_id} sim={best_similarity:.3f} "
                    f"camera={camera_id} local_id={local_track_id}"
                )

                # For v1: Just log (don't auto-correct)
                # For v2: Could merge global tracks if confident

def _compute_averaged_embedding(
    self,
    camera_id: int,
    local_track_id: int
) -> Optional[np.ndarray]:
    """Compute averaged embedding from track history."""

    global_id = self.local_to_global[camera_id].get(local_track_id)
    if global_id is None or global_id not in self.global_tracks:
        return None

    track = self.global_tracks[global_id]

    # Use top-K embeddings (already quality-filtered)
    if len(track.body_top_k) == 0:
        return None

    # Average top-K embeddings
    embeddings = [e.embedding for e in track.body_top_k]
    avg = np.mean(embeddings, axis=0)

    # Normalize
    avg = avg / np.linalg.norm(avg)

    return avg
```

### Integration:

```python
# In SmartOfficeEngine (when track is removed)
for removed_track in removed_tracks:
    # ... existing cleanup ...

    # NEW: Notify GlobalTrackManager
    if self.global_track_manager.enabled:
        self.global_track_manager.on_track_removed(
            camera_id=engine.camera_id,
            local_track_id=removed_track['track_id'],
            track_history=removed_track.get('history', [])
        )
```

### Deliverables (Phase 3):
- Track removal handling
- Averaged embedding computation
- Re-matching suggestions (logged, not auto-applied)

### Success Criteria:
- Logs show potential re-match opportunities
- Data for future merge decisions
- No auto-corrections yet (v1 conservative)

---

## Phase 4: Safety Mechanisms (Week 6)

**Goal**: Detect and handle incorrect merges.

### Mechanism 1: Detect Simultaneous Appearances

```python
def detect_impossible_merges(self):
    """Detect if same global ID appears on multiple cameras simultaneously.

    Only flags conflicts when simultaneity occurs between NON-overlapping cameras.
    Overlapping cameras (configured in overlapping_cameras) are allowed to see
    the same person at the same time.
    """

    conflicts = []

    for global_id, track in self.global_tracks.items():
        # Get currently active cameras for this global track
        active_cameras = []
        for cam_id, cam_track in track.camera_tracks.items():
            if cam_track.active:
                time_since = (datetime.now() - cam_track.last_seen).total_seconds()
                if time_since < 5.0:  # Active in last 5 seconds
                    active_cameras.append(cam_id)

        # If active on multiple cameras, check if they're allowed to overlap
        if len(active_cameras) > 1:
            # Check if all active cameras are in the same overlap group
            cameras_allowed = self._cameras_can_overlap(active_cameras)

            if not cameras_allowed:
                # Conflict: person on non-overlapping cameras simultaneously
                conflicts.append({
                    'global_id': global_id,
                    'cameras': active_cameras,
                    'severity': 'HIGH'
                })

                logger.error(
                    f"IMPOSSIBLE_MERGE | global_id={global_id} "
                    f"active_cameras={active_cameras} - person can't be in two non-overlapping places!"
                )

    return conflicts

def _cameras_can_overlap(self, camera_ids: List[int]) -> bool:
    """Check if given cameras are allowed to see the same person simultaneously.

    Args:
        camera_ids: List of camera IDs to check

    Returns:
        True if cameras are in the same overlap group, False otherwise
    """
    # Check if all cameras are in the same overlap group
    for overlap_group in self.overlapping_camera_groups:
        if all(cam_id in overlap_group for cam_id in camera_ids):
            return True  # All cameras in same group - allowed

    # Not all in same group - disallowed
    return False
```

### Mechanism 2: Split Incorrect Merges

```python
def split_global_track(self, global_id: int):
    """Split incorrectly merged global track."""

    track = self.global_tracks[global_id]

    # Find camera with longest presence (keep as primary)
    primary_camera = None
    max_duration = 0

    for cam_id, cam_track in track.camera_tracks.items():
        duration = (cam_track.last_seen - cam_track.first_seen).total_seconds()
        if duration > max_duration:
            max_duration = duration
            primary_camera = cam_id

    # Create new global IDs for other cameras
    for cam_id, cam_track in track.camera_tracks.items():
        if cam_id == primary_camera:
            continue  # Keep original global ID

        # Create new global track
        new_global_id = self.global_id_counter
        self.global_id_counter += 1

        new_track = GlobalTrack(new_global_id)
        new_track.camera_tracks[cam_id] = cam_track
        new_track.body_prototype = track.body_prototype.copy()  # Copy embeddings
        new_track.body_top_k = []  # Clear top-K (will rebuild)

        self.global_tracks[new_global_id] = new_track

        # Update mapping
        local_id = cam_track.local_track_id
        self.local_to_global[cam_id][local_id] = new_global_id

        logger.info(
            f"SPLIT_TRACK | old_global={global_id} new_global={new_global_id} "
            f"camera={cam_id} local_id={local_id}"
        )

    # Update original track (remove other cameras)
    track.camera_tracks = {primary_camera: track.camera_tracks[primary_camera]}
```

### Mechanism 3: Periodic Validation

```python
def periodic_validation(self):
    """Run periodic checks for data consistency."""

    # Check 1: Detect conflicts
    conflicts = self.detect_impossible_merges()

    for conflict in conflicts:
        # Auto-split if high severity
        if conflict['severity'] == 'HIGH':
            self.split_global_track(conflict['global_id'])

    # Check 2: Clean up old global tracks
    cutoff_time = datetime.now() - timedelta(minutes=10)
    inactive_tracks = []

    for global_id, track in self.global_tracks.items():
        if track.last_seen < cutoff_time:
            # All camera tracks inactive for >10 min
            if all(not ct.active for ct in track.camera_tracks.values()):
                inactive_tracks.append(global_id)

    # Archive inactive tracks (move to history)
    for global_id in inactive_tracks:
        track = self.global_tracks.pop(global_id)
        logger.info(f"ARCHIVE_TRACK | global_id={global_id} duration={(track.last_seen - track.first_seen).total_seconds():.1f}s")

    # Check 3: Cache cleanup
    self._cleanup_embedding_cache()

def _cleanup_embedding_cache(self):
    """Remove stale cache entries."""

    cutoff_time = datetime.now() - timedelta(minutes=5)
    stale_keys = []

    for key, cached in self.embedding_cache.items():
        if cached.timestamp < cutoff_time:
            stale_keys.append(key)

    for key in stale_keys:
        del self.embedding_cache[key]
```

### Integration: Periodic Task

```python
# In SmartOfficeEngine.run()
last_validation_time = time.time()
VALIDATION_INTERVAL = 30.0  # Every 30 seconds

while self.running:
    # ... process frames ...

    # Periodic validation
    if time.time() - last_validation_time > VALIDATION_INTERVAL:
        if self.global_track_manager.enabled:
            self.global_track_manager.periodic_validation()
        last_validation_time = time.time()
```

### Deliverables (Phase 4):
- Conflict detection
- Track splitting for incorrect merges
- Periodic validation task
- Cache cleanup

### Success Criteria:
- Conflicts detected and logged
- Automatic splitting when person on multiple cameras
- Old tracks archived properly

---

## Phase 5: Production Rollout (Week 7-8)

**Goal**: Enable in production with monitoring.

### Rollout Steps:

1. **Shadow Mode** (Week 7):
   - `ENABLE_GLOBAL_TRACKING=true`
   - Global IDs assigned but NOT sent to API yet
   - Log global IDs alongside local IDs
   - Collect metrics:
     - Match rate: % tracks matched vs new
     - Conflict rate: # conflicts per hour
     - Split rate: # splits per hour

2. **Partial Rollout** (Week 8):
   - Enable for 2 cameras (one room) first
   - Send global IDs to API for those cameras
   - Monitor for false merges
   - A/B test: compare with local-only tracking

3. **Full Rollout**:
   - Enable for all 10 cameras
   - Switch API to use global IDs as primary
   - Keep local IDs for debugging

### Monitoring Dashboard:

```python
class GlobalTrackingMetrics:
    """Metrics for monitoring."""

    # Matching metrics
    total_assignments: int = 0
    matched_to_existing: int = 0
    created_new: int = 0

    # Quality metrics
    avg_similarity_matched: float = 0.0
    avg_similarity_rejected: float = 0.0

    # Conflict metrics
    conflicts_detected: int = 0
    tracks_split: int = 0

    # Performance metrics
    avg_extraction_time_ms: float = 0.0
    avg_matching_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    def log_summary(self):
        match_rate = self.matched_to_existing / max(self.total_assignments, 1)

        logger.info(
            f"GLOBAL_TRACKING_METRICS | "
            f"match_rate={match_rate:.2%} "
            f"conflicts={self.conflicts_detected} "
            f"splits={self.tracks_split} "
            f"cache_hit_rate={self.cache_hit_rate:.2%} "
            f"avg_match_time={self.avg_matching_time_ms:.1f}ms"
        )
```

### Deliverables (Phase 5):
- Shadow mode testing
- Partial rollout
- Full production deployment
- Monitoring dashboard

### Success Criteria:
- <1% false merge rate
- >70% match rate (cross-camera)
- <50ms average matching latency
- Zero production incidents

---

## Future Work (Not in v1)

### Phase 6 (Optional): Face Integration

**After v1 is stable**, add face as high-confidence signal:

```python
# In matching logic
if face_embedding is not None and face_quality > 0.7:
    # High-confidence face available
    # Use face as primary signal
    face_similarity = cosine_similarity(face_embedding, candidate.face_prototype)

    if face_similarity > 0.85:  # Very high threshold
        # Face match - override body decision
        return candidate.global_id
```

**Conservative approach**:
- Face used only as high-confidence override
- Body remains primary signal (more reliable given visibility issues)
- Face requires quality >0.7 and similarity >0.85

### Phase 7 (Optional): In-Camera BoT-SORT Upgrade

**After v1 is stable**, optionally upgrade PersonTracker:
- Replace IoU-only with BoT-SORT
- Use body ReID within camera
- Reduce single-camera ID switches
- **Risk**: Changes existing stable tracker
- **Benefit**: Better within-camera tracking

**Decision**: Postpone until v1 proven in production.

### Phase 8 (Optional): Identity-Based Matching

**After v1 is stable**, add identity as signal:
- When face recognized and identity locked
- Use identity for fast-path matching
- Still validate with embeddings (don't trust identity alone)

---

## Critical Files

### Files to Create:
1. `src/person_tracking/core/global_track_manager.py` - Main class (500-700 LOC)
2. `src/person_tracking/core/global_track.py` - Data structures (200 LOC)
3. `configs/global_tracking.yaml` - Configuration
4. `tests/test_global_track_manager.py` - Unit tests

### Files to Modify:
1. `src/face_recognition/smart_office_engine.py` - Integration (50 LOC added)
   - Initialize GlobalTrackManager
   - Call assign_global_id() in processing loop
   - Call on_track_removed() for cleanup

2. `.env` or environment config - Add feature flag

### Files to Reference (No Changes):
1. `modules/yolo_tracking/boxmot/appearance/reid_auto_backend.py` - ReID backend
2. `src/person_tracking/core/person_tracker.py` - Stays unchanged in v1

---

## Configuration

### `configs/global_tracking.yaml`

```yaml
global_tracking:
  # Feature flag
  enabled: false  # Set to true to enable

  # Body ReID model
  body_reid:
    model_name: "osnet_x0_25_msmt17"
    weights_path: "weights/osnet_x0_25_msmt17.pt"
    device: "cuda:0"
    half_precision: false

  # Matching thresholds (conservative)
  matching:
    similarity_threshold: 0.70      # High (prefer false negatives)
    temporal_window_sec: 60         # 1 minute
    min_reentry_gap_sec: 10         # Avoid duplicates on same camera
    min_detection_quality: 0.5      # Reject low-quality detections
    removal_window_sec: 300         # 5 minutes for track-end matching

  # Crop quality requirements (before ReID extraction)
  crop_quality:
    min_height: 80                  # Minimum crop height in pixels
    min_width: 40                   # Minimum crop width in pixels
    min_area: 3200                  # Minimum crop area (80*40)
    min_aspect_ratio: 1.5           # Min height/width ratio (vertical person)
    max_aspect_ratio: 4.0           # Max height/width ratio (avoid extreme crops)

  # Embedding management
  embeddings:
    top_k_size: 5                   # Keep top-5 high-quality embeddings
    prototype_alpha: 0.2            # EMA smoothing factor
    extract_interval_frames: 10     # Extract every N frames

  # Safety mechanisms
  safety:
    validation_interval_sec: 30     # Run validation every 30s
    archive_after_inactive_min: 10  # Archive after 10 min inactive

    # Overlapping camera groups (cameras that can see the same person simultaneously)
    # Only split if simultaneity occurs between cameras NOT in the same group
    overlapping_cameras:
      - [1, 2]      # Room A - entrance and main area overlap
      - [3, 4, 5]   # Room B - three cameras with overlapping views
      # Cameras not listed are assumed non-overlapping with all others

  # Performance
  performance:
    batch_size: 16                  # Batch ReID extraction
    cache_ttl_sec: 300              # Cache TTL (5 minutes)
```

---

## API Changes

### External API (Enhanced with Global IDs)

**Before (Current)**:
```json
{
  "track_id": 3,
  "camera_id": 1,
  "camera_name": "Room A - Entrance",
  "name": "John Doe",
  "timestamp": "2026-01-10T10:30:00Z"
}
```

**After (v1 - Shadow Mode)**:
```json
{
  "track_id": 3,                    // Local track ID (unchanged)
  "camera_id": 1,
  "camera_name": "Room A - Entrance",
  "name": "John Doe",
  "timestamp": "2026-01-10T10:30:00Z",

  // NEW: Global tracking info (not used by API yet)
  "global_track_id": 1042,          // Cross-camera ID
  "global_tracking_enabled": true
}
```

**After (v2 - Production)**:
```json
{
  "global_track_id": 1042,          // PRIMARY ID
  "local_track_id": 3,              // For debugging
  "camera_id": 1,
  "camera_name": "Room A - Entrance",
  "name": "John Doe",
  "timestamp": "2026-01-10T10:30:00Z",

  // NEW: Cross-camera info
  "cameras_seen": [1, 2],           // Which cameras saw this person
  "first_seen_camera": 1,           // Where person first appeared
  "trajectory": [                   // Camera-to-camera path
    {"camera_id": 1, "entered": "10:29:50Z", "exited": "10:30:15Z"},
    {"camera_id": 2, "entered": "10:30:20Z", "exited": null}
  ]
}
```

---

## Testing Strategy

### Unit Tests (Phase 1):
```python
def test_conservative_matching():
    """Test that low similarity creates new track."""
    manager = GlobalTrackManager()

    # Create first track
    global_id_1 = manager.assign_global_id(
        camera_id=1, local_track_id=1,
        person_crop=crop_1, face_embedding=None,
        detection_confidence=0.8
    )

    # Try to match with dissimilar embedding
    global_id_2 = manager.assign_global_id(
        camera_id=2, local_track_id=1,
        person_crop=crop_2_different, face_embedding=None,
        detection_confidence=0.8
    )

    # Should create NEW global ID (conservative)
    assert global_id_1 != global_id_2

def test_temporal_gating():
    """Test that old tracks are not matched."""
    manager = GlobalTrackManager()

    # Create track
    global_id_1 = manager.assign_global_id(...)

    # Wait 2 minutes
    time.sleep(120)

    # Try to match (should fail due to temporal gate)
    global_id_2 = manager.assign_global_id(...)

    assert global_id_1 != global_id_2  # Too old, new ID created
```

### Integration Tests (Phase 3):
```python
def test_multi_camera_tracking():
    """Test person moving from Camera 1 to Camera 2."""
    manager = GlobalTrackManager()

    # Person appears on Camera 1
    global_id_cam1 = manager.assign_global_id(
        camera_id=1, local_track_id=1,
        person_crop=person_crop, ...
    )

    # Same person appears on Camera 2 (30 seconds later)
    time.sleep(30)
    global_id_cam2 = manager.assign_global_id(
        camera_id=2, local_track_id=1,
        person_crop=person_crop_similar, ...
    )

    # Should match (same global ID)
    assert global_id_cam1 == global_id_cam2
```

### Production Validation (Phase 5):
- Manual review of sample tracks
- Compare with ground truth (manual annotations)
- False merge rate: <1%
- Match rate: >70%

---

## Risk Mitigation

| Risk | Mitigation | Status |
|------|-----------|---------|
| Break existing tracking | Keep PersonTracker unchanged | ✓ Phase 1 |
| High GPU load | Interval-based extraction + caching | ✓ Phase 2 |
| False merges | Conservative thresholds, hard rejection | ✓ Phase 1 |
| Incorrect global IDs | Split mechanism, validation | ✓ Phase 4 |
| Production instability | Feature flag, shadow mode first | ✓ Phase 0 |
| Low face visibility | Use body ReID as primary signal | ✓ Phase 1 |

---

## Success Metrics

### Phase 1 (v1 Launch):
- ✓ Global IDs assigned to all tracks
- ✓ <1% false merge rate
- ✓ >70% cross-camera match rate
- ✓ <50ms average matching latency
- ✓ No production incidents
- ✓ Shadow mode data collected

### Long-term (v2+):
- >85% cross-camera match rate
- <0.5% false merge rate
- Support 20+ cameras
- <30ms matching latency
- Real-time identity tracking

---

## Summary

This plan provides a **safe, phased migration** to add global track IDs:

1. **Phase 0**: Instrumentation (no risk)
2. **Phase 1**: GlobalTrackManager v1 - body ReID only, conservative (low risk)
3. **Phase 2**: Optimization - caching, batching (performance)
4. **Phase 3**: Track-end matching (accuracy improvement)
5. **Phase 4**: Safety mechanisms (robustnes


s)
6. **Phase 5**: Production rollout (gradual)

**Future Work** (optional, after v1 stable):
- Face integration (high-confidence override)
- In-camera BoT-SORT upgrade (risky, postponed)
- Identity-based matching

**Philosophy**: Prefer duplicate global IDs over false merges. Conservative, stable, production-first.
