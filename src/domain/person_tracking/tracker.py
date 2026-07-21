"""
Person Tracker using BoT-SORT (or ByteTrack/OC-SORT).

Assigns and maintains stable TrackIDs for persons across video frames.
Uses boxmot library with external detections (single YOLO pass).
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from loguru import logger
import sys
from pathlib import Path

# Add boxmot to path
boxmot_path = Path(__file__).parent.parent.parent.parent / "modules" / "yolo_tracking"
if str(boxmot_path) not in sys.path:
    sys.path.insert(0, str(boxmot_path))

try:
    from boxmot.trackers.botsort.bot_sort import BoTSORT
    BOTSORT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BoT-SORT not available: {e}. Falling back to simple IoU tracking.")
    BOTSORT_AVAILABLE = False


class PersonTracker:
    """
    Multi-person tracker using BoT-SORT algorithm.

    Assigns unique TrackIDs to persons and maintains tracking across frames.
    Handles occlusions, brief disappearances, and track lifecycle management.
    """

    def __init__(
        self,
        tracker_type: str = "botsort",
        max_age: int = 120,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        model_size: str = "s",
        global_id_generator: Optional['GlobalTrackIDGenerator'] = None,
        global_track_manager: Optional['GlobalTrackManager'] = None,
        camera_id: int = 0,
        confidence_threshold: float = 0.5,
        with_reid: bool = True,
        frame_rate: int = 30,
        device: str = 'cuda:0'
    ):
        """
        Initialize Person Tracker.

        Args:
            tracker_type: Tracking algorithm ('botsort', 'bytetrack', 'ocsort')
            max_age: Maximum frames to keep track without updates (in frames, not seconds)
            min_hits: Minimum consecutive hits before track is confirmed
            iou_threshold: IoU threshold for track association
            model_size: YOLOv8 model size (legacy, not used)
            global_id_generator: Optional global ID generator for cross-camera unique IDs
            global_track_manager: Optional GlobalTrackManager for Phase 0 instrumentation
            camera_id: Camera identifier for logging
            confidence_threshold: Detection confidence threshold
            with_reid: Enable ReID appearance features (default: True)
            frame_rate: Frame rate for tracker buffer calculation
            device: Device for ReID model ('cuda:0' or 'cpu')
        """
        self.tracker_type = tracker_type.lower()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.global_id_generator = global_id_generator
        self.global_track_manager = global_track_manager
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.with_reid = with_reid
        self.frame_rate = frame_rate
        self.device = device

        # Initialize tracking state
        self.active_tracks: Dict[int, Dict] = {}  # {track_id: track_data}
        self.removed_tracks: List[Dict] = []  # Tracks that left the frame
        self.next_track_id = 1  # Only used if global_id_generator is None
        self.frame_count = 0

        # Track ID mapping (BoT-SORT ID -> our track ID)
        self.tracker_to_our_id: Dict[int, int] = {}

        # Initialize BoT-SORT tracker
        self.botsort = None
        if BOTSORT_AVAILABLE and tracker_type == 'botsort':
            try:
                # Calculate thresholds
                track_high_thresh = max(0.5, confidence_threshold)
                track_low_thresh = max(0.1, confidence_threshold * 0.5)
                new_track_thresh = min(0.7, max(0.4, confidence_threshold))

                self.botsort = BoTSORT(
                    model_weights=None,  # Will be loaded automatically if with_reid=True
                    device=device,
                    fp16=False,
                    per_class=False,
                    track_high_thresh=track_high_thresh,  # High confidence threshold (min 0.5)
                    track_low_thresh=track_low_thresh,  # Lower threshold for 2nd round (min 0.1)
                    new_track_thresh=new_track_thresh,  # Threshold for new tracks (0.4-0.7)
                    track_buffer=max_age,  # Buffer size = max_age
                    match_thresh=min(0.9, max(0.7, iou_threshold)),  # IoU threshold (0.7-0.9)
                    proximity_thresh=0.5,  # Proximity threshold for ReID
                    appearance_thresh=0.25,  # Appearance similarity threshold
                    cmc_method="sof",  # Camera motion compensation
                    frame_rate=frame_rate,
                    fuse_first_associate=False,
                    with_reid=with_reid,  # Enable/disable ReID
                    custom_features=None  # We'll provide custom features if needed
                )
                logger.debug(
                    f"BoT-SORT initialized: ReID={'enabled' if with_reid else 'disabled'}, "
                    f"device={device}, conf={confidence_threshold}, iou={iou_threshold}"
                )
            except Exception as e:
                logger.exception(f"Failed to initialize BoT-SORT: {e}")
                logger.warning("Falling back to simple IoU tracking")
                self.botsort = None
        else:
            logger.info(f"Using simple IoU tracking (BoT-SORT not available or type={tracker_type})")

        logger.debug(
            f"PersonTracker initialized: {tracker_type.upper()} "
            f"(max_age={max_age}, min_hits={min_hits}, iou={iou_threshold}, "
            f"global_ids={'enabled' if global_id_generator else 'disabled'}, "
            f"mode={'BoT-SORT' if self.botsort else 'Simple IoU'})"
        )

    def update(
        self,
        detections: List[Dict],
        frame: Optional[NDArray] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Update tracker with new detections.

        Args:
            detections: List of person detections from PersonDetector
                       Each dict contains: bbox, confidence, keypoints
            frame: Optional frame for appearance-based tracking

        Returns:
            Tuple of (active_tracks, removed_tracks)
            - active_tracks: Currently tracked persons
            - removed_tracks: Persons that left the frame in this update
        """
        self.frame_count += 1
        self.removed_tracks = []  # Reset removed tracks for this frame

        if not detections:
            self._age_tracks()
            return self._get_active_tracks(), self.removed_tracks

        # Use BoT-SORT if available, otherwise use simple IoU
        if self.botsort is not None and frame is not None:
            track_results = self._track_with_botsort(frame, detections)
        else:
            track_results = self._track_detections(detections)

        # Update active tracks
        self._update_active_tracks(track_results)

        # Remove old tracks
        self._age_tracks()

        # Verbose logging disabled to reduce log noise
        # logger.debug(
        #     f"Frame {self.frame_count}: {len(self.active_tracks)} active tracks, "
        #     f"{len(self.removed_tracks)} removed"
        # )

        return self._get_active_tracks(), self.removed_tracks

    def _track_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Perform tracking on detections.

        Args:
            detections: List of detection dicts

        Returns:
            List of tracked detections with track_id assigned
        """
        # Simple IoU-based tracking for MVP
        # For production, integrate with Ultralytics YOLO.track()

        tracked = []

        for det in detections:
            bbox = np.array(det['bbox'])

            # Try to match with existing tracks
            matched_track_id = self._match_detection_to_track(bbox)

            if matched_track_id is not None:
                # Update existing track
                track_id = matched_track_id
                self.active_tracks[track_id]['age'] = 0  # Reset age
                self.active_tracks[track_id]['hits'] += 1
            else:
                # Create new track with global or local ID
                if self.global_id_generator:
                    track_id = self.global_id_generator.get_next_id()
                else:
                    track_id = self.next_track_id
                    self.next_track_id += 1

                self.active_tracks[track_id] = {
                    'track_id': track_id,
                    'age': 0,
                    'hits': 1,
                    'first_frame': self.frame_count
                }

                # Phase 0: Log track creation
                if self.global_track_manager:
                    self.global_track_manager.on_track_created(
                        camera_id=self.camera_id,
                        local_track_id=track_id,
                        bbox=bbox,
                        frame_num=self.frame_count
                    )

            # Add tracking info to detection
            tracked_det = det.copy()
            tracked_det['track_id'] = track_id
            tracked_det['frame_num'] = self.frame_count
            tracked.append(tracked_det)

            # Update track bbox
            self.active_tracks[track_id]['bbox'] = bbox

            # Phase 0: Update track frame counter
            if self.global_track_manager:
                self.global_track_manager.on_track_update(
                    camera_id=self.camera_id,
                    local_track_id=track_id
                )

        return tracked

    def _track_with_botsort(self, frame: NDArray, detections: List[Dict]) -> List[Dict]:
        """
        Perform tracking using boxmot BoT-SORT with external detections.

        Args:
            frame: Input video frame
            detections: List of detections from PersonDetector

        Returns:
            List of tracked detections with track_id assigned
        """
        try:
            # Convert detections to BoT-SORT format: (N, 6) array [x1, y1, x2, y2, conf, class]
            if len(detections) == 0:
                dets = np.empty((0, 6))
            else:
                dets = []
                for det in detections:
                    bbox = det['bbox']  # [x1, y1, x2, y2]
                    conf = det['confidence']
                    cls = 0  # Person class
                    dets.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, cls])
                dets = np.array(dets, dtype=np.float32)

            # Run BoT-SORT tracking (uses external detections, not YOLO)
            tracks = self.botsort.update(dets, frame)  # Returns (N, 6) [x1, y1, x2, y2, track_id, conf, cls, det_ind]

            tracked = []

            if tracks is not None and len(tracks) > 0:
                for track in tracks:
                    # Parse BoT-SORT output: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
                    bbox = track[0:4]
                    botsort_id = int(track[4])
                    conf = float(track[5])
                    det_idx = int(track[7]) if len(track) > 7 else None

                    # Map BoT-SORT ID to our track ID (with global ID support)
                    if botsort_id not in self.tracker_to_our_id:
                        # New track - assign our ID
                        if self.global_id_generator:
                            our_track_id = self.global_id_generator.get_next_id()
                        else:
                            our_track_id = self.next_track_id
                            self.next_track_id += 1

                        self.tracker_to_our_id[botsort_id] = our_track_id

                        # Phase 0: Log track creation
                        if self.global_track_manager:
                            self.global_track_manager.on_track_created(
                                camera_id=self.camera_id,
                                local_track_id=our_track_id,
                                bbox=bbox,
                                frame_num=self.frame_count
                            )
                    else:
                        our_track_id = self.tracker_to_our_id[botsort_id]

                    # Get keypoints from original detection if we have the index
                    keypoints = None
                    if det_idx is not None and 0 <= det_idx < len(detections):
                        keypoints = detections[det_idx].get('keypoints')

                    tracked_det = {
                        'track_id': our_track_id,
                        'bbox': bbox.tolist(),
                        'confidence': conf,
                        'keypoints': keypoints,
                        'frame_num': self.frame_count,
                        'det_idx': det_idx,
                    }

                    tracked.append(tracked_det)

                    # Update track data
                    if our_track_id not in self.active_tracks:
                        self.active_tracks[our_track_id] = {
                            'track_id': our_track_id,
                            'age': 0,
                            'hits': 1,
                            'first_frame': self.frame_count,
                            'bbox': bbox
                        }
                    else:
                        self.active_tracks[our_track_id]['age'] = 0
                        self.active_tracks[our_track_id]['hits'] += 1
                        self.active_tracks[our_track_id]['bbox'] = bbox

                    # Phase 0: Update track frame counter
                    if self.global_track_manager:
                        self.global_track_manager.on_track_update(
                            camera_id=self.camera_id,
                            local_track_id=our_track_id
                        )

            return tracked

        except Exception as e:
            logger.exception(f"Error in BoT-SORT tracking: {e}")
            # Fallback to simple IoU tracking on error
            return self._track_detections(detections)

    def _match_detection_to_track(self, bbox: NDArray) -> Optional[int]:
        """
        Match a detection bbox to existing tracks using IoU.

        Args:
            bbox: Detection bounding box [x1, y1, x2, y2]

        Returns:
            Matched track_id or None
        """
        best_iou = 0.0
        best_track_id = None

        for track_id, track_data in self.active_tracks.items():
            if 'bbox' not in track_data:
                continue

            iou = self._calculate_iou(bbox, track_data['bbox'])

            if iou > self.iou_threshold and iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        return best_track_id

    @staticmethod
    def _calculate_iou(bbox1: NDArray, bbox2: NDArray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bboxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]

        Returns:
            IoU value (0.0 to 1.0)
        """
        # Intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _update_active_tracks(self, tracked_detections: List[Dict]) -> None:
        """
        Update active tracks with new detections.

        Args:
            tracked_detections: List of detections with track_id assigned
        """
        # Mark all tracks as not updated
        for track_id in self.active_tracks:
            self.active_tracks[track_id]['updated'] = False

        # Update tracks with new detections
        for det in tracked_detections:
            track_id = det['track_id']
            if track_id in self.active_tracks:
                det_idx = det.get('det_idx')
                is_detected = det_idx is not None and det_idx >= 0
                if is_detected:
                    # Real YOLO detection — reset age, keep alive
                    self.active_tracks[track_id]['updated'] = True
                    self.active_tracks[track_id]['last_detection'] = det
                    self.active_tracks[track_id]['last_frame'] = self.frame_count
                # Predicted tracks (det_idx < 0): do NOT set updated=True
                # so _age_tracks will increment their age and drop them

    def _age_tracks(self) -> None:
        """
        Age tracks and remove old ones.

        Tracks that haven't been updated for max_age frames are removed.
        """
        tracks_to_remove = []

        for track_id, track_data in self.active_tracks.items():
            if not track_data.get('updated', False):
                track_data['age'] += 1

                # Remove track if too old
                if track_data['age'] > self.max_age:
                    tracks_to_remove.append(track_id)

        # Remove old tracks
        for track_id in tracks_to_remove:
            removed_track = self.active_tracks.pop(track_id)

            # Only add to removed_tracks if track was confirmed (min_hits)
            if removed_track.get('hits', 0) >= self.min_hits:
                total_frames = self.frame_count - removed_track.get('first_frame', 0)

                self.removed_tracks.append({
                    'track_id': track_id,
                    'last_detection': removed_track.get('last_detection'),
                    'total_frames': total_frames,
                    'total_hits': removed_track.get('hits', 0)
                })

                # Phase 0: Log track removal
                if self.global_track_manager:
                    self.global_track_manager.on_track_removed(
                        camera_id=self.camera_id,
                        local_track_id=track_id,
                        total_frames=total_frames
                    )

                # Verbose logging disabled to reduce log noise
                # logger.debug(
                #     f"Track {track_id} removed after {removed_track.get('age', 0)} frames "
                #     f"of inactivity (total frames: {removed_track.get('hits', 0)})"
                # )

    def _get_active_tracks(self) -> List[Dict]:
        """
        Get list of currently active tracks.

        Returns:
            List of active track dicts with detection info
        """
        active = []

        for track_id, track_data in self.active_tracks.items():
            # Only return confirmed tracks (min_hits) that are currently being detected (age = 0)
            # This prevents ghost bboxes from aging tracks that left the frame
            if track_data.get('hits', 0) >= self.min_hits:
                if 'last_detection' in track_data and track_data.get('age', 0) == 0:
                    track_dict = track_data['last_detection'].copy()
                    track_dict['track_id'] = track_id
                    track_dict['track_age'] = track_data.get('age', 0)
                    track_dict['track_hits'] = track_data.get('hits', 0)
                    active.append(track_dict)

        return active

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Get information about a specific track.

        Args:
            track_id: Track ID to query

        Returns:
            Track info dict or None if not found
        """
        return self.active_tracks.get(track_id)
