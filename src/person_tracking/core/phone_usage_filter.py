"""
Phone Usage Temporal Filter with Sliding Window.

Implements temporal smoothing for phone usage detection to reduce false positives.
Uses a sliding window approach with consensus voting and minimum duration requirements.
"""

from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from enum import Enum
import time
import numpy as np
from loguru import logger


class PhoneUsageState(Enum):
    """Phone usage states."""
    NOT_USING = "not_using"
    USING = "using"


class PhoneUsageFilter:
    """
    Temporal filter for phone usage detection.

    Uses sliding window with consensus voting to smooth detections
    and reduce false positives from momentary phone appearances.

    State machine:
    NOT_USING -> USING: Requires N frames with ≥ consensus% showing usage
    USING -> NOT_USING: Requires N frames with ≥ consensus% showing no usage
    """

    def __init__(
        self,
        confirmation_frames: int = 8,
        confirmation_consensus: float = 0.75,
        min_duration_ms: int = 500,
        stop_confirmation_frames: int = 5
    ):
        """
        Initialize Phone Usage Filter.

        Args:
            confirmation_frames: Number of frames (N) for sliding window (not used for immediate detection)
            confirmation_consensus: Percentage of frames needed for confirmation (0.0-1.0) (not used for immediate detection)
            min_duration_ms: Minimum duration in milliseconds for state change (not used for immediate detection)
            stop_confirmation_frames: Number of consecutive frames without phone to confirm stopped (default: 5)
        """
        self.N = confirmation_frames
        self.consensus_threshold = confirmation_consensus
        self.min_duration_ms = min_duration_ms
        self.stop_confirmation_frames = stop_confirmation_frames

        # Track usage history per person
        # Format: {track_id: [(using_phone, confidence, timestamp), ...]}
        self.usage_history: Dict[int, List[Tuple[bool, float, float]]] = defaultdict(list)

        # Current phone usage state per person
        # Format: {track_id: PhoneUsageState}
        self.usage_state: Dict[int, PhoneUsageState] = {}

        # State change timestamps
        # Format: {track_id: timestamp}
        self.state_change_time: Dict[int, float] = {}

        # Counter for consecutive frames without phone (for stop detection)
        # Format: {track_id: count}
        self.no_phone_counter: Dict[int, int] = defaultdict(int)

        logger.info(
            f"PhoneUsageFilter initialized: immediate detection mode, "
            f"stop_delay={self.stop_confirmation_frames} frames"
        )

    def update(
        self,
        track_id: int,
        spatial_result: Dict
    ) -> bool:
        """
        Update phone usage with new spatial detection result.

        NEW LOGIC:
        - If phone detected in current frame -> immediately mark as USING
        - If no phone detected -> wait for stop_confirmation_frames before marking as NOT_USING

        Args:
            track_id: Person track identifier
            spatial_result: Result from PhoneUsageSpatialLogic:
                {
                    'using_phone': bool,
                    'confidence': float,
                    'checks_passed': int,
                    'details': {...}
                }

        Returns:
            Current phone usage state (True/False)
        """
        using_phone = spatial_result.get('using_phone', False)
        confidence = spatial_result.get('confidence', 0.0)
        timestamp = time.time()

        # Add to history (keep for statistics)
        self.usage_history[track_id].append((using_phone, confidence, timestamp))

        # Keep only last N frames
        if len(self.usage_history[track_id]) > self.N:
            self.usage_history[track_id] = self.usage_history[track_id][-self.N:]

        # Get current state
        current_state = self.usage_state.get(track_id, PhoneUsageState.NOT_USING)

        # IMMEDIATE DETECTION LOGIC
        if using_phone:
            # Phone detected -> immediately transition to USING
            if current_state != PhoneUsageState.USING:
                self._transition_state(track_id, PhoneUsageState.USING, confidence, 1)
            # Reset no-phone counter
            self.no_phone_counter[track_id] = 0
        else:
            # No phone detected
            if current_state == PhoneUsageState.USING:
                # Currently using phone, increment counter
                self.no_phone_counter[track_id] += 1

                # Only transition to NOT_USING after stop_confirmation_frames
                if self.no_phone_counter[track_id] >= self.stop_confirmation_frames:
                    self._transition_state(track_id, PhoneUsageState.NOT_USING, 0.0,
                                         self.no_phone_counter[track_id])
                    self.no_phone_counter[track_id] = 0

        # Return current state
        current_state = self.usage_state.get(track_id, PhoneUsageState.NOT_USING)
        return current_state == PhoneUsageState.USING

    def _update_state(self, track_id: int) -> None:
        """
        Update phone usage state based on history.

        Args:
            track_id: Track identifier
        """
        history = self.usage_history[track_id]

        # Need at least N frames
        if len(history) < self.N:
            return

        # Get current state (default to NOT_USING)
        current_state = self.usage_state.get(track_id, PhoneUsageState.NOT_USING)

        # Count usage in window
        usage_count = sum(1 for using, _, _ in history if using)
        consensus = usage_count / len(history)

        # Check window duration
        window_duration_ms = (history[-1][2] - history[0][2]) * 1000

        if window_duration_ms < self.min_duration_ms:
            return  # Not enough time has passed

        # State transition logic
        if current_state == PhoneUsageState.NOT_USING:
            # Transition to USING if consensus reached
            if consensus >= self.consensus_threshold:
                self._transition_state(
                    track_id,
                    PhoneUsageState.USING,
                    consensus,
                    usage_count
                )

        elif current_state == PhoneUsageState.USING:
            # Transition to NOT_USING if consensus for no usage
            no_usage_count = len(history) - usage_count
            no_usage_consensus = no_usage_count / len(history)

            if no_usage_consensus >= self.consensus_threshold:
                self._transition_state(
                    track_id,
                    PhoneUsageState.NOT_USING,
                    no_usage_consensus,
                    no_usage_count
                )

    def _transition_state(
        self,
        track_id: int,
        new_state: PhoneUsageState,
        confidence: float,
        frame_count: int
    ) -> None:
        """
        Transition to new phone usage state.

        Args:
            track_id: Track identifier
            new_state: New state
            confidence: Detection confidence
            frame_count: Number of frames for transition
        """
        old_state = self.usage_state.get(track_id, PhoneUsageState.NOT_USING)

        if old_state != new_state:
            self.usage_state[track_id] = new_state
            self.state_change_time[track_id] = time.time()

            logger.info(
                f"Track {track_id} phone usage: {old_state.value} -> {new_state.value} "
                f"(confidence={confidence:.2f}, frames={frame_count})"
            )

    def is_using_phone(self, track_id: int) -> bool:
        """
        Check if person is currently using phone.

        Args:
            track_id: Track identifier

        Returns:
            True if using phone
        """
        state = self.usage_state.get(track_id, PhoneUsageState.NOT_USING)
        return state == PhoneUsageState.USING

    def get_usage_confidence(self, track_id: int) -> float:
        """
        Get phone usage confidence for a track.

        Args:
            track_id: Track identifier

        Returns:
            Confidence score (0.0-1.0)
        """
        history = self.usage_history.get(track_id, [])

        if not history:
            return 0.0

        # Average confidence from recent frames
        recent_confidences = [conf for _, conf, _ in history[-self.N:]]
        return float(np.mean(recent_confidences))

    def get_usage_duration(self, track_id: int) -> Optional[float]:
        """
        Get duration of current phone usage session.

        Args:
            track_id: Track identifier

        Returns:
            Duration in seconds or None if not using phone
        """
        if not self.is_using_phone(track_id):
            return None

        if track_id not in self.state_change_time:
            return None

        start_time = self.state_change_time[track_id]
        duration = time.time() - start_time

        return duration

    def get_filter_status(self, track_id: int) -> Dict:
        """
        Get current filter status for a track.

        Args:
            track_id: Track identifier

        Returns:
            Status dictionary
        """
        history = self.usage_history.get(track_id, [])
        current_state = self.usage_state.get(track_id, PhoneUsageState.NOT_USING)

        # Count usage in window
        if history:
            usage_count = sum(1 for using, _, _ in history if using)
            consensus = usage_count / len(history)
        else:
            usage_count = 0
            consensus = 0.0

        status = {
            'track_id': track_id,
            'using_phone': current_state == PhoneUsageState.USING,
            'state': current_state.value,
            'frames_in_window': len(history),
            'required_frames': self.N,
            'usage_count': usage_count,
            'consensus': consensus,
            'consensus_required': self.consensus_threshold,
            'confidence': self.get_usage_confidence(track_id)
        }

        # Add duration if using phone
        duration = self.get_usage_duration(track_id)
        if duration is not None:
            status['usage_duration_seconds'] = duration

        return status

    def reset_track(self, track_id: int) -> None:
        """
        Reset all data for a track.

        Args:
            track_id: Track identifier
        """
        if track_id in self.usage_history:
            del self.usage_history[track_id]
        if track_id in self.usage_state:
            del self.usage_state[track_id]
        if track_id in self.state_change_time:
            del self.state_change_time[track_id]
        if track_id in self.no_phone_counter:
            del self.no_phone_counter[track_id]

    def get_statistics(self) -> Dict:
        """
        Get filter statistics.

        Returns:
            Dictionary with stats
        """
        total_tracks = len(self.usage_history)
        using_phone_count = sum(
            1 for state in self.usage_state.values()
            if state == PhoneUsageState.USING
        )

        return {
            'total_tracks': total_tracks,
            'using_phone': using_phone_count,
            'not_using_phone': total_tracks - using_phone_count,
            'config': {
                'N': self.N,
                'consensus_threshold': self.consensus_threshold,
                'min_duration_ms': self.min_duration_ms
            }
        }

    def reset(self) -> None:
        """Reset all filter data."""
        self.usage_history.clear()
        self.usage_state.clear()
        self.state_change_time.clear()
        self.no_phone_counter.clear()
        logger.info("Phone usage filter reset")

    def __repr__(self) -> str:
        """String representation."""
        using_count = sum(
            1 for s in self.usage_state.values()
            if s == PhoneUsageState.USING
        )
        return (
            f"PhoneUsageFilter(N={self.N}, consensus={self.consensus_threshold:.0%}, "
            f"using_phone={using_count}/{len(self.usage_history)})"
        )
