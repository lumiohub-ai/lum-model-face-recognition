"""
Idle Detection Temporal Filter.

Implements temporal smoothing for idle detection to reduce false positives.
Uses immediate detection for idle state and delayed confirmation for working state.
"""

from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from enum import Enum
import time
import numpy as np
from loguru import logger


class IdleState(Enum):
    """Idle states."""
    WORKING = "working"  # Person is looking at screen
    IDLE = "idle"        # Person is NOT looking at screen


class IdleFilter:
    """
    Temporal filter for idle detection.

    State machine:
    WORKING -> IDLE: Immediate transition when person looks away
    IDLE -> WORKING: Requires N consecutive frames showing person looking at screen

    This prevents flickering while ensuring we quickly detect when someone becomes idle.
    """

    def __init__(
        self,
        working_confirmation_frames: int = 5,
        min_duration_ms: int = 500
    ):
        """
        Initialize Idle Filter.

        Args:
            working_confirmation_frames: Consecutive frames needed to confirm working state
            min_duration_ms: Minimum duration for state tracking (not enforced for transitions)
        """
        self.working_confirmation_frames = working_confirmation_frames
        self.min_duration_ms = min_duration_ms

        # Track idle detection history per person
        # Format: {track_id: [(is_idle, confidence, timestamp), ...]}
        self.idle_history: Dict[int, List[Tuple[bool, float, float]]] = defaultdict(list)

        # Current idle state per person
        # Format: {track_id: IdleState}
        self.idle_state: Dict[int, IdleState] = {}

        # State change timestamps
        # Format: {track_id: timestamp}
        self.state_change_time: Dict[int, float] = {}

        # Counter for consecutive frames looking at screen (for working detection)
        # Format: {track_id: count}
        self.working_counter: Dict[int, int] = defaultdict(int)

        logger.info(
            f"IdleFilter initialized: immediate idle detection, "
            f"working_confirmation={self.working_confirmation_frames} frames"
        )

    def update(
        self,
        track_id: int,
        spatial_result: Dict
    ) -> bool:
        """
        Update idle state with new spatial detection result.

        Logic:
        - If person NOT looking at screen (idle) -> immediately mark as IDLE
        - If person IS looking at screen -> wait for working_confirmation_frames before marking as WORKING

        Args:
            track_id: Person track identifier
            spatial_result: Result from IdleDetectionLogic:
                {
                    'is_idle': bool,
                    'confidence': float,
                    'method': str,
                    'details': {...}
                }

        Returns:
            Current idle state (True=idle, False=working)
        """
        is_idle = spatial_result.get('is_idle', True)
        confidence = spatial_result.get('confidence', 0.0)
        timestamp = time.time()

        # Add to history (keep for statistics)
        self.idle_history[track_id].append((is_idle, confidence, timestamp))

        # Keep only last 10 frames
        if len(self.idle_history[track_id]) > 10:
            self.idle_history[track_id] = self.idle_history[track_id][-10:]

        # Get current state (default to WORKING)
        current_state = self.idle_state.get(track_id, IdleState.WORKING)

        # IMMEDIATE IDLE DETECTION LOGIC
        if is_idle:
            # Person is idle (not looking at screen) -> immediately transition to IDLE
            if current_state != IdleState.IDLE:
                self._transition_state(track_id, IdleState.IDLE, confidence, 1)
            # Reset working counter
            self.working_counter[track_id] = 0
        else:
            # Person is looking at screen (working)
            if current_state == IdleState.IDLE:
                # Currently idle, increment counter
                self.working_counter[track_id] += 1

                # Only transition to WORKING after working_confirmation_frames
                if self.working_counter[track_id] >= self.working_confirmation_frames:
                    self._transition_state(
                        track_id,
                        IdleState.WORKING,
                        confidence,
                        self.working_counter[track_id]
                    )
                    self.working_counter[track_id] = 0
            else:
                # Already working, reset counter
                self.working_counter[track_id] = 0

        # Return current state
        current_state = self.idle_state.get(track_id, IdleState.WORKING)
        return current_state == IdleState.IDLE

    def _transition_state(
        self,
        track_id: int,
        new_state: IdleState,
        confidence: float,
        frame_count: int
    ) -> None:
        """
        Transition to new idle state.

        Args:
            track_id: Track identifier
            new_state: New state
            confidence: Detection confidence
            frame_count: Number of frames for transition
        """
        old_state = self.idle_state.get(track_id, IdleState.WORKING)

        if old_state != new_state:
            self.idle_state[track_id] = new_state
            self.state_change_time[track_id] = time.time()

            logger.info(
                f"Track {track_id} idle state: {old_state.value} -> {new_state.value} "
                f"(confidence={confidence:.2f}, frames={frame_count})"
            )

    def is_idle(self, track_id: int) -> bool:
        """
        Check if person is currently idle.

        Args:
            track_id: Track identifier

        Returns:
            True if idle (not looking at screen)
        """
        state = self.idle_state.get(track_id, IdleState.WORKING)
        return state == IdleState.IDLE

    def get_idle_confidence(self, track_id: int) -> float:
        """
        Get idle detection confidence for a track.

        Args:
            track_id: Track identifier

        Returns:
            Confidence score (0.0-1.0)
        """
        history = self.idle_history.get(track_id, [])

        if not history:
            return 0.0

        # Average confidence from recent frames
        recent_confidences = [conf for _, conf, _ in history[-5:]]
        return float(np.mean(recent_confidences))

    def get_idle_duration(self, track_id: int) -> Optional[float]:
        """
        Get duration of current idle session.

        Args:
            track_id: Track identifier

        Returns:
            Duration in seconds or None if not idle
        """
        if not self.is_idle(track_id):
            return None

        if track_id not in self.state_change_time:
            return None

        start_time = self.state_change_time[track_id]
        duration = time.time() - start_time

        return duration

    def get_working_duration(self, track_id: int) -> Optional[float]:
        """
        Get duration of current working session.

        Args:
            track_id: Track identifier

        Returns:
            Duration in seconds or None if idle
        """
        if self.is_idle(track_id):
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
        history = self.idle_history.get(track_id, [])
        current_state = self.idle_state.get(track_id, IdleState.WORKING)

        # Count idle in window
        if history:
            idle_count = sum(1 for is_idle, _, _ in history if is_idle)
        else:
            idle_count = 0

        status = {
            'track_id': track_id,
            'is_idle': current_state == IdleState.IDLE,
            'state': current_state.value,
            'frames_in_window': len(history),
            'idle_count': idle_count,
            'confidence': self.get_idle_confidence(track_id),
            'working_counter': self.working_counter.get(track_id, 0)
        }

        # Add duration based on state
        if current_state == IdleState.IDLE:
            duration = self.get_idle_duration(track_id)
            if duration is not None:
                status['idle_duration_seconds'] = duration
        else:
            duration = self.get_working_duration(track_id)
            if duration is not None:
                status['working_duration_seconds'] = duration

        return status

    def reset_track(self, track_id: int) -> None:
        """
        Reset all data for a track.

        Args:
            track_id: Track identifier
        """
        if track_id in self.idle_history:
            del self.idle_history[track_id]
        if track_id in self.idle_state:
            del self.idle_state[track_id]
        if track_id in self.state_change_time:
            del self.state_change_time[track_id]
        if track_id in self.working_counter:
            del self.working_counter[track_id]

    def get_statistics(self) -> Dict:
        """
        Get filter statistics.

        Returns:
            Dictionary with stats
        """
        total_tracks = len(self.idle_history)
        idle_count = sum(
            1 for state in self.idle_state.values()
            if state == IdleState.IDLE
        )

        return {
            'total_tracks': total_tracks,
            'idle': idle_count,
            'working': total_tracks - idle_count,
            'config': {
                'working_confirmation_frames': self.working_confirmation_frames,
                'min_duration_ms': self.min_duration_ms
            }
        }

    def reset(self) -> None:
        """Reset all filter data."""
        self.idle_history.clear()
        self.idle_state.clear()
        self.state_change_time.clear()
        self.working_counter.clear()
        logger.info("Idle filter reset")

    def __repr__(self) -> str:
        """String representation."""
        idle_count = sum(
            1 for s in self.idle_state.values()
            if s == IdleState.IDLE
        )
        return (
            f"IdleFilter(working_conf={self.working_confirmation_frames}, "
            f"idle={idle_count}/{len(self.idle_history)})"
        )
