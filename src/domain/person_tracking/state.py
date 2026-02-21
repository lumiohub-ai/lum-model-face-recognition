"""
Person State Manager with Event System.

Manages comprehensive state for each tracked person and emits events
when state changes occur (identity locked, person entered/exited, etc.).
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from collections import deque
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from src.face_recognition.api.client import APIClient


class EventType(Enum):
    """Person tracking event types."""
    PERSON_ENTERED = "person_entered"
    PERSON_EXITED = "person_exited"
    IDENTITY_LOCKED = "identity_locked"
    IDENTITY_CHANGED = "identity_changed"


@dataclass
class PersonState:
    """State for a single person."""
    track_id: int
    camera_id: int

    # Identity
    identity: Optional[str] = None
    identity_locked: bool = False
    identity_confidence: float = 0.0

    # Timestamps
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    identity_locked_at: Optional[datetime] = None

    # Action Recognition
    last_action_check_time: float = 0.0  # Unix timestamp of last action recognition check
    last_detected_action: Optional[str] = None  # Last detected action type

    # Stats
    total_frames: int = 0


@dataclass
class PersonEvent:
    """Event emitted when person state changes."""
    event_type: EventType
    track_id: int
    camera_id: int
    timestamp: datetime

    # Optional event-specific data
    identity: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    metadata: Optional[Dict] = None


class PersonStateManager:
    """
    Manages state for all tracked persons and emits events.

    Maintains complete state including:
    - Identity (name, locked status, confidence)
    - Timestamps (first/last seen, state changes)
    - Statistics (total frames, etc.)

    Emits events when:
    - Person enters frame
    - Person exits frame
    - Identity locked
    - Identity changed
    """

    def __init__(self, camera_id: int = 1, name_to_id_map: Optional[Dict[str, int]] = None):
        """
        Initialize Person State Manager.

        Args:
            camera_id: Camera identifier
            name_to_id_map: Dictionary mapping user names to IDs
        """
        self.camera_id = camera_id
        self.name_to_id_map = name_to_id_map or {}

        # Person states
        # Format: {track_id: PersonState}
        self.person_states: Dict[int, PersonState] = {}

        # Event queue (bounded to prevent memory leak - events are logged but not consumed)
        self.event_queue: deque = deque(maxlen=1000)

        logger.info(f"PersonStateManager initialized for camera {camera_id}")

    def update_person(
        self,
        track_id: int,
        identity: Optional[str] = None,
        identity_locked: bool = False,
        identity_confidence: float = 0.0,
        proof_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Update state for a person.

        Args:
            track_id: Person track identifier
            identity: Person name (if recognized)
            identity_locked: Whether identity is locked
            identity_confidence: Identity confidence score
            proof_image: Optional image (frame or cropped bbox) for proof
        """
        now = datetime.now()

        # Create new state if person not tracked
        if track_id not in self.person_states:
            self.person_states[track_id] = PersonState(
                track_id=track_id,
                camera_id=self.camera_id,
                first_seen=now,
                last_seen=now
            )

            # Emit person entered event
            self._emit_event(
                EventType.PERSON_ENTERED,
                track_id,
                identity=identity
            )

        state = self.person_states[track_id]

        # Update timestamps
        state.last_seen = now
        state.total_frames += 1

        # Update identity
        self._update_identity(state, identity, identity_locked, identity_confidence)

    def _update_identity(
        self,
        state: PersonState,
        identity: Optional[str],
        identity_locked: bool,
        confidence: float
    ) -> None:
        """
        Update identity and emit events if changed.

        Args:
            state: Person state
            identity: New identity
            identity_locked: Whether locked
            confidence: Confidence score
        """
        old_identity = state.identity
        old_locked = state.identity_locked

        # Update identity
        state.identity = identity
        state.identity_confidence = confidence

        # Check if identity just locked
        if identity_locked and not old_locked:
            state.identity_locked = True
            state.identity_locked_at = datetime.now()

            self._emit_event(
                EventType.IDENTITY_LOCKED,
                state.track_id,
                identity=identity,
                confidence=confidence
            )

            # logger.info(
            #     f"Track {state.track_id}: Identity locked as '{identity}' "
            #     f"(confidence={confidence:.3f})"
            # )

        # Check if identity changed
        elif identity_locked and old_identity and identity != old_identity:
            state.identity_locked_at = datetime.now()

            self._emit_event(
                EventType.IDENTITY_CHANGED,
                state.track_id,
                identity=identity,
                confidence=confidence,
                metadata={'old_identity': old_identity}
            )

            logger.warning(
                f"Track {state.track_id}: Identity changed from '{old_identity}' "
                f"to '{identity}'"
            )

    def remove_person(self, track_id: int) -> Optional[PersonState]:
        """
        Remove person from tracking (when they leave frame).

        Args:
            track_id: Track identifier

        Returns:
            Final person state or None
        """
        if track_id not in self.person_states:
            return None

        state = self.person_states.pop(track_id)

        # Emit person exited event
        self._emit_event(
            EventType.PERSON_EXITED,
            track_id,
            identity=state.identity,
            metadata={
                'total_frames': state.total_frames
            }
        )

        return state

    def _emit_event(
        self,
        event_type: EventType,
        track_id: int,
        identity: Optional[str] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Emit an event.

        Args:
            event_type: Type of event
            track_id: Track identifier
            identity: Person identity (if known)
            confidence: Confidence score
            duration: Duration (for timed events)
            metadata: Additional metadata
        """
        event = PersonEvent(
            event_type=event_type,
            track_id=track_id,
            camera_id=self.camera_id,
            timestamp=datetime.now(),
            identity=identity,
            confidence=confidence,
            duration=duration,
            metadata=metadata
        )

        self.event_queue.append(event)

    def get_state(self, track_id: int) -> Optional[PersonState]:
        """
        Get state for a person.

        Args:
            track_id: Track identifier

        Returns:
            Person state or None
        """
        return self.person_states.get(track_id)

    def get_all_states(self) -> List[PersonState]:
        """
        Get all person states.

        Returns:
            List of person states
        """
        return list(self.person_states.values())
