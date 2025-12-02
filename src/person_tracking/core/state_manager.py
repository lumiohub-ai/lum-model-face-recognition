"""
Person State Manager with Event System.

Manages comprehensive state for each tracked person and emits events
when state changes occur (identity locked, phone usage started/stopped, etc.).
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import time
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
    PHONE_USAGE_STARTED = "phone_usage_started"
    PHONE_USAGE_STOPPED = "phone_usage_stopped"
    IDLE_STARTED = "idle_started"
    IDLE_STOPPED = "idle_stopped"


@dataclass
class PersonState:
    """State for a single person."""
    track_id: int
    camera_id: int

    # Identity
    identity: Optional[str] = None
    identity_locked: bool = False
    identity_confidence: float = 0.0

    # Phone usage
    using_phone: bool = False
    phone_confidence: float = 0.0
    phone_usage_duration: float = 0.0  # seconds

    # Idle detection (not looking at screen)
    is_idle: bool = False
    idle_confidence: float = 0.0

    # Timestamps
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    identity_locked_at: Optional[datetime] = None
    phone_usage_started_at: Optional[datetime] = None

    # Stats
    total_frames: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime to ISO format
        for key in ['first_seen', 'last_seen', 'identity_locked_at', 'phone_usage_started_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data


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

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = {
            'event_type': self.event_type.value,
            'track_id': self.track_id,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
        }

        if self.identity is not None:
            data['identity'] = self.identity
        if self.confidence is not None:
            data['confidence'] = self.confidence
        if self.duration is not None:
            data['duration'] = self.duration
        if self.metadata:
            data['metadata'] = self.metadata

        return data


class PersonStateManager:
    """
    Manages state for all tracked persons and emits events.

    Maintains complete state including:
    - Identity (name, locked status, confidence)
    - Phone usage (current state, duration)
    - Timestamps (first/last seen, state changes)
    - Statistics (total frames, etc.)

    Emits events when:
    - Person enters frame
    - Person exits frame
    - Identity locked
    - Identity changed
    - Phone usage started
    - Phone usage stopped
    """

    def __init__(self, camera_id: int = 1, api_client: Optional['APIClient'] = None, name_to_id_map: Optional[Dict[str, int]] = None):
        """
        Initialize Person State Manager.

        Args:
            camera_id: Camera identifier
            api_client: API client instance
            name_to_id_map: Dictionary mapping user names to IDs
        """
        self.camera_id = camera_id
        self.api_client = api_client
        self.name_to_id_map = name_to_id_map or {}

        # Person states
        # Format: {track_id: PersonState}
        self.person_states: Dict[int, PersonState] = {}

        # Event queue
        self.event_queue: List[PersonEvent] = []

        logger.info(f"PersonStateManager initialized for camera {camera_id}")

    def update_person(
        self,
        track_id: int,
        identity: Optional[str] = None,
        identity_locked: bool = False,
        identity_confidence: float = 0.0,
        using_phone: bool = False,
        phone_confidence: float = 0.0,
        is_idle: bool = False,
        idle_confidence: float = 0.0,
        proof_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Update state for a person.

        Args:
            track_id: Person track identifier
            identity: Person name (if recognized)
            identity_locked: Whether identity is locked
            identity_confidence: Identity confidence score
            using_phone: Whether person is using phone
            phone_confidence: Phone usage confidence
            is_idle: Whether person is idle (not looking at screen)
            idle_confidence: Idle detection confidence
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

        # Update phone usage
        self._update_phone_usage(state, using_phone, phone_confidence, proof_image)

        # Update idle status
        self._update_idle(state, is_idle, idle_confidence, proof_image)

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

    def _update_phone_usage(
        self,
        state: PersonState,
        using_phone: bool,
        confidence: float,
        proof_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Update phone usage and emit events if changed.

        Args:
            state: Person state
            using_phone: Whether using phone
            confidence: Confidence score
            proof_image: Optional image for proof
        """
        old_using_phone = state.using_phone

        # Update phone usage
        state.using_phone = using_phone
        state.phone_confidence = confidence

        # Phone usage started
        if using_phone and not old_using_phone:
            state.phone_usage_started_at = datetime.now()
            state.phone_usage_duration = 0.0

            self._emit_event(
                EventType.PHONE_USAGE_STARTED,
                state.track_id,
                identity=state.identity,
                confidence=confidence
            )

            logger.info(
                f"Track {state.track_id} ({state.identity or 'Unknown'}): "
                f"Phone usage started"
            )

            # Send activity to API if person is identified
            if self.api_client and state.identity and self.name_to_id_map:
                user_id = self.name_to_id_map.get(state.identity)
                if user_id:
                    logger.info(f"Sending phone_usage activity for user {user_id} ({state.identity})")
                    response = self.api_client.send_activities(
                        activity_type='phone_usage',
                        camera_id=state.camera_id,
                        user_id=user_id,
                        confidence_score=confidence,
                        proof_image=proof_image
                    )
                    if response and response.status_code in [200, 201]:
                        logger.info(f"Successfully sent phone_usage activity for {state.identity}")
                    else:
                        logger.error(f"Failed to send phone_usage activity for {state.identity}")

        # Phone usage stopped
        elif not using_phone and old_using_phone:
            # Calculate duration
            if state.phone_usage_started_at:
                duration = (datetime.now() - state.phone_usage_started_at).total_seconds()
                state.phone_usage_duration += duration
            else:
                duration = 0.0

            self._emit_event(
                EventType.PHONE_USAGE_STOPPED,
                state.track_id,
                identity=state.identity,
                duration=duration
            )

            if self.api_client and state.identity and self.name_to_id_map:
                user_id = self.name_to_id_map.get(state.identity)
                if user_id:
                    logger.info(f"Sending working activity for user {user_id} (stopped phone usage)")
                    response = self.api_client.send_activities(
                        activity_type='working',
                        camera_id=state.camera_id,
                        user_id=user_id
                    )
                    if response and response.status_code in [200, 201]:
                        logger.info(f"Successfully sent working activity for {state.identity}")
                    else:
                        logger.error(f"Failed to send working activity for {state.identity}")
            state.phone_usage_started_at = None

        # Update duration if currently using phone
        elif using_phone and state.phone_usage_started_at:
            duration = (datetime.now() - state.phone_usage_started_at).total_seconds()
            state.phone_usage_duration = duration

    def _update_idle(
        self,
        state: PersonState,
        is_idle: bool,
        confidence: float,
        proof_image: Optional[np.ndarray] = None
    ) -> None:
        """
        Update idle status and emit events if changed.

        Args:
            state: Person state
            is_idle: Whether person is idle (not looking at screen)
            confidence: Confidence score
            proof_image: Optional image for proof
        """
        old_is_idle = state.is_idle

        # Update idle status
        state.is_idle = is_idle
        state.idle_confidence = confidence

        # Idle started (person stopped looking at screen)
        if is_idle and not old_is_idle:
            self._emit_event(
                EventType.IDLE_STARTED,
                state.track_id,
                identity=state.identity,
                confidence=confidence
            )

            logger.info(
                f"Track {state.track_id} ({state.identity or 'Unknown'}): "
                f"Idle started (not looking at screen)"
            )

            # Send activity to API if person is identified
            if self.api_client and state.identity and self.name_to_id_map:
                user_id = self.name_to_id_map.get(state.identity)
                if user_id:
                    logger.info(f"Sending not_focusing activity for user {user_id} ({state.identity})")
                    response = self.api_client.send_activities(
                        activity_type='not_focusing',
                        camera_id=state.camera_id,
                        user_id=user_id,
                        confidence_score=confidence,
                        proof_image=proof_image
                    )
                    if response and response.status_code in [200, 201]:
                        logger.info(f"Successfully sent not_focusing activity for {state.identity}")
                    else:
                        logger.error(f"Failed to send not_focusing activity for {state.identity}")

        # Idle stopped (person started looking at screen again)
        elif not is_idle and old_is_idle:
            self._emit_event(
                EventType.IDLE_STOPPED,
                state.track_id,
                identity=state.identity,
                confidence=confidence
            )

            logger.info(
                f"Track {state.track_id} ({state.identity or 'Unknown'}): "
                f"Idle stopped (looking at screen)"
            )

            # Send working activity to API if person is identified
            if self.api_client and state.identity and self.name_to_id_map:
                user_id = self.name_to_id_map.get(state.identity)
                if user_id:
                    logger.info(f"Sending working activity for user {user_id} ({state.identity}) - started looking at screen")
                    response = self.api_client.send_activities(
                        activity_type='working',
                        camera_id=state.camera_id,
                        user_id=user_id
                    )
                    if response and response.status_code in [200, 201]:
                        logger.info(f"Successfully sent working activity for {state.identity}")
                    else:
                        logger.error(f"Failed to send working activity for {state.identity}")

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
                'total_frames': state.total_frames,
                'total_phone_usage_seconds': state.phone_usage_duration
            }
        )

        # logger.info(
        #     f"Track {track_id} ({state.identity or 'Unknown'}) exited: "
        #     f"{state.total_frames} frames, "
        #     f"{state.phone_usage_duration:.1f}s phone usage"
        # )

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
            duration: Duration (for phone usage events)
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

    def get_events(self, clear: bool = True) -> List[PersonEvent]:
        """
        Get queued events.

        Args:
            clear: Whether to clear event queue after retrieval

        Returns:
            List of events
        """
        events = self.event_queue.copy()

        if clear:
            self.event_queue.clear()

        return events

    def get_statistics(self) -> Dict:
        """
        Get state manager statistics.

        Returns:
            Dictionary with stats
        """
        total_persons = len(self.person_states)
        identified_persons = sum(
            1 for s in self.person_states.values()
            if s.identity_locked
        )
        using_phone_persons = sum(
            1 for s in self.person_states.values()
            if s.using_phone
        )
        idle_persons = sum(
            1 for s in self.person_states.values()
            if s.is_idle
        )

        total_phone_usage_time = sum(
            s.phone_usage_duration
            for s in self.person_states.values()
        )

        return {
            'camera_id': self.camera_id,
            'total_persons': total_persons,
            'identified_persons': identified_persons,
            'using_phone': using_phone_persons,
            'idle_persons': idle_persons,
            'total_phone_usage_seconds': total_phone_usage_time,
            'events_queued': len(self.event_queue)
        }

    def reset(self) -> None:
        """Reset all state data."""
        self.person_states.clear()
        self.event_queue.clear()
        logger.info("State manager reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PersonStateManager(camera={self.camera_id}, "
            f"persons={len(self.person_states)}, "
            f"events={len(self.event_queue)})"
        )
