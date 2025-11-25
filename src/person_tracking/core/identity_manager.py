"""
Identity Manager with Confidence-Weighted Temporal Voting.

Implements robust identity locking using a sliding window approach with
consensus voting to prevent identity flickering and false positives.
"""

from typing import Dict, Optional, List, Tuple
from collections import defaultdict, Counter
import time
import numpy as np
from loguru import logger


class IdentityManager:
    """
    Manages person identity assignment with temporal voting.

    Uses a sliding window of M frames to build consensus on person identity.
    Requires N% of frames to agree on the same identity before locking.

    This approach is more robust than simple frame counting because it:
    - Handles conflicting detections (same person matched to different identities)
    - Uses similarity scores for confidence weighting
    - Prevents premature locking from false positives
    - Configurable per camera
    """

    def __init__(
        self,
        identity_lock_frames: int = 5,
        identity_consensus: float = 0.60,
        min_window_duration_ms: int = 333,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize Identity Manager.

        Args:
            identity_lock_frames: Number of frames (M) for voting window
            identity_consensus: Percentage of frames needed for lock (0.0-1.0)
            min_window_duration_ms: Minimum window duration in milliseconds
            similarity_threshold: Minimum similarity for valid vote
        """
        self.M = identity_lock_frames
        self.consensus_threshold = identity_consensus
        self.min_window_duration_ms = min_window_duration_ms
        self.similarity_threshold = similarity_threshold

        # Track identity votes per person
        # Format: {track_id: [(identity, similarity, timestamp), ...]}
        self.identity_votes: Dict[int, List[Tuple[str, float, float]]] = defaultdict(list)

        # Locked identities
        # Format: {track_id: {'name': str, 'confidence': float, 'locked_at': float}}
        self.locked_identities: Dict[int, Dict] = {}

        logger.info(
            f"IdentityManager initialized: M={self.M} frames, "
            f"consensus={self.consensus_threshold:.0%}, "
            f"min_duration={self.min_window_duration_ms}ms"
        )

    def update_identity(
        self,
        track_id: int,
        recognition_result: Dict
    ) -> None:
        """
        Add identity vote for a track.

        Args:
            track_id: Person track identifier
            recognition_result: Face recognition result dict with:
                {
                    'recognized': bool,
                    'name': str or None,
                    'similarity': float,
                    'face_detected': bool
                }
        """
        # Skip if already locked
        if track_id in self.locked_identities:
            return

        # Skip if no face detected or not recognized
        if not recognition_result.get('face_detected') or not recognition_result.get('recognized'):
            return

        identity = recognition_result.get('name')
        similarity = recognition_result.get('similarity', 0.0)

        # Skip if similarity below threshold
        if similarity < self.similarity_threshold:
            return

        # Add vote
        timestamp = time.time()
        self.identity_votes[track_id].append((identity, similarity, timestamp))

        # Keep only last M votes
        if len(self.identity_votes[track_id]) > self.M:
            self.identity_votes[track_id] = self.identity_votes[track_id][-self.M:]

        # Try to lock identity
        self._try_lock_identity(track_id)

    def _try_lock_identity(self, track_id: int) -> None:
        """
        Try to lock identity if consensus reached.

        Args:
            track_id: Track identifier
        """
        votes = self.identity_votes[track_id]

        # Need at least M votes
        if len(votes) < self.M:
            return

        # Check window duration
        window_duration_ms = (votes[-1][2] - votes[0][2]) * 1000
        if window_duration_ms < self.min_window_duration_ms:
            return

        # Count votes by identity
        identity_counts = Counter([vote[0] for vote in votes])
        top_identity, count = identity_counts.most_common(1)[0]

        # Check if consensus reached
        consensus = count / len(votes)
        if consensus >= self.consensus_threshold:
            # Calculate average confidence for this identity
            identity_similarities = [
                vote[1] for vote in votes if vote[0] == top_identity
            ]
            avg_confidence = np.mean(identity_similarities)

            # Lock identity
            self.locked_identities[track_id] = {
                'name': top_identity,
                'confidence': float(avg_confidence),
                'locked_at': time.time(),
                'vote_count': count,
                'total_votes': len(votes)
            }

            logger.info(
                f"Identity locked for track {track_id}: {top_identity} "
                f"(confidence={avg_confidence:.3f}, votes={count}/{len(votes)})"
            )

    def is_identity_locked(self, track_id: int) -> bool:
        """
        Check if identity is locked for a track.

        Args:
            track_id: Track identifier

        Returns:
            True if identity is locked
        """
        return track_id in self.locked_identities

    def get_locked_identity(self, track_id: int) -> Optional[Dict]:
        """
        Get locked identity for a track.

        Args:
            track_id: Track identifier

        Returns:
            Identity dict or None if not locked
        """
        return self.locked_identities.get(track_id)

    def get_identity_name(self, track_id: int) -> Optional[str]:
        """
        Get identity name for a track.

        Args:
            track_id: Track identifier

        Returns:
            Identity name or None if not locked
        """
        identity = self.locked_identities.get(track_id)
        return identity['name'] if identity else None

    def unlock_identity(self, track_id: int) -> None:
        """
        Unlock identity for a track (force re-recognition).

        Args:
            track_id: Track identifier
        """
        if track_id in self.locked_identities:
            identity_name = self.locked_identities[track_id]['name']
            del self.locked_identities[track_id]
            logger.info(f"Identity unlocked for track {track_id}: {identity_name}")

        # Clear votes
        if track_id in self.identity_votes:
            del self.identity_votes[track_id]

    def reset_track(self, track_id: int) -> None:
        """
        Reset all data for a track.

        Args:
            track_id: Track identifier
        """
        if track_id in self.identity_votes:
            del self.identity_votes[track_id]
        if track_id in self.locked_identities:
            del self.locked_identities[track_id]

    def get_voting_status(self, track_id: int) -> Dict:
        """
        Get current voting status for a track.

        Args:
            track_id: Track identifier

        Returns:
            Status dictionary with voting details
        """
        votes = self.identity_votes.get(track_id, [])

        if not votes:
            return {
                'track_id': track_id,
                'locked': False,
                'votes': 0,
                'required_votes': self.M,
                'top_identity': None,
                'consensus': 0.0
            }

        # Count votes
        identity_counts = Counter([vote[0] for vote in votes])
        if identity_counts:
            top_identity, count = identity_counts.most_common(1)[0]
            consensus = count / len(votes)
        else:
            top_identity = None
            consensus = 0.0

        is_locked = track_id in self.locked_identities

        status = {
            'track_id': track_id,
            'locked': is_locked,
            'votes': len(votes),
            'required_votes': self.M,
            'top_identity': top_identity,
            'consensus': consensus,
            'consensus_required': self.consensus_threshold
        }

        if is_locked:
            status['locked_identity'] = self.locked_identities[track_id]

        return status

    def get_statistics(self) -> Dict:
        """
        Get identity manager statistics.

        Returns:
            Dictionary with stats
        """
        total_tracks = len(self.identity_votes)
        locked_tracks = len(self.locked_identities)
        pending_tracks = total_tracks - locked_tracks

        # Calculate average votes for pending tracks
        pending_votes = [
            len(votes) for tid, votes in self.identity_votes.items()
            if tid not in self.locked_identities
        ]
        avg_pending_votes = np.mean(pending_votes) if pending_votes else 0.0

        return {
            'total_tracks': total_tracks,
            'locked_tracks': locked_tracks,
            'pending_tracks': pending_tracks,
            'avg_pending_votes': float(avg_pending_votes),
            'config': {
                'M': self.M,
                'consensus_threshold': self.consensus_threshold,
                'min_window_duration_ms': self.min_window_duration_ms
            }
        }

    def reset(self) -> None:
        """Reset all identity data."""
        self.identity_votes.clear()
        self.locked_identities.clear()
        logger.info("Identity manager reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"IdentityManager(M={self.M}, consensus={self.consensus_threshold:.0%}, "
            f"locked={len(self.locked_identities)}, pending={len(self.identity_votes)})"
        )
