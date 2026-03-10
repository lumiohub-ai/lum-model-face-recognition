"""ID Switch Corrector for person tracking.

Detects and corrects track ID switches using face embeddings.
Implements multi-tier correction strategy for robust tracking.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class IDSwitchCorrector:
    """Corrects track ID switches using face embedding analysis.

    Multi-tier approach:
    - Tier 1: Prevention via face-based track matching
    - Tier 2: Immediate correction via identity consistency checks
    - Tier 3: Delayed correction via batch embedding analysis
    """

    def __init__(
        self,
        embedding_distance_threshold: float = 0.4,
        correction_interval_frames: int = 5,
        min_embedding_samples: int = 3
    ):
        """Initialize ID switch corrector.

        Args:
            embedding_distance_threshold: Max cosine distance for same person (0.4 = similar)
            correction_interval_frames: How often to run batch correction (every N frames)
            min_embedding_samples: Minimum embeddings needed for reliable comparison
        """
        self.embedding_threshold = embedding_distance_threshold
        self.correction_interval = correction_interval_frames
        self.min_samples = min_embedding_samples

        # Track embeddings: {track_id: [embeddings]}
        self.track_embeddings: Dict[int, List[np.ndarray]] = {}

        # Track identities: {track_id: identity_name}
        self.track_identities: Dict[int, str] = {}

        # Frame counter for delayed correction
        self.frame_count = 0

        # Correction history: {(wrong_id, correct_id): correction_count}
        self.correction_history: Dict[Tuple[int, int], int] = {}

        logger.debug(
            f"IDSwitchCorrector initialized: threshold={embedding_distance_threshold}, "
            f"interval={correction_interval_frames} frames"
        )

    def add_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
        identity_name: Optional[str] = None
    ) -> None:
        """Add face embedding for a track.

        Args:
            track_id: Track identifier
            embedding: Face embedding vector (512-d for InsightFace)
            identity_name: Optional locked identity name
        """
        if track_id not in self.track_embeddings:
            self.track_embeddings[track_id] = []

        self.track_embeddings[track_id].append(embedding)

        # Keep only last 10 embeddings per track (memory efficiency)
        if len(self.track_embeddings[track_id]) > 10:
            self.track_embeddings[track_id] = self.track_embeddings[track_id][-10:]

        # Store identity if provided
        if identity_name:
            self.track_identities[track_id] = identity_name

    def get_average_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get average embedding for a track.

        Args:
            track_id: Track identifier

        Returns:
            Average embedding vector or None
        """
        embeddings = self.track_embeddings.get(track_id, [])

        if len(embeddings) < self.min_samples:
            return None

        # Average of all embeddings
        avg_embedding = np.mean(embeddings, axis=0)

        # Normalize (L2 norm for cosine distance)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding

    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine distance (0 = identical, 1 = opposite, 0.4 = similar threshold)
        """
        # Normalize
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)

        # Cosine similarity: dot product of normalized vectors
        similarity = np.dot(emb1_norm, emb2_norm)

        # Convert to distance (1 - similarity)
        distance = 1.0 - similarity

        return float(distance)

    def check_identity_consistency(
        self,
        track_id: int,
        current_embedding: np.ndarray,
        locked_identity: str
    ) -> bool:
        """Check if current embedding matches track's historical embeddings (TIER 2).

        Args:
            track_id: Track identifier
            current_embedding: Current face embedding
            locked_identity: Locked identity name

        Returns:
            True if consistent, False if ID switch detected
        """
        avg_embedding = self.get_average_embedding(track_id)

        if avg_embedding is None:
            # Not enough samples yet
            return True

        distance = self.cosine_distance(current_embedding, avg_embedding)

        if distance > self.embedding_threshold:
            # logger.warning(
            #     f"ID SWITCH SUSPECTED: Track {track_id} (identity: {locked_identity}) "
            #     f"has inconsistent face embedding (distance: {distance:.3f} > {self.embedding_threshold})"
            # )
            return False

        return True

    def find_duplicate_tracks(self) -> List[Tuple[int, int, float]]:
        """Find tracks with same identity but different IDs (TIER 3 - Delayed).

        Returns:
            List of (track_id_1, track_id_2, distance) tuples for duplicates
        """
        duplicates = []
        track_ids = list(self.track_embeddings.keys())

        # Only check tracks with locked identities
        identity_tracks: Dict[str, List[int]] = {}
        for track_id, identity in self.track_identities.items():
            if identity not in identity_tracks:
                identity_tracks[identity] = []
            identity_tracks[identity].append(track_id)

        # Find duplicates within same identity
        for identity, tracks in identity_tracks.items():
            if len(tracks) < 2:
                continue  # No duplicates possible

            # Compare all pairs
            for i, track_id_1 in enumerate(tracks):
                for track_id_2 in tracks[i+1:]:
                    emb1 = self.get_average_embedding(track_id_1)
                    emb2 = self.get_average_embedding(track_id_2)

                    if emb1 is None or emb2 is None:
                        continue

                    distance = self.cosine_distance(emb1, emb2)

                    if distance < self.embedding_threshold:
                        duplicates.append((track_id_1, track_id_2, distance))
                        # logger.warning(
                        #     f"🔍 DUPLICATE TRACKS DETECTED: Track {track_id_1} and Track {track_id_2} "
                        #     f"(identity: {identity}, distance: {distance:.3f})"
                        # )

        return duplicates

    def should_run_correction(self) -> bool:
        """Check if delayed correction should run (TIER 3).

        Returns:
            True if correction interval reached
        """
        self.frame_count += 1
        return self.frame_count % self.correction_interval == 0

    def reset_track(self, track_id: int) -> None:
        """Reset tracking data for a track.

        Args:
            track_id: Track identifier
        """
        if track_id in self.track_embeddings:
            del self.track_embeddings[track_id]

        if track_id in self.track_identities:
            del self.track_identities[track_id]
