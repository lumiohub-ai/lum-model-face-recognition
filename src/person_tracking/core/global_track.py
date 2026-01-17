"""
Global Track Data Structures for Cross-Camera Tracking.

Phase 1: Body ReID only, conservative matching.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EmbeddingQuality:
    """Embedding with quality score for ranking."""
    embedding: np.ndarray  # 512-d normalized vector
    quality: float  # 0.0-1.0 (detection confidence)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Ensure embedding is normalized."""
        if self.embedding is not None:
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding = self.embedding / norm


@dataclass
class CameraTrackInfo:
    """Track info for one camera."""
    camera_id: int
    local_track_id: int
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    active: bool = True


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata for interval-based extraction."""
    embedding: np.ndarray
    frame_num: int
    quality: float
    timestamp: datetime = field(default_factory=datetime.now)


class GlobalTrack:
    """
    Represents a person across multiple cameras.

    Maintains:
    - Body ReID prototype (running EMA of embeddings)
    - Top-K high-quality embeddings for matching
    - Per-camera presence information
    - Lifecycle timestamps
    """

    def __init__(self, global_id: int):
        self.global_id = global_id

        # Embeddings: top-K + prototype
        self.body_prototype: Optional[np.ndarray] = None
        self.body_top_k: List[EmbeddingQuality] = []  # Max 5

        # Per-camera presence
        self.camera_tracks: Dict[int, CameraTrackInfo] = {}

        # Face identity (primary linking method)
        self.identity: Optional[str] = None  # Locked face recognition identity
        self.identity_locked: bool = False

        # Lifecycle
        self.first_seen: datetime = datetime.now()
        self.last_seen: datetime = datetime.now()
        self.active: bool = True

    def add_camera_track(
        self,
        camera_id: int,
        local_track_id: int
    ) -> None:
        """Add or update camera track info."""
        if camera_id in self.camera_tracks:
            # Update existing
            self.camera_tracks[camera_id].local_track_id = local_track_id
            self.camera_tracks[camera_id].last_seen = datetime.now()
            self.camera_tracks[camera_id].active = True
        else:
            # New camera
            self.camera_tracks[camera_id] = CameraTrackInfo(
                camera_id=camera_id,
                local_track_id=local_track_id
            )
        self.last_seen = datetime.now()

    def add_embedding(
        self,
        embedding: np.ndarray,
        quality: float,
        top_k_size: int = 5,
        prototype_alpha: float = 0.2
    ) -> None:
        """
        Add embedding using top-K + prototype strategy.

        Args:
            embedding: Body ReID embedding (512-d)
            quality: Detection confidence
            top_k_size: Maximum embeddings to keep
            prototype_alpha: EMA smoothing factor
        """
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Add to top-K list
        self.body_top_k.append(EmbeddingQuality(
            embedding=embedding.copy(),
            quality=quality,
            timestamp=datetime.now()
        ))

        # Sort by quality (descending)
        self.body_top_k.sort(key=lambda x: x.quality, reverse=True)

        # Keep only top-K
        if len(self.body_top_k) > top_k_size:
            self.body_top_k = self.body_top_k[:top_k_size]

        # Update prototype (running mean with EMA)
        if self.body_prototype is None:
            self.body_prototype = embedding.copy()
        else:
            # Exponential moving average
            self.body_prototype = (
                (1 - prototype_alpha) * self.body_prototype +
                prototype_alpha * embedding
            )
            # Normalize
            norm = np.linalg.norm(self.body_prototype)
            if norm > 0:
                self.body_prototype = self.body_prototype / norm

    def get_active_cameras(self, recency_threshold_sec: float = 5.0) -> List[int]:
        """Get list of cameras where this track is currently active."""
        active = []
        now = datetime.now()
        for cam_id, cam_track in self.camera_tracks.items():
            if cam_track.active:
                time_since = (now - cam_track.last_seen).total_seconds()
                if time_since < recency_threshold_sec:
                    active.append(cam_id)
        return active

    def mark_camera_inactive(self, camera_id: int) -> None:
        """Mark a camera track as inactive."""
        if camera_id in self.camera_tracks:
            self.camera_tracks[camera_id].active = False
            self.camera_tracks[camera_id].last_seen = datetime.now()

    def set_identity(self, identity: str, locked: bool = True) -> None:
        """Set face recognition identity for this global track."""
        self.identity = identity
        self.identity_locked = locked

    def compute_averaged_embedding(self) -> Optional[np.ndarray]:
        """Compute averaged embedding from top-K embeddings."""
        if len(self.body_top_k) == 0:
            return None

        # Average top-K embeddings
        embeddings = [e.embedding for e in self.body_top_k]
        avg = np.mean(embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        return avg

    def get_duration_seconds(self) -> float:
        """Get total duration this track has been active."""
        return (self.last_seen - self.first_seen).total_seconds()

    def __repr__(self) -> str:
        cameras = list(self.camera_tracks.keys())
        identity_str = f", identity='{self.identity}'" if self.identity else ""
        return (
            f"GlobalTrack(id={self.global_id}, cameras={cameras}, "
            f"active={self.active}, duration={self.get_duration_seconds():.1f}s{identity_str})"
        )
