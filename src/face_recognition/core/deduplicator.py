"""Temporal deduplication for unknown faces using embedding cache."""

from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UnknownDeduplicator:
    """Temporal deduplication for unknown faces using embedding cache.

    Prevents same person from generating multiple dashboard alerts by maintaining
    a cache of recent unknown face embeddings and comparing new unknowns against it.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 300,
        similarity_threshold: float = 0.85,
        cross_camera_window_seconds: int = 60
    ):
        """Initialize the deduplicator.

        Args:
            cache_ttl_seconds: Time-to-live for cache entries in seconds (default: 5 minutes)
            similarity_threshold: Cosine similarity threshold for considering duplicates
            cross_camera_window_seconds: Time window for cross-camera deduplication
        """
        self.cache_ttl = cache_ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.cross_camera_window = cross_camera_window_seconds

        # Cache structure: {camera_name: deque of (timestamp, embedding, quality, track_id)}
        self.camera_caches: Dict[str, deque] = {}

        # Global cache for cross-camera dedup
        self.global_cache = deque(maxlen=100)

    def should_send(
        self,
        embedding: np.ndarray,
        camera_name: str,
        quality_score: float,
        track_id: int
    ) -> Tuple[bool, str]:
        """Check if this unknown face is a duplicate and should be sent.

        Args:
            embedding: Face embedding vector
            camera_name: Name of the camera
            quality_score: Quality score of the face
            track_id: Track ID

        Returns:
            Tuple of (should_send, reason)
        """
        now = datetime.now()

        # 1. Check same-camera cache
        if camera_name in self.camera_caches:
            self._cleanup_expired(camera_name, now)

            for cached_time, cached_emb, cached_quality, cached_track_id in self.camera_caches[camera_name]:
                sim = cosine_similarity(
                    embedding.reshape(1, -1),
                    cached_emb.reshape(1, -1)
                )[0][0]

                if sim >= self.similarity_threshold:
                    # Duplicate detected
                    time_diff = (now - cached_time).total_seconds()

                    # If current quality is significantly better, replace in cache
                    if quality_score > cached_quality + 0.1:
                        self._replace_in_cache(
                            camera_name, cached_track_id,
                            now, embedding, quality_score, track_id
                        )
                        return True, f"duplicate_upgraded_{sim:.3f}"

                    return False, f"duplicate_same_camera_{time_diff:.0f}s_sim_{sim:.3f}"

        # 2. Check global cache (cross-camera, shorter window)
        recent_cutoff = now - timedelta(seconds=self.cross_camera_window)
        for cached_time, cached_emb, _, _ in self.global_cache:
            if cached_time < recent_cutoff:
                continue

            sim = cosine_similarity(
                embedding.reshape(1, -1),
                cached_emb.reshape(1, -1)
            )[0][0]

            # Stricter threshold for cross-camera
            if sim >= self.similarity_threshold + 0.05:
                time_diff = (now - cached_time).total_seconds()
                return False, f"duplicate_cross_camera_{time_diff:.0f}s_sim_{sim:.3f}"

        # 3. Not a duplicate - add to caches
        self._add_to_cache(camera_name, now, embedding, quality_score, track_id)
        return True, "unique"

    def _cleanup_expired(self, camera_name: str, now: datetime) -> None:
        """Remove expired entries from camera cache.

        Args:
            camera_name: Name of the camera
            now: Current timestamp
        """
        cutoff_time = now - timedelta(seconds=self.cache_ttl)

        if camera_name in self.camera_caches:
            self.camera_caches[camera_name] = deque(
                [entry for entry in self.camera_caches[camera_name]
                 if entry[0] > cutoff_time],
                maxlen=50
            )

    def _add_to_cache(
        self,
        camera_name: str,
        timestamp: datetime,
        embedding: np.ndarray,
        quality: float,
        track_id: int
    ) -> None:
        """Add new entry to caches.

        Args:
            camera_name: Name of the camera
            timestamp: Timestamp of the detection
            embedding: Face embedding
            quality: Quality score
            track_id: Track ID
        """
        if camera_name not in self.camera_caches:
            self.camera_caches[camera_name] = deque(maxlen=50)

        self.camera_caches[camera_name].append(
            (timestamp, embedding, quality, track_id)
        )
        self.global_cache.append((timestamp, embedding, quality, track_id))

    def _replace_in_cache(
        self,
        camera_name: str,
        old_track_id: int,
        new_time: datetime,
        new_emb: np.ndarray,
        new_quality: float,
        new_track_id: int
    ) -> None:
        """Replace lower-quality cache entry with better one.

        Args:
            camera_name: Name of the camera
            old_track_id: Old track ID to replace
            new_time: New timestamp
            new_emb: New embedding
            new_quality: New quality score
            new_track_id: New track ID
        """
        if camera_name not in self.camera_caches:
            return

        cache = self.camera_caches[camera_name]
        for i, (t, e, q, tid) in enumerate(cache):
            if tid == old_track_id:
                cache[i] = (new_time, new_emb, new_quality, new_track_id)
                break

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = sum(len(cache) for cache in self.camera_caches.values())

        return {
            'total_entries': total_entries,
            'num_cameras': len(self.camera_caches),
            'global_cache_size': len(self.global_cache),
        }
