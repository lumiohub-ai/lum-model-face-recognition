"""Face matching: compares face embeddings against a known-face database."""

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from ..ports import EmbeddingProvider


class FaceMatcher:
    """Matches face embeddings against a set of known identities.

    The embeddings are supplied by an :class:`~lum_vision.ports.EmbeddingProvider`
    rather than fetched directly, so this class needs no database of its own.
    """

    def __init__(self, provider: EmbeddingProvider, match_threshold: float = 0.3) -> None:
        """Initialize the matcher.

        Args:
            provider: Source of the known-face embeddings
            match_threshold: Minimum cosine similarity to consider a match

        Raises:
            TypeError: If provider does not implement ``get_all_embeddings``
        """
        if not isinstance(provider, EmbeddingProvider):
            raise TypeError(
                f"provider must implement get_all_embeddings(), got {type(provider).__name__}"
            )

        self.provider = provider
        self.match_threshold = match_threshold

        self.db_names, self.db_embs = self._load_embeddings()
        self._rebuild_name_index()

    def _load_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Fetch embeddings from the provider."""
        names, embeddings = self.provider.get_all_embeddings()
        logger.debug(f"Loaded {len(names)} embeddings from provider")
        return names, embeddings

    def _rebuild_name_index(self) -> None:
        """Rebuild the name-to-index mapping for O(1) lookups."""
        self.name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.db_names)
        }

    def reload_embeddings(self) -> None:
        """Re-fetch embeddings from the provider."""
        self.db_names, self.db_embs = self._load_embeddings()
        self._rebuild_name_index()
        logger.info(f"Reloaded {len(self.db_embs)} embeddings")

    def compute_similarities(self, face_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between input face embeddings and database embeddings.

        Args:
            face_embs: Face embeddings to compare (shape: [n_faces, embedding_dim])

        Returns:
            Matrix of similarity scores (shape: [n_faces, n_database_faces])
        """
        if len(self.db_embs) == 0:
            return np.empty((len(face_embs), 0), dtype=np.float32)

        # DB embeddings are pre-normalized at load time; normalize query here
        norms = np.linalg.norm(face_embs, axis=1, keepdims=True)
        q = face_embs / np.where(norms == 0, 1, norms)
        return np.dot(q, self.db_embs.T)

    def get_best_match(self, similarities: np.ndarray) -> Tuple[int, float]:
        """Find the best matching face embedding from the database.

        Args:
            similarities: Matrix of similarity scores (shape: [n_faces, n_database_faces])

        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        best_idx = np.argmax(max_sim_values)
        best_similarity = max_sim_values[best_idx]
        best_match_db_idx = max_sim_indices[best_idx]

        return best_match_db_idx, best_similarity
