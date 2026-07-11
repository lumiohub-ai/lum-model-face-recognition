"""Face recognition module for comparing face embeddings and identifying people."""

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np


class FaceRecognition:
    """Face recognition class for managing face embeddings and performing identity matching."""

    def __init__(self, args) -> None:
        self.args = args

        from infrastructure.storage import PgVectorStore
        self.pgvector_store = PgVectorStore(args.client_slug)

        # Guards db_names/db_embs/name_to_index: reload_embeddings() can run
        # (via a Redis reload notification) on a different thread than the
        # camera-worker threads doing lookups, and a match spans multiple
        # attribute reads (compute_similarities -> get_best_match -> name
        # lookup by index) that must all see the same snapshot, or a reload
        # landing mid-lookup can resolve an index against the wrong list.
        self._lock = threading.Lock()

        self.db_names, self.db_embs = self.load_embeddings_from_pgvector()
        self._rebuild_name_index()

    def load_embeddings_from_pgvector(self) -> Tuple[List[str], np.ndarray]:
        """Load face embeddings from pgvector database."""
        names, embeddings = self.pgvector_store.get_all_embeddings()
        self.args.logger.debug(f"Loaded {len(names)} embeddings from pgvector")
        return names, embeddings

    def _rebuild_name_index(self) -> None:
        """Rebuild the name-to-index mapping for O(1) lookups."""
        self.name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.db_names)
        }

    def reload_embeddings(self) -> None:
        """Reload embeddings from pgvector database."""
        # Build the new state off to the side first, then swap all three
        # references together under the lock so concurrent readers never see
        # a torn mix of old/new names+embeddings+index.
        new_names, new_embs = self.load_embeddings_from_pgvector()
        new_index = {name: idx for idx, name in enumerate(new_names)}
        with self._lock:
            self.db_names, self.db_embs = new_names, new_embs
            self.name_to_index = new_index
        self.args.logger.info(f"Reloaded {len(new_embs)} embeddings")

    def compute_similarities(self, face_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between input face embeddings and database embeddings.

        Args:
            face_embs: Face embeddings to compare (shape: [n_faces, embedding_dim])

        Returns:
            Matrix of similarity scores (shape: [n_faces, n_database_faces])
        """
        with self._lock:
            db_embs = self.db_embs
        if len(db_embs) == 0:
            return np.empty((len(face_embs), 0), dtype=np.float32)

        # DB embeddings are pre-normalized at load time; normalize query here
        norms = np.linalg.norm(face_embs, axis=1, keepdims=True)
        q = face_embs / np.where(norms == 0, 1, norms)
        return np.dot(q, db_embs.T)

    def get_best_match(self, similarities: np.ndarray) -> Tuple[int, float]:
        """Find the best matching face embedding from the database.

        Args:
            similarities: Matrix of similarity scores (shape: [n_faces, n_database_faces])

        Returns:
            Tuple of (best_match_index, similarity_score)

        Raises:
            ValueError: if similarities has zero columns (empty database).
        """
        if similarities.shape[1] == 0:
            raise ValueError("get_best_match called with an empty database (0 columns)")

        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        best_idx = np.argmax(max_sim_values)
        best_similarity = max_sim_values[best_idx]
        best_match_db_idx = max_sim_indices[best_idx]

        return best_match_db_idx, best_similarity

    def identify_best_match(self, face_embs: np.ndarray) -> Tuple[Optional[int], Optional[str], float]:
        """Compute similarities and resolve the best match's name atomically.

        Doing this as a single locked operation (rather than callers chaining
        compute_similarities() -> get_best_match() -> db_names[idx] against
        the live attributes) prevents a concurrent reload_embeddings() from
        swapping db_names out from under an index computed against the old
        db_embs.

        Returns:
            (best_index, best_name, best_similarity), or (None, None, 0.0)
            if the database is currently empty.
        """
        with self._lock:
            db_names, db_embs = self.db_names, self.db_embs
            if len(db_embs) == 0:
                return None, None, 0.0

            norms = np.linalg.norm(face_embs, axis=1, keepdims=True)
            q = face_embs / np.where(norms == 0, 1, norms)
            similarities = np.dot(q, db_embs.T)

            best_idx, best_similarity = self.get_best_match(similarities)
            return int(best_idx), db_names[best_idx], float(best_similarity)
