"""Face recognition module for comparing face embeddings and identifying people."""

import numpy as np
from typing import Dict, List, Tuple


class FaceRecognition:
    """Face recognition class for managing face embeddings and performing identity matching."""

    def __init__(self, args) -> None:
        self.args = args

        from infrastructure.storage import PgVectorStore
        self.pgvector_store = PgVectorStore(args.client_slug)
        self.db_names, self.db_embs = self.load_embeddings_from_pgvector()

        self._rebuild_name_index()

    def load_embeddings_from_pgvector(self) -> Tuple[List[str], np.ndarray]:
        """Load face embeddings from pgvector database."""
        names, embeddings = self.pgvector_store.get_all_embeddings()
        self.args.logger.info(f"Loaded {len(names)} embeddings from pgvector")
        return names, embeddings

    def _rebuild_name_index(self) -> None:
        """Rebuild the name-to-index mapping for O(1) lookups."""
        self.name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.db_names)
        }

    def reload_embeddings(self) -> None:
        """Reload embeddings from pgvector database."""
        self.db_names, self.db_embs = self.load_embeddings_from_pgvector()
        self._rebuild_name_index()
        self.args.logger.info(f"Reloaded {len(self.db_embs)} embeddings")

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
