"""Face recognition module for comparing face embeddings and identifying people."""

import pickle
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple


class FaceRecognition:
    """Face recognition class for managing face embeddings and performing identity matching.

    This class handles loading and updating face embeddings database, and provides
    methods to recognize faces based on similarity comparison.
    """
    def __init__(self, args) -> None:
        """Initialize the face recognition system.

        Args:
            args: Configuration arguments containing parameters for face recognition
        """
        self.args = args
        self.use_pgvector = getattr(args, 'use_pgvector', False)
        self.pgvector_store = None

        if self.use_pgvector:
            from ..storage.pgvector_store import PgVectorStore
            self.pgvector_store = PgVectorStore(args.client_slug)
            self.db_names, self.db_embs = self.load_embeddings_from_pgvector()
        else:
            self.db_names, self.db_embs = self.load_embeddings()

        self._rebuild_name_index()

    def load_embeddings(self):
        """Load face embeddings from the database file (pickle mode).

        Returns:
            Tuple containing lists of names and their corresponding face embeddings
        """
        with open(self.args.db_path, 'rb') as f:
            data = pickle.load(f)

        db_embs = data['embeddings']
        db_names = data['names']
        db_names = [name.split('_')[0] for name in db_names]

        return db_names, db_embs

    def load_embeddings_from_pgvector(self):
        """Load face embeddings from pgvector database.

        Returns:
            Tuple containing lists of names and their corresponding face embeddings
        """
        names, embeddings = self.pgvector_store.get_all_embeddings()
        # Names already processed by pgvector_store
        self.args.logger.info(f"Loaded {len(names)} embeddings from pgvector")
        return names, embeddings

    def _rebuild_name_index(self) -> None:
        """Rebuild the name-to-index mapping for O(1) lookups.

        This should be called whenever db_names is modified (add/delete operations).
        """
        self.name_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.db_names)
        }

    def reload_embeddings(self):
        """Reload embeddings from storage (pgvector or pickle).

        This method can be called to refresh the in-memory embeddings cache.
        """
        if self.use_pgvector:
            self.db_names, self.db_embs = self.load_embeddings_from_pgvector()
        else:
            self.db_names, self.db_embs = self.load_embeddings()

        self._rebuild_name_index()
        self.args.logger.info(f"Reloaded {len(self.db_embs)} embeddings")

    def compute_similarities(self, face_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between input face embeddings and database embeddings.

        Args:
            face_embs: Face embeddings to compare (shape: [n_faces, embedding_dim])

        Returns:
            Matrix of similarity scores (shape: [n_faces, n_database_faces])
        """
        # Handle empty database - return empty similarity matrix
        if len(self.db_embs) == 0:
            return np.empty((len(face_embs), 0), dtype=np.float32)

        return cosine_similarity(face_embs, self.db_embs)

    def get_best_match(self, similarities: np.ndarray) -> Tuple[int, float]:
        """Find the best matching face embedding from the database.

        Args:
            similarities: Matrix of similarity scores (shape: [n_faces, n_database_faces])

        Returns:
            Tuple of (best_match_index, similarity_score)
            - best_match_index: Index of the best matching person in the database
            - similarity_score: Cosine similarity score (0.0-1.0)
        """
        max_sim_indices = np.argmax(similarities, axis=1)
        max_sim_values = np.max(similarities, axis=1)
        best_idx = np.argmax(max_sim_values)
        best_similarity = max_sim_values[best_idx]
        best_match_db_idx = max_sim_indices[best_idx]

        return best_match_db_idx, best_similarity
