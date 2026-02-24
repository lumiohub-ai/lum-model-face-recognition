"""Face recognition module for comparing face embeddings and identifying people."""

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
from loguru import logger


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
            from infrastructure.storage import PgVectorStore
            self.pgvector_store = PgVectorStore(args.client_slug)
            self.db_names, self.db_embs = self.load_embeddings_from_pgvector()
        else:
            self.db_names, self.db_embs = self.load_embeddings()

        self._rebuild_name_index()

    def load_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load face embeddings from the database file (JSON format - secure).

        SECURITY: Uses JSON instead of pickle to prevent arbitrary code execution.
        Pickle deserialization can execute malicious code embedded in the file.

        Returns:
            Tuple containing lists of names and their corresponding face embeddings

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If file format is invalid or corrupted
        """
        db_path = self.args.db_path

        # Check file extension to determine format
        if db_path.endswith('.json'):
            return self._load_from_json(db_path)
        elif db_path.endswith('.npz'):
            return self._load_from_npz(db_path)
        elif db_path.endswith('.pkl') or db_path.endswith('.pickle'):
            # SECURITY WARNING: Pickle is deprecated due to RCE vulnerability
            # Migrate to JSON or NPZ format
            logger.warning(
                f"SECURITY WARNING: Loading from pickle file '{db_path}' is deprecated. "
                "Pickle deserialization can execute arbitrary code. "
                "Please migrate to JSON (.json) or NumPy (.npz) format."
            )
            return self._load_from_pickle_legacy(db_path)
        else:
            # Default to JSON for new files
            return self._load_from_json(db_path)

    def _load_from_json(self, path: str) -> Tuple[List[str], np.ndarray]:
        """Load embeddings from secure JSON format.

        Args:
            path: Path to JSON file

        Returns:
            Tuple of (names, embeddings)
        """
        if not os.path.exists(path):
            logger.warning(f"Embeddings file not found: {path}")
            return [], np.array([])

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate structure
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON structure in {path}: expected dict")

        db_names = data.get('names', [])
        embeddings_list = data.get('embeddings', [])

        if not isinstance(db_names, list) or not isinstance(embeddings_list, list):
            raise ValueError(f"Invalid data types in {path}")

        # Convert embeddings to numpy array
        db_embs = np.array(embeddings_list, dtype=np.float32) if embeddings_list else np.array([])

        # Process names (remove suffixes)
        db_names = [name.split('_')[0] for name in db_names]

        logger.info(f"Loaded {len(db_names)} embeddings from JSON: {path}")
        return db_names, db_embs

    def _load_from_npz(self, path: str) -> Tuple[List[str], np.ndarray]:
        """Load embeddings from NumPy NPZ format (secure binary).

        Args:
            path: Path to NPZ file

        Returns:
            Tuple of (names, embeddings)
        """
        if not os.path.exists(path):
            logger.warning(f"Embeddings file not found: {path}")
            return [], np.array([])

        # NPZ is safe - it only loads numpy arrays, no code execution
        data = np.load(path, allow_pickle=False)  # SECURITY: Disable pickle in npz

        db_names = data['names'].tolist() if 'names' in data else []
        db_embs = data['embeddings'] if 'embeddings' in data else np.array([])

        # Process names (remove suffixes)
        db_names = [name.split('_')[0] for name in db_names]

        logger.info(f"Loaded {len(db_names)} embeddings from NPZ: {path}")
        return db_names, db_embs

    def _load_from_pickle_legacy(self, path: str) -> Tuple[List[str], np.ndarray]:
        """Load embeddings from legacy pickle format (DEPRECATED - security risk).

        SECURITY WARNING: This method exists only for backward compatibility.
        Pickle can execute arbitrary code during deserialization.
        Migrate to JSON or NPZ format immediately.

        Args:
            path: Path to pickle file

        Returns:
            Tuple of (names, embeddings)
        """
        import pickle
        import hashlib

        if not os.path.exists(path):
            logger.warning(f"Embeddings file not found: {path}")
            return [], np.array([])

        # SECURITY: Log file hash for audit trail
        with open(path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        logger.warning(f"Loading pickle file with SHA256: {file_hash}")

        # Load with restricted unpickler would be ideal, but for compatibility we log and proceed
        with open(path, 'rb') as f:
            data = pickle.load(f)

        db_embs = data['embeddings']
        db_names = data['names']
        db_names = [name.split('_')[0] for name in db_names]

        # Encourage migration by logging
        logger.warning(
            f"MIGRATION RECOMMENDED: Convert pickle to JSON using: "
            f"FaceRecognition.convert_pickle_to_json('{path}')"
        )

        return db_names, db_embs

    @staticmethod
    def convert_pickle_to_json(pickle_path: str, json_path: str = None) -> str:
        """Convert legacy pickle file to secure JSON format.

        Args:
            pickle_path: Path to existing pickle file
            json_path: Output JSON path (default: same name with .json extension)

        Returns:
            Path to created JSON file
        """
        import pickle

        if json_path is None:
            json_path = pickle_path.rsplit('.', 1)[0] + '.json'

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'names': data['names'] if isinstance(data['names'], list) else data['names'].tolist(),
            'embeddings': data['embeddings'].tolist() if hasattr(data['embeddings'], 'tolist') else data['embeddings']
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)

        logger.info(f"Converted pickle to JSON: {pickle_path} -> {json_path}")
        return json_path

    def load_embeddings_from_pgvector(self) -> Tuple[List[str], np.ndarray]:
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
