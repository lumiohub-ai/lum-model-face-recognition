"""pgvector storage operations for face embeddings."""

from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from sqlalchemy import text
from loguru import logger
from .db_config import DatabaseConfig
from .validators import validate_client_slug, validate_schema_name
from .url_utils import normalize_image_url


class PgVectorStore:
    """Storage class for face embeddings using pgvector.

    This class provides CRUD operations for face embeddings in a PostgreSQL
    database with pgvector extension. Each organization has its own schema.
    """

    def __init__(self, client_slug: str):
        """Initialize pgvector store for a specific client.

        Args:
            client_slug: Organization slug (e.g., 'humblebee', 'dev')

        Raises:
            ValueError: If client_slug contains invalid characters
        """
        # SECURITY: Validate client_slug to prevent SQL injection
        validated_slug = validate_client_slug(client_slug)
        self.client_slug = validated_slug
        self.schema_name = f"org_{validated_slug}"

        # Double-check the schema name itself
        validate_schema_name(self.schema_name)

        self.db_config = DatabaseConfig()

        # Ensure schema exists (init_schema also validates)
        self.db_config.init_schema(self.client_slug)
        logger.info(f"PgVectorStore initialized for: {self.schema_name}")

    def add_embedding(
        self,
        user_id: str,
        user_name: str,
        image_url: str,
        embedding: np.ndarray,
        external_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[int]:
        """Add a single face embedding to database (idempotent).

        Uses INSERT ... ON CONFLICT DO NOTHING to prevent duplicates
        based on (user_id, normalized_image_url).

        Args:
            user_id: Backend user ID
            user_name: User's full name
            image_url: Source image URL from GCS
            embedding: Face embedding vector (512-dim numpy array)
            external_id: External employee ID (optional)
            metadata: Additional metadata (landmarks, bbox, etc.)

        Returns:
            int: ID of inserted embedding, or None if already exists

        Raises:
            Exception: If database operation fails
        """
        try:
            # Normalize URL for stable identity
            image_url_norm = normalize_image_url(image_url)

            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()

            # Convert metadata to JSON string if provided, otherwise use empty JSON object
            metadata_json = json.dumps(metadata) if metadata else '{}'

            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    INSERT INTO {self.schema_name}.face_embeddings
                    (user_id, user_name, external_id, image_url, image_url_norm, embedding, embedding_metadata)
                    VALUES (:user_id, :user_name, :external_id, :image_url, :image_url_norm, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                    ON CONFLICT (user_id, image_url_norm) DO NOTHING
                    RETURNING id
                """), {
                    'user_id': user_id,
                    'user_name': user_name,
                    'external_id': external_id,
                    'image_url': image_url,
                    'image_url_norm': image_url_norm,
                    'embedding': str(embedding_list),
                    'metadata': metadata_json
                })
                conn.commit()

                row = result.fetchone()
                if row:
                    embedding_id = row[0]
                    logger.info(f"Added embedding {embedding_id} for user {user_id} ({user_name}) - vector dim: {len(embedding_list)}")
                    return embedding_id
                else:
                    logger.debug(f"Embedding already exists for user {user_id}, image: {image_url_norm}")
                    return None

        except Exception as e:
            logger.error(f"Failed to add embedding for user {user_id}: {e}")
            raise

    def add_embeddings_batch(
        self,
        embeddings_data: List[Dict]
    ) -> List[int]:
        """Add multiple embeddings in a batch.

        Args:
            embeddings_data: List of dicts with keys:
                - user_id, user_name, image_url, embedding, external_id, metadata

        Returns:
            List[int]: IDs of inserted embeddings
        """
        inserted_ids = []

        try:
            with self.db_config.get_connection() as conn:
                for data in embeddings_data:
                    embedding_list = data['embedding'].tolist()
                    metadata_json = json.dumps(data.get('metadata')) if data.get('metadata') else '{}'

                    result = conn.execute(text(f"""
                        INSERT INTO {self.schema_name}.face_embeddings
                        (user_id, user_name, external_id, image_url, embedding, embedding_metadata)
                        VALUES (:user_id, :user_name, :external_id, :image_url, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                        RETURNING id
                    """), {
                        'user_id': data['user_id'],
                        'user_name': data['user_name'],
                        'external_id': data.get('external_id'),
                        'image_url': data['image_url'],
                        'embedding': str(embedding_list),
                        'metadata': metadata_json
                    })

                    inserted_ids.append(result.fetchone()[0])

                conn.commit()
                logger.info(f"Batch inserted {len(inserted_ids)} embeddings")

        except Exception as e:
            logger.error(f"Failed to batch insert embeddings: {e}")
            raise

        return inserted_ids

    def delete_all_for_user(self, user_id: str) -> int:
        """Delete all embeddings for a user.

        Args:
            user_id: Backend user ID

        Returns:
            int: Number of embeddings deleted
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.schema_name}.face_embeddings
                    WHERE user_id = :user_id
                """), {'user_id': user_id})
                conn.commit()

                count = result.rowcount
                logger.info(f"Deleted {count} embeddings for user {user_id}")
                return count

        except Exception as e:
            logger.error(f"Failed to delete embeddings for user {user_id}: {e}")
            raise

    def delete_by_image_url(self, user_id: str, image_url: str) -> int:
        """Delete embedding for a specific image.

        Args:
            user_id: Backend user ID
            image_url: Image URL to delete

        Returns:
            int: Number of embeddings deleted (should be 0 or 1)
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.schema_name}.face_embeddings
                    WHERE user_id = :user_id AND image_url = :image_url
                """), {'user_id': user_id, 'image_url': image_url})
                conn.commit()

                count = result.rowcount
                logger.info(f"Deleted {count} embedding for {image_url}")
                return count

        except Exception as e:
            logger.error(f"Failed to delete embedding for {image_url}: {e}")
            raise

    def delete_by_image_url_norm(self, user_id: str, image_url_norm: str) -> int:
        """Delete embedding by user_id and normalized image URL.

        Args:
            user_id: Backend user ID
            image_url_norm: Normalized image URL to delete

        Returns:
            int: Number of embeddings deleted (should be 0 or 1)
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.schema_name}.face_embeddings
                    WHERE user_id = :user_id AND image_url_norm = :image_url_norm
                """), {'user_id': user_id, 'image_url_norm': image_url_norm})
                conn.commit()

                count = result.rowcount
                if count > 0:
                    logger.debug(f"Deleted {count} embedding for normalized URL: {image_url_norm}")
                return count

        except Exception as e:
            logger.error(f"Failed to delete embedding for normalized URL {image_url_norm}: {e}")
            raise

    def delete_user_embeddings(self, user_name: str) -> int:
        """Delete all embeddings for a user by user_name.

        Args:
            user_name: User name to delete embeddings for

        Returns:
            int: Number of embeddings deleted
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.schema_name}.face_embeddings
                    WHERE user_name = :user_name
                """), {'user_name': user_name})
                conn.commit()

                count = result.rowcount
                logger.info(f"Deleted {count} embeddings for user_name: {user_name}")
                return count

        except Exception as e:
            logger.error(f"Failed to delete embeddings for user_name {user_name}: {e}")
            raise

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Get all embeddings for recognition.

        This method is compatible with the current FaceRecognition interface
        that expects (names, embeddings) tuple.

        Returns:
            Tuple of (names, embeddings):
                - names: List of user names
                - embeddings: numpy array of shape (N, 512)
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT user_name, embedding
                    FROM {self.schema_name}.face_embeddings
                    ORDER BY id
                """))

                rows = result.fetchall()

                if not rows:
                    logger.warning(f"No embeddings found in {self.schema_name}")
                    return [], np.empty((0, 512), dtype=np.float32)  # Return empty 2D array with correct shape

                names = [row[0] for row in rows]
                # pgvector returns vectors as strings like '[1,2,3,...]', convert to numpy arrays
                embeddings = []
                for row in rows:
                    # Parse the vector string and convert to numpy array
                    vec_str = row[1]
                    if isinstance(vec_str, str):
                        # Remove brackets and parse
                        vec_str = vec_str.strip('[]')
                        vec = np.fromstring(vec_str, dtype=np.float32, sep=',')
                    else:
                        # Already a list/array
                        vec = np.array(row[1], dtype=np.float32)
                    embeddings.append(vec)

                embeddings = np.array(embeddings, dtype=np.float32)

                logger.info(f"Loaded {len(names)} embeddings from {self.schema_name}")
                return names, embeddings

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: float = 0.3
    ) -> List[Dict]:
        """Search for similar faces using cosine similarity.

        Args:
            query_embedding: Face embedding to search for (512-dim)
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of dicts with keys: user_id, user_name, image_url, similarity
        """
        try:
            embedding_list = query_embedding.tolist()

            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT
                        user_id,
                        user_name,
                        image_url,
                        1 - (embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM {self.schema_name}.face_embeddings
                    WHERE 1 - (embedding <=> CAST(:query_embedding AS vector)) >= :threshold
                    ORDER BY embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :limit
                """), {
                    'query_embedding': str(embedding_list),
                    'threshold': threshold,
                    'limit': limit
                })

                matches = []
                for row in result:
                    matches.append({
                        'user_id': row[0],
                        'user_name': row[1],
                        'image_url': row[2],
                        'similarity': float(row[3])
                    })

                # logger.info(f"Found {len(matches)} similar faces above threshold {threshold}")
                return matches

        except Exception as e:
            logger.error(f"Failed to search similar faces: {e}")
            raise

    def update_user_name(self, user_id: str, new_name: str) -> int:
        """Update user name for all their embeddings.

        Args:
            user_id: Backend user ID
            new_name: New user name

        Returns:
            int: Number of embeddings updated
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    UPDATE {self.schema_name}.face_embeddings
                    SET user_name = :new_name, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                """), {'user_id': user_id, 'new_name': new_name})
                conn.commit()

                count = result.rowcount
                logger.info(f"Updated {count} embeddings with new name: {new_name}")
                return count

        except Exception as e:
            logger.error(f"Failed to update user name for {user_id}: {e}")
            raise

    def get_embedding_count(self) -> int:
        """Get total number of embeddings in this organization.

        Returns:
            int: Total embedding count
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM {self.schema_name}.face_embeddings
                """))
                count = result.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Failed to get embedding count: {e}")
            raise

    def get_user_embedding_count(self, user_id: str) -> int:
        """Get number of embeddings for a specific user.

        Args:
            user_id: Backend user ID

        Returns:
            int: Number of embeddings for this user
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM {self.schema_name}.face_embeddings
                    WHERE user_id = :user_id
                """), {'user_id': user_id})
                count = result.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Failed to get embedding count for user {user_id}: {e}")
            raise

    def clear_all_embeddings(self) -> int:
        """Clear all embeddings for this organization.

        Warning:
            This will delete all face embeddings for the organization!

        Returns:
            int: Number of embeddings deleted
        """
        try:
            with self.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.schema_name}.face_embeddings
                """))
                conn.commit()

                count = result.rowcount
                logger.warning(f"⚠️ Cleared {count} embeddings from {self.schema_name}")
                return count

        except Exception as e:
            logger.error(f"Failed to clear embeddings: {e}")
            raise
