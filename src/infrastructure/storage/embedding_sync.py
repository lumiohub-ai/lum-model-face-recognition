"""Service to sync face embeddings from backend user events."""

import os
import requests
from typing import Dict, List, Optional
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from domain.face_detection import FaceDetector
from .pgvector import PgVectorStore
from .url_utils import normalize_image_url
from .gcs import GCSClient


class EmbeddingSyncService:
    """Service to synchronize face embeddings with backend user data.

    This service handles:
    - User created: Download images, calculate embeddings, store in pgvector
    - User updated: Update embeddings for changed data
    - User deleted: Remove all embeddings from pgvector
    - Image added/deleted: Update specific embeddings
    """

    def __init__(self, client_slug: str, gpu_id: int = 0, config: Optional[Dict] = None):
        """Initialize embedding sync service.

        Args:
            client_slug: Organization slug (e.g., 'humblebee', 'dev')
            gpu_id: GPU device ID for face detection
            config: Optional config dict (from config.yaml)
        """
        self.client_slug = client_slug

        # Get face detection padding from config or environment (default: 20%)
        if config and 'face_detection_padding' in config:
            padding_percent = float(config.get('face_detection_padding', 20.0))
        else:
            padding_percent = float(os.getenv('FACE_DETECTION_PADDING', '20.0'))
        self.detector = FaceDetector(gpu_id=gpu_id, padding_percent=padding_percent)

        self.store = PgVectorStore(client_slug)
        self.image_fetcher = GCSClient()

        logger.info(f"EmbeddingSyncService initialized for: {client_slug}")

    def _process_single_image(self, img_data: Dict, user_id: str, user_name: str, external_id: Optional[str]) -> Optional[Dict]:
        """Process a single image: fetch, detect face, calculate embedding.

        Args:
            img_data: Image data dict with 'original' URL
            user_id: User ID
            user_name: User name
            external_id: External employee ID

        Returns:
            Dict with embedding data if successful, None if failed
        """
        original_url = img_data.get('original')

        if not original_url:
            logger.warning("Image data missing 'original' URL, skipping")
            return None

        try:
            # Fetch image
            image = self.image_fetcher.fetch_image(original_url)

            if image is None:
                logger.error(f"❌ Failed to fetch image: {original_url}")
                return None

            # Extract face features using detector
            features = self.detector.extract_face_features(image)

            if not features:
                logger.warning(f"⚠️  No face detected in: {original_url}")
                return None

            # Use first detected face (best quality)
            face = features[0]
            embedding = face['embedding']
            metadata = {
                'landmarks': face['landmarks'].tolist(),
                'bbox': [float(x) for x in face['bbox']]
            }

            return {
                'user_id': user_id,
                'user_name': user_name,
                'image_url': original_url,
                'embedding': embedding,
                'external_id': external_id,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"❌ Error processing image {original_url}: {e}")
            return None

    def handle_user_created(self, user_data: Dict) -> Dict:
        """Handle user creation event from backend (parallelized).

        Args:
            user_data: User data from backend containing:
                - id: User ID
                - full_name: User's full name
                - external_id: External employee ID (optional)
                - image_urls: List of dicts with 'original', 'thumb' URLs

        Returns:
            Dict with sync results:
                - user_id: User ID
                - user_name: User name
                - embeddings_added: Number of embeddings successfully added
                - failed_images: List of image URLs that failed
        """
        user_id = str(user_data.get('id'))
        user_name = user_data.get('full_name')
        external_id = user_data.get('external_id')
        image_urls = user_data.get('image_urls', [])

        logger.info(f"Syncing user created: {user_name} ({user_id}) with {len(image_urls)} images")

        results = {
            'user_id': user_id,
            'user_name': user_name,
            'embeddings_added': 0,
            'failed_images': []
        }

        # Process all images in parallel (up to 10 concurrent downloads)
        max_workers = min(10, len(image_urls)) if len(image_urls) > 0 else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all image processing tasks
            future_to_url = {
                executor.submit(self._process_single_image, img_data, user_id, user_name, external_id): img_data.get('original')
                for img_data in image_urls
                if img_data.get('original')
            }

            # Collect results as they complete
            for future in as_completed(future_to_url):
                original_url = future_to_url[future]
                try:
                    result = future.result()

                    if result is None:
                        results['failed_images'].append(original_url)
                        continue

                    # Save to pgvector database
                    self.store.add_embedding(
                        user_id=result['user_id'],
                        user_name=result['user_name'],
                        image_url=result['image_url'],
                        embedding=result['embedding'],
                        external_id=result['external_id'],
                        metadata=result['metadata']
                    )

                    results['embeddings_added'] += 1

                except Exception as e:
                    logger.error(f"❌ Error saving embedding for {original_url}: {e}")
                    results['failed_images'].append(original_url)

        logger.info(
            f"User sync complete: {results['embeddings_added']} embeddings added, "
            f"{len(results['failed_images'])} failed"
        )

        return results

    def sync_missing_embeddings(self) -> Dict:
        """Sync embeddings for users/images that exist in database but not in pgvector.

        This method:
        1. Fetches all users from database (direct query)
        2. Compares with existing embeddings in pgvector
        3. Calculates embeddings only for missing users/images

        Returns:
            Dict with sync results including users processed and embeddings added
        """
        logger.info(f"Syncing missing embeddings for {self.client_slug}")

        try:
            # Fetch all users from database directly
            from .repository import Repository
            repository = Repository(self.client_slug)
            all_users = repository.get_all_users()

            logger.info(f"Total users from database: {len(all_users)}")

            # Get existing embeddings from pgvector
            existing_names, existing_embs = self.store.get_all_embeddings()
            logger.info(f"Existing embeddings in pgvector: {len(existing_embs)}")

            # Build index of existing user_id -> normalized image_urls
            from sqlalchemy import text
            existing_images = {}
            with self.store.db_config.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT user_id, ARRAY_AGG(image_url_norm) as image_urls_norm
                    FROM {self.store.schema_name}.face_embeddings
                    WHERE image_url_norm IS NOT NULL
                    GROUP BY user_id
                """))

                for row in result:
                    user_id = row[0]
                    image_urls_norm = row[1] if row[1] else []
                    existing_images[user_id] = set(image_urls_norm)

            logger.info(f"Existing users in pgvector: {len(existing_images)}")

            # === TWO-WAY SYNC: Detect and remove stale embeddings ===

            # Build set of backend user IDs
            backend_user_ids = {str(user.get('id')) for user in all_users}
            pgvector_user_ids = set(existing_images.keys())

            # Find stale users (in pgvector but not in backend)
            stale_user_ids = pgvector_user_ids - backend_user_ids
            users_deleted = 0
            images_deleted = 0

            # Delete embeddings for stale users
            for user_id in stale_user_ids:
                logger.info(f"Stale user detected: {user_id} - removing all embeddings")
                count = self.store.delete_all_for_user(user_id)
                users_deleted += 1
                images_deleted += count

            # Build a mapping of user_id -> user data for backend users
            backend_users_by_id = {str(user.get('id')): user for user in all_users}

            # Find stale images for users that exist in both
            stale_images_by_user = {}
            for user_id in (pgvector_user_ids & backend_user_ids):
                user = backend_users_by_id.get(user_id)
                if not user:
                    continue

                raw_image_urls = user.get('image_urls', [])
                if not raw_image_urls:
                    # User has no images in backend but has embeddings in pgvector
                    # All their embeddings are stale
                    stale_images_by_user[user_id] = existing_images[user_id]
                    continue

                # Handle case where image_urls is a single string instead of array
                if isinstance(raw_image_urls, str):
                    raw_image_urls = [raw_image_urls]

                # Normalize backend image URLs
                backend_images_norm = set()
                for img in raw_image_urls:
                    if isinstance(img, dict):
                        original_url = img.get('original')
                    elif isinstance(img, str):
                        original_url = img
                    else:
                        continue
                    if original_url:
                        backend_images_norm.add(normalize_image_url(original_url))

                # Find images in pgvector but not in backend
                pgvector_images = existing_images[user_id]
                stale_images = pgvector_images - backend_images_norm

                if stale_images:
                    stale_images_by_user[user_id] = stale_images

            # Delete stale images
            for user_id, stale_images in stale_images_by_user.items():
                user = backend_users_by_id.get(user_id, {})
                user_name = user.get('full_name', 'Unknown')
                logger.info(f"Stale images detected for {user_name} ({user_id}): {len(stale_images)} images - removing embeddings")
                for image_url_norm in stale_images:
                    count = self.store.delete_by_image_url_norm(user_id, image_url_norm)
                    images_deleted += count

            if users_deleted > 0 or images_deleted > 0:
                logger.info(f"Stale embeddings removed: {users_deleted} users, {images_deleted} images")

            # === END TWO-WAY SYNC ===

            # Find missing users and images
            users_to_process = []

            for user in all_users:
                user_id = str(user.get('id'))
                user_name = user.get('full_name')
                external_id = user.get('external_id')
                raw_image_urls = user.get('image_urls', [])

                if not raw_image_urls:
                    continue

                # Handle case where image_urls is a single string instead of array
                if isinstance(raw_image_urls, str):
                    logger.warning(f"User {user_id} has image_urls as string instead of array, converting...")
                    raw_image_urls = [raw_image_urls]

                # Normalize image_urls format - handle both string and dict formats
                normalized_image_urls = []
                for img in raw_image_urls:
                    if isinstance(img, dict):
                        # Dict format: {'original': 'url', 'thumb': 'url'}
                        normalized_image_urls.append(img)
                    elif isinstance(img, str):
                        # String format: just the URL
                        normalized_image_urls.append({'original': img, 'thumb': img})
                    else:
                        logger.warning(f"Unexpected image_url format for user {user_id}: {type(img)}")
                        continue

                if not normalized_image_urls:
                    continue

                # Check if user exists in pgvector
                if user_id not in existing_images:
                    # New user - process all images
                    logger.info(f"🆕 New user detected: {user_name} ({user_id})")
                    # Update user dict with normalized image_urls
                    user_copy = user.copy()
                    user_copy['image_urls'] = normalized_image_urls
                    users_to_process.append(user_copy)
                else:
                    # User exists - check for new images by comparing normalized URLs
                    existing_user_images_norm = existing_images[user_id]
                    # Normalize backend URLs for comparison
                    backend_image_urls_norm = {
                        normalize_image_url(img.get('original'))
                        for img in normalized_image_urls
                        if img.get('original')
                    }

                    new_images_norm = backend_image_urls_norm - existing_user_images_norm

                    if new_images_norm:
                        logger.info(f"📸 New images detected for {user_name}: {len(new_images_norm)} images")
                        # Create user dict with only new images (match by normalized URL)
                        new_image_dicts = [
                            img for img in normalized_image_urls
                            if normalize_image_url(img.get('original')) in new_images_norm
                        ]
                        users_to_process.append({
                            'id': user.get('id'),
                            'full_name': user_name,
                            'external_id': external_id,
                            'image_urls': new_image_dicts
                        })

            if not users_to_process:
                if users_deleted > 0 or images_deleted > 0:
                    logger.info(f"✅ Two-way sync complete: 0 added, {users_deleted} users deleted, {images_deleted} images deleted")
                else:
                    logger.info("✅ No missing embeddings detected - database is up to date")
                return {
                    'success': True,
                    'users_processed': 0,
                    'embeddings_added': 0,
                    'users_deleted': users_deleted,
                    'images_deleted': images_deleted,
                    'message': 'Database is up to date' if (users_deleted == 0 and images_deleted == 0) else 'Stale embeddings removed'
                }

            # Process missing users/images
            total_users = len(users_to_process)
            total_images = sum(len(user.get('image_urls', [])) for user in users_to_process)
            logger.info(f"Processing {total_users} user(s) with {total_images} missing images")

            results = {
                'users_processed': 0,
                'embeddings_added': 0,
                'failed_users': []
            }

            for idx, user in enumerate(users_to_process, 1):
                try:
                    # Validate user format
                    if not isinstance(user, dict):
                        logger.error(f"Invalid user format (expected dict, got {type(user)}): {user}")
                        results['failed_users'].append(str(user))
                        continue

                    user_name = user.get('full_name', 'Unknown')
                    user_id = user.get('id', 'unknown')
                    logger.info(f"[{idx}/{total_users}] Processing user: {user_name} ({user_id})")

                    user_result = self.handle_user_created(user)
                    results['users_processed'] += 1
                    results['embeddings_added'] += user_result['embeddings_added']

                    logger.info(f"[{idx}/{total_users}] ✅ Completed {user_name}: {user_result['embeddings_added']} embeddings added")

                except Exception as e:
                    user_id = user.get('id') if isinstance(user, dict) else 'unknown'
                    user_name = user.get('full_name') if isinstance(user, dict) else 'unknown'
                    logger.error(f"Failed to sync user {user_name} ({user_id}): {e}", exc_info=True)
                    results['failed_users'].append(user_id)

            logger.info(
                f"✅ Two-way sync complete: {results['embeddings_added']} added, "
                f"{users_deleted} users deleted, {images_deleted} images deleted"
            )

            return {
                'success': True,
                'users_deleted': users_deleted,
                'images_deleted': images_deleted,
                **results
            }

        except Exception as e:
            logger.error(f"Failed to sync missing embeddings: {e}")
            return {
                'success': False,
                'error': str(e)
            }
