"""Service to sync face embeddings from backend user events."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import text

from lum_vision import FaceDetector
from .pgvector import PgVectorStore
from .repository import Repository
from .url_utils import normalize_image_url
from .gcs import ImageFetcher


class EmbeddingSyncService:
    """Service to synchronize face embeddings with backend user data.

    This service handles:
    - User created: Download images, calculate embeddings, store in pgvector
    - User updated: Update embeddings for changed data
    - User deleted: Remove all embeddings from pgvector
    - Image added/deleted: Update specific embeddings
    """

    def __init__(self, client_slug: str, gpu_id: int = 0, config: Optional[Dict] = None,
                 detector=None, store=None):
        """Initialize embedding sync service.

        Args:
            client_slug: Organization slug (e.g., 'humblebee', 'dev')
            gpu_id: GPU device ID for face detection
            config: Optional config dict (from config.yaml)
            detector: Optional existing FaceDetector to reuse (avoids re-loading model)
            store: Optional existing PgVectorStore to reuse (avoids re-init schema)
        """
        self.client_slug = client_slug

        if detector is not None:
            self.detector = detector
        else:
            # Build from the same vision config the engine uses, so this detector
            # reads its model zoo from the same place instead of re-downloading.
            from config.vision import build_vision_config

            vision_config = build_vision_config(config)
            self.detector = FaceDetector(
                gpu_id=gpu_id,
                model_name=vision_config.face_model_name,
                padding_percent=vision_config.face_detection_padding,
                model_root=vision_config.insightface_dir,
            )

        self.store = store if store is not None else PgVectorStore(client_slug)
        self.image_fetcher = ImageFetcher()

        logger.info(f"EmbeddingSyncService initialized for: {client_slug}")

    @staticmethod
    def _to_image_dicts(raw_image_urls) -> List[Dict]:
        """Normalize image_urls from any backend format to a list of dicts.

        Handles:
        - List of dicts: [{'original': url, 'thumb': url}, ...]
        - List of strings: ['url', ...]
        - Single string: 'url'

        Returns:
            List of dicts with at least an 'original' key.
        """
        if not raw_image_urls:
            return []
        if isinstance(raw_image_urls, str):
            raw_image_urls = [raw_image_urls]

        result = []
        for img in raw_image_urls:
            if isinstance(img, dict) and img.get('original'):
                result.append(img)
            elif isinstance(img, str) and img:
                result.append({'original': img, 'thumb': img})
        return result

    def _process_single_image(self, img_data: Dict, user_id: str, user_name: str,
                               external_id: Optional[str]) -> Optional[Dict]:
        """Fetch one image, detect face, return embedding dict or None."""
        original_url = img_data.get('original')
        if not original_url:
            logger.warning("Image data missing 'original' URL, skipping")
            return None

        try:
            image = self.image_fetcher.fetch_image(original_url)
            if image is None:
                logger.warning(f"❌ Failed to fetch image: {original_url}")
                return None

            features = self.detector.extract_face_features(image)
            if not features:
                logger.warning(f"⚠️  No face detected in: {original_url}")
                return None

            face = features[0]
            return {
                'user_id': user_id,
                'user_name': user_name,
                'image_url': original_url,
                'embedding': face['embedding'],
                'external_id': external_id,
                'metadata': {
                    'landmarks': face['landmarks'].tolist(),
                    'bbox': [float(x) for x in face['bbox']],
                },
            }

        except Exception as e:
            logger.exception(f"❌ Error processing image {original_url}: {e}")
            return None

    def handle_user_created(self, user_data: Dict) -> Dict:
        """Handle user creation event from backend (parallelized).

        Args:
            user_data: User data from backend containing:
                - id: User ID
                - full_name: User's full name
                - external_id: External employee ID (optional)
                - image_urls: List of dicts with 'original', 'thumb' URLs (or plain strings)

        Returns:
            Dict with keys: user_id, user_name, embeddings_added, failed_images
        """
        user_id = str(user_data.get('id'))
        user_name = user_data.get('full_name')
        external_id = user_data.get('external_id')
        image_dicts = self._to_image_dicts(user_data.get('image_urls', []))

        logger.info(f"Syncing user created: {user_name} ({user_id}) with {len(image_dicts)} images")

        results = {'user_id': user_id, 'user_name': user_name, 'embeddings_added': 0, 'failed_images': []}

        if not image_dicts:
            return results

        max_workers = min(10, len(image_dicts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._process_single_image, img, user_id, user_name, external_id): img.get('original')
                for img in image_dicts
            }

            for future in as_completed(future_to_url):
                original_url = future_to_url[future]
                try:
                    result = future.result()
                    if result is None:
                        results['failed_images'].append(original_url)
                        continue

                    self.store.add_embedding(
                        user_id=result['user_id'],
                        user_name=result['user_name'],
                        image_url=result['image_url'],
                        embedding=result['embedding'],
                        external_id=result['external_id'],
                        metadata=result['metadata'],
                    )
                    results['embeddings_added'] += 1

                except Exception as e:
                    logger.exception(f"❌ Error saving embedding for {original_url}: {e}")
                    results['failed_images'].append(original_url)

        logger.info(
            f"User sync complete: {results['embeddings_added']} embeddings added, "
            f"{len(results['failed_images'])} failed"
        )
        return results

    def _remove_stale_embeddings(
        self, backend_users_by_id: Dict, existing_images: Dict[str, set]
    ) -> tuple:
        """Delete embeddings for users/images no longer in the backend.

        Returns:
            Tuple of (users_deleted, images_deleted)
        """
        backend_user_ids = set(backend_users_by_id.keys())
        pgvector_user_ids = set(existing_images.keys())

        users_deleted = 0
        images_deleted = 0

        for user_id in pgvector_user_ids - backend_user_ids:
            logger.info(f"Stale user detected: {user_id} - removing all embeddings")
            images_deleted += self.store.delete_all_for_user(user_id)
            users_deleted += 1

        for user_id in pgvector_user_ids & backend_user_ids:
            user = backend_users_by_id[user_id]
            image_dicts = self._to_image_dicts(user.get('image_urls', []))
            backend_norm = {normalize_image_url(img['original']) for img in image_dicts}
            for norm_url in existing_images[user_id] - backend_norm:
                images_deleted += self.store.delete_by_image_url_norm(user_id, norm_url)

        if users_deleted or images_deleted:
            logger.info(f"Stale embeddings removed: {users_deleted} users, {images_deleted} images")

        return users_deleted, images_deleted

    def _add_missing_embeddings(
        self, all_users: List[Dict], existing_images: Dict[str, set]
    ) -> tuple:
        """Create embeddings for new users and new images.

        Returns:
            Tuple of (users_processed, embeddings_added, failed_users)
        """
        users_to_process = []

        for user in all_users:
            user_id = str(user.get('id'))
            image_dicts = self._to_image_dicts(user.get('image_urls', []))
            if not image_dicts:
                continue

            if user_id not in existing_images:
                logger.info(f"🆕 New user detected: {user.get('full_name')} ({user_id})")
                users_to_process.append({**user, 'image_urls': image_dicts})
            else:
                existing_norm = existing_images[user_id]
                new_dicts = [
                    img for img in image_dicts
                    if normalize_image_url(img['original']) not in existing_norm
                ]
                if new_dicts:
                    logger.info(f"📸 New images for {user.get('full_name')}: {len(new_dicts)}")
                    users_to_process.append({
                        'id': user.get('id'),
                        'full_name': user.get('full_name'),
                        'external_id': user.get('external_id'),
                        'image_urls': new_dicts,
                    })

        if not users_to_process:
            return 0, 0, []

        total = len(users_to_process)
        total_images = sum(len(u.get('image_urls', [])) for u in users_to_process)
        logger.info(f"Processing {total} user(s) with {total_images} missing images")

        users_processed = 0
        embeddings_added = 0
        failed_users = []

        for idx, user in enumerate(users_to_process, 1):
            user_name = user.get('full_name', 'Unknown')
            user_id = user.get('id', 'unknown')
            try:
                result = self.handle_user_created(user)
                users_processed += 1
                embeddings_added += result['embeddings_added']
                logger.info(f"[{idx}/{total}] ✅ {user_name}: {result['embeddings_added']} embeddings added")
            except Exception as e:
                logger.exception(f"Failed to sync user {user_name} ({user_id}): {e}")
                failed_users.append(user_id)

        return users_processed, embeddings_added, failed_users

    def sync_missing_embeddings(self) -> Dict:
        """Two-way sync: remove stale embeddings then add missing ones.

        Returns:
            Dict with sync results including users processed and embeddings added.
        """
        logger.info(f"Syncing missing embeddings for {self.client_slug}")

        try:
            all_users = Repository(self.client_slug).get_all_users()
            logger.info(f"Total users from database: {len(all_users)}")

            # Build index: user_id → set of normalized image URLs already in pgvector
            existing_images: Dict[str, set] = {}
            with self.store.db_config.get_connection() as conn:
                rows = conn.execute(text(f"""
                    SELECT user_id, ARRAY_AGG(image_url_norm) AS image_urls_norm
                    FROM {self.store.schema_name}.face_embeddings
                    WHERE image_url_norm IS NOT NULL
                    GROUP BY user_id
                """))
                for row in rows:
                    existing_images[row[0]] = set(row[1] or [])

            backend_users_by_id = {str(u.get('id')): u for u in all_users}

            users_deleted, images_deleted = self._remove_stale_embeddings(
                backend_users_by_id, existing_images
            )
            users_processed, embeddings_added, failed_users = self._add_missing_embeddings(
                all_users, existing_images
            )

            if users_processed == 0 and embeddings_added == 0:
                msg = ('Stale embeddings removed' if (users_deleted or images_deleted)
                       else 'Database is up to date')
                logger.info(f"✅ {msg}")
                return {
                    'success': True,
                    'users_processed': 0,
                    'embeddings_added': 0,
                    'users_deleted': users_deleted,
                    'images_deleted': images_deleted,
                    'message': msg,
                }

            logger.info(
                f"✅ Two-way sync complete: {embeddings_added} added, "
                f"{users_deleted} users deleted, {images_deleted} images deleted"
            )
            return {
                'success': True,
                'users_processed': users_processed,
                'embeddings_added': embeddings_added,
                'users_deleted': users_deleted,
                'images_deleted': images_deleted,
                'failed_users': failed_users,
            }

        except Exception as e:
            logger.exception(f"Failed to sync missing embeddings: {e}")
            return {'success': False, 'error': str(e)}
