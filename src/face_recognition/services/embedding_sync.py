"""Service to sync face embeddings from backend user events."""

import os
import requests
from typing import Dict, List, Optional
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.detector import FaceDetector
from ..storage.pgvector_store import PgVectorStore
from ..storage.url_utils import normalize_image_url
from .image_fetcher import ImageFetcher


class EmbeddingSyncService:
    """Service to synchronize face embeddings with backend user data.

    This service handles:
    - User created: Download images, calculate embeddings, store in pgvector
    - User updated: Update embeddings for changed data
    - User deleted: Remove all embeddings from pgvector
    - Image added/deleted: Update specific embeddings
    """

    def __init__(self, client_slug: str, gpu_id: int = 0):
        """Initialize embedding sync service.

        Args:
            client_slug: Organization slug (e.g., 'humblebee', 'dev')
            gpu_id: GPU device ID for face detection
        """
        self.client_slug = client_slug

        # Get face detection padding from environment (default: 20%)
        padding_percent = float(os.getenv('FACE_DETECTION_PADDING', '20.0'))
        self.detector = FaceDetector(gpu_id=gpu_id, padding_percent=padding_percent)

        self.store = PgVectorStore(client_slug)
        self.image_fetcher = ImageFetcher()

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

    def handle_user_updated(self, user_data: Dict) -> Dict:
        """Handle user update event from backend.

        This handles:
        - Name changes (update all embeddings)
        - New images added (calculate and add new embeddings)
        - Images removed (delete specific embeddings)

        Args:
            user_data: Updated user data from backend

        Returns:
            Dict with sync results
        """
        user_id = str(user_data.get('id'))
        new_name = user_data.get('full_name')
        image_urls = user_data.get('image_urls', [])

        logger.info(f"Updating user: {user_id} -> {new_name}")

        # Strategy: Delete all existing embeddings and re-add with updated data
        # This ensures complete consistency with backend state.
        # Differential updates could be added as optimization if needed.
        self.store.delete_all_for_user(user_id)

        # Re-add all embeddings
        return self.handle_user_created(user_data)

    def handle_user_deleted(self, user_id: str) -> Dict:
        """Handle user deletion event from backend.

        Args:
            user_id: User ID to delete

        Returns:
            Dict with deletion results
        """
        logger.info(f"Deleting user: {user_id}")

        count = self.store.delete_all_for_user(user_id)

        return {
            'user_id': user_id,
            'embeddings_deleted': count
        }

    def handle_image_added(self, user_id: str, user_name: str, image_url: str) -> Dict:
        """Handle individual image addition.

        Args:
            user_id: User ID
            user_name: User name
            image_url: URL of new image

        Returns:
            Dict with result
        """
        logger.info(f"Adding image for user {user_id}: {image_url}")

        try:
            # Fetch and process single image
            image = self.image_fetcher.fetch_image(image_url)

            if image is None:
                return {'success': False, 'error': 'Failed to fetch image'}

            features = self.detector.extract_face_features(image)

            if not features:
                return {'success': False, 'error': 'No face detected'}

            face = features[0]
            self.store.add_embedding(
                user_id=user_id,
                user_name=user_name,
                image_url=image_url,
                embedding=face['embedding'],
                metadata={'landmarks': face['landmarks'].tolist(), 'bbox': [float(x) for x in face['bbox']]}
            )

            return {'success': True}

        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            return {'success': False, 'error': str(e)}

    def handle_image_deleted(self, user_id: str, image_url: str) -> Dict:
        """Handle individual image deletion.

        Args:
            user_id: User ID
            image_url: URL of image to delete

        Returns:
            Dict with deletion result
        """
        logger.info(f"Deleting image for user {user_id}: {image_url}")

        count = self.store.delete_by_image_url(user_id, image_url)

        return {'embeddings_deleted': count}

    def rebuild_all_from_backend(self) -> Dict:
        """Fetch all users from backend API and rebuild embeddings.

        This calls the backend API to get all users for the organization,
        then rebuilds all embeddings from scratch.

        Returns:
            Dict with rebuild results
        """
        backend_url = os.getenv('SO_BACKEND_API_URL', 'http://localhost:7091')

        logger.info(f"Rebuilding all embeddings for {self.client_slug}")
        logger.info(f"Fetching users from: {backend_url}/api/org/{self.client_slug}/users")

        try:
            # Clear existing embeddings
            cleared = self.store.clear_all_embeddings()
            logger.info(f"Cleared {cleared} existing embeddings")

            # Fetch all users from backend with pagination
            page = 1
            all_users = []

            while True:
                response = requests.get(
                    f"{backend_url}/api/org/{self.client_slug}/users",
                    params={"page": page, "limit": 100},
                    timeout=30
                )

                if response.status_code != 200:
                    logger.error(f"Backend API returned {response.status_code}: {response.text}")
                    break

                users = response.json()

                if not users or len(users) == 0:
                    break

                all_users.extend(users)
                logger.info(f"Fetched page {page}: {len(users)} users")

                page += 1

                # Safety limit
                if page > 100:
                    logger.warning("Reached page limit (100), stopping")
                    break

            logger.info(f"Total users fetched: {len(all_users)}")

            # Rebuild embeddings for all users
            return self.rebuild_all(all_users)

        except Exception as e:
            logger.error(f"Failed to rebuild from backend: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def rebuild_all(self, users: List[Dict]) -> Dict:
        """Rebuild all embeddings for a list of users.

        Args:
            users: List of user data dicts

        Returns:
            Dict with rebuild results
        """
        logger.info(f"Rebuilding embeddings for {len(users)} users")

        results = {
            'total_users': len(users),
            'successful': 0,
            'failed': 0,
            'total_embeddings_added': 0,
            'failed_users': []
        }

        for user in users:
            try:
                user_result = self.handle_user_created(user)
                results['successful'] += 1
                results['total_embeddings_added'] += user_result['embeddings_added']

                if user_result.get('failed_images'):
                    logger.warning(f"User {user.get('id')} had failed images: {user_result['failed_images']}")

            except Exception as e:
                logger.error(f"Failed to sync user {user.get('id')}: {e}")
                results['failed'] += 1
                results['failed_users'].append(user.get('id'))

        logger.info(
            f"✅ Rebuild complete: {results['successful']}/{results['total_users']} users, "
            f"{results['total_embeddings_added']} embeddings"
        )

        return results

    def sync_missing_embeddings(self, api_client=None) -> Dict:
        """Sync embeddings for users/images that exist in backend but not in pgvector.

        This method:
        1. Fetches all users from backend API
        2. Compares with existing embeddings in pgvector
        3. Calculates embeddings only for missing users/images

        Args:
            api_client: Optional authenticated APIClient instance for fetching users

        Returns:
            Dict with sync results including users processed and embeddings added
        """
        logger.info(f"🔄 Syncing missing embeddings for {self.client_slug}")

        try:
            # Fetch all users from backend
            all_users = []
            seen_user_ids = set()  # Track seen user IDs to avoid duplicates

            if api_client:
                # Use authenticated API client
                logger.info("Using authenticated API client to fetch users")
                page = 1
                while True:
                    try:
                        response = api_client.session.get(
                            f"{api_client.base_url}/org/{self.client_slug}/users",
                            params={"page": page, "limit": 100},
                            timeout=30
                        )

                        if response.status_code != 200:
                            logger.error(f"Backend API returned {response.status_code}: {response.text}")
                            break

                        users = response.json()

                        if not users or len(users) == 0:
                            break

                        # Deduplicate users by ID
                        new_users = []
                        for user in users:
                            user_id = str(user.get('id'))
                            if user_id not in seen_user_ids:
                                seen_user_ids.add(user_id)
                                new_users.append(user)

                        if not new_users:
                            # All users on this page were duplicates, stop pagination
                            logger.info(f"Page {page} contained only duplicates, stopping pagination")
                            break

                        all_users.extend(new_users)
                        logger.info(f"Fetched page {page}: {len(new_users)} new users ({len(users)} total returned)")

                        page += 1

                        if page > 100:  # Safety limit
                            logger.warning("Reached page limit (100), stopping")
                            break

                    except Exception as e:
                        logger.error(f"Error fetching users from API: {e}")
                        break
            else:
                # Fallback to direct unauthenticated request (for backward compatibility)
                backend_url = os.getenv('SO_BACKEND_API_URL', 'http://localhost:7091')
                logger.warning("No API client provided, attempting unauthenticated request")
                page = 1

                while True:
                    response = requests.get(
                        f"{backend_url}/api/org/{self.client_slug}/users",
                        params={"page": page, "limit": 100},
                        timeout=30
                    )

                    if response.status_code != 200:
                        logger.error(f"Backend API returned {response.status_code}: {response.text}")
                        break

                    users = response.json()

                    if not users or len(users) == 0:
                        break

                    # Deduplicate users by ID
                    new_users = []
                    for user in users:
                        user_id = str(user.get('id'))
                        if user_id not in seen_user_ids:
                            seen_user_ids.add(user_id)
                            new_users.append(user)

                    if not new_users:
                        logger.info(f"Page {page} contained only duplicates, stopping pagination")
                        break

                    all_users.extend(new_users)
                    logger.info(f"Fetched page {page}: {len(new_users)} new users ({len(users)} total returned)")

                    page += 1

                    if page > 100:  # Safety limit
                        logger.warning("Reached page limit (100), stopping")
                        break

            logger.info(f"Total users from backend: {len(all_users)}")

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
                logger.info("✅ No missing embeddings detected - database is up to date")
                return {
                    'success': True,
                    'users_processed': 0,
                    'embeddings_added': 0,
                    'message': 'Database is up to date'
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
                f"✅ Sync complete: {results['users_processed']} users processed, "
                f"{results['embeddings_added']} embeddings added"
            )

            return {
                'success': True,
                **results
            }

        except Exception as e:
            logger.error(f"Failed to sync missing embeddings: {e}")
            return {
                'success': False,
                'error': str(e)
            }
