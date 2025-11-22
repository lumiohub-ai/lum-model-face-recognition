"""Service to sync face embeddings from backend user events."""

import os
import requests
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from ..core.detector import FaceDetector
from ..storage.pgvector_store import PgVectorStore
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
        self.detector = FaceDetector(gpu_id=gpu_id)
        self.store = PgVectorStore(client_slug)
        self.image_fetcher = ImageFetcher()

        logger.info(f"EmbeddingSyncService initialized for: {client_slug}")

    def handle_user_created(self, user_data: Dict) -> Dict:
        """Handle user creation event from backend.

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

        # Process each image
        for img_data in image_urls:
            original_url = img_data.get('original')

            if not original_url:
                logger.warning("Image data missing 'original' URL, skipping")
                continue

            try:
                # Fetch image
                image = self.image_fetcher.fetch_image(original_url)

                if image is None:
                    results['failed_images'].append(original_url)
                    logger.error(f"❌ Failed to fetch image: {original_url}")
                    continue

                # Extract face features using detector
                features = self.detector.extract_face_features(image)

                if not features:
                    results['failed_images'].append(original_url)
                    logger.warning(f"⚠️  No face detected in: {original_url}")
                    continue

                # Use first detected face (best quality)
                face = features[0]
                embedding = face['embedding']
                metadata = {
                    'landmarks': face['landmarks'].tolist(),
                    'bbox': [float(x) for x in face['bbox']]  # Convert numpy types to Python float
                }

                # Save to pgvector database
                self.store.add_embedding(
                    user_id=user_id,
                    user_name=user_name,
                    image_url=original_url,
                    embedding=embedding,
                    external_id=external_id,
                    metadata=metadata
                )

                results['embeddings_added'] += 1
                logger.info(f"✅ Added embedding for {user_name} from {original_url}")

            except Exception as e:
                logger.error(f"❌ Error processing image {original_url}: {e}")
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

        # Simple approach: Delete all existing embeddings and re-add
        # TODO: Implement differential update (only add new images)
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
