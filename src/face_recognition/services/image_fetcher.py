"""Fetch images from Google Cloud Storage or HTTP URLs."""

import os
import requests
from typing import Optional
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from loguru import logger

# Try to import GCS client, but make it optional
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not installed, GCS download disabled")


class ImageFetcher:
    """Fetch and process images from various sources.

    Supports:
    - Google Cloud Storage (gs:// URLs)
    - HTTP/HTTPS URLs
    - Local file paths (for testing)
    """

    def __init__(self):
        """Initialize image fetcher with GCS credentials if available."""
        self.gcs_credentials = os.getenv('GCS_CREDENTIALS_PATH')
        self.gcs_bucket = os.getenv('GCS_BUCKET', 'hbai-general-data')
        self.gcs_client = None

        # Initialize GCS client if credentials exist
        if GCS_AVAILABLE and self.gcs_credentials and os.path.exists(self.gcs_credentials):
            try:
                self.gcs_client = storage.Client.from_service_account_json(
                    self.gcs_credentials
                )
                logger.info(f"GCS client initialized with bucket: {self.gcs_bucket}")
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}")
                self.gcs_client = None
        else:
            logger.warning(
                "GCS credentials not found or invalid. "
                "GCS downloads will fail. Using HTTP fallback."
            )

    def fetch_image(self, url: str) -> Optional[np.ndarray]:
        """Fetch image from GCS, HTTP, or local path.

        Args:
            url: Image URL (gs://, http://, https://, or file path)

        Returns:
            np.ndarray: Image in BGR format (OpenCV), or None if failed
        """
        try:
            if url.startswith('gs://'):
                return self._fetch_from_gcs(url)
            elif url.startswith('http://') or url.startswith('https://'):
                return self._fetch_from_http(url)
            elif os.path.exists(url):
                return self._fetch_from_local(url)
            else:
                logger.error(f"Unsupported URL format or file not found: {url}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch image from {url}: {e}")
            return None

    def _fetch_from_gcs(self, gs_url: str) -> Optional[np.ndarray]:
        """Fetch from Google Cloud Storage.

        Args:
            gs_url: GCS URL in format gs://bucket/path

        Returns:
            np.ndarray: Image in BGR format, or None if failed
        """
        if not self.gcs_client:
            logger.error("GCS client not initialized, cannot fetch from GCS")
            return None

        try:
            # Parse gs://bucket/path format
            parts = gs_url.replace('gs://', '').split('/', 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ''

            # Get bucket and blob
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download to memory
            image_bytes = blob.download_as_bytes()

            # Convert to OpenCV format
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)

            # Convert RGB to BGR (OpenCV format)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            logger.info(f"Fetched image from GCS: {gs_url} (shape: {image_np.shape})")
            return image_np

        except Exception as e:
            logger.error(f"Failed to fetch from GCS {gs_url}: {e}")
            return None

    def _fetch_from_http(self, url: str) -> Optional[np.ndarray]:
        """Fetch from HTTP/HTTPS URL.

        Args:
            url: HTTP/HTTPS URL

        Returns:
            np.ndarray: Image in BGR format, or None if failed
        """
        try:
            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Convert to OpenCV format
            image = Image.open(BytesIO(response.content))
            image_np = np.array(image)

            # Convert RGB to BGR (OpenCV format)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            logger.info(f"Fetched image from HTTP: {url} (shape: {image_np.shape})")
            return image_np

        except Exception as e:
            logger.error(f"Failed to fetch from HTTP {url}: {e}")
            return None

    def _fetch_from_local(self, file_path: str) -> Optional[np.ndarray]:
        """Fetch from local file system.

        Args:
            file_path: Local file path

        Returns:
            np.ndarray: Image in BGR format, or None if failed
        """
        try:
            # Read image using OpenCV (already in BGR)
            image_np = cv2.imread(file_path)

            if image_np is None:
                logger.error(f"Failed to read image from {file_path}")
                return None

            logger.info(f"Fetched image from local: {file_path} (shape: {image_np.shape})")
            return image_np

        except Exception as e:
            logger.error(f"Failed to fetch from local {file_path}: {e}")
            return None

    def fetch_images_batch(self, urls: list) -> list:
        """Fetch multiple images in batch.

        Args:
            urls: List of image URLs

        Returns:
            List of tuples: (url, image_np or None)
        """
        results = []

        for url in urls:
            image = self.fetch_image(url)
            results.append((url, image))

        successful = sum(1 for _, img in results if img is not None)
        logger.info(f"Batch fetch complete: {successful}/{len(urls)} successful")

        return results

    def convert_http_to_gs(self, http_url: str) -> str:
        """Convert HTTP GCS URL to gs:// format.

        Args:
            http_url: HTTP URL like https://storage.googleapis.com/bucket/path

        Returns:
            str: gs:// formatted URL

        Example:
            https://storage.googleapis.com/hbai-general-data/2025/image.jpg
            -> gs://hbai-general-data/2025/image.jpg
        """
        if http_url.startswith('https://storage.googleapis.com/'):
            path = http_url.replace('https://storage.googleapis.com/', '')
            return f"gs://{path}"
        return http_url
