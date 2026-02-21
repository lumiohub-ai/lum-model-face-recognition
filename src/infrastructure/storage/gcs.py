"""Fetch and upload images to/from Google Cloud Storage or HTTP URLs."""

import os
import uuid
import requests
from datetime import datetime
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
        """Fetch from HTTP/HTTPS URL with SSL retry logic.

        Args:
            url: HTTP/HTTPS URL

        Returns:
            np.ndarray: Image in BGR format, or None if failed
        """
        max_retries = 3
        img_bytes = None

        # Retry loop for handling intermittent SSL errors
        for attempt in range(max_retries):
            try:
                # Download image with SSL verification
                response = requests.get(url, timeout=30, verify=True)
                response.raise_for_status()
                img_bytes = response.content
                break  # Success, exit retry loop

            except requests.exceptions.SSLError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"SSL error (attempt {attempt + 1}/{max_retries}), retrying... {url}"
                    )
                    # Add small delay before retry
                    import time
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Failed to fetch after {max_retries} SSL retry attempts: {url}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP request failed for {url}: {e}")
                return None

            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        if img_bytes is None:
            logger.error(f"Failed to fetch image bytes from {url}")
            return None

        try:
            # Convert to OpenCV format
            image = Image.open(BytesIO(img_bytes))
            image_np = np.array(image)

            # Convert RGB to BGR (OpenCV format)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            logger.info(f"Fetched image from HTTP: {url} (shape: {image_np.shape})")
            return image_np

        except Exception as e:
            logger.error(f"Failed to process image from {url}: {e}")
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

    def upload_image(
        self,
        image: np.ndarray,
        prefix: str = "unrecognized_faces",
        client_slug: Optional[str] = None
    ) -> Optional[str]:
        """Upload image to Google Cloud Storage.

        Args:
            image: Image in BGR format (numpy array)
            prefix: GCS path prefix (e.g., "unrecognized_faces", "attendance_proofs")
            client_slug: Organization slug for path organization

        Returns:
            str: Public URL of uploaded image, or None if failed
        """
        if not self.gcs_client:
            logger.warning("GCS client not initialized, cannot upload image")
            return None

        try:
            # Generate unique filename with timestamp
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.jpg"

            # Build GCS path matching backend pattern
            year = now.strftime("%Y")
            if client_slug:
                blob_path = f"{year}/cv.face-recognition/{client_slug}/{prefix}/{filename}"
            else:
                blob_path = f"{year}/cv.face-recognition/{prefix}/{filename}"

            # Convert BGR to RGB and encode as JPEG
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Encode to JPEG bytes
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()

            # Upload to GCS
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(image_bytes, content_type='image/jpeg')

            # For buckets with uniform bucket-level access, use the public URL directly
            # The bucket should be configured for public access at the bucket level
            # or use signed URLs for private access
            public_url = f"https://storage.googleapis.com/{self.gcs_bucket}/{blob_path}"

            logger.info(f"Uploaded image to GCS: {public_url}")
            return public_url

        except Exception as e:
            logger.error(f"Failed to upload image to GCS: {e}")
            return None


