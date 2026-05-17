"""Fetch and upload images to/from Google Cloud Storage or HTTP URLs."""

import os
import re
import uuid
import ipaddress
import requests
from datetime import datetime
from typing import Optional, Set, List
from urllib.parse import urlparse
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


# SECURITY: URL validation to prevent SSRF attacks
class URLValidator:
    """Validates URLs to prevent Server-Side Request Forgery (SSRF) attacks.

    SSRF allows attackers to make the server fetch resources from internal
    networks or cloud metadata endpoints.
    """

    # Allowed URL schemes
    ALLOWED_SCHEMES: Set[str] = {'https', 'http', 'gs'}

    # Blocked IP ranges (internal networks, cloud metadata, localhost)
    BLOCKED_IP_RANGES: List[str] = [
        '127.0.0.0/8',      # Localhost
        '10.0.0.0/8',       # Private network (Class A)
        '172.16.0.0/12',    # Private network (Class B)
        '192.168.0.0/16',   # Private network (Class C)
        '169.254.0.0/16',   # Link-local / AWS metadata
        '100.64.0.0/10',    # Carrier-grade NAT
        '0.0.0.0/8',        # Current network
        '224.0.0.0/4',      # Multicast
        '240.0.0.0/4',      # Reserved
        '::1/128',          # IPv6 localhost
        'fc00::/7',         # IPv6 private
        'fe80::/10',        # IPv6 link-local
    ]

    # Blocked hostnames
    BLOCKED_HOSTNAMES: Set[str] = {
        'localhost',
        'metadata.google.internal',
        'metadata.google.com',
        '169.254.169.254',  # Cloud metadata endpoint
        'metadata',
    }

    # Allowed GCS buckets (whitelist approach)
    ALLOWED_GCS_BUCKETS: Set[str] = set()  # Empty = allow all; populate for stricter security

    # Allowed HTTP domains (whitelist approach)
    ALLOWED_HTTP_DOMAINS: Set[str] = {
        'storage.googleapis.com',
        'storage.cloud.google.com',
    }

    @classmethod
    def configure(cls, allowed_gcs_buckets: List[str] = None, allowed_http_domains: List[str] = None):
        """Configure allowed buckets and domains.

        Args:
            allowed_gcs_buckets: List of allowed GCS bucket names
            allowed_http_domains: List of allowed HTTP domains
        """
        if allowed_gcs_buckets:
            cls.ALLOWED_GCS_BUCKETS = set(allowed_gcs_buckets)
        if allowed_http_domains:
            cls.ALLOWED_HTTP_DOMAINS.update(allowed_http_domains)

    @classmethod
    def is_safe_url(cls, url: str) -> tuple[bool, str]:
        """Check if a URL is safe to fetch.

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_safe, reason)
        """
        if not url:
            return False, "Empty URL"

        # Handle GCS URLs separately
        if url.startswith('gs://'):
            return cls._validate_gcs_url(url)

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Invalid URL format: {e}"

        # Check scheme
        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            return False, f"Blocked scheme: {parsed.scheme}"

        # Check hostname
        hostname = parsed.hostname
        if not hostname:
            return False, "No hostname in URL"

        hostname_lower = hostname.lower()

        # Check blocked hostnames
        if hostname_lower in cls.BLOCKED_HOSTNAMES:
            return False, f"Blocked hostname: {hostname}"

        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            # Check against blocked IP ranges
            for blocked_range in cls.BLOCKED_IP_RANGES:
                if ip in ipaddress.ip_network(blocked_range, strict=False):
                    return False, f"Blocked IP range: {hostname}"
        except ValueError:
            # Not an IP address, check domain whitelist
            if cls.ALLOWED_HTTP_DOMAINS:
                domain_allowed = any(
                    hostname_lower == domain or hostname_lower.endswith('.' + domain)
                    for domain in cls.ALLOWED_HTTP_DOMAINS
                )
                if not domain_allowed:
                    return False, f"Domain not in allowlist: {hostname}"

        return True, "URL is safe"

    @classmethod
    def _validate_gcs_url(cls, url: str) -> tuple[bool, str]:
        """Validate GCS URL.

        Args:
            url: GCS URL (gs://bucket/path)

        Returns:
            Tuple of (is_safe, reason)
        """
        # Parse gs://bucket/path
        match = re.match(r'^gs://([^/]+)(/.*)?$', url)
        if not match:
            return False, "Invalid GCS URL format"

        bucket_name = match.group(1)

        # If whitelist is configured, check it
        if cls.ALLOWED_GCS_BUCKETS and bucket_name not in cls.ALLOWED_GCS_BUCKETS:
            return False, f"GCS bucket not in allowlist: {bucket_name}"

        return True, "GCS URL is safe"


class ImageFetcher:
    """Fetch and process images from various sources.

    Supports:
    - Google Cloud Storage (gs:// URLs)
    - HTTP/HTTPS URLs (with SSRF protection)
    - Local file paths (for testing)
    """

    def __init__(self, allowed_gcs_buckets: List[str] = None, allowed_http_domains: List[str] = None):
        """Initialize image fetcher with GCS credentials if available.

        Args:
            allowed_gcs_buckets: Optional list of allowed GCS bucket names
            allowed_http_domains: Optional list of allowed HTTP domains
        """
        from config.settings import settings
        self.use_gcs = settings.use_gcs
        self.gcs_credentials = settings.gcs_credentials_path
        self.gcs_bucket = settings.gcs_bucket
        self.gcs_client = None

        # Configure URL validator
        if allowed_gcs_buckets:
            URLValidator.configure(allowed_gcs_buckets=allowed_gcs_buckets)
        if allowed_http_domains:
            URLValidator.configure(allowed_http_domains=allowed_http_domains)

        # Initialize GCS client if credentials exist
        if self.use_gcs and GCS_AVAILABLE and self.gcs_credentials and os.path.exists(self.gcs_credentials):
            try:
                self.gcs_client = storage.Client.from_service_account_json(
                    self.gcs_credentials
                )
                logger.debug(f"GCS client initialized with bucket: {self.gcs_bucket}")
            except Exception as e:
                logger.exception(f"Failed to initialize GCS client: {e}")
                self.gcs_client = None
        else:
            logger.info(
                "GCS disabled or unavailable. Using local image storage."
            )

    def fetch_image(self, url: str) -> Optional[np.ndarray]:
        """Fetch image from GCS, HTTP, or local path.

        SECURITY: All URLs are validated to prevent SSRF attacks.

        Args:
            url: Image URL (gs://, http://, https://, or file path)

        Returns:
            np.ndarray: Image in BGR format (OpenCV), or None if failed
        """
        try:
            # SECURITY: Validate URL before fetching
            if url.startswith('gs://') or url.startswith('http://') or url.startswith('https://'):
                is_safe, reason = URLValidator.is_safe_url(url)
                if not is_safe:
                    logger.warning(f"SSRF_BLOCKED: {reason} - URL: {url}")
                    return None

            # Convert https://storage.googleapis.com/{bucket}/... to gs:// so the
            # authenticated GCS client is used instead of unauthenticated HTTP.
            if url.startswith('https://storage.googleapis.com/') and self.gcs_client:
                path = url[len('https://storage.googleapis.com/'):]
                url = f'gs://{path}'

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
            logger.exception(f"Failed to fetch image from {url}: {e}")
            return None

    @staticmethod
    def _decode_image_bytes(data: bytes) -> np.ndarray:
        """Decode image bytes to a BGR numpy array (OpenCV format).

        Handles both RGB and RGBA source images.
        """
        image = Image.open(BytesIO(data))
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_np

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
            image_np = self._decode_image_bytes(image_bytes)
            logger.info(f"Fetched image from GCS: {gs_url} (shape: {image_np.shape})")
            return image_np

        except Exception as e:
            logger.exception(f"Failed to fetch from GCS {gs_url}: {e}")
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
                logger.exception(f"HTTP request failed for {url}: {e}")
                return None

            except Exception as e:
                logger.exception(f"Unexpected error fetching {url}: {e}")
                return None

        if img_bytes is None:
            logger.error(f"Failed to fetch image bytes from {url}")
            return None

        try:
            image_np = self._decode_image_bytes(img_bytes)
            logger.info(f"Fetched image from HTTP: {url} (shape: {image_np.shape})")
            return image_np
        except Exception as e:
            logger.exception(f"Failed to process image from {url}: {e}")
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
            logger.exception(f"Failed to fetch from local {file_path}: {e}")
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
            return self._upload_image_locally(image, prefix, client_slug)

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
            if image is None or image.size == 0:
                logger.warning("Cannot upload empty image to GCS")
                return None
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
            logger.exception(f"Failed to upload image to GCS: {e}")
            return None

    def _upload_image_locally(
        self,
        image: np.ndarray,
        prefix: str = "unrecognized_faces",
        client_slug: Optional[str] = None
    ) -> Optional[str]:
        """Save image to local disk and return an HTTP URL served by the metrics server."""
        try:
            if image is None or image.size == 0:
                return None

            local_dir = os.environ.get(
                "SO_LOCAL_IMAGE_DIR",
                "/app/volumes/storage/person-tracking/images"
            )
            os.makedirs(local_dir, exist_ok=True)

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.jpg"

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(os.path.join(local_dir, filename), format="JPEG", quality=85)

            host = os.environ.get("SO_LOCAL_IMAGE_HOST", "localhost")
            port = os.environ.get("SO_METRICS_PORT", "8765")
            url = f"http://{host}:{port}/images/{filename}"
            logger.info(f"Saved image locally: {url}")
            return url
        except Exception as e:
            logger.exception(f"Failed to save image locally: {e}")
            return None

