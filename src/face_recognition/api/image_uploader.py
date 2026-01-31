"""Image upload helper for GCS uploads."""

from typing import Optional
import numpy as np
from loguru import logger

# Lazy-loaded ImageFetcher instance
_image_fetcher = None


def get_image_fetcher():
    """Get or create ImageFetcher instance."""
    global _image_fetcher
    if _image_fetcher is None:
        from ..services.image_fetcher import ImageFetcher
        _image_fetcher = ImageFetcher()
    return _image_fetcher


def upload_proof_image(
    image: np.ndarray,
    prefix: str,
    client_slug: str
) -> Optional[str]:
    """Upload proof image to GCS.

    Args:
        image: Image in BGR format (numpy array)
        prefix: GCS path prefix (e.g., "activity_proofs", "attendance_proofs")
        client_slug: Organization slug

    Returns:
        str: Public URL of uploaded image, or None if failed
    """
    try:
        fetcher = get_image_fetcher()
        url = fetcher.upload_image(
            image=image,
            prefix=prefix,
            client_slug=client_slug
        )
        if url:
            logger.debug(f"Uploaded proof image to GCS: {url}")
        return url
    except Exception as e:
        logger.error(f"Failed to upload proof image: {e}")
        return None
