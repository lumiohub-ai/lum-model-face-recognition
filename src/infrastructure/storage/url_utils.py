"""URL utilities for stable embedding identity."""

from urllib.parse import urlparse, urlunparse


def normalize_image_url(url: str) -> str:
    """Normalize image URL by removing query params and fragments.

    This ensures stable identity for embeddings even when URLs have
    changing tokens, timestamps, or other query parameters.

    Args:
        url: Raw image URL (may contain query params)

    Returns:
        Normalized URL without query params or fragments

    Example:
        >>> normalize_image_url("https://storage.com/image.jpg?token=abc123")
        "https://storage.com/image.jpg"
        >>> normalize_image_url("https://storage.com/image.jpg#section")
        "https://storage.com/image.jpg"
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)
        # Reconstruct URL without query params and fragments
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # No params
            '',  # No query
            ''   # No fragment
        ))
        return normalized
    except Exception:
        # If parsing fails, return original URL
        return url
