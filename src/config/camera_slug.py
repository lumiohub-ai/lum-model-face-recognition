"""Canonical camera-name -> MediaMTX path slug (LSO-30).

CANONICAL SPEC: mediamtx-edge/generator/camera_paths.py::slug
Shared contract vectors: mediamtx-edge/generator/slug_vectors.json — keep this in
sync with the edge generator, the backend, and the frontend resolver.

Rule: lowercase, then keep only [a-z0-9]. Kept dependency-free so it can be
imported and tested in isolation (no settings / loguru).
"""

import re


def mediamtx_path(camera_name: str) -> str:
    """'Vision_1' -> 'vision1', 'Meeting room' -> 'meetingroom'."""
    return re.sub(r"[^a-z0-9]", "", str(camera_name).lower())
