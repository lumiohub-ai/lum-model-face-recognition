"""
Camera processor module for managing shared frame buffers.
Uses in-memory storage for efficient frame sharing within same process.
"""

import cv2
import numpy as np
import time
from typing import Dict, Optional, Tuple
from collections import defaultdict
from threading import Lock
from loguru import logger


class CameraProcessor:
    """Manages frame buffers for multiple cameras using in-memory storage.

    This class provides a thread-safe way to share video frames between
    the face recognition pipeline and the streaming server in the same process.
    """

    def __init__(self, max_frame_age: float = 2.0, store_compressed: bool = True):
        """Initialize the camera processor.

        Args:
            max_frame_age: Maximum age of frames in seconds before they're considered stale
            store_compressed: If True, store frames as compressed JPEG to save memory
        """
        self.max_frame_age = max_frame_age
        self.store_compressed = store_compressed
        self.frame_queues: Dict[str, Tuple[np.ndarray, float]] = {}
        self.locks: Dict[str, Lock] = defaultdict(Lock)
        logger.info(f"CameraProcessor initialized with in-memory frame storage (compressed={store_compressed})")

    def put_frame(self, camera_id: str, frame: np.ndarray) -> bool:
        """Store a frame for a specific camera.

        Args:
            camera_id: Unique identifier for the camera
            frame: Video frame as numpy array

        Returns:
            True if frame was stored successfully, False otherwise
        """
        try:
            if frame is None or frame.size == 0:
                return False

            # Store frame (compressed or raw depending on setting)
            if self.store_compressed:
                # Encode frame as JPEG to save memory
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    logger.warning(f"Failed to encode frame for camera {camera_id}")
                    return False

                with self.locks[camera_id]:
                    self.frame_queues[camera_id] = (buffer, time.time())
            else:
                # Store raw frame for faster streaming (no decode/re-encode)
                with self.locks[camera_id]:
                    self.frame_queues[camera_id] = (frame.copy(), time.time())

            return True

        except Exception as e:
            logger.error(f"Error storing frame for camera {camera_id}: {e}")
            return False

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Retrieve the latest frame for a specific camera.

        Args:
            camera_id: Unique identifier for the camera

        Returns:
            Latest frame as numpy array, or None if no frame available
        """
        try:
            with self.locks[camera_id]:
                if camera_id not in self.frame_queues:
                    return None

                data, timestamp = self.frame_queues[camera_id]

                # Check if frame is too old
                if time.time() - timestamp > self.max_frame_age:
                    logger.debug(f"Frame for camera {camera_id} is stale")
                    return None

                # Decode JPEG if compressed, otherwise return raw frame
                if self.store_compressed:
                    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                else:
                    frame = data

                return frame

        except Exception as e:
            logger.error(f"Error retrieving frame for camera {camera_id}: {e}")
            return None

    def get_camera_ids(self) -> list:
        """Get list of all active camera IDs.

        Returns:
            List of camera IDs that have frames available
        """
        return list(self.frame_queues.keys())

    def cleanup_stale_frames(self):
        """Remove frames that are older than max_frame_age."""
        current_time = time.time()
        stale_cameras = []

        for camera_id in list(self.frame_queues.keys()):
            with self.locks[camera_id]:
                if camera_id in self.frame_queues:
                    _, timestamp = self.frame_queues[camera_id]
                    if current_time - timestamp > self.max_frame_age:
                        stale_cameras.append(camera_id)

        for camera_id in stale_cameras:
            with self.locks[camera_id]:
                if camera_id in self.frame_queues:
                    del self.frame_queues[camera_id]
                    logger.debug(f"Cleaned up stale frame for camera {camera_id}")


# Global instance for sharing across modules
_camera_processor_instance = None


def setup_cameras() -> CameraProcessor:
    """Get or create the global camera processor instance.

    Returns:
        CameraProcessor instance
    """
    global _camera_processor_instance
    if _camera_processor_instance is None:
        # Disable compression for faster streaming (avoid double JPEG encoding)
        _camera_processor_instance = CameraProcessor(store_compressed=False)
    return _camera_processor_instance


def get_camera_processor() -> Optional[CameraProcessor]:
    """Get the existing camera processor instance.

    Returns:
        CameraProcessor instance if initialized, None otherwise
    """
    return _camera_processor_instance
