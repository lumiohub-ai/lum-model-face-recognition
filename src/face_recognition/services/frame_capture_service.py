"""Frame capture service for camera calibration."""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger

from ..api.client import APIClient
from ..storage.cloud_storage import CloudStorageManager
from ..video.stream_handler import StreamHandler


class FrameCaptureService:
    """Service for capturing frames from camera streams for calibration.
    
    This service manages temporary RTSP connections to capture frames
    and uploads them to Google Cloud Storage for calibration purposes.
    """
    
    def __init__(
        self,
        api_client: APIClient,
        cloud_storage: Optional[CloudStorageManager] = None
    ):
        """Initialize the frame capture service.
        
        Args:
            api_client: API client for fetching camera configurations
            cloud_storage: Cloud storage manager (will create if not provided)
        """
        self.api_client = api_client
        self.cloud_storage = cloud_storage or CloudStorageManager()
        self.timeout = int(os.getenv('FRAME_CAPTURE_TIMEOUT', '30'))
        self.default_quality = int(os.getenv('FRAME_CAPTURE_DEFAULT_QUALITY', '85'))
        
        # Cache for temporary stream handlers
        self._stream_cache: Dict[int, StreamHandler] = {}
    
    def capture_frame_for_calibration(
        self,
        org_slug: str,
        camera_id: int,
        frame_index: int,
        quality: Optional[int] = None
    ) -> Dict[str, any]:
        """Capture a frame from a camera for calibration.
        
        Args:
            org_slug: Organization slug
            camera_id: Camera ID
            frame_index: Frame index (1-30) for calibration grid
            quality: JPEG quality (1-100), defaults to env var or 85
            
        Returns:
            Dictionary containing:
                - success: bool
                - frame_url: GCS path (gs://...)
                - signed_url: Signed URL for browser access
                - captured_at: ISO 8601 timestamp
                - metadata: Frame metadata (width, height, size, etc.)
                
        Raises:
            ValueError: If camera not found or invalid parameters
            TimeoutError: If frame capture times out
            Exception: If upload fails
        """
        if quality is None:
            quality = self.default_quality
        
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be between 1 and 100, got {quality}")
        
        if not 1 <= frame_index <= 30:
            raise ValueError(f"Frame index must be between 1 and 30, got {frame_index}")
        
        logger.info(
            f"Capturing frame for calibration: org={org_slug}, "
            f"camera_id={camera_id}, frame_index={frame_index}, quality={quality}"
        )
        
        # Fetch camera configuration
        camera_config = self.api_client.get_camera_by_id(camera_id)
        if not camera_config:
            raise ValueError(f"Camera with ID {camera_id} not found")
        
        camera_name = camera_config.get('name', f'Camera {camera_id}')
        stream_url = camera_config.get('stream_url')
        
        if not stream_url:
            raise ValueError(f"Camera {camera_id} does not have a stream_url configured")
        
        logger.info(f"Camera configuration: name={camera_name}, stream_url={stream_url[:50]}...")
        
        # Capture frame from stream
        frame = self._capture_frame_from_stream(stream_url, camera_id)
        
        if frame is None or frame.size == 0:
            raise Exception(f"Failed to capture frame from camera {camera_id}")
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Prepare metadata
        captured_at = datetime.now(timezone.utc)
        metadata = {
            "cameraId": str(camera_id),
            "cameraName": camera_name,
            "frameIndex": str(frame_index),
            "captureSource": "face-recognition-api",
            "width": str(width),
            "height": str(height),
            "capturedAt": captured_at.isoformat()
        }
        
        # Upload to GCS
        try:
            upload_result = self.cloud_storage.upload_frame(
                frame=frame,
                org_slug=org_slug,
                camera_id=camera_id,
                frame_index=frame_index,
                quality=quality,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to upload frame to GCS: {e}")
            raise Exception(f"GCS upload failed: {str(e)}")
        
        # Build response
        response = {
            "success": True,
            "frame_url": upload_result['gcs_path'],
            "signed_url": upload_result['signed_url'],
            "captured_at": captured_at.isoformat(),
            "metadata": {
                "width": width,
                "height": height,
                "size_bytes": upload_result['size_bytes'],
                "source": "face-recognition-api",
                "frame_index": frame_index,
                "camera_name": camera_name
            }
        }
        
        logger.info(
            f"Successfully captured and uploaded frame: "
            f"camera_id={camera_id}, frame_index={frame_index}, "
            f"size={upload_result['size_bytes']} bytes"
        )
        
        return response
    
    def _capture_frame_from_stream(
        self,
        stream_url: str,
        camera_id: int
    ) -> Optional[np.ndarray]:
        """Capture a single frame from an RTSP stream.
        
        Args:
            stream_url: RTSP/HTTP stream URL
            camera_id: Camera ID (for logging)
            
        Returns:
            Captured frame as numpy array, or None if capture failed
            
        Raises:
            TimeoutError: If frame capture times out
        """
        start_time = time.time()
        
        try:
            # Create temporary stream handler
            logger.debug(f"Creating stream handler for camera {camera_id}")
            
            # Use a simple logger for StreamHandler
            import logging
            stream_logger = logging.getLogger(f"stream_{camera_id}")
            
            stream_handler = StreamHandler(stream_url, stream_logger)
            
            # Try to read a frame with timeout
            max_attempts = 5
            frame = None
            
            for attempt in range(max_attempts):
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(
                        f"Frame capture timed out after {self.timeout} seconds"
                    )
                
                ret, current_frame = stream_handler.cap.read()
                
                if ret and current_frame is not None and current_frame.size > 0:
                    frame = current_frame
                    logger.debug(
                        f"Successfully captured frame from camera {camera_id} "
                        f"on attempt {attempt + 1}"
                    )
                    break
                
                logger.debug(
                    f"Frame capture attempt {attempt + 1} failed, retrying..."
                )
                time.sleep(0.5)
            
            # Clean up stream handler
            stream_handler.stop()
            
            if frame is None:
                logger.error(
                    f"Failed to capture frame from camera {camera_id} "
                    f"after {max_attempts} attempts"
                )
            
            return frame
            
        except TimeoutError:
            raise
        except Exception as e:
            logger.error(f"Error capturing frame from camera {camera_id}: {e}")
            raise
    
    def cleanup_stream_cache(self):
        """Clean up any cached stream handlers."""
        for camera_id, handler in list(self._stream_cache.items()):
            try:
                handler.stop()
                logger.debug(f"Stopped stream handler for camera {camera_id}")
            except Exception as e:
                logger.warning(f"Error stopping stream handler for camera {camera_id}: {e}")
        
        self._stream_cache.clear()
