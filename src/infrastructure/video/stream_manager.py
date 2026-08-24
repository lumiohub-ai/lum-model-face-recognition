"""Stream manager for handling video streams and output writers.

This module manages:
- Multiple video streams (RTSP, file, webcam)
- Video output writers for recording
- Stream lifecycle (start, read, stop)
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from loguru import logger

from .stream_handler import StreamHandler


class StreamManager:
    """Manages video streams and output writers.

    Handles initialization, reading, and cleanup of multiple video streams
    and optional video recording.
    """

    def __init__(self, camera_configs: List[Dict[str, Any]]):
        """Initialize stream manager.

        Args:
            camera_configs: List of camera configuration dictionaries
                Each config should have: stream_url, camera_name, cam_type
        """
        self.camera_configs = camera_configs
        # Keyed by DB camera id, not list position (LSO-130). Positions shift
        # when a camera leaves the set, which silently re-points every later
        # camera's stream handler at the wrong camera.
        self.streams: Dict[int, StreamHandler] = {}
        self.video_writers: Dict[int, Optional[cv2.VideoWriter]] = {}
        self._initialized = False

    def init_streams(self) -> Dict[int, StreamHandler]:
        """Initialize stream handlers for all cameras, keyed by camera id."""
        self.streams = {}

        for config in self.camera_configs:
            camera_id = config.get('camera_id')
            stream = StreamHandler(
                src=config['stream_url'],
                logger=logger
            )
            self.streams[camera_id] = stream
            if stream.connected:
                logger.info(
                    f"Stream ready: {config.get('camera_name', 'Unknown')} "
                    f"(camera_id={config.get('camera_id')})"
                )
            else:
                logger.warning(
                    f"Stream unavailable at startup: {config.get('camera_name', 'Unknown')} "
                    f"(camera_id={config.get('camera_id')}) — reconnecting in background"
                )

        connected = sum(1 for s in self.streams.values() if s.connected)
        logger.info(
            f"Stream init complete: {connected}/{len(self.streams)} connected, "
            f"{len(self.streams) - connected} reconnecting"
        )

        self._initialized = True
        return self.streams

    def init_video_writers(self, output_dir: str) -> Dict[int, Optional[cv2.VideoWriter]]:
        """Initialize video writers for saving output.

        Args:
            output_dir: Directory to save video files

        Returns:
            List of VideoWriter objects (or None for failed writers)
        """
        self.video_writers = {}

        now = datetime.now()
        date = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        # Create output directory and check permissions
        try:
            os.makedirs(output_dir, exist_ok=True)
            test_file = os.path.join(output_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Output directory ready: {output_dir}")
        except Exception as e:
            logger.exception(f"Output directory not writable: {output_dir} - {e}")
            return self.video_writers

        for config in self.camera_configs:
            camera_id = config.get('camera_id')
            camera_name = config['camera_name'].replace(' ', '_')
            status = config.get('cam_type', 'IN').upper()
            filename = f"{output_dir}/{status}_{camera_name}_{date}_{time_str}.mp4"

            # Get frame dimensions and fps from stream
            w, h, fps = 1920, 1080, 20
            stream = self.streams.get(camera_id)
            if stream is not None:
                if stream.cap is not None:
                    cap_w = int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cap_h = int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap_fps = stream.cap.get(cv2.CAP_PROP_FPS)
                    if cap_w > 0:
                        w = cap_w
                    if cap_h > 0:
                        h = cap_h
                    if 1 < cap_fps < 120:
                        fps = int(cap_fps)

            writer = self._create_video_writer(filename, w, h, fps)
            self.video_writers[camera_id] = writer

        return self.video_writers

    def _create_video_writer(
        self,
        filename: str,
        width: int,
        height: int,
        fps: int = 20
    ) -> Optional[cv2.VideoWriter]:
        """Create a video writer.

        Args:
            filename: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            VideoWriter object or None if creation failed
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            if writer.isOpened():
                logger.info(f"Video writer initialized: {filename} ({width}x{height} @ {fps}fps) [mp4v]")
                return writer
            else:
                writer.release()
                logger.error(f"Failed to open video writer: {filename}")
                return None
        except Exception as e:
            logger.exception(f"Exception initializing video writer: {filename} - {e}")
            return None

    def start_streams(self) -> None:
        """Start all non-video file streams (RTSP, webcam)."""
        if not self._initialized:
            self.init_streams()

        for stream in self.streams.values():
            if not stream.is_video:
                stream.start()

        logger.info("Started video streams")

    def stop_streams(self) -> None:
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()
        logger.info("Stopped all video streams")

    def release_writers(self) -> None:
        """Release all video writers."""
        for writer in self.video_writers.values():
            if writer is not None:
                writer.release()
        self.video_writers = {}
        logger.info("Released all video writers")

    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Return the latest frame for a specific camera without blocking the pipeline.

        Reads directly from the StreamHandler's background thread buffer.
        Does not start a new RTSP connection.

        Args:
            camera_id: The camera ID to capture from

        Returns:
            Latest frame as numpy array, or None if camera not found / no frame yet
        """
        stream = self.streams.get(camera_id)
        if stream is None:
            return None
        ret, frame = stream.read()
        if not ret or frame is None:
            return None
        return frame

    def get_fresh_frame(
        self, camera_id: int, max_age_sec: float = 5.0
    ) -> Optional[np.ndarray]:
        """Return a recent frame only if it was received within max_age_sec."""
        stream = self.streams.get(camera_id)
        if stream is None:
            return None
        ret, frame = stream.read()
        if not ret or frame is None:
            return None
        age = stream.frame_age_sec()
        if age is None or age > max_age_sec:
            return None
        return frame

    def add_stream(self, config: Dict[str, Any]) -> Optional[StreamHandler]:
        """Open a stream for a camera joining the running set (LSO-130)."""
        camera_id = config.get('camera_id')
        if camera_id in self.streams:
            logger.debug(f"Stream for camera_id={camera_id} already open")
            return self.streams[camera_id]
        stream = StreamHandler(src=config['stream_url'], logger=logger)
        self.streams[camera_id] = stream
        stream.start()
        logger.info(
            f"Stream added: {config.get('camera_name', 'Unknown')} "
            f"(camera_id={camera_id}, connected={stream.connected})"
        )
        return stream

    def remove_stream(self, camera_id: int) -> None:
        """Stop and drop one camera's stream, leaving the others untouched."""
        stream = self.streams.pop(camera_id, None)
        if stream is None:
            logger.debug(f"No stream to remove for camera_id={camera_id}")
            return
        try:
            stream.stop()
        except Exception as e:
            logger.warning(f"Error stopping stream camera_id={camera_id}: {e}")
        writer = self.video_writers.pop(camera_id, None)
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        logger.info(f"Stream removed: camera_id={camera_id}")

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.stop_streams()
        self.release_writers()
