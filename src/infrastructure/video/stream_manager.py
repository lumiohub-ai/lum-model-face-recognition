"""Stream manager for handling video streams and output writers.

This module manages:
- Multiple video streams (RTSP, file, webcam)
- Video output writers for recording
- Stream lifecycle (start, read, stop)
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

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
        self.streams: List[StreamHandler] = []
        self.video_writers: List[Optional[cv2.VideoWriter]] = []
        self.frame_nums: List[int] = []
        self._initialized = False
        self._last_write_time: List[float] = []
        self._write_interval: float = 1.0 / 20

    def init_streams(self) -> List[StreamHandler]:
        """Initialize stream handlers for all cameras.

        Returns:
            List of initialized StreamHandler objects
        """
        self.streams = []
        self.frame_nums = []

        for config in self.camera_configs:
            stream = StreamHandler(
                src=config['stream_url'],
                logger=logger
            )
            self.streams.append(stream)
            self.frame_nums.append(0)

        self._initialized = True
        return self.streams

    def init_video_writers(self, output_dir: str) -> List[Optional[cv2.VideoWriter]]:
        """Initialize video writers for saving output.

        Args:
            output_dir: Directory to save video files

        Returns:
            List of VideoWriter objects (or None for failed writers)
        """
        self.video_writers = []

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
            logger.error(f"Output directory not writable: {output_dir} - {e}")
            return self.video_writers

        self._last_write_time = []

        for i, config in enumerate(self.camera_configs):
            camera_name = config['camera_name'].replace(' ', '_')
            status = config.get('cam_type', 'IN').upper()
            filename = f"{output_dir}/{status}_{camera_name}_{date}_{time_str}.mp4"

            # Get frame dimensions and fps from stream
            w, h, fps = 1920, 1080, 20
            if i < len(self.streams):
                stream = self.streams[i]
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
            self.video_writers.append(writer)
            self._last_write_time.append(0.0)

        self._write_interval = 1.0 / (fps if fps > 0 else 20)
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
            logger.error(f"Exception initializing video writer: {filename} - {e}")
            return None

    def start_streams(self) -> None:
        """Start all non-video file streams (RTSP, webcam)."""
        if not self._initialized:
            self.init_streams()

        for stream in self.streams:
            if not stream.is_video:
                stream.start()

        logger.info("Started video streams")

    def read_all_frames(self) -> List[Tuple[int, np.ndarray, int]]:
        """Read frames from all cameras.

        Returns:
            List of tuples: (camera_idx, frame, frame_num)
            Only includes successfully read frames
        """
        frames = []

        for i, stream in enumerate(self.streams):
            ret, frame = stream.read()
            if not ret:
                logger.warning(f"Camera {i}: Failed to read frame")
                continue

            self.frame_nums[i] += 1
            frames.append((i, frame, self.frame_nums[i]))

        return frames

    def write_frames(self, annotated_frames: List[Tuple[int, np.ndarray]]) -> None:
        """Write annotated frames to video files.

        Throttles writes to match the declared fps so playback duration
        matches real-world recording time.

        Args:
            annotated_frames: List of (camera_idx, frame) tuples
        """
        now = time.monotonic()
        for camera_idx, frame in annotated_frames:
            if camera_idx >= len(self.video_writers):
                continue
            writer = self.video_writers[camera_idx]
            if writer is None:
                continue
            # Skip if not enough time has passed since last write
            if now - self._last_write_time[camera_idx] < self._write_interval:
                continue
            writer.write(frame)
            self._last_write_time[camera_idx] = now

    def stop_streams(self) -> None:
        """Stop all streams."""
        for stream in self.streams:
            stream.stop()
        logger.info("Stopped all video streams")

    def release_writers(self) -> None:
        """Release all video writers."""
        for writer in self.video_writers:
            if writer is not None:
                writer.release()
        self.video_writers = []
        logger.info("Released all video writers")

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.stop_streams()
        self.release_writers()
