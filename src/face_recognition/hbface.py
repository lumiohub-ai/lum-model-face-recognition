"""Main face recognition system for processing video streams and recognizing faces."""

import os
import warnings
from typing import List, Optional, Union
import time
import csv
from collections import deque
import cv2
import numpy as np
import psutil
from numpy.typing import NDArray
from loguru import logger
import datetime
import pytz  # For timezone support

# Local imports
from .core.engine import FaceEngine
from .system_setup import FaceSetup
from .video.frame_processor import FrameProcessor

warnings.filterwarnings("ignore", category=FutureWarning)
# Configure RTSP options for better stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "timeout;60000000|"  # 60 seconds in microseconds
    "stimeout;60000000"  # socket timeout
)


# Main class that merges configuration and processing
class HBFace:
    """Main face recognition system for processing video streams and identifying people.

    This class integrates all components of the face recognition system, including
    video handling, face detection and recognition, and result logging.
    """
    def __init__(self, cam_types: Optional[List[str]] = None, video_path: Optional[Union[str, List[str]]] = None,
                 multi_camera: bool = True, config_path: str = "configs/config.yaml", **kwargs) -> None:
        """Initialize the face recognition system with optional camera types and video paths.

        Args:
            cam_types: List of camera types (e.g., "entry", "exit")
            video_path: Path(s) to video file(s) or stream URL(s)
            multi_camera: Whether to process multiple cameras simultaneously
            config_path: Path to the configuration file
            **kwargs: Additional configuration parameters
        """
        # Initialize configuration
        self.config = FaceSetup(cam_types, video_path, multi_camera, config_path, **kwargs)

        # For easier access
        self.streams = self.config.streams
        self.engines = self.config.engines
        self.entry_logger = self.config.entry_logger
        self.video_writers = self.config.video_writers
        self.FR_SLUG = os.getenv("FR_SLUG")
        self.client_slug = self.entry_logger.client_slug

        # Get timezone from config, default to UTC if not specified
        self.timezone = getattr(self.config, 'timezone', 'UTC')

        # Initialize frame processor
        self.frame_processor = FrameProcessor(self.entry_logger, self.timezone)

        # FPS and performance monitoring
        self.fps_history = deque(maxlen=120)  # Store last 120 FPS readings (2 min at 1fps)
        self.fps_log_interval = 3600  # Write to file every 1 hour (seconds)
        self.fps_console_interval = 600  # Print to console every 10 minutes (seconds)
        self.fps_check_interval = 30  # Calculate FPS every 30 frames
        self.fps_threshold_warning = 15  # Warn if FPS < 15
        self.fps_threshold_critical = 10  # Critical if FPS < 10
        self.last_fps_log_time = time.time()
        self.last_fps_console_time = time.time()
        self.last_fps_check_time = time.time()

        # Setup FPS log file
        log_dir = os.path.join(os.getcwd(), f'/app/volumes/storage/{self.FR_SLUG}/logs')
        os.makedirs(log_dir, exist_ok=True)
        self.fps_log_path = os.path.join(log_dir, f'{self.client_slug}_fps_performance.csv')

        # Create CSV header if file doesn't exist
        if not os.path.exists(self.fps_log_path):
            with open(self.fps_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'avg_fps', 'min_fps', 'max_fps', 'memory_mb', 'status'])

        logger.info(f"📊 FPS monitoring enabled. Log file: {self.fps_log_path}")

    def run(self) -> None:
        """Run the face recognition system and process video streams."""
        try:
            new_users = self.entry_logger.new_users
            deleted_users = self.entry_logger.deleted_users

            # Start streams for non-video sources
            for i, stream in enumerate(self.streams):
                if not stream.is_video:
                    stream.start()

                if self.engines[i].args.production:
                    self.engines[i].update_database(new_users, deleted_users)

            frame_nums = [0] * len(self.streams)
            total_frames = 0
            start_time = time.time()

            while True:
                frames = []
                for i, stream in enumerate(self.streams):
                    ret, frame = stream.read()
                    if not ret:
                        logger.info("End of video stream reached")
                        return
                    frame_nums[i] += 1
                    total_frames += 1
                    frames.append(frame)

                annotated_frames = self._process_frames(frames, frame_nums)

                self._save_frames(annotated_frames)
                self._display_frames(annotated_frames)

                # FPS monitoring (every fps_check_interval frames)
                if total_frames % self.fps_check_interval == 0:
                    self._check_and_log_fps()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        except cv2.error as e:
            logger.error(f"OpenCV error during processing: {e}", exc_info=True)
            raise

        except OSError as e:
            logger.error(f"File system error during processing: {e}", exc_info=True)
            raise

        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}", exc_info=True)
            raise

        finally:
            self._cleanup()
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Average FPS: {avg_fps:.2f}")

    def _process_frames(self, frames: List[NDArray], frame_nums: List[int]) -> List[NDArray]:
        """Process multiple frames for face detection, tracking and recognition.

        Args:
            frames: List of video frames to process
            frame_nums: List of frame numbers corresponding to each frame

        Returns:
            List of annotated frames with visualization
        """
        return [self._process_single_frame(frame, frame_nums[i], self.engines[i])
                for i, frame in enumerate(frames)]

    def _process_single_frame(self, frame: NDArray, frame_num: int, engine: FaceEngine) -> NDArray:
        """Process a single frame for face detection, tracking and recognition.

        Args:
            frame: Video frame to process
            frame_num: Frame number in the sequence
            engine: FaceEngine instance to use for processing

        Returns:
            Annotated frame with visualization
        """
        roi = engine.args.roi
        frame_cropped = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame

        # Track faces in the current frame
        active_tracks, removed_tracks = engine.track(frame_cropped)

        # Prune tracks that have been active for too long and add them to the removed list
        expired_tracks = engine.prune_long_lived_tracks()
        removed_tracks.extend(expired_tracks)

        frame_annotated = engine.visualize_tracks(frame_cropped)

        if active_tracks:
            # Process active tracks and recognize faces in removed tracks
            engine.process_active_tracks(active_tracks, frame_cropped, frame_num)

        persons_recognized = engine.recognize_removed_tracks(removed_tracks, last_frame=False)

        # Process recognition results using FrameProcessor
        self.frame_processor.process_recognition_results(
            persons_recognized,
            camera_type=engine.args.cam_type,
            camera_name=engine.args.camera_name,
            camera_id=engine.args.camera_id,
            save_recognized_callback=self.save_recognized_frame,
            save_recognized_enabled=engine.args.save_recognized_frame
        )

        # Annotate frame with visualizations
        self.frame_processor.annotate_frame(
            frame_annotated,
            line_points=engine.args.line_points,
            add_timestamp=True,
            add_entries=True
        )

        return frame_annotated

    def _display_frames(self, frames: List[NDArray]) -> None:
        """Display processed frames if configured to show output.

        Args:
            frames: List of processed frames to display
        """
        if self.engines[0].args.show:
            # Simple concatenation for multiple frames (horizontal)
            if len(frames) > 1:
                combined_frame = cv2.hconcat(frames)
            else:
                combined_frame = frames[0]

            cv2.imshow("Face Recognition", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("ESC key pressed, exiting")

    def _check_time_interval(self) -> bool:
        """Check if current time is within the defined recording intervals.

        Program must save the frames from 7 AM to 9 AM and 5 PM to 7 PM
        every day with Uzbekistan time.

        Returns:
            True if current time is within recording intervals, False otherwise
        """
        tz = pytz.timezone('UTC')  # Use 'Asia/Seoul' for Korea timezone
        current_time = datetime.datetime.now(tz)
        start_morning = current_time.replace(hour=7, minute=0, second=0, microsecond=0)
        end_morning = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
        start_evening = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
        end_evening = current_time.replace(hour=19, minute=0, second=0, microsecond=0)

        return (start_morning <= current_time <= end_morning) or (start_evening <= current_time <= end_evening)

    def _save_frames(self, frames: List[NDArray]) -> None:
        """Save processed frames to video files if enabled.

        Args:
            frames: List of processed frames to save
        """
        if not self.engines[0].args.record_always:
            if not self._check_time_interval():
                return

        if self.engines[0].args.save_video:
            for i, frame in enumerate(frames):
                if self.video_writers[i]:
                    self.video_writers[i].write(frame)

    def _process_rest_tracks(self) -> None:
        """Process remaining tracks after video stream ends."""
        for engine in self.engines:
            persons_recognized = engine.recognize_removed_tracks([], last_frame=True)
            # Log any final recognized persons
            for name, (track_id, appear_time, recognized, image, recognition_info) in persons_recognized.items():
                status = engine.args.cam_type
                camera_name = engine.args.camera_name
                camera_id = engine.args.camera_id

                if recognized == 'recognized':
                    recorded = self.entry_logger.log_person_entry(name, status, appear_time, camera_name, camera_id)
                    if self.engines[0].args.save_recognized_frame and recorded:
                        self.save_recognized_frame(name, image, status)
                elif recognized == 'unrecognized':
                    # Send to API only if image is valid (None check)
                    if image is not None:
                        self.entry_logger.send_unrecognized_face(face=image, status=status, camera_id=camera_id)

    def save_recognized_frame(self, name: str, image: NDArray, status: str) -> None:
        """Save recognized frame to the specified directory.

        Args:
            name: Name of the recognized person
            image: Image of the recognized face
            status: Status of the recognition (e.g., "entry", "exit")
        """
        # Use storage configuration for base path (TODO: refactor to use StorageConfig from config.py)
        storage_base_path = os.getenv("STORAGE_BASE_PATH", "/app/volumes/storage")
        recognized_dir = os.path.join(storage_base_path, self.FR_SLUG, "data", self.client_slug, "recognized_frames")

        try:
            if not os.path.exists(recognized_dir):
                os.makedirs(recognized_dir, exist_ok=True)

            # Create status subdirectory if it doesn't exist
            status_dir = os.path.join(recognized_dir, status)
            if not os.path.exists(status_dir):
                os.makedirs(status_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            save_path = os.path.join(status_dir, filename)

            # Add error handling for cv2.imwrite
            success = cv2.imwrite(save_path, image)
            if not success:
                logger.error(f"Failed to save recognized frame for {name} at {save_path}")
            else:
                logger.debug(f"Saved recognized frame for {name} at {save_path}")
        except OSError as e:
            logger.error(f"Error saving recognized frame for {name}: {e}")

    def _check_and_log_fps(self) -> None:
        """Check current FPS and log to file if needed."""
        current_time = time.time()
        elapsed = current_time - self.last_fps_check_time

        if elapsed > 0:
            current_fps = self.fps_check_interval / elapsed
            self.fps_history.append(current_fps)
            self.last_fps_check_time = current_time

            # Calculate statistics
            if len(self.fps_history) > 0:
                avg_fps = np.mean(self.fps_history)
                min_fps = np.min(self.fps_history)
                max_fps = np.max(self.fps_history)

                # Get memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                # Determine status and log accordingly
                time_since_console = current_time - self.last_fps_console_time

                if avg_fps < self.fps_threshold_critical:
                    status = "critical"
                    # CRITICAL: Always print immediately
                    logger.error(f"🔴 CRITICAL: FPS={avg_fps:.1f} Memory={memory_mb:.0f}MB - RESTART RECOMMENDED")
                    # Write immediately to file
                    self._write_fps_to_file(avg_fps, min_fps, max_fps, memory_mb, status)
                    self.last_fps_console_time = current_time

                elif avg_fps < self.fps_threshold_warning:
                    status = "warning"
                    # WARNING: Always print immediately
                    logger.warning(f"⚠️  WARNING: FPS={avg_fps:.1f} Memory={memory_mb:.0f}MB - Performance degrading")
                    # Write immediately to file
                    self._write_fps_to_file(avg_fps, min_fps, max_fps, memory_mb, status)
                    self.last_fps_console_time = current_time

                else:
                    status = "healthy"
                    # HEALTHY: Only print every 10 minutes
                    if time_since_console >= self.fps_console_interval:
                        logger.info(f"✅ FPS: {avg_fps:.1f} Memory: {memory_mb:.0f}MB - System healthy")
                        self.last_fps_console_time = current_time

                # Check if it's time for hourly file log (for healthy status)
                time_since_last_log = current_time - self.last_fps_log_time
                if time_since_last_log >= self.fps_log_interval:
                    self._write_fps_to_file(avg_fps, min_fps, max_fps, memory_mb, status)
                    self.last_fps_log_time = current_time

    def _write_fps_to_file(self, avg_fps: float, min_fps: float, max_fps: float,
                           memory_mb: float, status: str) -> None:
        """Write FPS metrics to CSV file.

        Args:
            avg_fps: Average FPS
            min_fps: Minimum FPS
            max_fps: Maximum FPS
            memory_mb: Memory usage in MB
            status: Status (healthy/warning/critical)
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.fps_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    f"{avg_fps:.2f}",
                    f"{min_fps:.2f}",
                    f"{max_fps:.2f}",
                    f"{memory_mb:.1f}",
                    status
                ])
            logger.info(f"📊 FPS log written: {avg_fps:.1f} FPS, {memory_mb:.0f} MB, Status: {status}")
        except Exception as e:
            logger.error(f"Failed to write FPS log: {e}")

    def _cleanup(self) -> None:
        """Cleanup resources and finalize the face recognition system."""
        # Release resources and perform final recognition on remaining tracks
        self._process_rest_tracks()

        # Stop Redis subscribers (if using pgvector mode)
        for engine in self.engines:
            if hasattr(engine, 'redis_subscriber') and engine.redis_subscriber is not None:
                try:
                    logger.info("Stopping Redis subscriber...")
                    engine.redis_subscriber.stop()
                    logger.info("Redis subscriber stopped successfully")
                except Exception as e:
                    logger.warning(f"Error stopping Redis subscriber: {e}")

        # Stop streams and release video writers
        for stream in self.streams:
            stream.stop()
        for writer in self.video_writers:
            if writer:
                writer.release()

        if self.engines[0].args.show:
            cv2.destroyAllWindows()

        text = self.entry_logger.save_status_info()
        logger.info(f"{text}")