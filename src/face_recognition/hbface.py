"""Main face recognition system for processing video streams and recognizing faces."""

import os
import warnings
from typing import List, Optional, Union
import time
import cv2
from numpy.typing import NDArray
from loguru import logger
import datetime
import pytz  # For timezone support

# Local imports
from .engine import FaceEngine
from .system_setup import FaceSetup

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


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
        self.visualize = self.config.visualize
        self.entry_logger = self.config.entry_logger
        self.video_writers = self.config.video_writers
        self.FR_SLUG = os.getenv("FR_SLUG")
        self.client_slug = self.entry_logger.client_slug

        # Get timezone from config, default to UTC if not specified
        self.timezone = getattr(self.config, 'timezone', 'UTC')

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

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        except Exception as e:
            logger.error(f"Error during processing: {e}")

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

        # Log recognized persons
        for name, (_, appear_time, recognized, image) in persons_recognized.items():
            status = engine.args.cam_type
            if recognized == 'recognized':
                recorded = self.entry_logger.log_person_entry(name, status, appear_time)
                logger.info(f"Person recognized: {name}, recorded={recorded}, save_enabled={engine.args.save_recognized_frame}")
                if engine.args.save_recognized_frame and recorded:
                    logger.info(f"Calling save_recognized_frame for {name}")
                    self.save_recognized_frame(name, image, status)
            else:
                # Log unrecognized faces
                self.entry_logger.send_unrecognized_face(face = image, status=status)

        # Draw counting line if configured
        if engine.args.line_points:
            cv2.line(frame_annotated, engine.args.line_points[0], engine.args.line_points[1], (0, 255, 0), 3)

        # Add visualization of recognized entries
        self.entry_logger.visualize_entries(frame_annotated)

        # Add timestamp to the frame
        self._add_timestamp(frame_annotated)
        # self.entry_logger.`send_annotated_frame`(frame_annotated, engine.args.cam_type)


        return frame_annotated

    def _add_timestamp(self, frame: NDArray) -> None:
        """Add timestamp to the frame.

        Args:
            frame: Frame to add timestamp to
        """
        try:
            # Get current time in the configured timezone
            tz = pytz.timezone(self.engines[0].args.timezone)
            current_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

            # Add timestamp to the top-left corner
            text = f"{current_time}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            logger.error(f"Error adding timestamp: {e}")

    def _display_frames(self, frames: List[NDArray]) -> None:
        """Display processed frames if configured to show output.

        Args:
            frames: List of processed frames to display
        """
        if self.engines[0].args.show:
            combined_frame = self.visualize.concat_frames(*frames) if len(frames) > 1 else frames[0]

            if self.engines[0].args.show:
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
            for name, (_, appear_time, recognized, image) in persons_recognized.items():
                status = engine.args.cam_type
                if recognized == 'recognized':
                    recorded = self.entry_logger.log_person_entry(name, status, appear_time)
                    if self.engines[0].args.save_recognized_frame and recorded:
                        self.save_recognized_frame(name, image, status)
                else:
                    # Log unrecognized faces
                    self.entry_logger.send_unrecognized_face(face = image, status=status)

    def save_recognized_frame(self, name: str, image: NDArray, status: str) -> None:
        """Save recognized frame to the specified directory.

        Args:
            name: Name of the recognized person
            image: Image of the recognized face
            status: Status of the recognition (e.g., "entry", "exit")
        """
        try:
            # Use absolute path that matches Docker volume mount
            recognized_dir = os.path.join(f"/app/volumes/storage/{self.FR_SLUG}/data/{self.client_slug}", f"recognized_frames/")
            if not os.path.exists(recognized_dir):
                os.makedirs(recognized_dir)

            # Create status subdirectory if it doesn't exist
            status_dir = os.path.join(recognized_dir, status)
            if not os.path.exists(status_dir):
                os.makedirs(status_dir)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            save_path = os.path.join(status_dir, filename)

            # Check if image is valid
            if image is None or image.size == 0:
                logger.error(f"Cannot save - image is None or empty for {name}")
                return

            # Add error handling for cv2.imwrite
        cv2.imwrite(save_path, image)


    def _cleanup(self) -> None:
        """Cleanup resources and finalize the face recognition system."""
        # Release resources and perform final recognition on remaining tracks
        self._process_rest_tracks()

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