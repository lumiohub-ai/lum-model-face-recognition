import warnings
from typing import List, Optional, Union
import time
import cv2
from numpy.typing import NDArray
from loguru import logger

# Local imports
from .engine import FaceEngine
from .system_setup import FaceSetup

warnings.filterwarnings("ignore", category=FutureWarning)


# Main class that merges configuration and processing
class HBFace:
    def __init__(self, cam_types: Optional[List[str]] = None, video_path: Optional[Union[str, List[str]]] = None,
                 multi_camera: bool = True, config_path: str = "configs/config.yaml", **kwargs) -> None:
        """Initialize the face recognition system with optional camera types and video paths."""
        # Initialize configuration
        self.config = FaceSetup(cam_types, video_path, multi_camera, config_path, **kwargs)
        
        # For easier access
        self.streams = self.config.streams
        self.engines = self.config.engines
        self.visualize = self.config.visualize
        self.entry_logger = self.config.entry_logger
        self.video_writers = self.config.video_writers
    
    def setup_cameras(self, cam_types: Optional[List[str]], video_paths: Optional[Union[str, List[str]]], **kwargs) -> None:
        """Set up camera streams based on configuration."""
        self.config.setup_cameras(cam_types, video_paths, **kwargs)
    
    def run(self) -> None:
        """Run the face recognition system and process video streams."""
        try:
            # Start streams for non-video sources
            for stream in self.streams:
                if not stream.is_video:
                    stream.start()

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

                if cv2.waitKey(1) == 27:  # ESC key
                    logger.info("ESC key pressed, exiting")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self._cleanup()
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0 
            logger.info(f"Average FPS: {avg_fps:.2f}")

    def _process_frames(self, frames: List[NDArray], frame_nums: List[int]) -> List[NDArray]:
        """Process multiple frames for face detection, tracking and recognition."""
        return [self._process_single_frame(frame, frame_nums[i], self.engines[i]) 
                for i, frame in enumerate(frames)]

    def _process_single_frame(self, frame: NDArray, frame_num: int, engine: FaceEngine) -> NDArray:
        """Process a single frame for face detection, tracking and recognition."""
        roi = engine.args.roi
        frame_cropped = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame

        # Track faces in the current frame
        active_tracks, removed_tracks = engine.track(frame_cropped)
        frame_annotated = engine.visualize_tracks(frame_cropped)

        if active_tracks:
            # Process active tracks and recognize faces in removed tracks
            engine.process_active_tracks(active_tracks, frame_num)
            persons_recognized = engine.recognize_removed_tracks(removed_tracks, last_frame=False)
            
            # Log recognized persons
            for name, (track_id, appear_time) in persons_recognized.items():
                status = engine.args.cam_type
                
                log_message = f"{name} -> {status} -> {appear_time.strftime('%H:%M:%S')}, track_id: {track_id}"
                logger.debug(log_message)

                self.entry_logger.log_person_entry(name, status, appear_time)
        
        else:
            frame_annotated = frame_cropped

        # Draw counting line if configured
        if engine.args.line_points:
            cv2.line(frame_annotated, engine.args.line_points[0], engine.args.line_points[1], (0, 255, 0), 2)

        # Add visualization of recognized entries
        self.entry_logger.visualize_entries(frame_annotated)

        return frame_annotated

    def _display_frames(self, frames: List[NDArray]) -> None:
        """Display processed frames if configured to show output."""
        if self.engines[0].args.show:
            combined_frame = self.visualize.concat_frames(*frames) if len(frames) > 1 else frames[0]
            
            if self.engines[0].args.show:
                cv2.imshow("Face Recognition", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("ESC key pressed, exiting")
                    return
    
    def _save_frames(self, frames: List[NDArray]) -> None:
        """Save processed frames to video files if enabled."""
        if self.engines[0].args.save_video:      
            for i, frame in enumerate(frames):
                if self.video_writers[i]:
                    self.video_writers[i].write(frame)

    def _process_rest_tracks(self) -> None:
        """Process remaining tracks after video stream ends."""
        for engine in self.engines:
            persons_recognized = engine.recognize_removed_tracks([], last_frame=True)
            # Log any final recognized persons
            for name, (track_id, appear_time) in persons_recognized.items():
                status = engine.args.cam_type
                log_message = f"{name} -> {status} -> {appear_time.strftime('%H:%M:%S')}, track_id: {track_id}"
                logger.info(log_message)
                self.entry_logger.log_person_entry(name, status, appear_time)

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
        
        logger.info("Cleanup complete")