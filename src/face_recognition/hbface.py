import sys
import os
from typing import List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime

import cv2
import yaml
from numpy.typing import NDArray

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
sys.path.append(os.curdir)

from src.face_recognition.engine import FaceEngine
from src.face_recognition.utils import Visualization, StreamHandler, EntryLogger, ColorLogger

class HBFace:
    def __init__(self, cam_types: Optional[List[str]] = None, video_path: Optional[Union[str, List[str]]] = None,
                 multi_camera: bool = True, config_path: str = "src/face_recognition/cfg/config.yaml", **kwargs) -> None:
        self.multi_camera = multi_camera
        self.streams: List[StreamHandler] = []
        self.engines: List[FaceEngine] = []
        self.visualize = Visualization()
        self.logger = ColorLogger(log_file=kwargs.get('log_file', None))
        self.entry_logger = EntryLogger(backend_url=kwargs.get('backend_url', 'http://backend:4000/graphql'))
        self.video_writers: List[Optional[cv2.VideoWriter]] = []
        self.config_path = config_path

        self.setup_cameras(cam_types, video_path, **kwargs)

    def setup_cameras(self, cam_types, video_paths, **kwargs):
        """ Setup cameras based on the provided types and paths. """
        if self.multi_camera:
            self.setup_multi_camera(cam_types, video_paths, **kwargs)
        else:
            self.setup_single_camera(cam_types, video_paths, **kwargs)

    def setup_multi_camera(self, cam_types, video_paths, **kwargs):
        """ Setup multiple cameras for multi-camera mode. """
        if not cam_types or not isinstance(video_paths, list):
            raise ValueError("Multi-camera setup requires cam_types and a list of video_paths.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (cam_type, vid_path) in enumerate(zip(cam_types, video_paths)):
            args = self.load_config(video_path=vid_path, cam_type=cam_type, **kwargs)
            self.initialize_camera(args, vid_path, cam_type, timestamp, i)

    def setup_single_camera(self, cam_type, video_path, **kwargs):
        """ Setup a single camera for single-camera mode. """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args = self.load_config(video_path=video_path, cam_type=cam_type, **kwargs)
        self.initialize_camera(args, video_path, cam_type, timestamp)

    def initialize_camera(self, args, vid_path, cam_type, timestamp, index=None):
        """ Initialize a camera with the given arguments. """
        args.logger = self.logger
        args.visualize = self.visualize
        args.cam_type = cam_type
        args.roi = args.roi[index] if index is not None else args.roi
        args.line_points = args.line_points[index] if index is not None else args.line_points

        stream = StreamHandler(vid_path)
        self.streams.append(stream)
        self.engines.append(FaceEngine(args=args))

        frame = stream.frame
        width, height = frame.shape[1], frame.shape[0]

        if args.roi:
            roi = args.roi
            width, height = roi[2] - roi[0], roi[3] - roi[1]

        if args.save_video:
            os.makedirs("saved_videos", exist_ok=True)
            video_filename = f"saved_videos/{cam_type}_{index if index is not None else ''}_{timestamp}.avi"
            writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
            self.video_writers.append(writer)
        else:
            self.video_writers.append(None)

        self.show_config(args)

    def load_config(self, **kwargs) -> Any:
        """ Load configuration from YAML file and override with provided arguments. """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            config = {}

        args = type('Args', (), {})()
        for key, value in {**config, **kwargs}.items():
            setattr(args, key, value)

        args.save_video = getattr(args, "save_video", True)

        return args

    def show_config(self, args: Any) -> None:
        """ Display the configuration settings. """
        self.logger.info("\n=== Configuration Settings ===")
        for key, value in args.__dict__.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("===========================\n")

    def run(self) -> None:
        """ Start the face recognition process. """
        try:
            for stream in self.streams:
                if not stream.is_video:
                    stream.start()

            frame_nums = [0] * len(self.streams)

            while True:
                frames = []
                for i, stream in enumerate(self.streams):
                    ret, frame = stream.read()
                    if not ret:
                        return
                    frame_nums[i] += 1
                    frames.append(frame)

                annotated_frames = self.process_frames(frames, frame_nums)


                self.save_frames(annotated_frames)
                self.display_frames(annotated_frames)

                if cv2.waitKey(1) == 27:
                    break

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")

        finally:
            self.cleanup()

    def process_frames(self, frames: List[NDArray], frame_nums: List[int]) -> List[NDArray]:
        """ Process frames from multiple cameras. """
        return [self.process_single_frame(frame, frame_nums[i], self.engines[i]) for i, frame in enumerate(frames)]

    def process_single_frame(self, frame: NDArray, frame_num: int, engine: FaceEngine) -> NDArray:
        """ Process a single frame for face recognition. """
        roi = engine.args.roi if engine.args.roi else None
        frame_cropped = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame

        current_dets, removed_tracks = engine.track(frame_cropped)

        if current_dets:
            frame_annotated = engine.process_detections(current_dets, frame_cropped, frame_num)
            recognized = engine.recognize_tracks(removed_tracks, frame_num == self.streams[0].last_frame)
            
            for name, (track_id, appear_time) in recognized.items():
                self.entry_logger.log_person_entry(name, engine.args.cam_type, track_id, appear_time)
        else:
            frame_annotated = frame_cropped

        if engine.args.line_points:
            cv2.line(frame_annotated, engine.args.line_points[0], engine.args.line_points[1], (0, 255, 0), 2)

        return frame_annotated

    def display_frames(self, frames: List[NDArray]) -> None:
        """ Display frames from multiple cameras. """
        if self.engines[0].args.show:
            combined_frame = self.visualize.concat_frames(*frames) if len(frames) > 1 else frames[0]
            self.visualize.display(combined_frame, window_name="Camera Feed")
    
    def save_frames(self, frames: List[NDArray]) -> None:
        """ Save frames to video files if enabled. """
        if self.engines[0].args.save_video:      
            for i, frame in enumerate(frames):
                if self.video_writers[i]:
                    self.video_writers[i].write(frame)

    def cleanup(self) -> None:
        """ Cleanup resources. """
        for stream in self.streams:
            stream.stop()
        for writer in self.video_writers:
            if writer:
                writer.release()
        cv2.destroyAllWindows()

        self.logger.info("Cleanup complete.")
