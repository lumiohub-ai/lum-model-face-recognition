import sys
import os
import signal
from typing import List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime

import cv2
import yaml
from numpy.typing import NDArray
# Set RTSP environment variable for OpenCV
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Add current directory to path
sys.path.append(os.curdir)

# Import local modules
from src.face_recognition.engine import FaceEngine
from src.face_recognition.utils import Visualization, StreamHandler, EntryLogger, ColorLogger

@dataclass
class CameraConfig:
    cam_type: str
    video_path: str
    roi: Optional[Tuple[int, int, int, int]] = None
    line_points: Optional[List[Tuple[int, int]]] = None

class HBFace:
    def __init__(
        self,
        cam_type: Optional[str] = None,
        cam_types: Optional[List[str]] = None,
        video_path: Optional[Union[str, List[str]]] = None,
        multi_camera: bool = True,
        config_path: str = "src/face_recognition/cfg/config.yaml",
        **kwargs
    ) -> None:
        self.multi_camera = multi_camera
        self.streams: List[StreamHandler] = []
        self.engines: List[FaceEngine] = []
        self.visualize = Visualization()
        self.logger = ColorLogger()
        self.entry_logger = EntryLogger()
        self.video_writers: List[Optional[cv2.VideoWriter]] = []

        if multi_camera:
            self._setup_multi_camera(cam_types, video_path, config_path, **kwargs)
        else:
            self._setup_single_camera(cam_type, video_path, config_path, **kwargs)

    def _setup_multi_camera(self, cam_types, video_paths, config_path, **kwargs):
        if not cam_types or not isinstance(video_paths, list):
            raise ValueError("For multi-camera setup, cam_types and video_paths (list) are required")

        self.cam_types = cam_types
        self.video_paths = video_paths
        roi_all = kwargs.pop('roi', None)
        line_points_all = kwargs.pop('line_points', None)

        os.makedirs("saved_videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (cam_type, vid_path) in enumerate(zip(cam_types, video_paths)):
            cam_kwargs = kwargs.copy()
            cam_kwargs['video_path'] = vid_path
            args = self._load_config(config_path, **cam_kwargs)

            args.roi = roi_all[i] if roi_all and i < len(roi_all) else None
            args.line_points = line_points_all[i] if line_points_all and i < len(line_points_all) else None
            args.logger = self.logger
            args.visualize = self.visualize
            args.cam_type = cam_type

            stream = StreamHandler(vid_path)
            self.streams.append(stream)
            self.engines.append(FaceEngine(args=args))

            if args.roi is not None:
                width = args.roi[2] - args.roi[0]
                height = args.roi[3] - args.roi[1]
            else:
                frame = stream.frame
                width, height = frame.shape[1], frame.shape[0]

            if args.save_video:
                output_path = f"saved_videos/{cam_type}_{i}_{timestamp}.avi"
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
                self.video_writers.append(writer)
            else:
                self.video_writers.append(None)

        self._show_config(self.engines[0].args)
        self._show_config(self.engines[1].args)

    def _setup_single_camera(self, cam_type, video_path, config_path, **kwargs):
        self.cam_type = cam_type or "Camera"
        args = self._load_config(config_path, video_path=video_path, **kwargs)
        args.roi = kwargs.get('roi')
        args.line_points = kwargs.get('line_points')
        args.cam_type = self.cam_type
        args.logger = self.logger
        args.visualize = self.visualize

        stream = StreamHandler(args.video_path)
        self.streams = [stream]
        self.engines = [FaceEngine(args=args)]

        os.makedirs("saved_videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.roi is not None:
            width = args.roi[2] - args.roi[0]
            height = args.roi[3] - args.roi[1]
        else:
            frame = stream.frame
            width, height = frame.shape[1], frame.shape[0]

        if args.save_video:
            output_path = f"saved_videos/{self.cam_type}_{timestamp}.avi"
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
            self.video_writers = [writer]
        else:
            self.video_writers = [None]

        self._show_config(args)

    def _load_config(self, config_path: str, **kwargs) -> Any:
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            config = {}

        args = type('Args', (), {})()
        for key, value in config.items():
            if key not in kwargs:
                setattr(args, key, value)
        for key, value in kwargs.items():
            setattr(args, key, value)

        # Default save_video to True if not provided
        if not hasattr(args, "save_video"):
            args.save_video = True

        return args

    def _show_config(self, args: Any) -> None:
        self.logger.info("\n=== Configuration Settings ===")
        for key, value in args.__dict__.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("===========================\n")

    def run(self) -> None:
        try:
            for stream in self.streams:
                if not stream.is_video:
                    stream.start()

            cam_count = len(self.streams)
            frame_nums = [0] * cam_count
            frames = [None] * cam_count

            while True:
                all_frames_read = True
                for i, stream in enumerate(self.streams):
                    ret, frame = stream.read()
                    if not ret:
                        all_frames_read = False
                        break
                    frames[i] = frame
                    frame_nums[i] += 1

                if not all_frames_read:
                    break

                annotated_frames = self._process_frames(frames, frame_nums)

                # Write frames to video if enabled
                for i, frame in enumerate(annotated_frames):
                    if self.video_writers[i] is not None:
                        self.video_writers[i].write(frame)

                self._display_frames(annotated_frames)

                if cv2.waitKey(1) == 27:
                    break

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Cleaning up...")
        finally:
            self._cleanup()

    def _process_frames(self, frames: List[NDArray], frame_nums: List[int]) -> List[NDArray]:
        annotated_frames = []

        for i, frame in enumerate(frames):
            cam_type = self.cam_types[i] if self.multi_camera else self.cam_type
            engine = self.engines[i]
            frame_num = frame_nums[i]
            last_frame = (frame_num == self.streams[i].last_frame)

            roi = engine.args.roi
            if roi:
                x, y, x2, y2 = roi
                w, h = x2 - x, y2 - y
                frame = frame[y:y+h, x:x+w]

            detections = engine.track(frame)

            if detections is None:
                frame = self._draw_line(frame, engine.args.line_points)
                annotated_frames.append(frame)
                continue

            annotated_frame = engine.process_detections(frame, frame_num)
            recognized_persons = engine.recognize_tracks(detections, last_frame=last_frame)

            for name, (track_id, appear_time) in recognized_persons.items():
                self.entry_logger.log_person_entry(name, cam_type, track_id, appear_time)


            if engine.args.line_points is not None:
                annotated_frame = self._draw_line(annotated_frame, engine.args.line_points)
                
            cv2.putText(
                annotated_frame,
                f"Camera: {cam_type}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            self.entry_logger.visualize_entries(annotated_frame)
            annotated_frames.append(annotated_frame)

        return annotated_frames

    def _draw_line(self, frame: NDArray, line_points: Optional[List[Tuple[int, int]]]) -> NDArray:
        if line_points is not None:
            cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)
        return frame

    def _display_frames(self, annotated_frames: List[NDArray]) -> None:
        if not self.engines[0].args.show:
            return

        if self.multi_camera and len(annotated_frames) > 1:
            display_frame = self.visualize.concat_frames(
                annotated_frames[0],
                annotated_frames[1],
                mode="horizontal"
            )
            display_frame = cv2.resize(display_frame, (1280, 720))
            self.visualize.display(display_frame, window_name="Multi-Camera System")
        else:
            self.visualize.display(
                annotated_frames[0],
                window_name=self.cam_types[0] if self.multi_camera else self.cam_type
            )

    def _cleanup(self) -> None:
        for stream in self.streams:
            stream.stop()
        for writer in self.video_writers:
            if writer is not None:
                writer.release()
        cv2.destroyAllWindows()
        self.logger.info("Exiting... Successfully processed all frames and saved videos.")
