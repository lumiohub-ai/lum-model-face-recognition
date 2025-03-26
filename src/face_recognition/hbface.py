import sys
import os
import threading
sys.path.append(os.curdir)

import cv2
import yaml
import numpy as np

from src.face_recognition.engine import FaceEngine
from src.face_recognition.utils import Visualization, StreamHandler, EntryLogger, ColorLogger

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class HBFace:
    def __init__(self, cam_type=None, cam_types=None,
                 video_path=None, multi_camera=False,
                 config_path="src/face_recognition/cfg/config.yaml", **kwargs) -> None:
        
        self.multi_camera = multi_camera
        
        self.streams = []
        self.engines = []

        self.visualize = Visualization()
        self.logger = ColorLogger()
        self.entry_logger = EntryLogger()

        self.recognized_names = set()
    
        # For multi-camera setup
        if multi_camera and cam_types and isinstance(video_path, list):
            self.cam_types = cam_types
            self.video_paths = video_path
            
            # Initialize each camera
            for i, (cam_type, vid_path) in enumerate(zip(cam_types, video_path)):
                # Create a copy of kwargs for this camera
                cam_kwargs = kwargs.copy()
                cam_kwargs['video_path'] = vid_path
                
                # Load config
                args = self.load_cfgs(config_path, **cam_kwargs)
                
                # Set camera-specific args
                args.cam_type = cam_type
                args.logger = self.logger
                args.entry_logger = self.entry_logger
                args.visualize = self.visualize    

                # Create stream and entry logger for this camera
                self.streams.append(StreamHandler(vid_path))
                self.engines.append(FaceEngine(args=args))         
                
            self.show_configs(self.engines[0].args)
            
        # For single camera setup
        else:
            self.cam_type = cam_type or "Camera"
            args = self.load_cfgs(config_path, **kwargs)
            
            self.streams = [StreamHandler(args.video_path)]
            self.entry_loggers = [EntryLogger()]
            
            # Add logger to the args
            args.cam_type = self.cam_type
            args.logger = self.logger
            args.entry_logger = self.entry_loggers[0]
            args.visualize = self.visualize
            
            engine = FaceEngine(args=args)
            self.engines = [engine]
            self.show_configs(args)

    def show_configs(self, args) -> None:
        """Display configuration settings."""
        print("\n=== Configuration Settings ===")
        for key, value in args.__dict__.items():
            self.logger.info(f"{key}: {value}")
        print("===========================\n")

    def load_cfgs(self, config_path, **kwargs):
        # Load config from yaml file
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception:
            self.logger.error("Error loading config file.")
            config = {}
        
        # Create args object to store all parameters
        args = type('Args', (), {})()
        
        # Merge config with kwargs (kwargs take precedence)
        for key, value in config.items():
            if key not in kwargs:  # Only set from config if not explicitly provided
                setattr(args, key, value)

        # Add explicitly provided parameters
        for key, value in kwargs.items():
            setattr(args, key, value)
            
        return args
        
    def run(self) -> None:
        # Start all streams if they're not video files
        for stream in self.streams:
            if not stream.is_video:
                stream.start()
        
        # Get camera count
        cam_count = len(self.streams)
        
        # Initialize frame counters and other tracking variables
        frame_nums = [0] * cam_count
        frames = [None] * cam_count
        
        while True:
            # Read frames from all cameras
            all_frames_read = True
            
            for i, stream in enumerate(self.streams):
                ret, frame = stream.read()
                if not ret:
                    all_frames_read = False
                    break
                    
                frames[i] = frame
                frame_nums[i] += 1
            
            # Exit if any camera fails to provide a frame
            if not all_frames_read:
                break
                
            # Process each camera's frame
            annotated_frames = []
            
            for i, frame in enumerate(frames):
                # Get camera info
                cam_type = self.cam_types[i] if self.multi_camera else self.cam_type

                engine = self.engines[i]
                frame_num = frame_nums[i]
                last_frame = (frame_num == self.streams[i].last_frame)
                
                # Track faces in the frame using this camera's dedicated engine
                detections = engine.track(frame)
                
                if detections is None:
                    annotated_frames.append(frame)
                    continue
                
                # Process detections
                annotated_frame = engine.process_detections(frame, frame_num)
                
                # Process removed tracks and recognize faces
                recognized_persons = engine.recognize_tracks(detections, last_frame=last_frame)
                
                for name, track_id in recognized_persons.items():
                    self.entry_logger.log_person_entry(name, cam_type, track_id)
                    self.recognized_names.add(name)
                
                # Draw line points
                if engine.args.line_points is not None:
                    cv2.line(annotated_frame, engine.args.line_points[0], 
                            engine.args.line_points[1], (0, 255, 0), 2)
                
                # Add camera label to the frame
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
            
            # Display the frames
            if self.engines[0].args.show:  # Using first engine's args for consistency
                if self.multi_camera and len(annotated_frames) > 1:
                    # Create a combined frame (horizontal layout)
                    display_frame = self.visualize.concat_frames(
                        annotated_frames[0], 
                        annotated_frames[1], 
                        mode="horizontal"
                    )
                    display_frame = cv2.resize(display_frame, (1900, 720))
                    self.visualize.display(display_frame, window_name="Multi-Camera System")
                else:
                    # Display single camera
                    self.visualize.display(annotated_frames[0], 
                                          window_name=self.cam_types[0] if self.multi_camera else self.cam_type)
            
        # Cleanup
        for stream in self.streams:
            stream.stop()
        
        cv2.destroyAllWindows()
        self.logger.info("Exiting... Successfully processed all frames.")
