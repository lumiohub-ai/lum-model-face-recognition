import os
import sys
import cv2
import yaml
from loguru import logger
from datetime import datetime
from typing import List, Optional, Union, Any

# Local imports
from .engine import FaceEngine
from .visualize import Visualization
from .stream_handler import StreamHandler
from .entry_logger import EntryLogger


class FaceSetup:
    """Handles system configuration and initialization for the face recognition system."""
    
    def __init__(self, cam_types: Optional[List[str]] = None, video_path: Optional[Union[str, List[str]]] = None,
                 multi_camera: bool = True, config_path: str = "src/face_recognition/cfg/config.yaml", **kwargs) -> None:
        # Core configuration
        self.multi_camera = multi_camera
        self.streams: List[StreamHandler] = []
        self.engines: List[FaceEngine] = []
        self.visualize = Visualization()
        self.video_writers: List[Optional[cv2.VideoWriter]] = []
        self.config_path = config_path
        
        # Configure logger
        self._setup_logger(kwargs.get('log_file'), kwargs.get('debug', True))
        
        # Setup camera streams
        args = self.setup_cameras(cam_types, video_path, **kwargs)

        self.entry_logger = EntryLogger(args=args)
    
    def setup_cameras(self, cam_types: Optional[List[str]], video_paths: Optional[Union[str, List[str]]], **kwargs) -> None:
        """Set up camera streams based on configuration."""
        if self.multi_camera:
            args = self._setup_multi_camera(cam_types, video_paths, **kwargs)
        else:
            args = self._setup_single_camera(cam_types, video_paths, **kwargs)

        return args

    def _setup_multi_camera(self, cam_types: Optional[List[str]], video_paths: Optional[List[str]], **kwargs) -> None:
        """Configure multiple camera streams."""
        if not cam_types or not isinstance(video_paths, list):
            raise ValueError("Multi-camera setup requires cam_types and a list of video_paths")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (cam_type, vid_path) in enumerate(zip(cam_types, video_paths)):
            args = self._load_config(video_path=vid_path, cam_type=cam_type, **kwargs)
            self._initialize_camera(args, vid_path, cam_type, timestamp, i)
        
        return args

    def _setup_single_camera(self, cam_type: Optional[str], video_path: Optional[str], **kwargs) -> None:
        """Configure a single camera stream."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args = self._load_config(video_path=video_path, cam_type=cam_type, **kwargs)
        self._initialize_camera(args, video_path, cam_type, timestamp)

        return args

    def _initialize_camera(self, args: Any, vid_path: str, cam_type: str, timestamp: str, index: Optional[int] = None) -> None:
        """Initialize a camera with configuration and prepare video writer if needed."""
        args.cam_type = cam_type
        
        # Handle ROI configuration
        if hasattr(args, 'roi') and args.roi is not None:
            if index is not None and isinstance(args.roi, (list, tuple)) and len(args.roi) > index:
                args.roi = args.roi[index]
        else:
            args.roi = None
        
        # Handle line points configuration
        if hasattr(args, 'line_points') and args.line_points is not None:
            if index is not None and isinstance(args.line_points, (list, tuple)) and len(args.line_points) > index:
                args.line_points = args.line_points[index]
        else:
            args.line_points = None

        # Initialize stream and engine
        stream = StreamHandler(vid_path)
        self.streams.append(stream)
        self.engines.append(FaceEngine(args=args))

        # Configure video output if enabled
        frame = stream.frame
        width, height = frame.shape[1], frame.shape[0]

        if args.roi:
            roi = args.roi
            width, height = roi[2] - roi[0], roi[3] - roi[1]

        if args.save_video:
            os.makedirs("saved_videos", exist_ok=True)
            camera_id = f"{cam_type}_{index if index is not None else ''}"
            video_filename = f"saved_videos/{camera_id}_{timestamp}.avi"
            writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
            self.video_writers.append(writer)
        else:
            self.video_writers.append(None)

        self._show_config(args)

    def _load_config(self, **kwargs) -> Any:
        """Load configuration from YAML file and override with provided arguments."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            config = {}

        # Create args object and set attributes
        args = type('Args', (), {})()
        for key, value in {**config, **kwargs}.items():
            setattr(args, key, value)

        # Set default values
        args.save_video = getattr(args, "save_video", True)
        
        # Add logger to args
        args.logger = logger
        
        return args

    def _show_config(self, args: Any) -> None:
        """Display the configuration settings."""
        logger.info("\n=== Configuration Settings ===")
        for key, value in args.__dict__.items():
            logger.info(f"{key}: {value}")
        logger.info("===========================\n")
    
    def _setup_logger(self, log_file: Optional[str] = None, debug: bool = False) -> None:
        """Setup the logger with different levels and formats."""
        logger.remove()  # Clear default handlers

        # Define levels and their formats
        level_formats = {
            "DEBUG": "<bold><red>{level: <8}</red> | <red>{message}</red></bold>",
            "INFO": "<bold><green>{level: <8}</green> | <green>{message}</green></bold>",
            "WARNING": "<bold><yellow>{level: <8}</yellow> | <yellow>{message}</yellow></bold>",
            "ERROR": "<bold><magenta>{level: <8}</magenta> | <magenta>{message}</magenta></bold>",
            "CRITICAL": "<bold><RED>{level: <8}</RED> | <RED>{message}</RED></bold>",
        }

        # Only add DEBUG handler if debug mode is on
        for level, fmt in level_formats.items():
            if level == "DEBUG" and not debug:
                continue  # Skip DEBUG logs in non-debug mode

            logger.add(
                sys.stderr,
                level=level,
                format=fmt,
                colorize=True,
                filter=lambda record, lvl=level: record["level"].name == lvl
            )

        if log_file:
            logger.add(log_file, rotation="10 MB", level="DEBUG" if debug else "INFO")
