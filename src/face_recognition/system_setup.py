"""System setup module for configuring the face recognition system."""

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
        """Initialize the system configuration.

        Args:
            cam_types: List of camera types (e.g., "entry", "exit")
            video_path: Path(s) to video file(s) or stream URL(s)
            multi_camera: Whether to process multiple cameras simultaneously
            config_path: Path to the configuration file
            **kwargs: Additional configuration parameters
        """
        # Core configuration
        self.multi_camera = multi_camera
        self.streams: List[StreamHandler] = []
        self.engines: List[FaceEngine] = []
        self.visualize = Visualization()
        self.video_writers: List[Optional[cv2.VideoWriter]] = []
        self.config_path = config_path
        self.client_slug = kwargs.get('client_slug', 'default_client')
        self.FR_SLUG = os.getenv("FR_SLUG", "face-recognition")

        # Configure logger (pass client_slug directly)
        self._setup_logger(kwargs.get('log_file'), kwargs.get('debug', True), self.client_slug)

        # Setup camera streams
        self.args = self.setup_cameras(cam_types, video_path, **kwargs)
        self.FR_SLUG = os.getenv("FR_SLUG")


        self.entry_logger = EntryLogger(args=self.args)

    def setup_cameras(self, cam_types: Optional[List[str]], video_paths: Optional[Union[str, List[str]]], **kwargs) -> None:
        """Set up camera streams based on configuration.

        Args:
            cam_types: List of camera types
            video_paths: Path(s) to video file(s) or stream URL(s)
            **kwargs: Additional configuration parameters

        Returns:
            Configuration arguments
        """
        if self.multi_camera:
            args = self._setup_multi_camera(cam_types, video_paths, **kwargs)
        else:
            args = self._setup_single_camera(cam_types, video_paths, **kwargs)

        return args

    def _setup_multi_camera(self, cam_types: Optional[List[str]], video_paths: Optional[List[str]], **kwargs) -> None:
        """Configure multiple camera streams.

        Args:
            cam_types: List of camera types
            video_paths: List of paths to video files or stream URLs
            **kwargs: Additional configuration parameters

        Returns:
            Configuration arguments

        Raises:
            ValueError: If cam_types is None or video_paths is not a list
        """
        if not cam_types or not isinstance(video_paths, list):
            raise ValueError("Multi-camera setup requires cam_types and a list of video_paths")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, (cam_type, vid_path) in enumerate(zip(cam_types, video_paths)):
            args = self._load_config(video_path=vid_path, cam_type=cam_type, **kwargs)
            self._initialize_camera(args, vid_path, cam_type, timestamp, i)

        return args

    def _setup_single_camera(self, cam_type: Optional[str], video_path: Optional[str], **kwargs) -> None:
        """Configure a single camera stream.

        Args:
            cam_type: Camera type
            video_path: Path to video file or stream URL
            **kwargs: Additional configuration parameters

        Returns:
            Configuration arguments
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args = self._load_config(video_path=video_path, cam_type=cam_type, **kwargs)
        self._initialize_camera(args, video_path, cam_type, timestamp)

        return args

    def _initialize_camera(self, args: Any, vid_path: str, cam_type: str, timestamp: str, index: Optional[int] = None) -> None:
        """Initialize a camera with configuration and prepare video writer if needed.

        Args:
            args: Configuration arguments
            vid_path: Path to video file or stream URL
            cam_type: Camera type
            timestamp: Current timestamp string
            index: Camera index (for multi-camera setup)
        """
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

        # Handle match_threshold configuration (can be single value or list)
        if hasattr(args, 'match_threshold') and isinstance(args.match_threshold, list):
            if index is not None and index < len(args.match_threshold):
                args.match_threshold = args.match_threshold[index]
            else:
                # Use default if list is too short
                args.match_threshold = 0.3
                logger.warning(f"match_threshold list is too short for camera index {index}, using default 0.3")

        # Handle camera_names configuration
        if hasattr(args, 'camera_names') and args.camera_names is not None:
            if index is not None and isinstance(args.camera_names, (list, tuple)) and len(args.camera_names) > index:
                args.camera_name = args.camera_names[index]
            else:
                args.camera_name = f"Camera_{index}" if index is not None else "Camera_0"
        else:
            args.camera_name = f"Camera_{index}" if index is not None else "Camera_0"

        # Handle camera_ids configuration
        if hasattr(args, 'camera_ids') and args.camera_ids is not None:
            if index is not None and isinstance(args.camera_ids, (list, tuple)) and len(args.camera_ids) > index:
                args.camera_id = args.camera_ids[index]
            else:
                args.camera_id = str(index) if index is not None else "0"
        else:
            args.camera_id = str(index) if index is not None else "0"

        # Initialize stream and engine
        stream_handler = StreamHandler(vid_path, args.logger)
        engine = FaceEngine(args=args)

        args.db_names = engine.face_recognition.db_names
        args.fps = stream_handler.fps

        self.streams.append(stream_handler)
        self.engines.append(engine)

        # Configure video output if enabled
        frame = stream_handler.frame
        width, height = frame.shape[1], frame.shape[0]

        if args.roi:
            roi = args.roi
            width, height = roi[2] - roi[0], roi[3] - roi[1]

        if args.save_video:
            client_slug = self.client_slug
            dir = f"/app/volumes/storage/{self.FR_SLUG}/data/{self.client_slug}/saved_videos/"
            os.makedirs(f"{dir}", exist_ok=True)

            camera_id = f"{cam_type}_{index if index is not None else ''}"
            video_filename = f"{dir}/{camera_id}_{timestamp}.avi"
            writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
            self.video_writers.append(writer)
        else:
            self.video_writers.append(None)

        self._show_config(args)

    def _load_config(self, **kwargs) -> Any:
        """Load configuration from YAML file and override with provided arguments.

        Args:
            **kwargs: Configuration parameters to override

        Returns:
            Configuration arguments object

        Raises:
            Exception: If there's an error loading the configuration file
        """
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
        """Display the configuration settings.

        Args:
            args: Configuration arguments to display
        """
        logger.info("\n=== Configuration Settings ===")
        needed_keys = ['cam_type', 'video_path', 'match_threshold', 'db_path', 'roi', 'line_points', 'show',
                       'record_always', 'eval']
        for key, value in args.__dict__.items():
            if key in needed_keys:
                logger.info(f"{key}: {value}")
        logger.info("===========================\n")

    def _setup_logger(self, log_file: Optional[str] = None, debug: bool = False, client_slug: str = 'default') -> None:
        """Setup the logger with different levels and formats.

        Args:
            log_file: Path to log file (if None, logging to file is disabled)
            debug: Whether to enable debug logging
            client_slug: Client slug for organizing logs
        """
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

        # Always add file logging with rotation
        FR_SLUG = os.getenv("FR_SLUG", "face-recognition")
        default_log_path = f'/app/volumes/storage/{FR_SLUG}/logs/{client_slug}/app.log'

        # Create log directory if it doesn't exist
        log_path = log_file if log_file else default_log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        logger.add(
            log_path,
            rotation="10 MB",
            retention="2 weeks",
            level="DEBUG" if debug else "INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8"
        )
