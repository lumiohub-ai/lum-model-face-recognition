"""Constants for the face recognition system.

This module centralizes all magic numbers, default values, and configuration constants
used throughout the face recognition system.
"""

# ============================================================================
# Tracking Constants
# ============================================================================

# Default maximum lifetime for a tracked face in seconds
DEFAULT_TRACK_LIFETIME_SECONDS = 30

# Maximum number of passed tracks to keep in history
MAX_PASSED_TRACKS = 5000

# Minimum track lifetime to consider for recognition (seconds)
MIN_TRACK_LIFETIME_SECONDS = 1

# ============================================================================
# Recognition Constants
# ============================================================================

# Default face recognition similarity threshold
DEFAULT_MATCH_THRESHOLD = 0.3

# Alpha value for embedding normalization
EMBEDDING_NORMALIZATION_ALPHA = 0.9

# Minimum number of frames required for recognition
MIN_FRAMES_FOR_RECOGNITION = 3

# Minimum face size (in pixels) for detection
DEFAULT_MINIMUM_FACE_SIZE = 30

# ============================================================================
# Enhanced Filter Constants (New)
# ============================================================================

# Quality thresholds
DEFAULT_MIN_FRONTALITY_SCORE = 0.65
DEFAULT_MIN_LAPLACIAN_VARIANCE = 100.0
DEFAULT_MIN_BRIGHTNESS = 40
DEFAULT_MAX_BRIGHTNESS = 220
DEFAULT_MIN_LANDMARK_SPREAD = 0.12

# Recognition confidence zones
DEFAULT_RECOGNIZED_THRESHOLD = 0.40
DEFAULT_TRUE_UNKNOWN_THRESHOLD = 0.22
DEFAULT_MIN_TOP2_MARGIN = 0.08
DEFAULT_MAX_UNKNOWN_SIMILARITY = 0.25

# Temporal deduplication
DEFAULT_DEDUP_CACHE_TTL = 300  # 5 minutes
DEFAULT_DEDUP_SIMILARITY_THRESHOLD = 0.85
DEFAULT_DEDUP_CROSS_CAMERA_WINDOW = 60  # 1 minute

# Track quality
DEFAULT_MIN_TRACK_QUALITY_SCORE = 0.60
DEFAULT_MIN_QUALITY_FRAMES = 3
DEFAULT_MIN_UNRECOGNIZED_TRACK_LIFETIME = 2.5  # seconds

# Rate limiting
DEFAULT_MAX_UNKNOWNS_PER_CAMERA_PER_MINUTE = 8

# ============================================================================
# Video Processing Constants
# ============================================================================

# Default frame dimensions for display
DISPLAY_FRAME_WIDTH = 1280
DISPLAY_FRAME_HEIGHT = 720

# Concatenated frame dimensions
CONCAT_FRAME_WIDTH = 1920
CONCAT_FRAME_HEIGHT = 720

# FPS calculation window (number of frames)
FPS_CALCULATION_WINDOW = 30

# ============================================================================
# API Constants
# ============================================================================

# Default API timeout in seconds
DEFAULT_API_TIMEOUT = 30

# Maximum retry attempts for API calls
MAX_API_RETRY_ATTEMPTS = 3

# Retry delay (exponential backoff base in seconds)
API_RETRY_BASE_DELAY = 1

# ============================================================================
# Storage Constants
# ============================================================================

# Default storage base path (can be overridden by env var)
DEFAULT_STORAGE_BASE_PATH = "/app/volumes/storage"

# Subdirectory names
RECOGNIZED_FRAMES_DIR = "recognized_frames"
UNRECOGNIZED_FRAMES_DIR = "unrecognized_frames"
DATA_COLLECTION_DIR = "collection"
EMBEDDINGS_DIR = "embeddings"

# ============================================================================
# Dashboard Constants
# ============================================================================

# Default dashboard port
DEFAULT_DASHBOARD_PORT = 5001

# Default dashboard host
DEFAULT_DASHBOARD_HOST = "0.0.0.0"

# JPEG encoding quality for streaming
JPEG_QUALITY = 80

# ============================================================================
# Model Constants
# ============================================================================

# Default GPU ID
DEFAULT_GPU_ID = 0

# InsightFace model name
DEFAULT_FACE_MODEL = "buffalo_l"

# ============================================================================
# Logging Constants
# ============================================================================

# Log levels
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"

# ============================================================================
# Timezone Constants
# ============================================================================

# Default timezone
DEFAULT_TIMEZONE = "UTC"

# Common timezones
TIMEZONE_UTC = "UTC"
TIMEZONE_TASHKENT = "Asia/Tashkent"

# ============================================================================
# File Constants
# ============================================================================

# Default configuration file path
DEFAULT_CONFIG_PATH = "configs/config.yaml"

# Default database path
DEFAULT_DB_PATH = "volumes/src/embeddings/main.pkl"

# Image file extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Video file extensions
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# ============================================================================
# Camera/Stream Constants
# ============================================================================

# Camera types
CAMERA_TYPE_IN = "IN"
CAMERA_TYPE_OUT = "OUT"
CAMERA_TYPE_MANAGEMENT = "MANAGEMENT"

# Stream reconnection settings
STREAM_RECONNECT_DELAY_SECONDS = 5
STREAM_MAX_RECONNECT_ATTEMPTS = 10

# Frame queue size for streaming
FRAME_QUEUE_MAXSIZE = 1

# ============================================================================
# Recognition Status Constants
# ============================================================================

STATUS_RECOGNIZED = "RECOGNIZED"
STATUS_UNRECOGNIZED = "UNRECOGNIZED"
STATUS_PARTIAL_MATCH = "PARTIAL_MATCH"

# Entry/Exit status
STATUS_IN = "IN"
STATUS_OUT = "OUT"

# ============================================================================
# Validation Constants
# ============================================================================

# Minimum values for validation
MIN_MATCH_THRESHOLD = 0.0
MAX_MATCH_THRESHOLD = 1.0

MIN_GPU_ID = 0
MAX_GPU_ID = 7  # Typical maximum GPU count

MIN_FACE_SIZE = 10
MAX_FACE_SIZE = 10000

# ============================================================================
# HTTP Status Codes (for reference)
# ============================================================================

HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500
