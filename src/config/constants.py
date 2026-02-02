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
# Validation Constants
# ============================================================================

# Minimum values for validation
MIN_MATCH_THRESHOLD = 0.0
MAX_MATCH_THRESHOLD = 1.0

MIN_GPU_ID = 0
MAX_GPU_ID = 7  # Typical maximum GPU count

MIN_FACE_SIZE = 10
MAX_FACE_SIZE = 10000

