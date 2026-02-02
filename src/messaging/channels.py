"""
MDA Channel Definitions

All Redis Pub/Sub channel names used for communication between services.
"""

# Channels for Backend -> AI Service
EMBEDDING_REQUESTS = "embedding.requests"  # User CRUD triggers embedding sync
CAMERA_CONFIG = "camera.config"  # Camera configuration changes

# Channels for AI Service -> Backend
ATTENDANCE_RECORDS = "attendance.records"  # Attendance IN/OUT events
UNRECOGNIZED_FACES = "unrecognized.faces"  # Unknown face detections
ACTIVITY_RECORDS = "activity.records"  # Activity tracking (phone, sleeping)
USER_LOCATIONS = "user.locations"  # User location at camera
EMBEDDING_RESULTS = "embedding.results"  # Embedding processing results


# Channel groups for easy subscription
SUBSCRIBE_CHANNELS = [
    EMBEDDING_REQUESTS,
    CAMERA_CONFIG,
]

PUBLISH_CHANNELS = [
    ATTENDANCE_RECORDS,
    UNRECOGNIZED_FACES,
    ACTIVITY_RECORDS,
    USER_LOCATIONS,
    EMBEDDING_RESULTS,
]
