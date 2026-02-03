"""
MDA Channel Definitions

Commands: Backend → AI Service (Redis Streams)
Events: AI Service → Backend (Redis Pub/Sub)
"""

# ============================================================
# COMMANDS (Backend → AI Service) - Redis Streams
# These are now handled by stream_consumer.py, not Pub/Sub
# ============================================================
COMMAND_STREAMS = {
    'EMBEDDING': 'commands:embedding',
    'CAMERA': 'commands:camera',
}

# Legacy Pub/Sub channels (deprecated - use streams instead)
EMBEDDING_REQUESTS = "embedding.requests"  # DEPRECATED
CAMERA_CONFIG = "camera.config"  # DEPRECATED


# ============================================================
# EVENTS (AI Service → Backend) - Redis Pub/Sub
# Ephemeral notifications - data already persisted in DB
# ============================================================
EVENT_CHANNELS = {
    'ATTENDANCE': 'events:attendance',
    'UNRECOGNIZED': 'events:unrecognized',
    'ACTIVITY': 'events:activity',
    'LOCATION': 'events:location',
    'EMBEDDING': 'events:embedding',
}

# Event types
EVENT_TYPES = {
    # Attendance
    'ATTENDANCE_RECORDED': 'AttendanceRecorded',

    # Unrecognized faces
    'UNRECOGNIZED_FACE_SAVED': 'UnrecognizedFaceSaved',

    # Activity
    'ACTIVITY_DETECTED': 'ActivityDetected',

    # Location
    'USER_LOCATION_UPDATED': 'UserLocationUpdated',

    # Embedding
    'EMBEDDING_CREATED': 'EmbeddingCreated',
    'EMBEDDING_FAILED': 'EmbeddingFailed',
}

# ============================================================
# INTERNAL CHANNELS (AI Service internal communication)
# Used for notifying camera engine to reload embeddings
# ============================================================
INTERNAL_CHANNELS = {
    'EMBEDDING_RELOAD': 'internal:embedding:reload',
}

# Legacy channel names (for backward compatibility during migration)
ATTENDANCE_RECORDS = EVENT_CHANNELS['ATTENDANCE']
UNRECOGNIZED_FACES = EVENT_CHANNELS['UNRECOGNIZED']
ACTIVITY_RECORDS = EVENT_CHANNELS['ACTIVITY']
USER_LOCATIONS = EVENT_CHANNELS['LOCATION']
EMBEDDING_RESULTS = EVENT_CHANNELS['EMBEDDING']


# Channel groups
SUBSCRIBE_CHANNELS = []  # No longer subscribing via Pub/Sub - using Streams

PUBLISH_CHANNELS = list(EVENT_CHANNELS.values())
