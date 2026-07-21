"""
MDA Channel Definitions

Commands: Backend -> AI Service (Redis Streams)
Events: AI Service -> Backend (Redis Pub/Sub)
"""

# ============================================================
# COMMANDS (Backend -> AI Service) - Redis Streams
# ============================================================
COMMAND_STREAMS = {
    'EMBEDDING': 'commands:embedding',
    'CAMERA': 'commands:camera',
}


# ============================================================
# EVENTS (AI Service -> Backend) - Redis Pub/Sub
# Ephemeral notifications - data already persisted in DB
# ============================================================
EVENT_CHANNELS = {
    'ATTENDANCE': 'events:attendance',
    'UNRECOGNIZED': 'events:unrecognized',
    'ACTIVITY': 'events:activity',
    'LOCATION': 'events:location',
    'EMBEDDING': 'events:embedding',
    'FRAME_CAPTURE': 'events:frame_captured',
    'CALIBRATION': 'events:calibration',
    'METRICS': 'events:metrics',
    'CAMERA': 'events:camera',
}

# Event types
EVENT_TYPES = {
    'ATTENDANCE_RECORDED': 'AttendanceRecorded',
    'UNRECOGNIZED_FACE_SAVED': 'UnrecognizedFaceSaved',
    'ACTIVITY_DETECTED': 'ActivityDetected',
    'USER_LOCATION_UPDATED': 'UserLocationUpdated',
    'EMBEDDING_CREATED': 'EmbeddingCreated',
    'EMBEDDING_FAILED': 'EmbeddingFailed',
    'FRAME_CAPTURED': 'FrameCaptured',
    'FRAME_CAPTURE_FAILED': 'FrameCaptureFailed',
    'CALIBRATION_COMPLETE': 'CalibrationComplete',
    'CALIBRATION_FAILED': 'CalibrationFailed',
    'TEST_CALIBRATION_COMPLETE': 'TestCalibrationComplete',
    'HOMOGRAPHY_CALIBRATED': 'HomographyCalibrated',
    'HOMOGRAPHY_FAILED': 'HomographyFailed',
    'SYSTEM_METRICS': 'SystemMetrics',
    'SYSTEM_ALERT': 'SystemAlert',
    'CAMERA_HEARTBEAT': 'CameraHeartbeat',
}


# ============================================================
# INTERNAL CHANNELS (AI Service internal communication)
# Used for notifying camera engine to reload embeddings/status
# ============================================================
INTERNAL_CHANNELS = {
    'EMBEDDING_RELOAD': 'internal:embedding:reload',
    'STATUS_RELOAD': 'internal:status:reload',
}
