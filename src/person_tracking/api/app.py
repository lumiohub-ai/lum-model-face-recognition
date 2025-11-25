"""
FastAPI Application for Person Tracking (Mock Implementation).

Provides REST API endpoints for:
- Logging phone usage events
- Querying person tracking status
- Health checks

This is a MOCK implementation for MVP. Real backend integration
will be implemented in Post-MVP phase.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger

# Create FastAPI app
app = FastAPI(
    title="Person Tracking API",
    description="Person tracking and phone usage detection API (Mock)",
    version="0.1.0"
)


# ============================================================================
# Pydantic Models
# ============================================================================

class EventCreate(BaseModel):
    """Model for creating a person tracking event."""
    track_id: int = Field(..., description="Person track ID")
    camera_id: int = Field(..., description="Camera ID")
    event_type: str = Field(..., description="Event type")
    person_name: Optional[str] = Field(None, description="Person name (if recognized)")
    using_phone: Optional[bool] = Field(None, description="Phone usage status")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    duration: Optional[float] = Field(None, ge=0.0, description="Duration in seconds")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": 1,
                "camera_id": 1,
                "event_type": "phone_usage_started",
                "person_name": "John Doe",
                "using_phone": True,
                "confidence": 0.95,
                "timestamp": "2025-11-24T10:30:00"
            }
        }


class PersonStatus(BaseModel):
    """Model for person tracking status."""
    track_id: int
    camera_id: int
    identity: Optional[str] = None
    identity_locked: bool = False
    identity_confidence: float = 0.0
    using_phone: bool = False
    phone_confidence: float = 0.0
    phone_usage_duration: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_frames: int = 0


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    camera_id: int
    total_persons: int
    identified_persons: int
    using_phone_count: int
    persons: List[PersonStatus]


# ============================================================================
# Mock Data Store (in-memory)
# ============================================================================

# Store events in memory (for MVP demonstration)
mock_events_store: List[Dict] = []

# Store current person states
mock_person_states: Dict[int, Dict] = {}


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "Person Tracking API",
        "version": "0.1.0",
        "status": "running",
        "mode": "mock"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "person-tracking",
        "timestamp": datetime.now().isoformat(),
        "mode": "mock",
        "events_logged": len(mock_events_store),
        "active_persons": len(mock_person_states)
    }


@app.post("/api/v1/person-tracking/events", tags=["Events"])
async def create_event(event: EventCreate):
    """
    Log a person tracking event.

    This is a MOCK endpoint that logs events to console and in-memory storage.
    In production, this will call the real backend API.

    Args:
        event: Event data

    Returns:
        Success response
    """
    # Log event to console
    logger.info(
        f"[MOCK API] Event received: {event.event_type} | "
        f"Track={event.track_id} | "
        f"Person={event.person_name or 'Unknown'} | "
        f"Phone={event.using_phone} | "
        f"Confidence={event.confidence}"
    )

    # Store event in mock database
    event_dict = event.model_dump()
    if event_dict.get('timestamp'):
        event_dict['timestamp'] = event_dict['timestamp'].isoformat()

    mock_events_store.append(event_dict)

    # Update mock person state
    if event.track_id not in mock_person_states:
        mock_person_states[event.track_id] = {
            'track_id': event.track_id,
            'camera_id': event.camera_id,
            'identity': event.person_name,
            'using_phone': event.using_phone or False,
            'events': []
        }

    mock_person_states[event.track_id]['events'].append(event.event_type)
    if event.using_phone is not None:
        mock_person_states[event.track_id]['using_phone'] = event.using_phone
    if event.person_name:
        mock_person_states[event.track_id]['identity'] = event.person_name

    return {
        "success": True,
        "message": "Event logged successfully (mock)",
        "event_id": len(mock_events_store),
        "track_id": event.track_id
    }


@app.get("/api/v1/person-tracking/status", response_model=StatusResponse, tags=["Status"])
async def get_status(camera_id: Optional[int] = None):
    """
    Get current person tracking status.

    Args:
        camera_id: Optional camera ID to filter by

    Returns:
        Current status for all tracked persons
    """
    # Filter by camera if specified
    persons = list(mock_person_states.values())
    if camera_id is not None:
        persons = [p for p in persons if p.get('camera_id') == camera_id]

    # Convert to PersonStatus objects
    person_statuses = []
    for p in persons:
        person_statuses.append(PersonStatus(
            track_id=p['track_id'],
            camera_id=p.get('camera_id', 1),
            identity=p.get('identity'),
            identity_locked=p.get('identity') is not None,
            using_phone=p.get('using_phone', False),
            total_frames=len(p.get('events', []))
        ))

    # Calculate stats
    identified_count = sum(1 for p in person_statuses if p.identity is not None)
    using_phone_count = sum(1 for p in person_statuses if p.using_phone)

    return StatusResponse(
        camera_id=camera_id or 0,
        total_persons=len(person_statuses),
        identified_persons=identified_count,
        using_phone_count=using_phone_count,
        persons=person_statuses
    )


@app.get("/api/v1/person-tracking/events", tags=["Events"])
async def get_events(
    limit: int = 100,
    track_id: Optional[int] = None,
    event_type: Optional[str] = None
):
    """
    Get logged events (for debugging/testing).

    Args:
        limit: Maximum number of events to return
        track_id: Filter by track ID
        event_type: Filter by event type

    Returns:
        List of events
    """
    events = mock_events_store.copy()

    # Apply filters
    if track_id is not None:
        events = [e for e in events if e.get('track_id') == track_id]

    if event_type is not None:
        events = [e for e in events if e.get('event_type') == event_type]

    # Apply limit
    events = events[-limit:]

    return {
        "total": len(mock_events_store),
        "returned": len(events),
        "events": events
    }


@app.delete("/api/v1/person-tracking/reset", tags=["Admin"])
async def reset_data():
    """
    Reset all mock data (for testing).

    Returns:
        Success response
    """
    mock_events_store.clear()
    mock_person_states.clear()

    logger.info("[MOCK API] All data reset")

    return {
        "success": True,
        "message": "All mock data cleared"
    }


@app.get("/api/v1/person-tracking/statistics", tags=["Statistics"])
async def get_statistics():
    """
    Get API statistics.

    Returns:
        Statistics about API usage
    """
    event_types = {}
    for event in mock_events_store:
        event_type = event.get('event_type', 'unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1

    return {
        "total_events": len(mock_events_store),
        "active_persons": len(mock_person_states),
        "event_types": event_types,
        "mode": "mock"
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("Person Tracking API started (MOCK mode)")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown."""
    logger.info("Person Tracking API shutting down")
