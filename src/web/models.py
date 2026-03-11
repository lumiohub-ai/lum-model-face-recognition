"""Pydantic models for API request and response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class CaptureFrameRequest(BaseModel):
    """Request model for capturing a frame for calibration."""
    
    frame_index: int = Field(
        ...,
        ge=1,
        le=30,
        description="Frame index for calibration grid (1-30)"
    )
    quality: int = Field(
        85,
        ge=1,
        le=100,
        description="JPEG compression quality (1-100, default 85)"
    )


class FrameMetadata(BaseModel):
    """Metadata about a captured frame."""
    
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    size_bytes: int = Field(..., description="Frame size in bytes")
    source: str = Field("face-recognition-api", description="Source of the frame capture")
    frame_index: int = Field(..., description="Frame index in calibration grid")
    camera_name: Optional[str] = Field(None, description="Camera name")


class CaptureFrameResponse(BaseModel):
    """Response model for frame capture."""
    
    success: bool = Field(..., description="Whether the capture was successful")
    frame_url: str = Field(..., description="GCS path to the frame (gs://...)")
    signed_url: str = Field(..., description="Signed URL for browser access")
    captured_at: str = Field(..., description="ISO 8601 timestamp of capture")
    metadata: FrameMetadata = Field(..., description="Frame metadata")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
