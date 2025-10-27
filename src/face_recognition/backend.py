"""
FastAPI backend - Production-ready video streaming server for face recognition dashboard
Supports global deployment with CORS, async streaming, and proper error handling
"""
import asyncio
import cv2
import time
import os
from typing import List
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn
from .camera_processor import setup_cameras

# Cleanup task
cleanup_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    global cleanup_task

    # Startup
    print("🚀 Starting FastAPI dashboard server...")
    print(f"📊 Dashboard: http://0.0.0.0:{os.getenv('DASHBOARD_PORT', '5000')}")

    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    # Shutdown
    print("🛑 Shutting down dashboard server...")
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Face Recognition Dashboard API",
    description="Real-time video streaming dashboard for face recognition system",
    version="2.0.0",
    lifespan=lifespan
)

# Initialize camera processor
camera_processor = setup_cameras()

# Production-ready middleware configurations

# CORS - Allow global access (configure based on your security needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# GZip compression for better bandwidth usage
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Cleanup configuration
CLEANUP_INTERVAL = float(os.getenv("CLEANUP_INTERVAL", "5.0"))


async def periodic_cleanup():
    """Background task to cleanup stale frames periodically"""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            camera_processor.cleanup_stale_frames()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in cleanup task: {e}")


async def generate_stream(camera_id: str):
    """Async MJPEG stream generator

    Args:
        camera_id: Camera identifier to stream

    Yields:
        MJPEG frame data
    """
    consecutive_failures = 0
    max_failures = int(os.getenv("MAX_STREAM_FAILURES", "50"))

    try:
        while consecutive_failures < max_failures:
            frame = camera_processor.get_frame(camera_id)

            if frame is None:
                consecutive_failures += 1
                await asyncio.sleep(0.1)  # Non-blocking wait
                continue

            consecutive_failures = 0  # Reset on successful frame

            # Encode as JPEG
            ret, buffer = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(os.getenv("JPEG_QUALITY", "85"))]
            )

            if not ret:
                continue

            # Yield MJPEG frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

            # Target FPS control
            await asyncio.sleep(1.0 / int(os.getenv("TARGET_FPS", "30")))

    except asyncio.CancelledError:
        print(f"Stream cancelled for camera: {camera_id}")
    except Exception as e:
        print(f"Error in stream generation for {camera_id}: {e}")


@app.get("/", response_class=FileResponse)
async def index():
    """Serve the dashboard frontend

    Returns:
        HTML dashboard page
    """
    # frontend.html is in project root, go up two levels from src/face_recognition/
    frontend_path = os.path.join(os.path.dirname(__file__), "../..", "frontend.html")

    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="Frontend not found")

    return FileResponse(frontend_path)


@app.get("/video/{camera_id}")
async def video_stream(camera_id: str):
    """Stream video from a specific camera

    Args:
        camera_id: Camera identifier

    Returns:
        MJPEG video stream

    Raises:
        HTTPException: If camera not found
    """
    # Check if camera exists
    if camera_id not in camera_processor.frame_queues:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found. Available cameras: {list(camera_processor.frame_queues.keys())}"
        )

    return StreamingResponse(
        generate_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@app.get("/cameras")
async def list_cameras():
    """List all available camera streams

    Returns:
        JSON with camera IDs and metadata
    """
    cameras = list(camera_processor.frame_queues.keys())

    return JSONResponse({
        "success": True,
        "count": len(cameras),
        "cameras": cameras,
        "timestamp": time.time()
    })


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers

    Returns:
        JSON with service health status
    """
    return JSONResponse({
        "status": "healthy",
        "service": "face-recognition-dashboard",
        "version": "2.0.0",
        "active_cameras": len(camera_processor.frame_queues),
        "timestamp": time.time()
    })


@app.get("/api/stats")
async def get_stats():
    """Get system statistics

    Returns:
        JSON with system statistics
    """
    return JSONResponse({
        "total_cameras": len(camera_processor.frame_queues),
        "camera_ids": list(camera_processor.frame_queues.keys()),
        "cleanup_interval": CLEANUP_INTERVAL,
        "max_frame_age": camera_processor.max_frame_age,
    })


def run_server():
    """Run the FastAPI server with uvicorn (for threading)"""
    port = int(os.getenv("DASHBOARD_PORT", "5001"))
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="error",  # Only show errors to reduce log clutter
        access_log=False,
    )


def main():
    """Run the FastAPI server with uvicorn (for standalone execution)"""
    port = int(os.getenv("DASHBOARD_PORT", "5001"))
    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", "1"))

    uvicorn.run(
        "backend:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        proxy_headers=True,  # Important for deployment behind reverse proxy
        forwarded_allow_ips="*",  # Trust proxy headers (configure for production)
    )


if __name__ == "__main__":
    main()