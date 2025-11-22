"""WebRTC streaming server using aiortc for dual camera face recognition streams."""

import os
import asyncio
import cv2
import logging
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from face_recognition.video.stream_handler import StreamHandler
from face_recognition.video.frame_processor import FrameProcessor

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStreamTrack(VideoStreamTrack):
    """Video track that streams from RTSP camera with face recognition annotations."""

    def __init__(self, rtsp_url: str, camera_name: str, frame_processor: Optional[FrameProcessor] = None):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.frame_processor = frame_processor

        # Initialize stream handler
        self.stream_handler = StreamHandler(rtsp_url, logger)
        self.stream_handler.start()

        logger.info(f"Camera stream initialized: {camera_name} from {rtsp_url[:30]}...")

    async def recv(self):
        """Receive the next video frame."""
        # Read frame from stream
        ret, frame = self.stream_handler.read()

        if not ret or frame is None:
            # Return blank frame if no frame available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"No Signal - {self.camera_name}", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Annotate frame if processor is available
            if self.frame_processor:
                frame = self.frame_processor.annotate_frame(
                    frame.copy(),
                    add_timestamp=True,
                    add_entries=False
                )
            else:
                # Simple annotation
                cv2.putText(frame, self.camera_name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert to VideoFrame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = self.stream_handler.cap.get(cv2.CAP_PROP_POS_FRAMES)
        video_frame.time_base = "1/30"

        # Small delay to control frame rate
        await asyncio.sleep(1/30)

        return video_frame

    def stop(self):
        """Stop the stream."""
        super().stop()
        if self.stream_handler:
            self.stream_handler.stop()
            logger.info(f"Stopped camera stream: {self.camera_name}")


class DualCameraStreamingServer:
    """WebRTC server for streaming two camera feeds with face recognition."""

    def __init__(self, port: int = 2233):
        self.port = port
        self.pcs = set()  # Peer connections
        self.camera_tracks: Dict[str, CameraStreamTrack] = {}

        # Camera configuration from environment
        self.rtsp1 = os.getenv("RTSP1", "")
        self.rtsp2 = os.getenv("RTSP2", "")

        if not self.rtsp1 or not self.rtsp2:
            logger.warning("RTSP1 or RTSP2 not set in environment variables")

    async def offer(self, request):
        """Handle WebRTC offer from client."""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        camera_id = params.get("camera", "camera1")

        # Create peer connection
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for {camera_id}: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                self.pcs.discard(pc)

        # Set remote description
        await pc.setRemoteDescription(offer)

        # Get or create camera track
        if camera_id not in self.camera_tracks:
            if camera_id == "camera1" and self.rtsp1:
                self.camera_tracks[camera_id] = CameraStreamTrack(self.rtsp1, "Camera 1")
            elif camera_id == "camera2" and self.rtsp2:
                self.camera_tracks[camera_id] = CameraStreamTrack(self.rtsp2, "Camera 2")
            else:
                logger.error(f"Invalid camera ID or RTSP URL not configured: {camera_id}")
                return web.Response(status=400, text="Camera not configured")

        # Add track to peer connection
        pc.addTrack(self.camera_tracks[camera_id])

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=f'{{"sdp": "{pc.localDescription.sdp}", "type": "{pc.localDescription.type}"}}'
        )

    async def index(self, request):
        """Serve the HTML client page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Dual Camera Face Recognition Stream</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #ffffff;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin-bottom: 30px;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(800px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }
        .camera-container {
            background: #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 2px solid #4CAF50;
        }
        .camera-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .camera-title {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #666;
            animation: pulse 2s ease-in-out infinite;
        }
        .status-dot.connected {
            background: #4CAF50;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        video {
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #000;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        .btn-start {
            background: #4CAF50;
            color: white;
        }
        .btn-start:hover {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(76,175,80,0.3);
        }
        .btn-stop {
            background: #f44336;
            color: white;
        }
        .btn-stop:hover {
            background: #da190b;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(244,67,54,0.3);
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .info-box {
            background: #333;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #4CAF50;
        }
        .info-box p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Dual Camera Face Recognition Stream</h1>

        <div class="info-box">
            <p><strong>Server:</strong> localhost:2233</p>
            <p><strong>Technology:</strong> WebRTC (aiortc)</p>
            <p><strong>Time:</strong> <span id="time"></span></p>
        </div>

        <div class="camera-grid">
            <!-- Camera 1 -->
            <div class="camera-container">
                <div class="camera-header">
                    <div class="camera-title">Camera 1</div>
                    <div class="status">
                        <span id="status1">Disconnected</span>
                        <div class="status-dot" id="dot1"></div>
                    </div>
                </div>
                <video id="video1" autoplay playsinline></video>
                <div class="controls">
                    <button class="btn-start" onclick="startStream('camera1', 'video1', 'status1', 'dot1')">Start Stream</button>
                    <button class="btn-stop" onclick="stopStream('camera1', 'video1', 'status1', 'dot1')">Stop Stream</button>
                </div>
            </div>

            <!-- Camera 2 -->
            <div class="camera-container">
                <div class="camera-header">
                    <div class="camera-title">Camera 2</div>
                    <div class="status">
                        <span id="status2">Disconnected</span>
                        <div class="status-dot" id="dot2"></div>
                    </div>
                </div>
                <video id="video2" autoplay playsinline></video>
                <div class="controls">
                    <button class="btn-start" onclick="startStream('camera2', 'video2', 'status2', 'dot2')">Start Stream</button>
                    <button class="btn-stop" onclick="stopStream('camera2', 'video2', 'status2', 'dot2')">Stop Stream</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let pc1 = null;
        let pc2 = null;

        function updateTime() {
            document.getElementById('time').textContent = new Date().toLocaleString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        async function startStream(cameraId, videoId, statusId, dotId) {
            const video = document.getElementById(videoId);
            const status = document.getElementById(statusId);
            const dot = document.getElementById(dotId);

            status.textContent = 'Connecting...';

            // Create peer connection
            const pc = new RTCPeerConnection({
                iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
            });

            // Store reference
            if (cameraId === 'camera1') pc1 = pc;
            else pc2 = pc;

            // Handle incoming tracks
            pc.ontrack = (event) => {
                video.srcObject = event.streams[0];
                status.textContent = 'Connected';
                dot.classList.add('connected');
            };

            // Handle connection state
            pc.onconnectionstatechange = () => {
                console.log(`${cameraId} connection state:`, pc.connectionState);
                if (pc.connectionState === 'connected') {
                    status.textContent = 'Live';
                } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                    status.textContent = 'Disconnected';
                    dot.classList.remove('connected');
                }
            };

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Send offer to server
            const response = await fetch('/offer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                    camera: cameraId
                })
            });

            const answer = await response.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
        }

        function stopStream(cameraId, videoId, statusId, dotId) {
            const video = document.getElementById(videoId);
            const status = document.getElementById(statusId);
            const dot = document.getElementById(dotId);

            const pc = cameraId === 'camera1' ? pc1 : pc2;
            if (pc) {
                pc.close();
                if (cameraId === 'camera1') pc1 = null;
                else pc2 = null;
            }

            video.srcObject = null;
            status.textContent = 'Disconnected';
            dot.classList.remove('connected');
        }

        // Auto-start streams on page load (optional)
        // window.onload = () => {
        //     startStream('camera1', 'video1', 'status1', 'dot1');
        //     startStream('camera2', 'video2', 'status2', 'dot2');
        // };
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type="text/html")

    async def on_shutdown(self, app):
        """Cleanup on server shutdown."""
        # Close all peer connections
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

        # Stop all camera tracks
        for track in self.camera_tracks.values():
            track.stop()
        self.camera_tracks.clear()

    def run(self):
        """Start the WebRTC streaming server."""
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)
        app.on_shutdown.append(self.on_shutdown)

        logger.info(f"Starting WebRTC streaming server on port {self.port}")
        logger.info(f"Open http://localhost:{self.port} in your browser")

        web.run_app(app, host="0.0.0.0", port=self.port)


def main():
    """Main entry point."""
    server = DualCameraStreamingServer(port=2233)
    server.run()


if __name__ == "__main__":
    main()
