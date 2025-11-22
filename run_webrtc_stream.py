#!/usr/bin/env python3
"""
Simple launcher for WebRTC dual camera streaming server.

Usage:
    python run_webrtc_stream.py

Then open http://localhost:2233 in your browser.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from web.webrtc_streaming import DualCameraStreamingServer

if __name__ == "__main__":
    print("=" * 60)
    print("WebRTC Dual Camera Streaming Server")
    print("=" * 60)
    print("\nMake sure to configure RTSP1 and RTSP2 in your .env file:")
    print("  RTSP1='rtsp://username:password@ip:port'")
    print("  RTSP2='rtsp://username:password@ip:port'")
    print("\n" + "=" * 60)
    print("\nStarting server...")

    server = DualCameraStreamingServer(port=2233)
    server.run()
