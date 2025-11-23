#!/usr/bin/env python3
"""Test RTSP connection from inside Docker container."""

import cv2
import sys

def test_rtsp(url):
    """Test RTSP connection and read one frame."""
    print(f"Testing RTSP URL: {url}")
    print("=" * 80)

    # Try different RTSP transport modes
    transports = [
        ("TCP", {"CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*'H264'),
                 "CAP_PROP_BUFFERSIZE": 1}),
        ("UDP", {}),
    ]

    for transport_name, params in transports:
        print(f"\nTrying {transport_name} transport...")

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        # Set parameters
        for key, val in params.items():
            cap.set(getattr(cv2, key), val)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ SUCCESS with {transport_name}!")
                print(f"   Frame shape: {frame.shape}")
                print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                cap.release()
                return True
            else:
                print(f"❌ Opened but failed to read frame with {transport_name}")
        else:
            print(f"❌ Failed to open with {transport_name}")

        cap.release()

    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_rtsp.py <rtsp_url>")
        print("Example: python test_rtsp.py 'rtsp://admin:pass@192.168.1.100:554'")
        sys.exit(1)

    rtsp_url = sys.argv[1]
    success = test_rtsp(rtsp_url)

    sys.exit(0 if success else 1)
