#!/bin/bash
echo "Starting 1h15m dual camera recording..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Start both recordings
ffmpeg -t 8100 -i "rtsp://admin:qwerty12.@192.168.1.68/Streaming/Channels/101" -c copy face-recognition/video_recordong/IN_$TIMESTAMP.mp4 &
PID1=$!
ffmpeg -t 8100 -i "rtsp://admin:qwerty12.@192.168.1.69/Streaming/Channels/101" -c copy face-recognition/video_recordong/OUT_$TIMESTAMP.mp4 &
PID2=$!

echo "Recording started at $(date)"
echo "Will stop automatically after 1 hour 15 minutes"
echo "Press Ctrl+C to stop early if needed"

# Wait for both processes to complete
wait $PID1
wait $PID2

echo "Recording completed at $(date)"