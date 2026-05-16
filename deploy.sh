#!/bin/bash
# Deploy local source changes into the running person-tracking container.
# Run this after every ./compose.sh start (or any time files change).

set -e

CONTAINER=somodel-face-recognition-person-tracking-1

echo "==> Copying updated source files..."
docker cp src/domain/action_recognition/__init__.py     $CONTAINER:/app/src/domain/action_recognition/__init__.py
docker cp src/domain/action_recognition/recognizer.py   $CONTAINER:/app/src/domain/action_recognition/recognizer.py
docker cp src/domain/action_recognition/phone_detector.py $CONTAINER:/app/src/domain/action_recognition/phone_detector.py
docker cp src/domain/face_detection/recognizer.py       $CONTAINER:/app/src/domain/face_detection/recognizer.py
docker cp src/domain/face_detection/model_factory.py    $CONTAINER:/app/src/domain/face_detection/model_factory.py
docker cp src/domain/person_tracking/state.py           $CONTAINER:/app/src/domain/person_tracking/state.py
docker cp src/pipeline/camera_engine.py                 $CONTAINER:/app/src/pipeline/camera_engine.py
docker cp src/pipeline/camera_worker.py                 $CONTAINER:/app/src/pipeline/camera_worker.py
docker cp src/pipeline/engine.py                        $CONTAINER:/app/src/pipeline/engine.py
docker cp src/infrastructure/video/annotator.py         $CONTAINER:/app/src/infrastructure/video/annotator.py

echo "==> Clearing pycache..."
docker exec $CONTAINER find /app/src -name "*.pyc" -delete
docker exec $CONTAINER find /app/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

echo "==> Restarting person-tracking service..."
docker compose restart person-tracking

echo "==> Done. Follow logs with: docker compose logs -f person-tracking"
