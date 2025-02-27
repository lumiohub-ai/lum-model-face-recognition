import sys
import os
sys.path.append(os.curdir)

from ultralytics import YOLO
import cv2

# Load the YOLO11 model
model = YOLO("/home/hb-nano/mirsaid/face-recognition/models/yolov8n-face.pt")

# Export the model to TensorRT format
model.export(format="engine", int8=True, imgsz=640, simplify=True)




