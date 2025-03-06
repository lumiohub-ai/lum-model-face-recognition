## 🖥️ NVIDIA Jetson Nano Setup

To optimize face recognition performance on NVIDIA Jetson Nano, you'll need to convert the YOLO model to TensorRT format.

### Prerequisites

#### Set up PyTorch and torchvision compatible with your Jetpack version:
  - Follow the official guide at [Ultralytics Jetson Setup](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Model Conversion

#### 1. Use the provided conversion script to create a TensorRT engine:

```python
# src/jetson/convert_model.py
import sys
import os
sys.path.append(os.curdir)

from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov11s-face.pt")

# Export the model to TensorRT format
model.export(format="engine", int8=True, imgsz=640, simplify=True)
```

#### 2. Customize the conversion:
  * Change the model path to your specific YOLO model
  * Adjust quantization options:
     * Use `int8=True` for maximum speed (default)
     * Use `int8=False` for higher accuracy
     * Use `half=True` for fp16 quantization
  * Modify `imgsz` based on your application needs

#### 3. After conversion, update the `config.yaml` file:
  * Change the `model_path` from `.pt` to `.engine`:

```yaml
model_path: "models/yolov8m-face.engine"  # Updated path to TensorRT engine
```

#### 4. Run the application normally

```
python src/face_recognition/main.py
```
