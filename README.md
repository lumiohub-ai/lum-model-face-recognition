# Face recognition and Smart-Office System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/bybatkhuu/model.python-template/2.build-publish.yml?logo=GitHub)](https://github.com/bybatkhuu/model.python-template/actions/workflows/2.build-publish.yml)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/bybatkhuu/model.python-template?logo=GitHub&color=blue)](https://github.com/bybatkhuu/model.python-template/releases)

## 📋 Table of Contents

- [✨ Features](#features)
- [🛠 Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Download Repository](#download-or-clone-the-repository)
  - [Package Installation](#install-the-package)
- [⚙️ Configuration](docs/configurations.md)
- [🚸 Usage/Examples](#usageexamples)
- [📊 Evaluation](docs/evaluation.md)
- [🖥️ NVIDIA Jetson Nano setup](docs/jetson_nano.md)
- [📚 Documentation](#-documentation)
- [📑 Research References](#--research-references)


## ✨ Features

- Face Detection/Tracking
- Face-Recognition
- Counter
- Desk presence Detection
- NVIDIA Jetson converter

---

## 🛠 Installation

### 1. 🚧 Prerequisites

- Install **Python (>= v3.10)** and **pip (>= 25.0.1)**:
    - *[RECOMMENDED][Python virutal environment] [venv](https://docs.python.org/3/library/venv.html)*
    - *[Miniconda (v3)](https://www.anaconda.com/docs/getting-started/miniconda/install)*

- *[OPTIONAL]* For **GPU (NVIDIA)**:
    - **NVIDIA CUDA (>= v12.6)**

[OPTIONAL] For **DEVELOPMENT** environment:

- Install [**git**](https://git-scm.com/downloads)
- Setup an [**SSH key**](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) ([video tutorial](https://www.youtube.com/watch?v=snCP3c7wXw0))

### 2. 📥 Download or clone the repository

[TIP] Skip this step, if you're going to install the package directly from **GitHub** repository.

**2.1.** Prepare projects directory (if not exists):

```sh
# Create projects directory:
mkdir -pv ~/workspaces/projects

# Enter into projects directory:
cd ~/workspaces/projects
```

**2.2.** Follow one of the below options **[A]**, **[B]** or **[C]**:

**OPTION A.** Clone the repository:

```sh
git clone https://github.com/humblebeeintel/face-recognition.git && \
    cd face-recognition
```

**OPTION B.** Clone the repository (for **DEVELOPMENT**: git + ssh key):

```sh
git git@github.com:humblebeeintel/face-recognition.git && \
    cd face-recognition
```

**OPTION C.** Download source code:

1. Download archived **zip** file from [**releases**](https://github.com/humblebeeintel/face-recognition).
2. Extract it into the projects directory.

### 3. 📦 Install the package

Install for **DEVELOPMENT** environment:

```sh
pip install -r requirements.txt

# Clone needed repository for evaluating the model
git clone https://github.com/humblebeeintel/TrackEval
```

## ⚙️ Configuration

Before running setup configurations first!!!

[**`src/face_recognition/cfg/config.yaml`**](https://github.com/humblebeeintel/face-recognition/blob/main/src/face_recognition/cfg/config.yaml):

```yaml
# Device settings
device: "cuda:0"  # Use GPU (CUDA) or switch to "cpu" if GPU is not available

# Database settings
face_crops_path: "data/database"  # Path to the folder containing face images for recognition
output_video_path: "data/output.mp4"  # Path to save the output video with annotations

# Camera input settings (RTSP links or video file paths)
in_camera: ""  # RTSP link or video file path for the entry camera
out_camera: ""  # RTSP link or video file path for the exit camera

# Define entry and exit regions (coordinates of the regions of interest)
in_region_points:  
  - [553, 106]  # Point 1 (X, Y)
  - [543, 349]  # Point 2 (X, Y)

out_region_points:  
  - [234, 217]  # Point 1 (X, Y)
  - [14, 680]   # Point 2 (X, Y)

# Database settings for recent entries
recent_entries_max_len: 3  # Maximum number of recent face entries to store for tracking

# Model configuration
model_path: "models/yolov8m-face.pt"  # Path to the face detection model
match_threshold: 0.7  # Threshold for face similarity matching
detection_threshold: 0.25  # Confidence threshold for face detection
imgsz: 960  # Image size for inference (higher values improve accuracy but increase processing time)
tracker: "bytetrack.yaml"  # Path to the tracker configuration file

# Video processing settings
skip_frames: 5  # Number of frames to skip between detections (reduces processing load)                       
```

## 🚸 Usage/Examples

#### For running a face recognition application, use the following command:

```
python src/face_recognition/main.py
```

---
## 📊 Evaluation

To properly evaluate the face recognition system's performance and identify areas for improvement, follow these steps:

### 1. Video Labeling and Ground Truth

1. Use Label Studio to label video tracking data
2. Download annotations in `json-mini` format
3. Prepare an ID-to-name mapping dictionary:

```python
id_to_name = {
    1: 'Azamat',
    2: 'Oybek',
    3: 'Maruf',
    4: 'Bahodir',
    5: 'Sarvar',
    6: 'MuhammadAmin',
    7: 'Batkhuu',
    8: 'Mirsaid',
}
```

### 2. Converting Annotations to MOT Format

Run the conversion script to transform Label Studio annotations to MOT format:

```bash
python src/face-recognition/evaluation/json_mini_converter.py \
    -j path/to/annotations.json \
    -v path/to/video/directory \
    --label_studio_fps 25 \
    --output_txt_path output_mot_10.txt \
    --verify True
```

Arguments:
* `-j, --json_path`: Path to JSON annotations (required)
* `-v, --video_dir`: Path to directory containing video files
* `--label_studio_fps`: Label Studio FPS (default: 25)
* `--output_txt_path`: Path to output txt file (default: 'output_mot_10.txt')
* `--verify`: Verify the output video (default: True)

### 3. Run Evaluation

```
python src/face_recognition/evaluation/evaluate_system.py
```
Required Parameters

--videos: A list of video names to process. Multiple videos can be specified by separating them with spaces.
--video_dir: The directory containing the video files to be processed.
--alg_name: The name of the face recognition algorithm to evaluate.
--benchmark: The benchmark dataset to use for evaluation.
--output: The filename for the output CSV file containing evaluation results.


---
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

---

## 📚 Documentation

- [Description](https://docs.google.com/document/d/1DaPsSgqk6UXJVogyn9UbPGN5JYFbu2do8p9r11yKAKA)
- [Methodology and Evaluation](https://docs.google.com/document/d/1SsCB4fBA2nK6PQISYrcaki0moe4ID7Mwm4J2cF_g9i0)

--- 

## 📑  Research References

- [Face ReID Model](https://github.com/timesler/facenet-pytorch)
- [Track Evaluation](https://github.com/JonathonLuiten/TrackEval)
- [Detection and Tracking Model](https://github.com/ultralytics/ultralytics)
- [Face Detection Models](https://github.com/akanametov/yolo-face)

