# Face recognition and Smart-Office System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/bybatkhuu/model.python-template/2.build-publish.yml?logo=GitHub)](https://github.com/bybatkhuu/model.python-template/actions/workflows/2.build-publish.yml)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/bybatkhuu/model.python-template?logo=GitHub&color=blue)](https://github.com/bybatkhuu/model.python-template/releases)

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠 Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Download Repository](#download-or-clone-the-repository)
  - [Package Installation](#install-the-package)
- [⚙️ Configuration](#️-configuration)
- [🚸 Usage/Examples](#-usageexamples)
- [📊 Evaluation](#-evaluation)
- [🖥️ NVIDIA Jetson Nano setup](#️-nvidia-jetson-nano-setup)
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
bash scripts/setup.sh

source myenv/bin/activate
```

### 4. 📥 Download databases and models

1. Download models from here: [LINK](https://drive.google.com/drive/folders/140jyB_uM2PF9-CBVtQCR4hFJJ4Ql0TFs?usp=sharing)

2. Download database from here: [LINK](https://drive.google.com/drive/folders/1A6s3MBQvj1PXJDmGKSL4R0ZR6Sc-iicw?usp=sharing)

3. Place put them in the main directory of the repository 

## ⚙️ Configuration

[**`configs/config.yaml`**](https://github.com/humblebeeintel/face-recognition/blob/main/src/face_recognition/cfg/config.yaml):

```yaml
# Device settings
device: "cuda:0"  # Use GPU (CUDA) or switch to "cpu" if GPU is not available

# Database settings
db_path: "data/embeddings/hb-kor.pkl" # Must be in pkl format
video_path: "data/videos/sample.mp4"  # Path to the input video file or RTSP stream

# Face recognition settings
match_threshold: 0.3  # Threshold for face similarity matching

# Other settings
show: True  # Show the output video with annotations
save_video: True  # Save the output video with annotations

# Evaluation settings
eval: False  # Enable evaluation mode

# ROI settings
roi: null  # Enable region of interest (ROI) mode
line_points: null # Define the ROI line points (e.g., [(0, 0), (1280, 720)])

# Timezone settings
timezone: "Asia/Seoul"  # Timezone for tracking times
debug: True  # Enable debug mode for detailed logging
log_file: "logs/debug.log"  # Path to save the debug log file               
```

## 🚸 Usage/Examples

#### For running a face recognition application, use the following command:

```bash
python examples/test.py
```

**For arguments passed to the class, reference the config.yaml file where all parameters can be configured.**

```python
from face_recognition import HBFace

# RTSP streams for IN and OUT cameras
in_camera = 'hb-videos/in.mp4' 
out_camera = 'hb-videos/out.mp4'

# Initialize HBFace for multi-camera setup
face_engine_multi = HBFace(
    cam_types=["IN", "OUT"],
    video_path=[in_camera, out_camera],
    multi_camera=True,
    show=True,  # Disable display, just process and save
    match_threshold=0.3,
    db_path='data/embeddings/hb-kor-camera.pkl',
)

# Run the face recognition system
face_engine_multi.run()
```

---
## 📊 Evaluation

To properly evaluate the face recognition system's performance and identify areas for improvement, follow these steps:

### 1. Video Labeling and Ground Truth

1. Prepare an ID-to-name mapping dictionary:

 ```json { "1": "Azamat", "2": "Oybek", "3": "Maruf", "4": "Bahodir", "5": "Sarvar", "6": "MuhammadAmin", "7": "Batkhuu", "8": "Mirsaid" } ```

2. Rename it same with video name and save it ```annotations``` directory.

### 3. Run Evaluation

```bash
python3 tests/eval.py \
  --alg_name "fps-calculation" \
  --benchmark "ilhan" \
  --videos videoa1-1 videoa1-2 videoa1-3 videoa1-4 videoa1-5 \
  --video_dir "data/tests/ilhan_videos" \
  --db_path "data/embeddings/ilhan.pkl" \
  --output "results/" \
  --match_threshold 0.3 \
  --show
```

| Argument            | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `--alg_name`        | Name of the algorithm used for evaluation (e.g., `"fps-calculation"`) |
| `--benchmark`       | Name of the benchmark dataset or group (e.g., `"ilhan"`)              |
| `--videos`          | List of video names in ```video_dir``` (space-separated)                                 |
| `--video_dir`       | Path to the directory containing videos files                          |
| `--db_path`         | Path to the `.pkl` file with saved embeddings                         |
| `--output`          | Directory to save evaluation results                                  |
| `--match_threshold` | Similarity threshold to match faces (default is `0.3`)                |
| `--show` (optional) | Add this flag to visualize matching results during evaluation         |


#### Required Parameters

- `--videos`: Specify one or more video filenames for processing. Multiple entries should be space-delimited.
- `--video_dir`: Designate the directory path containing the target video files.
- `--alg_name`: Indicate the face recognition algorithm to be evaluated.
- `--benchmark`: Select the benchmark dataset for performance evaluation.
- `--output`: Define the destination CSV filename where evaluation results will be stored.

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
python examples/test.py
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

