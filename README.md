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
- [🐳 Docker Setup](#-docker-setup)
  - [Prerequisites for Docker](#prerequisites-for-docker)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running the Docker Container](#running-the-docker-container)
- [⚙️ Configuration](docs/configurations.md)
- [🚸 Usage/Examples](#-usageexamples)
- [📊 Evaluation](docs/evaluation.md)
- [📊 Eval Metrics](docs/evalmetrics.md)
- [🖥️ NVIDIA Jetson Nano setup](docs/jetson_nano.md)
- [📚 Documentation](#-documentation)
- [Video Recording using ffmpeg](#video-recording-using-ffmpeg)
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

## 🐳 Docker Setup

### Prerequisites for Docker

1. Install Docker:
   - [Docker Engine](https://docs.docker.com/engine/install/) for Linux
   - [Docker Desktop](https://docs.docker.com/desktop/) for Windows/macOS

2. For GPU support, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Building the Docker Image

1. From the project root directory, build the Docker image:

```sh
# Basic build
docker build -t face-recognition:latest .

# Build with specific tag
docker build -t face-recognition:v1.0.0 .
```

### Running the Docker Container

#### Basic Usage:

```sh
# Run with GPU support
docker run --rm -it --gpus all face-recognition:latest
```

#### Running Examples with commands:

```sh
# Run a test example
docker run --rm -it \
  --gpus all \
  -v $(pwd)/data:/usr/src/vision-app/data \
  -v $(pwd)/configs:/usr/src/vision-app/configs \
  face-recognition:latest \
  python examples/test.py
```

##### For Production Run:

```sh
# Run with access to host webcam
docker run --rm -it \
  --gpus all \
  -v $(pwd):/usr/src/vision-app \
  face-recognition:latest \
  python examples/clients/ilhan.py
```

##### Development Mode:

```sh
# Run in development mode with source code mounted
docker run --rm -it \
  --gpus all \
  -v $(pwd):/usr/src/vision-app \
  face-recognition:latest \
  bash
```

## 🐳 Docker Setup

### Prerequisites for Docker

1. Install Docker:
   - [Docker Engine](https://docs.docker.com/engine/install/) for Linux
   - [Docker Desktop](https://docs.docker.com/desktop/) for Windows/macOS

2. For GPU support, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Building the Docker Image

1. From the project root directory, build the Docker image:

```sh
# Basic build
docker build -t face-recognition:latest .

# Build with specific tag
docker build -t face-recognition:v1.0.0 .
```

### Running the Docker Container

#### Basic Usage:

```sh
# Run with GPU support
docker run --rm -it --gpus all face-recognition:latest
```

#### Running Examples with commands:

```sh
# Run a test example
docker run --rm -it \
  --gpus all \
  -v $(pwd)/data:/usr/src/vision-app/data \
  -v $(pwd)/configs:/usr/src/vision-app/configs \
  face-recognition:latest \
  python examples/test.py
```

##### For Production Run:

```sh
# Run with access to host webcam
docker run --rm -it \
  --gpus all \
  -v $(pwd):/usr/src/vision-app \
  face-recognition:latest \
  python examples/clients/ilhan.py
```

##### Development Mode:

```sh
# Run in development mode with source code mounted
docker run --rm -it \
  --gpus all \
  -v $(pwd):/usr/src/vision-app \
  face-recognition:latest \
  bash
```

---
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
    match_threshold=0.5,
    db_path='data/embeddings/hb-kor-camera.pkl',
)

# Run the face recognition system
face_engine_multi.run()
```
---

## ⚙️ Configuration

Please refer to [this page](docs/configurations.md).

---
## 📊 Evaluation

Please refer to [this page](docs/evaluation.md).

---
## 🖥️ NVIDIA Jetson Nano Setup

Please refer to [this page](docs/jetson_nano).

---



## 📚 Documentation

- [Description](https://docs.google.com/document/d/1DaPsSgqk6UXJVogyn9UbPGN5JYFbu2do8p9r11yKAKA)
- [Methodology and Evaluation](https://docs.google.com/document/d/1SsCB4fBA2nK6PQISYrcaki0moe4ID7Mwm4J2cF_g9i0)

--- 

## Video recording using ffmpeg
### Simple video recording 
```bash 
ffmpeg -rtsp_transport tcp -i "rtsp://<camera-link>" -c copy output.mp4
```

### Scheduled Video Recording

You can schedule video recording from an RTSP camera using `ffmpeg` together with the `at` command.

#### When the camera **has audio**

```bash
echo 'ffmpeg -rtsp_transport tcp -i "rtsp://<camera-link>" -t 3000 -an -c:v copy output.mp4' | at 17:45
```
#### When the camera does not have audio
```bash
echo 'ffmpeg -rtsp_transport tcp -i "rtsp://<camera-link>" -t 3000 -c copy output.mp4' | at 17:45
```

rtsp://<camera-link> → Replace with your camera’s RTSP stream URL.

-t 3000 → Duration of recording in seconds (adjust as needed).

-an → Disable audio (useful when you only want video).

-c:v copy / -c copy → Copy streams without re-encoding for efficiency.

at 17:45 → Time to schedule the recording (24-hour format).

#### Check scheduled jobs
```bash
atq
```
#### Remove a scheduled job
```bash
atrm <job-number>
```



## 📑  Research References

- [Face ReID Model](https://github.com/timesler/facenet-pytorch)
- [Track Evaluation](https://github.com/JonathonLuiten/TrackEval)
- [Detection and Tracking Model](https://github.com/ultralytics/ultralytics)
- [Face Detection Models](https://github.com/akanametov/yolo-face)

