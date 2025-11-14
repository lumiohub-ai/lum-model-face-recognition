# Face recognition and Smart-Office System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/bybatkhuu/model.python-template/2.build-publish.yml?logo=GitHub)](https://github.com/bybatkhuu/model.python-template/actions/workflows/2.build-publish.yml)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/bybatkhuu/model.python-template?logo=GitHub&color=blue)](https://github.com/bybatkhuu/model.python-template/releases)


# Installation Guide
### 1. Clone the Repository
```bash
git clone --recursive so.model-face-recognition
cd so.model-face-recognition```

### 2. Prepare Configuration Files

Copy example configuration templates:

```bash
cp templates/compose/compose.override.dev.yml compose.override.dev.yml
cp .env.example .env
```
### 3. Update Environment Variables
Open .env and update the following:

- Set input source (RTSP or video path)
```bash
HB_IN = rtsp://your_camera_stream
```
- Adjust any other variables as needed for your local environment.

### 4. Update Client and Configuration Files

Edit examples/clients/main.py to match your client configuration.
Update the api_host field in configs/config.yaml to point to your API endpoint.

### 5. Add Face Embeddings5. Add Face Embeddings
Place your face embedding file (main.pkl) in: ```volumes/src/embeddings/```

# Build and Run the Application

## Build Docker Containers

 ```bash
 ./compose.sh build
```
## Run the Application
```bash
./compose.sh test -l
```

## Stop Containers
```bash
./compose.sh stop
```


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

