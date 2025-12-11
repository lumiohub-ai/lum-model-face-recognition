# Face recognition and Smart-Office System

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/bybatkhuu/model.python-template/2.build-publish.yml?logo=GitHub)](https://github.com/bybatkhuu/model.python-template/actions/workflows/2.build-publish.yml)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/bybatkhuu/model.python-template?logo=GitHub&color=blue)](https://github.com/bybatkhuu/model.python-template/releases)


# Installation Guide
### 1. Clone the Repository
```bash
git clone --recursive https://github.com/humblebeeai/so.model-face-recognition.git
cd so.model-face-recognition
```

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

# Check stream in VLC
vlc "rtsp://your_stream"

# Verify environment variables
docker exec face-recognition env | grep HB_IN
```



## Research References

- [InsightFace](https://github.com/deepinsight/insightface) - Face detection and recognition
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [DeepOCSORT](https://github.com/mikel-brostrom/yolo_tracking) - Multi-object tracking
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) - Tracking evaluation
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- HumbleBee AI team for development and testing
- InsightFace team for face detection models
- Ultralytics for YOLO models
- pgvector team for vector database extension

## Support

For issues, questions, or feature requests:
- Create an issue on GitHub
- Check existing documentation
- Contact the development team

---

**Version**: 2.0.0
**Last Updated**: December 2025
**Status**: Production Ready
