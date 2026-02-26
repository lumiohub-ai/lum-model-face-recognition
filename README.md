# Face Recognition — R&D Branch

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Offline face recognition pipeline for research and evaluation.
Processes a list of video files against a face embedding database and outputs per-video JSON result files for annotation-based evaluation.

> **Branch:** `rnd/liteFaceR` — production dependencies (API, Redis, pgvector, GCS) are removed.

---

## How It Works

```
volumes/src/images/folder_1/{name}_{n}.jpg   ← face images
        ↓
build_embeddings.py              ← generates volumes/src/embeddings/main.pkl
        ↓
configs/rnd_config.yaml          ← define video list + annotation files
        ↓
./compose.sh start -l            ← runs pipeline on each video
        ↓
volumes/rnd_results/{video}_results.json     ← recognition events per video
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA 12.2+ (recommended)
- **CPU**: 4+ cores (CPU-only mode: pass `--gpu -1`)
- **RAM**: 8 GB minimum
- **Storage**: 10 GB+ for models and data

---

## Quick Start

### 1. Clone and configure environment

```bash
git clone --recursive https://github.com/humblebeeai/so.model-face-recognition.git
cd so.model-face-recognition
git checkout rnd/liteFaceR

cp .env.example .env
```

### 2. Prepare Face Images

Put one or more images per person in `volumes/src/images/{folder_name}/`, named `{name}_{index}.jpg`:

```
volumes/src/images/
└── folder_1/
    ├── oybek_1.jpg
    ├── oybek_2.jpg
    └── john_1.jpg
```

### 3. Build Embedding Database

**With Docker:**
```bash
./compose.sh build
docker run --rm --gpus all \
    -v $(pwd)/volumes/src:/app/face-recognition/volumes/src \
    humblebeeintel/face-recognition \
    python scripts/tools/build_embeddings.py \
        --input volumes/src/images/folder_1 \
        --output volumes/src/embeddings/main.pkl
```

**Without Docker:**
```bash
conda create -n fr python=3.10 -y && conda activate fr
pip install ./modules/insightface
pip install ./modules/yolo_tracking
pip install -e .
pip install -r requirements.rnd.txt

python scripts/tools/build_embeddings.py \
    --input volumes/src/images/folder_1 \
    --output volumes/src/embeddings/main.pkl \
    --gpu 0          # use --gpu -1 for CPU
```

> InsightFace model weights (`buffalo_l`) download automatically on first run.

### 4. Configure Videos

Edit `configs/rnd_config.yaml`:

```yaml
db_path: "volumes/src/embeddings/main.pkl"
output_dir: "volumes/rnd_results"

videos:
  - path: "data/videos/clip1.mp4"
    annotation: "data/annotations/clip1.json"
    camera_name: "cam1"
    cam_type: "IN"
```

### 5. Run Pipeline

```bash
./compose.sh start -l
```

Results are written to `volumes/rnd_results/{video_stem}_results.json`.

---

## Output Format

```json
{
  "video": "data/videos/clip1.mp4",
  "annotation_file": "data/annotations/clip1.json",
  "total_frames": 1800,
  "results": [
    {
      "frame_num": 142,
      "track_id": 3,
      "name": "oybek",
      "recognized": "recognized",
      "status": "RECOGNIZED",
      "similarity": 0.7812,
      "confidence": 0.84,
      "appear_time": "2026-02-18T10:00:05.123"
    }
  ]
}
```

---

## Project Structure

```
configs/
├── config.yaml           # base recognition settings
└── rnd_config.yaml       # R&D video list and settings

data/
├── videos/               # input video files
└── annotations/          # ground-truth annotation JSON files (TBD)

volumes/src/
├── images/folder_1/      # {name}_{index}.jpg  ← input for build_embeddings
└── embeddings/main.pkl   # generated face database

src/face_recognition/
├── core/                 # detector, tracker, recognizer, engine
├── video/                # stream handler, frame processor
└── rnd/                  # rnd_runner.py

scripts/tools/
└── build_embeddings.py   # builds main.pkl from face images

examples/clients/
└── rnd_main.py           # pipeline entry point

volumes/
├── src/embeddings/       # main.pkl  ← face embedding database
└── rnd_results/          # per-video output JSON files

compose.rnd.yml           # standalone R&D compose (no Redis/Postgres)
requirements.rnd.txt      # minimal deps for local (non-Docker) use
.env.example              # copy to .env before first run
```

---

## compose.sh Reference

```bash
./compose.sh build        # build Docker image
./compose.sh start -l     # start pipeline + stream logs
./compose.sh stop         # stop
./compose.sh enter        # shell into container
./compose.sh logs         # view logs
```

---

## rnd_config.yaml Settings

| Key | Default | Description |
|---|---|---|
| `gpu_id` | `0` | GPU device (-1 for CPU) |
| `match_threshold` | `0.3` | Cosine similarity threshold for recognition |
| `db_path` | `volumes/src/embeddings/main.pkl` | Face embedding database |
| `output_dir` | `volumes/rnd_results` | Where JSON results are saved |
| `max_track_lifetime_seconds` | `30` | Max duration of a tracked face |
| `minimum_face_size` | `50` | Minimum face size in pixels |

---

## References

- [InsightFace](https://github.com/deepinsight/insightface) — face detection & embeddings
- [DeepOCSORT](https://github.com/mikel-brostrom/yolo_tracking) — multi-object tracking
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) — tracking evaluation metrics

---

**Branch**: `rnd/liteFaceR` | **Base**: `main` (production)
