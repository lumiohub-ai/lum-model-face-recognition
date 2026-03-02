# Face Recognition — R&D Branch

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Offline face recognition pipeline for research and evaluation.
Processes frame directories or video files against a face embedding database and outputs per-source JSON result files for annotation-based evaluation.

> **Branch:** `rnd/liteFaceR` — production dependencies (API, Redis, pgvector, GCS) are removed.

---

## How It Works

```
volumes/src/images/{folder}/       ← enrollment face images (.jpg, .png, .pgm, …)
        ↓
build_embeddings.py                ← generates volumes/src/embeddings/main.pkl
        ↓
configs/rnd_config.yaml            ← define sources (frame dirs or videos) + annotations
        ↓
./compose.sh start -l              ← runs pipeline on each source
        ↓
volumes/rnd_results/{source}_results.json   ← recognition events per source
```

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA 12.2+ (recommended)
- **CPU**: 4+ cores (CPU-only mode: set `gpu_id: -1` in config)
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

Place one or more images per person under `volumes/src/images/{folder_name}/`.

**Supported naming conventions:**
- `{name}_{index}.jpg` — multiple images per person: `oybek_1.jpg`, `oybek_2.jpg`
- `{name}.jpg` — single image per person: `oybek.jpg`, `0001.pgm`

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.pgm`

```
volumes/src/images/
└── folder_1/
    ├── oybek_1.jpg
    ├── oybek_2.jpg
    ├── john_1.jpg
    └── 0001.pgm       ← plain name, PGM format also works
```

> **Note:** Ensure files are readable by the Docker container: `chmod -R a+r volumes/src/images/`

### 3. Build Embedding Database

**With Docker:**

Update `--input` to point to your images folder, then run:

```bash
./scripts/tools/docker_build_embeddings.sh --input volumes/src/images/folder_1
```

The script mounts `volumes/src/`, `scripts/`, and `src/` so any local code changes are picked up without rebuilding the image.

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

### 4. Configure Sources

Edit `configs/rnd_config.yaml`. Sources can be **video files** or **frame directories**:

```yaml
gpu_id: 0
match_threshold: 0.3
db_path: "volumes/src/embeddings/main.pkl"
output_dir: "volumes/rnd_results"
fps: 25            # used for time calculations in eval mode
eval: false        # set true to write per-recognition CSV alongside JSON

videos:
  # Frame directory
  - path: "volumes/src/datset/chockpoint/P1E_S1/P1E_S1_C1"
    annotation: "volumes/src/annotation/chockpoint/P1E/P1E_S1/P1E_S1_C1.json"
    camera_name: "P1E_S1_C1"
    cam_type: "IN"

  # Video file
  - path: "volumes/src/videos/clip1.mp4"
    annotation: "volumes/src/annotation/clip1.json"
    camera_name: "cam1"
    cam_type: "IN"
```

> **Permissions:** If using a dataset extracted from an archive, ensure the container can read the files:
> `sudo chmod -R a+rX volumes/src/datset/`

### 5. Run Pipeline

```bash
./compose.sh start -l
```

Results are written to `volumes/rnd_results/{camera_name}_results.json`.

---

## Output Format

```json
{
  "source": "volumes/src/datset/chockpoint/P1E_S1/P1E_S1_C1",
  "annotation_file": "volumes/src/annotation/chockpoint/P1E/P1E_S1/P1E_S1_C1.json",
  "total_frames": 2294,
  "results": [
    {
      "frame_num": 142,
      "track_id": 3,
      "name": "0003",
      "recognized": "recognized",
      "status": "RECOGNIZED",
      "similarity": 0.87,
      "confidence": 0.84,
      "appear_time": "2026-02-18T10:00:05.123"
    }
  ]
}
```

When `eval: true`, a CSV is also written to `volumes/rnd_results/{camera_name}_eval.csv`:

```
time,name,cam_type
0:05,0003,IN
0:12,0007,IN
```

---

## Project Structure

```
configs/
└── rnd_config.yaml       # R&D sources and settings

volumes/src/
├── images/               # enrollment face images → input for build_embeddings
├── embeddings/main.pkl   # generated face database
├── datset/               # dataset frame directories (gitignored)
└── annotation/           # ground-truth annotation JSON files (gitignored)

volumes/rnd_results/      # per-source output JSON (and CSV if eval: true)

src/face_recognition/
├── core/                 # detector, tracker, recognizer, engine
├── video/                # stream handler, frame processor
└── rnd/                  # rnd_runner.py

scripts/tools/
└── build_embeddings.py   # builds main.pkl from face images

examples/clients/
└── rnd_main.py           # pipeline entry point

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
| `partial_match_threshold` | `0.15` | Partial match threshold |
| `db_path` | `volumes/src/embeddings/main.pkl` | Face embedding database |
| `output_dir` | `volumes/rnd_results` | Where JSON results are saved |
| `max_track_lifetime_seconds` | `30` | Max duration of a tracked face |
| `minimum_face_size` | `50` | Minimum face size in pixels |
| `fps` | `25` | Frames per second (used for time calculations) |
| `eval` | `false` | Write per-recognition CSV alongside JSON |
| `timezone` | `UTC` | Timezone for appear_time timestamps |
| `debug` | `true` | Enable debug logging |

---

## References

- [InsightFace](https://github.com/deepinsight/insightface) — face detection & embeddings
- [DeepOCSORT](https://github.com/mikel-brostrom/yolo_tracking) — multi-object tracking
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) — tracking evaluation metrics

---

**Branch**: `rnd/liteFaceR` | **Base**: `main` (production)
