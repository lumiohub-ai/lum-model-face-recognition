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
