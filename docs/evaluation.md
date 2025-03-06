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

### 3. Generate MOT Format Predictions

Run the prediction script to generate tracking results in MOT format:

```bash
python src/face_recognition/evaluation/predict_mot.py
```

Note: You must adjust video paths and configurations inside the code.

### 4. Evaluate Tracking Performance

1. Create dataset and tracker configurations following [TrackEval MOT Challenge guide](https://github.com/humblebeeintel/TrackEval/blob/main/docs/MOTChallenge-Official/Readme.md)

2. Run the evaluation script:

```bash
python TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK <YOUR-CHALLENGE> \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL MPNTrack \
    --METRICS HOTA CLEAR Identity VACE \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1
```

### 5. Recognition Accuracy Evaluation

1. Locate the recognition results file (created alongside MOT predictions with `_recognition` suffix)
2. Run the recognition evaluation script:

```bash
python src/face_recognition/evaluation/reca.py
```

3. Update the following paths in the code:
   ```python
   gt_path = 'output_mot_11.txt'
   pred_path = 'output_11_processed_fps_recognition.txt'
   ```

4. Ensure the `id_to_name` dictionary in the code matches your dataset