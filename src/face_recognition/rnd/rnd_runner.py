"""R&D pipeline runner - processes video files or frame directories sequentially."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import yaml
from loguru import logger

from ..video.stream_handler import StreamHandler
from ..core.engine import FaceEngine


# --------------------------------------------------------------------------- #
# Frame directory reader — same interface as StreamHandler                    #
# --------------------------------------------------------------------------- #

class FrameDirReader:
    """Reads frames from a directory of sorted image files.

    Frames are read in ascending numeric order based on filename
    (e.g. 00000293.jpg → frame_num 293).
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.is_video = False

        files = sorted([
            f for f in os.listdir(directory)
            if Path(f).suffix.lower() in self.IMAGE_EXTS
        ])
        if not files:
            raise ValueError(f"No image files found in: {directory}")

        self._files: List[str] = files
        self._index: int = 0

        # Read first frame to expose .frame property (matches StreamHandler API)
        first_path = os.path.join(directory, self._files[0])
        self.frame = cv2.imread(first_path)
        if self.frame is None:
            raise ValueError(f"Cannot read first frame: {first_path}")

        self.fps = 1          # frame dirs have no inherent FPS
        self.stopped = False

    def _frame_num_from_name(self, filename: str) -> int:
        """Extract numeric frame number from filename (e.g. '00000293.jpg' → 293)."""
        return int(Path(filename).stem)

    def read(self) -> Tuple[bool, Optional[Any]]:
        if self._index >= len(self._files):
            return False, None

        path = os.path.join(self.directory, self._files[self._index])
        frame = cv2.imread(path)
        self._index += 1

        if frame is None:
            logger.warning(f"Cannot read frame: {path}, skipping")
            return self.read()  # skip unreadable files

        return True, frame

    def current_frame_num(self) -> int:
        """Return the numeric frame number of the frame most recently returned by read()."""
        idx = max(0, self._index - 1)
        return self._frame_num_from_name(self._files[idx])

    def stop(self) -> None:
        self.stopped = True

    def __len__(self) -> int:
        return len(self._files)


# --------------------------------------------------------------------------- #
# R&D runner                                                                  #
# --------------------------------------------------------------------------- #

class RnDRunner:
    """Offline face recognition pipeline for R&D.

    Supports both video files (.mp4, .avi, …) and directories of image frames.
    Reads config from rnd_config.yaml and saves per-source JSON result files.
    """

    def __init__(self, config_path: str = "configs/rnd_config.yaml") -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config.get("output_dir", "volumes/rnd_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logger()

    def _setup_logger(self) -> None:
        import sys
        logger.remove()
        level = "DEBUG" if self.config.get("debug", False) else "INFO"
        logger.add(sys.stderr, level=level, colorize=True,
                   format="<green>{level: <8}</green> | <green>{message}</green>")

    def _build_args(self, video_entry: Dict[str, Any]) -> Any:
        """Build an args namespace for FaceEngine from config + per-source entry."""
        args = type("Args", (), {})()

        args.gpu_id = self.config.get("gpu_id", 0)
        args.match_threshold = self.config.get("match_threshold", 0.3)
        args.partial_match_threshold = self.config.get("partial_match_threshold", 0.15)
        args.db_path = self.config.get("db_path", "volumes/src/embeddings/main.pkl")
        args.timezone = self.config.get("timezone", "UTC")
        args.max_track_lifetime_seconds = self.config.get("max_track_lifetime_seconds", 30)
        args.minimum_face_size = self.config.get("minimum_face_size", 50)
        args.client_slug = self.config.get("client_slug", "rnd")

        args.use_pgvector = False
        args.shadow_mode = False
        args.production = False
        args.debug = self.config.get("debug", True)
        args.logger = logger

        args.cam_type = video_entry.get("cam_type", "IN")
        args.camera_name = video_entry.get("camera_name", "cam")
        args.camera_id = video_entry.get("camera_id", None)
        args.video_path = video_entry["path"]
        args.roi = video_entry.get("roi", None)
        args.line_points = video_entry.get("line_points", None)

        return args

    def _open_source(self, path: str):
        """Return a FrameDirReader or StreamHandler depending on path type."""
        if os.path.isdir(path):
            logger.info(f"Source is a frame directory ({len(os.listdir(path))} files)")
            return FrameDirReader(path)
        else:
            logger.info("Source is a video file")
            return StreamHandler(path, logger)

    def _process_source(self, video_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Run the recognition pipeline on a video file or frame directory."""
        src_path = video_entry["path"]
        annotation_file = video_entry.get("annotation", None)

        logger.info(f"Processing: {src_path}")

        args = self._build_args(video_entry)
        source = self._open_source(src_path)
        engine = FaceEngine(args=args)

        results: List[Dict[str, Any]] = []
        frame_num = 0

        while True:
            ret, frame = source.read()
            if not ret:
                break

            # Use filename-embedded number for frame dirs, sequential for video
            if isinstance(source, FrameDirReader):
                frame_num = source.current_frame_num()
            else:
                frame_num += 1

            roi = args.roi
            frame_crop = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame

            active_tracks, removed_tracks = engine.track(frame_crop)
            expired = engine.prune_long_lived_tracks()
            removed_tracks.extend(expired)

            if active_tracks:
                engine.process_active_tracks(active_tracks, frame_crop, frame_num)

            persons = engine.recognize_removed_tracks(removed_tracks)
            self._collect_results(persons, frame_num, results)

        # Finalize any tracks still open at end
        persons = engine.recognize_removed_tracks([], last_frame=True)
        self._collect_results(persons, frame_num, results)

        source.stop()

        total = len(source) if isinstance(source, FrameDirReader) else frame_num
        logger.info(f"Done: {total} frames processed, {len(results)} recognition events")

        output: Dict[str, Any] = {
            "source": src_path,
            "annotation_file": annotation_file,
            "total_frames": total,
            "results": results,
        }

        if annotation_file:
            evaluation = self._evaluate(results, annotation_file)
            output["evaluation"] = evaluation
            self._log_evaluation(evaluation)

        return output

    def _collect_results(
        self,
        persons_recognized: Dict[str, List],
        frame_num: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Append recognition events into results list."""
        for name, (track_id, appear_time, recognized_status, image, info) in persons_recognized.items():
            results.append({
                "frame_num": frame_num,
                "track_id": int(track_id),
                "name": name,
                "recognized": recognized_status,
                "status": info.get("status"),
                "similarity": round(float(info.get("similarity", 0.0)), 4),
                "confidence": round(float(info.get("confidence", 0.0)), 4),
                "appear_time": appear_time.isoformat() if appear_time else None,
            })

    def _evaluate(self, results: List[Dict[str, Any]], annotation_file: str) -> Dict[str, Any]:
        """Compare predicted entry sequence against ground-truth annotation.

        Annotation format: {"person_id": "frame_number"} — ordered by appearance.
        Predicted sequence: first recognition event per person, sorted by frame_num.
        """
        with open(annotation_file) as f:
            annotation: Dict[str, str] = json.load(f)  # preserves insertion order (py3.7+)

        gt_sequence: List[str] = list(annotation.keys())  # already ordered by frame number

        # First recognition event per person
        first_seen: Dict[str, int] = {}
        for r in results:
            name = r["name"]
            if name not in first_seen or r["frame_num"] < first_seen[name]:
                first_seen[name] = r["frame_num"]

        pred_sequence: List[str] = sorted(first_seen, key=lambda n: first_seen[n])

        gt_set = set(gt_sequence)
        pred_set = set(pred_sequence)

        correctly_identified = gt_set & pred_set
        missed = gt_set - pred_set
        false_positives = pred_set - gt_set
        identification_rate = len(correctly_identified) / len(gt_set) if gt_set else 0.0

        # Accuracy_event = (1/N) * sum_{i=1..N} I(p_i == g_i)
        # N = len(gt_sequence); pred_sequence may be shorter (missing persons = wrong)
        n = len(gt_sequence)
        matches = sum(
            1 for i in range(n)
            if i < len(pred_sequence) and pred_sequence[i] == gt_sequence[i]
        )
        accuracy_event = matches / n if n else 0.0

        return {
            "gt_sequence": gt_sequence,
            "pred_sequence": pred_sequence,
            "correctly_identified": sorted(correctly_identified),
            "missed": sorted(missed),
            "false_positives": sorted(false_positives),
            "total_gt": n,
            "total_predicted": len(pred_set),
            "identification_rate": round(identification_rate, 4),
            "accuracy_event": round(accuracy_event, 4),
        }

    def _log_evaluation(self, ev: Dict[str, Any]) -> None:
        """Log a compact evaluation summary."""
        logger.info(
            f"Evaluation — accuracy_event: {ev['accuracy_event']:.4f}  "
            f"identified: {len(ev['correctly_identified'])}/{ev['total_gt']} "
            f"({ev['identification_rate']*100:.1f}%)  "
            f"missed: {len(ev['missed'])}  "
            f"false_positives: {len(ev['false_positives'])}"
        )
        if ev["missed"]:
            logger.info(f"  Missed persons: {ev['missed']}")
        if ev["false_positives"]:
            logger.info(f"  False positives: {ev['false_positives']}")

    def run(self) -> None:
        """Process all sources listed in the config sequentially."""
        videos = self.config.get("videos", [])
        if not videos:
            logger.warning("No videos defined in rnd_config.yaml.")
            return

        logger.info(f"R&D pipeline started — {len(videos)} source(s) to process")

        for i, video_entry in enumerate(videos, 1):
            logger.info(f"[{i}/{len(videos)}] {video_entry['path']}")

            output = self._process_source(video_entry)

            stem = Path(video_entry["path"]).stem
            out_path = os.path.join(self.output_dir, f"{stem}_results.json")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

            logger.info(f"Saved → {out_path}")

        logger.info("R&D pipeline complete.")
