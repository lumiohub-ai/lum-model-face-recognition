"""R&D pipeline runner - processes video files sequentially and saves recognition results as JSON."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from ..video.stream_handler import StreamHandler
from ..core.engine import FaceEngine


class RnDRunner:
    """Offline face recognition pipeline for R&D.

    Reads a list of videos from rnd_config.yaml, runs the recognition
    pipeline on each, and saves per-video JSON result files.
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
        """Build an args namespace for FaceEngine from config + per-video entry."""
        args = type("Args", (), {})()

        # Global config
        args.gpu_id = self.config.get("gpu_id", 0)
        args.match_threshold = self.config.get("match_threshold", 0.3)
        args.partial_match_threshold = self.config.get("partial_match_threshold", 0.15)
        args.db_path = self.config.get("db_path", "volumes/src/embeddings/main.pkl")
        args.timezone = self.config.get("timezone", "UTC")
        args.max_track_lifetime_seconds = self.config.get("max_track_lifetime_seconds", 30)
        args.minimum_face_size = self.config.get("minimum_face_size", 50)
        args.client_slug = self.config.get("client_slug", "rnd")

        # Disabled for R&D
        args.use_pgvector = False
        args.shadow_mode = False
        args.production = False
        args.debug = self.config.get("debug", True)
        args.logger = logger

        # Per-video settings
        args.cam_type = video_entry.get("cam_type", "IN")
        args.camera_name = video_entry.get("camera_name", "cam")
        args.camera_id = video_entry.get("camera_id", None)
        args.video_path = video_entry["path"]
        args.roi = video_entry.get("roi", None)
        args.line_points = video_entry.get("line_points", None)

        return args

    def _process_video(self, video_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Run the recognition pipeline on a single video file."""
        video_path = video_entry["path"]
        annotation_file = video_entry.get("annotation", None)

        logger.info(f"Processing: {video_path}")

        args = self._build_args(video_entry)
        stream = StreamHandler(video_path, logger)
        engine = FaceEngine(args=args)

        results: List[Dict[str, Any]] = []
        frame_num = 0

        while True:
            ret, frame = stream.read()
            if not ret:
                break

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

        # Finalize any tracks still open at end of video
        persons = engine.recognize_removed_tracks([], last_frame=True)
        self._collect_results(persons, frame_num, results)

        stream.stop()

        logger.info(f"Done: {frame_num} frames, {len(results)} recognition events")
        return {
            "video": video_path,
            "annotation_file": annotation_file,
            "total_frames": frame_num,
            "results": results,
        }

    def _collect_results(
        self,
        persons_recognized: Dict[str, List],
        frame_num: int,
        results: List[Dict[str, Any]],
    ) -> None:
        """Append recognition events from one frame pass into results list."""
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

    def run(self) -> None:
        """Process all videos listed in the config sequentially."""
        videos = self.config.get("videos", [])
        if not videos:
            logger.warning("No videos defined in rnd_config.yaml.")
            return

        logger.info(f"R&D pipeline started — {len(videos)} video(s) to process")

        for i, video_entry in enumerate(videos, 1):
            logger.info(f"[{i}/{len(videos)}] {video_entry['path']}")

            output = self._process_video(video_entry)

            stem = Path(video_entry["path"]).stem
            out_path = os.path.join(self.output_dir, f"{stem}_results.json")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

            logger.info(f"Saved → {out_path}")

        logger.info("R&D pipeline complete.")
