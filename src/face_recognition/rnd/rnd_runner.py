"""R&D pipeline runner - processes video files or frame directories sequentially."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
        args.eval = self.config.get("eval", False)
        args.logger = logger

        args.cam_type = video_entry.get("cam_type", "IN")
        args.camera_name = video_entry.get("camera_name", "cam")
        args.camera_id = video_entry.get("camera_id", None)
        args.video_path = video_entry["path"]
        args.roi = video_entry.get("roi", None)
        args.line_points = video_entry.get("line_points", None)
        args.fps = self.config.get("fps", 25)

        output_dir = self.config.get("output_dir", "volumes/rnd_results")
        args.txt_path = os.path.join(output_dir, f"{args.camera_name}_eval.csv")

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
            entry = {
                "frame_num": frame_num,
                "track_id": int(track_id),
                "name": name,
                "recognized": recognized_status,
                "status": info.get("status"),
                "similarity": round(float(info.get("similarity", 0.0)), 4),
                "confidence": round(float(info.get("confidence", 0.0)), 4),
                "appear_time": appear_time.isoformat() if appear_time else None,
            }
            results.append(entry)
            logger.debug(
                f"[COLLECT] frame={frame_num} track={track_id} name={name!r} "
                f"status={recognized_status} sim={entry['similarity']:.4f}"
            )

    def _evaluate(self, results: List[Dict[str, Any]], annotation_file: str) -> Dict[str, Any]:
        """Compare predicted entry sequence against ground-truth annotation.

        Annotation format: {"person_id": "frame_number"} — ordered by appearance.
        Predicted sequence: first recognition event per person, sorted by frame_num.
        """
        with open(annotation_file) as f:
            annotation: Dict[str, str] = json.load(f)  # preserves insertion order (py3.7+)

        gt_sequence: List[str] = list(annotation.keys())  # already ordered by frame number

        logger.debug(f"[EVAL] GT sequence ({len(gt_sequence)}): {gt_sequence}")
        logger.debug(f"[EVAL] All results ({len(results)} entries):")
        for r in results:
            logger.debug(
                f"  frame={r['frame_num']} track={r['track_id']} name={r['name']!r} "
                f"status={r['recognized']} sim={r['similarity']:.4f}"
            )

        # First recognition event per person
        first_seen: Dict[str, int] = {}
        for r in results:
            name = r["name"]
            if name not in first_seen or r["frame_num"] < first_seen[name]:
                first_seen[name] = r["frame_num"]
                logger.debug(
                    f"[EVAL] first_seen[{name!r}] = frame {r['frame_num']} "
                    f"(status={r['recognized']} sim={r['similarity']:.4f})"
                )

        pred_sequence: List[str] = sorted(first_seen, key=lambda n: first_seen[n])
        logger.debug(f"[EVAL] Pred sequence ({len(pred_sequence)}): {pred_sequence}")

        gt_set = set(gt_sequence)
        pred_set = set(pred_sequence)

        correctly_identified = gt_set & pred_set
        missed = gt_set - pred_set
        false_positives = pred_set - gt_set
        identification_rate = len(correctly_identified) / len(gt_set) if gt_set else 0.0

        # Accuracy_event = (1/N) * sum_{i=1..N} I(p_i == g_i)
        # N = len(gt_sequence); pred_sequence may be shorter (missing persons = wrong)
        n = len(gt_sequence)
        logger.debug("[EVAL] Position-by-position sequence match:")
        matches = 0
        for i in range(n):
            pred = pred_sequence[i] if i < len(pred_sequence) else "<missing>"
            gt = gt_sequence[i]
            match = pred == gt
            if match:
                matches += 1
            logger.debug(f"  [{i:02d}] gt={gt!r}  pred={pred!r}  {'✓' if match else '✗'}")
        accuracy_event = matches / n if n else 0.0

        curves = self._compute_biometric_curves(results, annotation)

        stem = Path(annotation_file).stem
        plot_path = os.path.join(self.output_dir, f"{stem}_det_curve.png")
        self._plot_det_curve(curves, plot_path)

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
            "biometric_curves": curves,
        }

    def _compute_biometric_curves(
        self,
        results: List[Dict[str, Any]],
        annotation: Dict[str, str],
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute TPIR/FPIR curves using track-level probe scoring per NIST FRVT / ISO 19795-1.

        Each result entry = one probe/transaction (one track).
        TPIR(T) = correct-identity probes accepted at T / all genuine probes
        FPIR(T) = wrong-identity probes accepted at T / all impostor probes

        Definitions:
          Genuine probe  : any track whose GT window belongs to a known person
          Impostor probe : a recognized (non-unrecognized) track claiming the wrong identity
          FN             : GT person with zero track results → one missed probe at score 0.0
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 101)

        # Build per-person frame windows from annotation {person_id: first_frame}
        gt_sorted = sorted(annotation.items(), key=lambda x: int(x[1]))
        windows: List[Tuple[str, int, float]] = []
        for i, (pid, first_frame) in enumerate(gt_sorted):
            start = int(first_frame)
            end = int(gt_sorted[i + 1][1]) - 1 if i + 1 < len(gt_sorted) else float("inf")
            windows.append((pid, start, end))

        def find_true_id(frame_num: int) -> Optional[str]:
            for pid, start, end in windows:
                if start <= frame_num <= end:
                    return pid
            return None

        # Each entry: (score, is_correct_identity)
        genuine_probes: List[Tuple[float, bool]] = []
        impostor_scores: List[float] = []  # wrong-identity accepted claims

        logger.debug(f"[BIO] GT windows ({len(windows)}): {[(p, s, e) for p, s, e in windows]}")
        for r in results:
            true_id = find_true_id(r["frame_num"])
            if true_id is None:
                logger.debug(
                    f"[BIO] SKIP frame={r['frame_num']} name={r['name']!r} — outside GT windows"
                )
                continue

            is_correct = (r["name"] == true_id)
            genuine_probes.append((r["similarity"], is_correct))

            tag = "GENUINE-CORRECT" if is_correct else "GENUINE-WRONG"
            logger.debug(
                f"[BIO] {tag} frame={r['frame_num']} track={r['track_id']} "
                f"name={r['name']!r} true_id={true_id!r} sim={r['similarity']:.4f} "
                f"status={r['recognized']}"
            )

            # Wrong-identity recognized claim → impostor probe
            if r["recognized"] == "recognized" and not is_correct:
                impostor_scores.append(r["similarity"])
                logger.debug(f"[BIO] → also counted as IMPOSTOR score={r['similarity']:.4f}")

        # GT persons with zero track results → one missed probe at score 0.0
        persons_with_results = {
            find_true_id(r["frame_num"])
            for r in results
            if find_true_id(r["frame_num"]) is not None
        }
        for pid, _, _ in windows:
            if pid not in persons_with_results:
                genuine_probes.append((0.0, False))
                logger.debug(f"[BIO] FN-INJECT pid={pid!r} — no tracks, injecting score=0.0")

        n_genuine = len(genuine_probes)
        n_impostor = len(impostor_scores)

        tpir_values, fpir_values = [], []
        for t in thresholds:
            tp = sum(1 for score, correct in genuine_probes if score >= t and correct)
            fp = sum(1 for score in impostor_scores if score >= t)
            tpir_values.append(tp / n_genuine if n_genuine else 0.0)
            fpir_values.append(fp / n_impostor if n_impostor else 0.0)

        n_correct_tracks = sum(1 for score, correct in genuine_probes if score >= 0.5 and correct)

        return {
            "thresholds": [round(t, 4) for t in thresholds.tolist()],
            "tpir": [round(v, 4) for v in tpir_values],
            "fpir": [round(v, 4) for v in fpir_values],
            "n_genuine_probes": n_correct_tracks,
            "n_genuine_total": n_genuine,
            "n_impostor_probes": n_impostor,
        }

    def _plot_det_curve(self, curves: Dict[str, Any], output_path: str) -> None:
        """Plot TPIR vs FPIR curve and save to output_path."""
        fpir = curves["fpir"]
        tpir = curves["tpir"]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpir, tpir, "b-", linewidth=2)
        ax.set_xlabel("FPIR (False Positive Identification Rate)")
        ax.set_ylabel("TPIR (True Positive Identification Rate)")
        ax.set_title("TPIR vs FPIR — DET Curve")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True)
        fig.tight_layout()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        logger.info(f"DET curve saved → {output_path}")

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

        curves = ev.get("biometric_curves", {})
        if curves:
            logger.info(
                f"  Biometric probes — genuine tracks: {curves['n_genuine_probes']}/{curves['n_genuine_total']}, "
                f"impostor: {curves['n_impostor_probes']}"
            )
            # Log TPIR/FPIR at the mid-threshold (T=0.5) as a reference operating point
            thresholds = curves["thresholds"]
            idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.5))
            logger.info(
                f"  @ T=0.50 — TPIR: {curves['tpir'][idx]:.4f}, FPIR: {curves['fpir'][idx]:.4f}"
            )

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
