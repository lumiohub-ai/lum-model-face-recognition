"""R&D pipeline runner - processes video files or frame directories sequentially."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

from ..video.stream_handler import StreamHandler
from ..core.engine import FaceEngine
from .gt_matcher import parse_xml_groundtruth, match_track_to_gt
from .visualize_tracks import draw_frame_overlays, gt_active_at, draw_track_table, \
                               find_gt_id_for_track, draw_gt_labels


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

        # Load XML groundtruth for spatial matching if available
        xml_gt = None
        groundtruth_dir = self.config.get("groundtruth_dir")
        camera_name = video_entry.get("camera_name")
        if groundtruth_dir and camera_name:
            xml_path = os.path.join(groundtruth_dir, f"{camera_name}.xml")
            if os.path.exists(xml_path):
                xml_gt = parse_xml_groundtruth(xml_path)
                logger.info(f"Loaded XML GT: {xml_path} ({len(xml_gt)} annotated frames)")
            else:
                logger.debug(f"No XML GT found at {xml_path}, using temporal fallback")

        # --- visualization setup ---
        visualize = self.config.get("visualize", False)
        vis_proc = None
        vis_annotation: Dict[str, Any] = {}
        vis_out_path = ""
        track_registry: Dict[int, Dict[str, int]] = {}
        if visualize:
            vis_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(vis_dir, exist_ok=True)
            vis_out_path = os.path.join(vis_dir, f"{args.camera_name}_tracks.mp4")
            if annotation_file and os.path.exists(annotation_file):
                with open(annotation_file) as _f:
                    vis_annotation = json.load(_f)

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

            # --- visualization frame ---
            if visualize:
                active_ids = {t.id for t in active_tracks}
                for t in active_tracks:
                    if t.id not in track_registry:
                        # Birth: temporal fallback only (XML may not have coverage yet)
                        gt_id = find_gt_id_for_track(vis_annotation, frame_num) if vis_annotation else None
                        track_registry[t.id] = {"first": frame_num, "last": frame_num, "gt_id": gt_id}
                    elif t.time_since_update == 0:
                        track_registry[t.id]["last"] = frame_num

                    # Continuously correct GT with spatial XML whenever frame data is available
                    if xml_gt and frame_num in xml_gt and t.history_observations:
                        single_frame_boxes = {frame_num: list(t.history_observations[-1])}
                        updated_gt = find_gt_id_for_track(
                            vis_annotation, frame_num,
                            track_boxes=single_frame_boxes,
                            xml_gt=xml_gt,
                        )
                        track_registry[t.id]["gt_id"] = updated_gt

                # Per-frame similarity for live overlay (embeddings are ready after track())
                track_sims: Dict[int, Tuple[str, float]] = engine.get_active_track_sims(active_tracks)

                vis = engine.visualize_tracks(frame_crop)
                draw_gt_labels(vis, active_tracks, track_registry, track_sims)
                active_persons = gt_active_at(vis_annotation, frame_num) if vis_annotation else []
                draw_frame_overlays(vis, frame_num, active_persons, vis_annotation)

                panel = draw_track_table(vis.shape[0], track_registry, active_ids)
                combined = np.hstack([vis, panel])

                if vis_proc is None:
                    ch, cw = combined.shape[:2]
                    vis_proc = subprocess.Popen(
                        [
                            "ffmpeg", "-y",
                            "-f", "rawvideo",
                            "-vcodec", "rawvideo",
                            "-pix_fmt", "bgr24",
                            "-s", f"{cw}x{ch}",
                            "-r", str(self.config.get("vis_fps", 25)),
                            "-i", "pipe:0",
                            "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-preset", "fast",
                            vis_out_path,
                        ],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    logger.info(f"[VIS] Writing {cw}x{ch} → {vis_out_path}")
                vis_proc.stdin.write(combined.tobytes())

            expired = engine.prune_long_lived_tracks()
            removed_tracks.extend(expired)

            if active_tracks:
                engine.process_active_tracks(active_tracks, frame_crop, frame_num)

            persons = engine.recognize_removed_tracks(removed_tracks)
            if visualize:
                for info in persons.values():
                    tid, _, _, _, rec_info = info[0], info[1], info[2], info[3], info[4]
                    if tid in track_registry:
                        track_registry[tid]["sim"] = round(rec_info.get("similarity", 0.0), 2)
                        track_registry[tid]["rec"] = rec_info.get("name", "?")
            self._collect_results(persons, frame_num, results, xml_gt)

        # Finalize any tracks still open at end
        persons = engine.recognize_removed_tracks([], last_frame=True)
        if visualize:
            for info in persons.values():
                tid, _, _, _, rec_info = info[0], info[1], info[2], info[3], info[4]
                if tid in track_registry:
                    track_registry[tid]["sim"] = round(rec_info.get("similarity", 0.0), 2)
                    track_registry[tid]["rec"] = rec_info.get("name", "?")
        self._collect_results(persons, frame_num, results, xml_gt)

        source.stop()
        if vis_proc is not None:
            vis_proc.stdin.close()
            vis_proc.wait()
            logger.info(f"[VIS] Saved: {vis_out_path}")

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
            if xml_gt:
                evaluation.update(self._compute_frame_metrics(results, xml_gt))
            output["evaluation"] = evaluation
            self._log_evaluation(evaluation)

        # Strip internal/redundant fields from result entries before serialization
        _STRIP = {"confidence", "_debug_n_embeddings", "_debug_embed_span",
                  "_xml_frames_checked", "appear_time"}
        for r in output["results"]:
            for f in _STRIP:
                r.pop(f, None)

        return output

    def _collect_results(
        self,
        persons_recognized: Dict[str, List],
        frame_num: int,
        results: List[Dict[str, Any]],
        xml_gt: Optional[Dict] = None,
    ) -> None:
        """Append recognition events into results list."""
        for _key, (track_id, appear_time, recognized_status, image, info) in persons_recognized.items():
            name = info.get('name', _key)  # key is track_id str in eval mode
            track_boxes = info.pop("_track_boxes", {})
            if xml_gt and track_boxes:
                true_gt_id, xml_frames_checked = match_track_to_gt(track_boxes, xml_gt)
            else:
                true_gt_id, xml_frames_checked = None, 0

            entry = {
                "first_frame_num": info.get('first_frame_num', frame_num),
                "last_frame_num": info.get('last_frame_num', frame_num),
                "track_id": int(track_id),
                "name": name,
                "recognized": recognized_status,
                "status": info.get("status"),
                "similarity": round(float(info.get("similarity", 0.0)), 4),
                "confidence": round(float(info.get("confidence", 0.0)), 4),
                "appear_time": appear_time.isoformat() if appear_time else None,
                "true_gt_id": true_gt_id,
                "_xml_frames_checked": xml_frames_checked,
                "_debug_n_embeddings": info.get('_debug_n_embeddings'),
                "_debug_embed_span":   info.get('_debug_embed_span'),
            }
            results.append(entry)

    def _evaluate(self, results: List[Dict[str, Any]], annotation_file: str) -> Dict[str, Any]:
        """Compare predicted entry sequence against ground-truth annotation.

        Annotation format: {"person_id": {"start": "frame_num", "end": "frame_num"}} — ordered by appearance.
        Predicted sequence: first recognition event per person, sorted by frame_num.
        """
        with open(annotation_file) as f:
            annotation: Dict[str, str] = json.load(f)  # preserves insertion order (py3.7+)

        gt_sequence: List[str] = list(annotation.keys())  # already ordered by frame number



        # Partition results by spatial validity — exclude tracks where XML confirms wrong identity
        # or confirms bystander (XML frames checked but no GT eye in bbox).
        # has_spatial=True when any track's frames overlapped with XML annotations.
        _MIN_XML_FRAMES = 3  # minimum frames checked to conclude bystander (vs no-data edge case)
        has_spatial = any(r.get("_xml_frames_checked", 0) > 0 for r in results)
        spatially_rejected: List[Dict[str, Any]] = []
        valid_results: List[Dict[str, Any]] = results

        if has_spatial:
            valid_results = []
            for r in results:
                true_gt_id = r.get("true_gt_id")
                xml_checked = r.get("_xml_frames_checked", 0)

                if true_gt_id is not None and true_gt_id != r["name"]:
                    # Case 1: XML spatially confirmed a different identity
                    spatially_rejected.append(r)
                    logger.debug(
                        f"[EVAL] SPATIAL-REJECT (wrong-id) track={r['track_id']} name={r['name']!r} "
                        f"true_gt_id={true_gt_id!r} sim={r['similarity']:.4f}"
                    )
                elif true_gt_id is None and xml_checked >= _MIN_XML_FRAMES:
                    # Case 2: XML was present for N frames, no GT eye ever landed in bbox → bystander
                    spatially_rejected.append(r)
                    logger.debug(
                        f"[EVAL] SPATIAL-REJECT (bystander) track={r['track_id']} name={r['name']!r} "
                        f"xml_checked={xml_checked} sim={r['similarity']:.4f} — no GT eye in bbox"
                    )
                else:
                    valid_results.append(r)

        if spatially_rejected:
            logger.debug(f"[EVAL] {len(spatially_rejected)} spatially-rejected tracks excluded from evaluation")


        gt_set = set(gt_sequence)
        n = len(gt_sequence)

        # Spatial per-person matching using true_gt_id (XML eye-coordinate GT)
        has_spatial = any(r.get("true_gt_id") is not None for r in results)

        if not has_spatial:
            logger.warning("[EVAL] No XML GT spatial data — PLA requires spatial matching, skipped")
            pla = None
            fnr = None
            sir = None
            correctly_identified: set = set()
            missed: set = gt_set.copy()
            misidentified: set = set()
        else:
            results_by_true_id: Dict[str, List] = {}
            for r in results:
                tid = r.get("true_gt_id")
                if tid and tid in gt_set:
                    results_by_true_id.setdefault(tid, []).append(r)

            correctly_identified = set()
            missed = set()
            misidentified = set()

            for pid in gt_sequence:
                tracks = results_by_true_id.get(pid, [])
                if not tracks:
                    missed.add(pid)
                elif any(r["name"] == pid for r in tracks):
                    correctly_identified.add(pid)
                else:
                    misidentified.add(pid)

            pla = len(correctly_identified) / n if n else 0.0
            fnr = len(missed) / n if n else 0.0
            sir = len(misidentified) / n if n else 0.0

        probe_curves = self._compute_biometric_curves(results, annotation)

        return {
            "gt_sequence": gt_sequence,
            "correctly_identified": sorted(correctly_identified),
            "missed": sorted(missed),
            "misidentified": sorted(misidentified),
            "total_gt": n,
            "pla": round(pla, 4) if pla is not None else None,
            "fnr": round(fnr, 4) if fnr is not None else None,
            "sir": round(sir, 4) if sir is not None else None,
            "n_spatial_false_positives": len(spatially_rejected),
            "spatial_false_positives": [
                {
                    "track_id": r["track_id"],
                    "name": r["name"],
                    "true_gt_id": r.get("true_gt_id"),
                    "xml_frames_checked": r.get("_xml_frames_checked", 0),
                    "rejection_reason": (
                        "misidentified" if r.get("true_gt_id") is not None
                        else "confirmed_bystander"
                    ),
                    "similarity": r["similarity"],
                    "first_frame_num": r["first_frame_num"],
                }
                for r in spatially_rejected
            ],
            "probe_curves": probe_curves,
        }

    def _compute_biometric_curves(
        self,
        results: List[Dict[str, Any]],
        annotation: Dict[str, Dict[str, str]],
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

        # Each entry: (score, is_correct_identity)
        genuine_probes: List[Tuple[float, bool]] = []
        impostor_scores: List[float] = []  # wrong-identity accepted claims

        has_spatial_gt = any(r.get("_xml_frames_checked", 0) > 0 for r in results)
        # Minimum XML frames to confirm bystander (same constant as _evaluate)
        _MIN_XML_FRAMES_BIO = 3

        if has_spatial_gt:
            gt_persons = set(annotation.keys())
            for r in results:
                true_id = r.get("true_gt_id")
                xml_checked = r.get("_xml_frames_checked", 0)
                n_emb = r.get("_debug_n_embeddings")

                if true_id is not None and true_id in gt_persons:
                    # Genuine probe: enrolled person (correct or wrong label)
                    # GENUINE-WRONG stays in genuine denominator — does NOT affect FPIR
                    is_correct = (r["name"] == true_id)
                    genuine_probes.append((r["similarity"], is_correct))

                elif true_id is None and (xml_checked >= _MIN_XML_FRAMES_BIO or n_emb == 0):
                    # Impostor probe: XML-confirmed bystander, or null probe (face too small/unprocessable)
                    # Score=0.0 for null probes → correct rejection at any threshold > 0
                    impostor_scores.append(r["similarity"])

                # else: spurious track — insufficient spatial data to classify, skip

            # FN injection: enrolled persons with zero tracks → missed probe at score 0.0
            persons_with_results = {r["true_gt_id"] for r in results if r.get("true_gt_id") in gt_persons}
            for pid in annotation:
                if pid not in persons_with_results:
                    genuine_probes.append((0.0, False))

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
            "n_genuine_accepted": n_correct_tracks,
            "n_genuine": n_genuine,
            "n_impostors": n_impostor,
        }

    def _compute_frame_metrics(
        self,
        results: List[Dict[str, Any]],
        xml_gt: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute frame-level identification accuracy.

        For each XML-annotated frame f with person p: correct if a track exists
        where true_gt_id == p AND name == p AND track spans frame f.
        frame_accuracy = correct_frames / total_xml_frames (FNMR = 1 - frame_accuracy).
        """
        total = 0
        correct = 0
        for frame_num, persons_in_frame in xml_gt.items():
            for pid in persons_in_frame:
                total += 1
                for r in results:
                    if (r.get("true_gt_id") == pid
                            and r["name"] == pid
                            and r["first_frame_num"] <= frame_num <= r["last_frame_num"]):
                        correct += 1
                        break
        return {
            "n_frames": total,
            "n_frames_correct": correct,
            "fir": round(correct / total, 4) if total else None,
            "fnmr": round(1 - correct / total, 4) if total else None,
        }

    def _log_evaluation(self, ev: Dict[str, Any]) -> None:
        """Log a compact evaluation summary."""
        pla = ev["pla"]
        fnr = ev["fnr"]
        sir = ev["sir"]
        n_correct = len(ev['correctly_identified'])
        n_total = ev['total_gt']
        logger.info(
            f"Evaluation — PLA: {f'{pla:.4f}' if pla is not None else 'N/A'}  "
            f"identified: {n_correct}/{n_total} "
            f"({f'{pla*100:.1f}' if pla is not None else 'N/A'}%)  "
            f"FNR: {f'{fnr:.4f}' if fnr is not None else 'N/A'}  "
            f"SIR: {f'{sir:.4f}' if sir is not None else 'N/A'}"
        )
        if ev["missed"]:
            logger.info(f"  Missed persons: {ev['missed']}")
        if ev.get("misidentified"):
            logger.info(f"  Misidentified persons: {ev['misidentified']}")
        spfp = ev.get("n_spatial_false_positives", 0)
        if spfp:
            logger.info(f"  Spatial rejections (XML-confirmed): {spfp}")

        curves = ev.get("probe_curves", {})
        if curves:
            thresholds_ref = curves["thresholds"]
            idx = min(range(len(thresholds_ref)), key=lambda i: abs(thresholds_ref[i] - 0.5))
            logger.info(
                f"  [Probe-level]  genuine: {curves['n_genuine_accepted']}/{curves['n_genuine']}  "
                f"impostors: {curves['n_impostors']}  "
                f"@ T=0.50 — TPIR: {curves['tpir'][idx]:.4f}, FPIR: {curves['fpir'][idx]:.4f}"
            )

        fir = ev.get("fir")
        if fir is not None:
            fnmr = ev.get("fnmr", 0.0)
            logger.info(
                f"  [Frame-level]  FIR: {fir:.4f}  FNMR: {fnmr:.4f}  "
                f"frames: {ev.get('n_frames_correct', 0)}/{ev.get('n_frames', 0)}"
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

            stem = Path(video_entry["path"]).name
            out_path = os.path.join(self.output_dir, f"{stem}_results.json")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

            logger.info(f"Saved → {out_path}")

        logger.info("R&D pipeline complete.")
