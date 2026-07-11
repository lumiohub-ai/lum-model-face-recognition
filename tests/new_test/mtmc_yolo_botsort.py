"""
LumioHub MTMC baseline:
- Person detection: YOLO26s / YOLO-compatible Ultralytics weights
- Local tracker: BoT-SORT
- Global tracker: cosine similarity over appearance embeddings
- Input: cam1.mp4 + cam2.mp4
- Output: annotated videos + CSV/JSON metrics in output folder

Example:
    python mtmc_yolo_botsort.py \
        --source1 cam1.mp4 \
        --source2 cam2.mp4 \
        --weights yolo26s.pt \
        --out output \
        --sim-thres 0.72
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from ultralytics import YOLO


BBox = Tuple[int, int, int, int]


class OSNetReID:
    """Body-ReID embedder backed by boxmot's OSNet, matching production's
    approach in src/domain/person_tracking/global_track.py (_init_body_reid_model /
    _extract_body_embedding), rather than a generic detector-classification-head
    embedding. Auto-downloads weights (via gdown) on first use if missing.
    """

    def __init__(self, weights: str, device: str, half: bool = False) -> None:
        self.backend = ReidAutoBackend(
            weights=Path(weights), device=torch.device(device), half=half
        ).model

    def __call__(self, frame: np.ndarray, xyxys: Sequence[BBox]) -> List[np.ndarray]:
        if len(xyxys) == 0:
            return []
        features = self.backend.get_features(np.array(xyxys, dtype=np.float32), frame)
        return list(np.atleast_2d(features))


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def clamp_bbox(xyxy: Sequence[float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def color_for_id(global_id: int) -> Tuple[int, int, int]:
    """
    Deterministic BGR color for OpenCV drawing.
    """
    rng = np.random.default_rng(global_id * 9973)
    color = rng.integers(40, 230, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


# Crop-quality gate before ReID extraction, matching configs/global_tracking.yaml's
# crop_quality thresholds (production's _validate_crop_quality in
# src/domain/person_tracking/global_track.py) so low-quality/degenerate crops
# don't feed noisy embeddings into cross-camera cosine matching.
MIN_CROP_HEIGHT = 80
MIN_CROP_WIDTH = 40
MIN_CROP_AREA = 3200
MIN_ASPECT_RATIO = 1.5  # height / width; person crops should be roughly vertical
MAX_ASPECT_RATIO = 4.0


def is_valid_crop(bbox: BBox) -> bool:
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h < MIN_CROP_HEIGHT or w < MIN_CROP_WIDTH:
        return False
    if h * w < MIN_CROP_AREA:
        return False
    aspect_ratio = h / w if w > 0 else 0
    return MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO

# Camera -> floor-plan homographies, derived by manually matching floor-level
# landmarks (wall corners) between each camera frame and floorplan.png, then
# solving with src/domain/calibration/homography.py:compute_homography(). Only
# valid for this specific camera setup / floor plan. Recalibrated against the
# updated floorplan.png (1438x970).
MAP_IMAGE_PATH = "floorplan.png"
HOMOGRAPHIES: Dict[str, np.ndarray] = {
    "cam1": np.array(
        [
            [0.2716879989585372, -1.0576735573836094, 1476.7908253445646],
            [2.7676228222135615, 2.4734047498693803, -603.8635385939165],
            [0.0013738058043704857, 0.004126082222423953, 1.0],
        ],
        dtype=np.float64,
    ),
    "cam2": np.array(
        [
            [-0.06232521929255049, 1.4943664701790436, -348.53961134303216],
            [-0.6129954396401657, 0.6273681100378701, 1297.9394172019167],
            [-8.659274498633057e-05, 0.001170006767231962, 1.0],
        ],
        dtype=np.float64,
    ),
}


def bbox_foot_point(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def project_to_map(bbox: BBox, H: np.ndarray) -> Tuple[float, float]:
    foot = np.array([[list(bbox_foot_point(bbox))]], dtype=np.float64)
    mapped = cv2.perspectiveTransform(foot, H).reshape(2)
    return float(mapped[0]), float(mapped[1])


@dataclass
class GlobalTrack:
    global_id: int
    embedding: np.ndarray
    first_frame: int
    last_frame: int
    first_timestamp_sec: float
    last_timestamp_sec: float
    last_camera_id: str
    last_local_track_id: int
    observations: int = 1
    cameras_seen: Set[str] = field(default_factory=set)
    local_track_keys: Set[str] = field(default_factory=set)
    max_similarity_seen: float = 0.0
    last_bbox: Optional[BBox] = None

    def update(
        self,
        embedding: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        camera_id: str,
        local_track_id: int,
        bbox: BBox,
        similarity: float,
        ema_alpha: float,
    ) -> None:
        self.embedding = l2_normalize(ema_alpha * self.embedding + (1.0 - ema_alpha) * embedding)
        self.last_frame = frame_idx
        self.last_timestamp_sec = timestamp_sec
        self.last_camera_id = camera_id
        self.last_local_track_id = int(local_track_id)
        self.observations += 1
        self.cameras_seen.add(camera_id)
        self.local_track_keys.add(f"{camera_id}:{local_track_id}")
        self.max_similarity_seen = max(self.max_similarity_seen, float(similarity))
        self.last_bbox = bbox


class GlobalTrackManager:
    """
    Assigns one global_person_id across cameras.

    Matching policy:
    1. If (camera_id, local_track_id) was already mapped, reuse the same global ID.
    2. Otherwise, compare current appearance embedding with active global tracks.
    3. If best cosine similarity >= threshold, attach to that global ID.
    4. Else create a new global ID.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        max_inactive_frames: int = 450,
        ema_alpha: float = 0.85,
        archive_after_frames: Optional[int] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_inactive_frames = max_inactive_frames
        self.ema_alpha = ema_alpha
        # Fully drop tracks well past max_inactive_frames so global_tracks/
        # local_to_global don't grow unboundedly over a long-running stream.
        self.archive_after_frames = archive_after_frames or max_inactive_frames * 4

        self.next_global_id = 1
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.local_to_global: Dict[str, int] = {}

    def has_mapping(self, camera_id: str, local_track_id: int) -> bool:
        """Whether (camera_id, local_track_id) already has an established global ID."""
        return f"{camera_id}:{int(local_track_id)}" in self.local_to_global

    def cleanup_stale(self, frame_idx: int) -> int:
        """Drop global tracks inactive well beyond max_inactive_frames. Returns count removed."""
        stale_gids = [
            gid for gid, track in self.global_tracks.items()
            if frame_idx - track.last_frame > self.archive_after_frames
        ]
        for gid in stale_gids:
            del self.global_tracks[gid]
        if stale_gids:
            stale_set = set(stale_gids)
            self.local_to_global = {
                k: v for k, v in self.local_to_global.items() if v not in stale_set
            }
        return len(stale_gids)

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        bbox: BBox,
        blocked_global_ids: Optional[Set[int]] = None,
    ) -> Tuple[int, float, str]:
        blocked_global_ids = blocked_global_ids or set()
        local_key = f"{camera_id}:{int(local_track_id)}"
        embedding = l2_normalize(embedding)

        # Existing local track mapping.
        if local_key in self.local_to_global:
            gid = self.local_to_global[local_key]
            track = self.global_tracks[gid]
            sim = cosine_similarity(embedding, track.embedding)
            track.update(
                embedding=embedding,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                camera_id=camera_id,
                local_track_id=int(local_track_id),
                bbox=bbox,
                similarity=sim,
                ema_alpha=self.ema_alpha,
            )
            return gid, sim, "mapped_local"

        # Search active global tracks.
        best_gid: Optional[int] = None
        best_sim = -1.0

        for gid, track in self.global_tracks.items():
            if gid in blocked_global_ids:
                continue
            if frame_idx - track.last_frame > self.max_inactive_frames:
                continue

            sim = cosine_similarity(embedding, track.embedding)
            if sim > best_sim:
                best_gid = gid
                best_sim = sim

        if best_gid is not None and best_sim >= self.similarity_threshold:
            gid = best_gid
            self.local_to_global[local_key] = gid
            self.global_tracks[gid].update(
                embedding=embedding,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                camera_id=camera_id,
                local_track_id=int(local_track_id),
                bbox=bbox,
                similarity=best_sim,
                ema_alpha=self.ema_alpha,
            )
            return gid, best_sim, "matched_global"

        # Create new global track.
        gid = self.next_global_id
        self.next_global_id += 1
        gt = GlobalTrack(
            global_id=gid,
            embedding=embedding,
            first_frame=frame_idx,
            last_frame=frame_idx,
            first_timestamp_sec=timestamp_sec,
            last_timestamp_sec=timestamp_sec,
            last_camera_id=camera_id,
            last_local_track_id=int(local_track_id),
            cameras_seen={camera_id},
            local_track_keys={local_key},
            last_bbox=bbox,
        )
        self.global_tracks[gid] = gt
        self.local_to_global[local_key] = gid
        return gid, 1.0, "new_global"

    def to_rows(self) -> List[dict]:
        rows = []
        for gid in sorted(self.global_tracks):
            t = self.global_tracks[gid]
            rows.append({
                "global_person_id": t.global_id,
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "duration_frames": t.last_frame - t.first_frame + 1,
                "first_timestamp_sec": round(t.first_timestamp_sec, 3),
                "last_timestamp_sec": round(t.last_timestamp_sec, 3),
                "duration_sec": round(max(0.0, t.last_timestamp_sec - t.first_timestamp_sec), 3),
                "observations": t.observations,
                "cameras_seen": "|".join(sorted(t.cameras_seen)),
                "num_cameras_seen": len(t.cameras_seen),
                "local_track_keys": "|".join(sorted(t.local_track_keys)),
                "max_similarity_seen": round(t.max_similarity_seen, 4),
            })
        return rows


def draw_detection(
    frame: np.ndarray,
    bbox: BBox,
    global_id: int,
    local_id: int,
    camera_id: str,
    similarity: float,
    status: str,
    conf: float,
) -> None:
    x1, y1, x2, y2 = bbox
    color = color_for_id(global_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    status_text = "" if status == "mapped_local" else f" {status}"
    label = f"{camera_id} G:{global_id} L:{local_id}{status_text} sim:{similarity:.2f} conf:{conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_text), (min(frame.shape[1] - 1, x1 + tw + 4), y_text + th + 6), color, -1)
    cv2.putText(frame, label, (x1 + 2, y_text + th + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def video_meta(cap: cv2.VideoCapture) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Video source reported invalid frame size ({width}x{height}); "
            "the capture may not be ready or the source is unreadable."
        )
    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame
    interp = cv2.INTER_AREA if (w > width or h > height) else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interp)


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
    return writer


def draw_map_marker(map_frame: np.ndarray, pt: Tuple[float, float], global_id: int) -> None:
    h, w = map_frame.shape[:2]
    x, y = int(round(pt[0])), int(round(pt[1]))
    if not (-50 <= x <= w + 50 and -50 <= y <= h + 50):
        return  # far outside the map, likely an extreme-edge extrapolation artifact
    color = color_for_id(global_id)
    cv2.circle(map_frame, (x, y), 8, color, -1)
    cv2.circle(map_frame, (x, y), 8, (255, 255, 255), 1)
    cv2.putText(
        map_frame, f"G{global_id}", (x + 10, y + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    camera_id: str,
    frame_idx: int,
    timestamp_sec: float,
    tracker_yaml: str,
    reid_encoder: OSNetReID,
    global_manager: GlobalTrackManager,
    used_global_ids_this_frame: Set[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[dict]]:
    annotated = frame.copy()

    # Person-only tracking: COCO class 0 = person.
    result_list = model.track(
        source=frame,
        persist=True,
        tracker=tracker_yaml,
        classes=[0],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=None if args.device == "auto" else args.device,
        verbose=False,
    )
    if not result_list:
        return annotated, []

    result = result_list[0]
    if result.boxes is None or result.boxes.id is None:
        return annotated, []

    boxes = result.boxes
    xyxys = boxes.xyxy.detach().cpu().numpy()
    local_ids = boxes.id.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(local_ids), dtype=np.float32)

    h, w = frame.shape[:2]
    bboxes = [clamp_bbox(xyxy, w, h) for xyxy in xyxys]
    keep_mask = np.array([is_valid_crop(b) for b in bboxes])
    if not np.any(keep_mask):
        return annotated, []

    bboxes = [b for b, keep in zip(bboxes, keep_mask) if keep]
    local_ids = local_ids[keep_mask]
    confs = confs[keep_mask]
    embeddings = reid_encoder(frame, bboxes)
    if len(embeddings) != len(bboxes):
        # Can't safely pair embeddings to boxes if the ReID backend dropped a
        # row; skip this frame's detections for this camera rather than risk
        # scoring one person's box against another's embedding.
        print(
            f"[WARN] {camera_id} frame {frame_idx}: reid_encoder returned "
            f"{len(embeddings)} embeddings for {len(bboxes)} boxes, skipping frame"
        )
        return annotated, []

    rows: List[dict] = []

    # Process already-established local tracks first so their global IDs land
    # in used_global_ids_this_frame before any brand-new local track's
    # embedding-similarity search can coincidentally claim the same ID.
    detections = list(zip(bboxes, local_ids, confs, embeddings))
    detections.sort(key=lambda d: 0 if global_manager.has_mapping(camera_id, int(d[1])) else 1)

    for bbox, local_id, conf, emb in detections:
        gid, sim, status = global_manager.assign(
            camera_id=camera_id,
            local_track_id=int(local_id),
            embedding=emb,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            bbox=bbox,
            blocked_global_ids=used_global_ids_this_frame,
        )
        used_global_ids_this_frame.add(gid)

        draw_detection(
            annotated,
            bbox=bbox,
            global_id=gid,
            local_id=int(local_id),
            camera_id=camera_id,
            similarity=sim,
            status=status,
            conf=float(conf),
        )

        x1, y1, x2, y2 = bbox
        rows.append({
            "frame": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "camera_id": camera_id,
            "local_track_id": int(local_id),
            "global_person_id": gid,
            "match_status": status,
            "similarity": round(float(sim), 4),
            "confidence": round(float(conf), 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return annotated, rows


def make_combined_frame(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Creates side-by-side preview. Resizes cam2 to cam1 height.
    """
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    if h2 != h1:
        new_w2 = int(w2 * (h1 / float(h2)))
        frame2 = cv2.resize(frame2, (new_w2, h1), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([frame1, frame2])


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap1 = open_video(args.source1)
    cap2 = open_video(args.source2)
    meta1 = video_meta(cap1)
    meta2 = video_meta(cap2)

    # Separate YOLO instances keep BoT-SORT local tracking state separate per camera.
    model1 = YOLO(args.weights)
    model2 = YOLO(args.weights)

    reid_device = args.device
    if reid_device == "auto":
        reid_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reid_encoder = OSNetReID(weights=args.reid_weights, device=reid_device)
    global_manager = GlobalTrackManager(
        similarity_threshold=args.sim_thres,
        max_inactive_frames=args.max_inactive_frames,
        ema_alpha=args.ema_alpha,
    )

    # Process and output both cameras at the same resolution (smaller of the two, to avoid upscaling).
    if meta1["width"] * meta1["height"] <= meta2["width"] * meta2["height"]:
        target_w, target_h = meta1["width"], meta1["height"]
    else:
        target_w, target_h = meta2["width"], meta2["height"]

    fps_out = min(meta1["fps"], meta2["fps"]) if args.output_fps <= 0 else args.output_fps
    writer1 = make_writer(out_dir / "cam1_annotated.mp4", fps_out, target_w, target_h)
    writer2 = make_writer(out_dir / "cam2_annotated.mp4", fps_out, target_w, target_h)

    combined_writer = None

    # 2D map projection: detections are on the (possibly downscaled) processing
    # frame, but HOMOGRAPHIES were calibrated against each camera's native
    # resolution, so scale bboxes back up before projecting.
    base_map = cv2.imread(MAP_IMAGE_PATH)
    if base_map is None:
        raise FileNotFoundError(f"Could not load map image: {MAP_IMAGE_PATH}")
    map_h, map_w = base_map.shape[:2]
    map_writer = make_writer(out_dir / "map_projection.mp4", fps_out, map_w, map_h)
    proj_scale = {
        "cam1": (meta1["width"] / target_w, meta1["height"] / target_h),
        "cam2": (meta2["width"] / target_w, meta2["height"] / target_h),
    }
    map_trails: Dict[int, List[Tuple[int, int]]] = {}

    all_detection_rows: List[dict] = []
    local_tracks_by_camera: Dict[str, Set[int]] = {"cam1": set(), "cam2": set()}

    start_time = time.time()
    frame_idx = 0
    CLEANUP_EVERY_FRAMES = 500

    try:
        while True:
            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()
            if not ok1 or not ok2:
                break

            frame1 = resize_to(frame1, target_w, target_h)
            frame2 = resize_to(frame2, target_w, target_h)

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            # Basic frame-index synchronization. For real RTSP, use capture timestamps instead.
            timestamp_sec = frame_idx / fps_out

            used_cam1: Set[int] = set()
            used_cam2: Set[int] = set()

            ann1, rows1 = process_frame(
                frame=frame1,
                model=model1,
                camera_id="cam1",
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                tracker_yaml=args.tracker,
                reid_encoder=reid_encoder,
                global_manager=global_manager,
                used_global_ids_this_frame=used_cam1,
                args=args,
            )

            ann2, rows2 = process_frame(
                frame=frame2,
                model=model2,
                camera_id="cam2",
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                tracker_yaml=args.tracker,
                reid_encoder=reid_encoder,
                global_manager=global_manager,
                used_global_ids_this_frame=used_cam2,
                args=args,
            )

            for r in rows1:
                local_tracks_by_camera["cam1"].add(r["local_track_id"])
            for r in rows2:
                local_tracks_by_camera["cam2"].add(r["local_track_id"])

            all_detection_rows.extend(rows1)
            all_detection_rows.extend(rows2)

            map_frame = base_map.copy()
            for row in rows1 + rows2:
                sx, sy = proj_scale[row["camera_id"]]
                bbox = (row["x1"] * sx, row["y1"] * sy, row["x2"] * sx, row["y2"] * sy)
                map_pt = project_to_map(bbox, HOMOGRAPHIES[row["camera_id"]])
                gid = row["global_person_id"]
                map_trails.setdefault(gid, []).append((int(round(map_pt[0])), int(round(map_pt[1]))))
                draw_map_marker(map_frame, map_pt, gid)
            map_writer.write(map_frame)

            writer1.write(ann1)
            writer2.write(ann2)

            if args.save_combined:
                combined = make_combined_frame(ann1, ann2)
                if combined_writer is None:
                    ch, cw = combined.shape[:2]
                    combined_writer = make_writer(out_dir / "combined_annotated.mp4", fps_out, cw, ch)
                combined_writer.write(combined)

            if args.show:
                preview = make_combined_frame(ann1, ann2)
                cv2.imshow("LumioHub MTMC - press q to quit", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % CLEANUP_EVERY_FRAMES == 0:
                global_manager.cleanup_stale(frame_idx)

            if args.log_every > 0 and frame_idx % args.log_every == 0:
                print(f"[INFO] Processed frame {frame_idx}")
    finally:
        # Always release capture/writer handles, even if an exception broke out
        # of the loop above — otherwise output videos are left without their
        # trailer/moov atom (unplayable) and OS handles leak.
        cap1.release()
        cap2.release()
        writer1.release()
        writer2.release()
        map_writer.release()
        if combined_writer is not None:
            combined_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    trails_frame = base_map.copy()
    for gid, pts in map_trails.items():
        color = color_for_id(gid)
        if len(pts) >= 2:
            cv2.polylines(trails_frame, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        draw_map_marker(trails_frame, pts[-1], gid)
    cv2.imwrite(str(out_dir / "map_projection_trails.png"), trails_frame)

    global_rows = global_manager.to_rows()
    write_csv(out_dir / "detections.csv", all_detection_rows)
    write_csv(out_dir / "global_tracks.csv", global_rows)

    cross_camera_ids = [
        row["global_person_id"]
        for row in global_rows
        if int(row["num_cameras_seen"]) >= 2
    ]

    avg_conf = float(np.mean([r["confidence"] for r in all_detection_rows])) if all_detection_rows else 0.0
    avg_sim = float(np.mean([r["similarity"] for r in all_detection_rows])) if all_detection_rows else 0.0

    summary = {
        "inputs": {
            "cam1": args.source1,
            "cam2": args.source2,
        },
        "weights": args.weights,
        "tracker": args.tracker,
        "person_class_filter": "COCO class 0",
        "processing_frame_size": {"width": target_w, "height": target_h},
        "similarity_threshold": args.sim_thres,
        "max_inactive_frames": args.max_inactive_frames,
        "frames_processed": frame_idx,
        "processing_time_sec": round(elapsed, 3),
        "processing_fps_pair": round(frame_idx / elapsed, 3) if elapsed > 0 else 0.0,
        "detections_total": len(all_detection_rows),
        "avg_detection_confidence": round(avg_conf, 4),
        "avg_assignment_similarity": round(avg_sim, 4),
        "unique_local_tracks": {
            "cam1": len(local_tracks_by_camera["cam1"]),
            "cam2": len(local_tracks_by_camera["cam2"]),
        },
        "unique_global_person_ids": len(global_rows),
        "cross_camera_global_person_ids": cross_camera_ids,
        "num_cross_camera_global_person_ids": len(cross_camera_ids),
        "video_meta": {
            "cam1": meta1,
            "cam2": meta2,
        },
        "outputs": {
            "cam1_annotated": str(out_dir / "cam1_annotated.mp4"),
            "cam2_annotated": str(out_dir / "cam2_annotated.mp4"),
            "combined_annotated": str(out_dir / "combined_annotated.mp4") if args.save_combined else None,
            "map_projection": str(out_dir / "map_projection.mp4"),
            "map_projection_trails": str(out_dir / "map_projection_trails.png"),
            "detections_csv": str(out_dir / "detections.csv"),
            "global_tracks_csv": str(out_dir / "global_tracks.csv"),
            "metrics_summary_json": str(out_dir / "metrics_summary.json"),
        },
    }

    (out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[DONE] Output folder:", out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26s + BoT-SORT + cosine global tracker baseline")
    parser.add_argument("--source1", default="cam1.mp4", help="Path to first camera video")
    parser.add_argument("--source2", default="cam2.mp4", help="Path to second camera video")
    parser.add_argument("--weights", default="yolo26m.pt", help="YOLO weights path, e.g. yolo26m.pt")
    parser.add_argument(
        "--reid-weights",
        default="osnet_x0_25_msmt17.pt",
        help="Body-ReID model for cross-camera appearance embedding (boxmot/OSNet, matches "
        "production's src/domain/person_tracking/global_track.py body ReID model)",
    )
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--sim-thres", type=float, default=0.72, help="Cosine threshold for global ID matching")
    parser.add_argument("--max-inactive-frames", type=int, default=450, help="Frames before a global track is inactive")
    parser.add_argument("--ema-alpha", type=float, default=0.85, help="Embedding smoothing factor")
    parser.add_argument("--max-frames", type=int, default=-1, help="Debug limit; -1 means full videos")
    parser.add_argument("--output-fps", type=float, default=-1, help="Output FPS; <=0 uses min input FPS")
    parser.add_argument("--save-combined", action="store_true", help="Save side-by-side annotated video")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N frames; 0 disables")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
