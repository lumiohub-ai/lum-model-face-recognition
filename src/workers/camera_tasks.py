"""Celery task for per-camera CPU work (LSO-67, Stage 1).

Replaces CameraWorker._process_one_frame's Steps 4-9 (camera_worker.py) for
whichever camera is flagged onto Celery. Steps 1-3 (read frame, apply ROI,
frame-skip) stay in the main process, which still owns the StreamHandler's
persistent RTSP connection — a Celery task instance is stateless per call and
cannot hold that connection open the way a thread does. The main process
writes the cropped frame into a `frame_store.CameraFrameSlot` and enqueues
this task with the resulting handle; everything from "submit to GPU" onward
happens here, in a separate OS process, escaping the GIL that motivated this
migration in the first place.

No torch/ultralytics/insightface/onnxruntime import at module scope, and no
GPU model construction anywhere in this file — this process holds zero
models. GPU inference is reached via gpu_worker_rpc (never local), and
GlobalTrackManager via gpu_rpc/global_track_adapter (also never local). Both
principles carried over directly from the benchmark harnesses'
model_holder.py convention, generalised here to "this process touches no
main-process-owned object directly, only through an RPC client."

## What this task does NOT yet handle (explicitly out of scope for Stage 1)

  - Cross-camera global identity: RemoteGlobalTrackManager is wired in, but
    GlobalTrackIDGenerator (PersonTracker's local-track-ID counter) is left
    at its own local-fallback default (global_id_generator=None) — Stage 2's
    Redis INCR work, not duplicated here.
  - save_video / annotated debug output: tied to a local cv2.VideoWriter
    file handle, which doesn't cross a process boundary meaningfully. The
    flagged camera simply does not support it while running on Celery.
  - Sticky-routing enforcement: this task assumes whatever routes it here
    keeps sending the same camera_id to the same worker process (so the
    per-process caches below stay valid) — the routing mechanism itself is
    Stage 2.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from loguru import logger

from workers.celery_app import celery
from workers.frame_store import FrameHandle, RoiBatchHandle, attach_and_read_roi_batch
from workers.gpu_rpc import GpuRpcClient
from workers.gpu_worker_rpc import GpuWorkerRpcClient
from workers.global_track_adapter import RemoteGlobalTrackManager
from workers.rpc_framing import recv_framed  # noqa: F401  (re-export sanity import)


class _CameraContext:
    """Everything one camera's worker-process-local processing needs,
    constructed once and reused across every task call for that camera_id —
    the same lazy-per-process-singleton pattern the benchmark harnesses used
    for model loading, applied here to CameraEngine/PersonTracker instead.

    Deliberately holds NO reference to any main-process object. Every field
    here is either a plain value (config), a worker-local instance with its
    own DB/GCS connections (FaceMatcher, EntryLogger, AsyncLogger,
    HomographyRegistry), or an RPC client.
    """

    def __init__(self, camera_id: int):
        # Imports deferred to construction time, not module scope — this
        # module must be importable (Celery's `include=` loads it to
        # register the task) without pulling in DB/GCS/lum_vision.
        from config.settings import settings
        from config.camera_loader import load_cameras_from_db
        from config.vision import build_vision_config
        from infrastructure.storage import PgVectorStore
        from infrastructure.entry_logger import EntryLogger
        from infrastructure.async_logger import AsyncLogger
        from domain.calibration.homography_registry import HomographyRegistry
        from lum_vision import ActionRecognizer, FaceMatcher
        from pipeline.action_worker import ActionRecognitionWorker
        from pipeline.camera_engine import CameraEngine

        self.camera_id = camera_id
        self.client_slug = settings.client_slug

        camera_config = self._load_camera_config(camera_id, load_cameras_from_db)
        self.camera_config = camera_config
        self.recognition_interval = int(
            camera_config.get("pipeline", {}).get("recognition_interval", 5)
        )

        vision_config = build_vision_config(_load_yaml_config())

        # FaceMatcher: its own PgVectorStore, no reference to the main
        # process's copy. The Redis reload fan-out (embedding_tasks.py)
        # already exists and works process-agnostically — a hot reload
        # published from anywhere reaches every subscriber, this one
        # included, so a worker-local instance does not go stale relative
        # to the main process's.
        self.face_matcher = FaceMatcher(
            provider=PgVectorStore(self.client_slug),
            match_threshold=vision_config.match_threshold,
        )

        # PersonDetector stand-in: CameraEngine/PersonTracker read exactly
        # two scalars off `person_detector` (.confidence_threshold, .device)
        # and never call a method on it — verified by tracing every
        # `self.person_detector.` reference in camera_engine.py. The real
        # PersonDetector loads YOLO weights on construction; this process
        # must never do that, since GPU inference is centralised in the
        # main process specifically to avoid N model copies.
        self._person_detector_stub = _PersonDetectorStub(
            confidence_threshold=vision_config.person_detection_threshold,
            device="cpu",  # this process does no local GPU inference
        )

        self.rpc_client = GpuRpcClient()
        self.gpu_worker_client = GpuWorkerRpcClient()
        self.rpc_client.on_fallback = lambda: logger.warning(
            f"camera_tasks[cam={camera_id}]: GlobalTrackManager RPC fallback fired"
        )
        self.gpu_worker_client.on_fallback = lambda: logger.warning(
            f"camera_tasks[cam={camera_id}]: GPU worker RPC fallback fired"
        )
        self.global_track_manager = RemoteGlobalTrackManager(
            self.rpc_client, enabled=bool(_load_yaml_config().get("enable_global_tracking", False))
        )

        self.homography_registry = HomographyRegistry()

        entry_logger_args = _make_entry_logger_args(
            client_slug=self.client_slug,
            db_names=self.face_matcher.db_names,
            camera_config=camera_config,
        )
        self.entry_logger = EntryLogger(args=entry_logger_args)
        self.async_logger = AsyncLogger(entry_logger=self.entry_logger)
        self.async_logger.start()

        # NOT self.entry_logger.name_to_id — that is a different, list-of-
        # dicts value EntryLogger computes for its own new/deleted-user
        # bookkeeping (repository.check_new_and_deleted_users). CameraEngine's
        # name_to_id_map wants engine.py's _build_name_to_id_map() shape: a
        # plain {name: id} dict from a distinct query
        # (repository.get_user_name_to_id()). Reuses entry_logger's own
        # Repository instance rather than opening a second DB connection.
        name_to_id_map = {
            u["name"]: u["id"]
            for u in self.entry_logger.repository.get_user_name_to_id()
            if u.get("name") and u.get("id")
        }

        # Action recognition (LSO-67): safe to run per-worker because the
        # recognizer only talks to Ollama over HTTP -- no GPU weights are
        # loaded here, so this costs no VRAM the way YOLO/face would.
        #
        # The per-identity throttle inside ActionRecognitionWorker is what
        # makes several of these instances safe to run at once: it is backed
        # by Redis (workers/identity_throttle.py), so "classify this person
        # once per interval" still holds across worker processes. With the
        # old in-memory dict, each worker would have claimed the same person
        # independently and fired N duplicate VLM inferences.
        action_cfg = _load_yaml_config().get("action_recognition", {}) or {}
        self.action_worker = ActionRecognitionWorker(
            recognizer=ActionRecognizer(vision_config.action),
            client_slug=self.client_slug,
            max_queue_size=action_cfg.get("max_queue_size", 50),
            num_workers=action_cfg.get("async_workers", 1),
            min_crop_height=action_cfg.get("min_crop_height", 0),
            min_crop_width=action_cfg.get("min_crop_width", 0),
            min_crop_area=action_cfg.get("min_crop_area", 0),
        )
        self.action_worker.set_async_logger(self.async_logger)
        self.action_worker.start_workers()

        self.camera_engine = CameraEngine(
            camera_config=camera_config,
            face_detector=None,  # never called by CameraEngine — see module docstring
            face_recognizer=self.face_matcher,
            person_detector=self._person_detector_stub,
            client_slug=self.client_slug,
            global_id_generator=None,  # Stage 2: Redis INCR, not duplicated here
            name_to_id_map=name_to_id_map,
            global_track_manager=self.global_track_manager,
            action_recognizer=self.action_worker,
            homography_registry=self.homography_registry,
        )

        self._detection_frame_num = 0

        logger.info(f"camera_tasks: worker-local context ready for camera {camera_id}")

    @staticmethod
    def _load_camera_config(camera_id: int, load_cameras_from_db) -> Dict[str, Any]:
        # Reuses the exact same branch-scoped, stream-URL-resolving loader
        # the main process uses (src/config/camera_loader.py) rather than a
        # second, divergent DB query — filtered down to this one camera_id
        # since load_cameras_from_db returns every camera for the org.
        from config.settings import settings

        cameras = load_cameras_from_db(
            client_slug=settings.client_slug, applications=["attendance"]
        )
        for cam in cameras:
            if cam.get("camera_id") == camera_id:
                return cam
        raise RuntimeError(
            f"camera_tasks: camera_id={camera_id} not found in "
            f"load_cameras_from_db — was it removed after this worker started?"
        )

    def process_frame(
        self, frame_handle: FrameHandle, frame_num: int
    ) -> Optional[Dict[str, Any]]:
        """Steps 4-9 of the old CameraWorker._process_one_frame, ported.
        Returns a small summary dict for the task result (mainly for Stage
        1's measurement work), or None if the frame was dropped upstream.
        """
        from workers import frame_store

        frame = frame_store.attach_and_read(frame_handle)
        if frame is None:
            # Slot vanished (camera removed) or this reply is for a frame
            # already superseded — same "nothing to do" case
            # CameraWorker.submit_frame's False return produces today.
            return None

        # No frame-skip here: the producer already gated on detection_interval
        # before writing shared memory and paying the broker hop, so every
        # task call IS a detection frame. A second gate here would compound
        # to interval² and halve the detection rate.
        self._detection_frame_num += 1

        detections = self.gpu_worker_client.detect(
            camera_id=self.camera_id, frame_handle=frame_handle, frame_num=frame_num
        )

        active_tracks, removed_tracks, person_rois = self.camera_engine.update_tracking(
            detections, frame, frame_num
        )

        run_recognition = (
            self._detection_frame_num % self.recognition_interval == 0
        )
        if run_recognition and person_rois:
            track_ids = [tid for tid, _roi, _off in person_rois]
            rois = [roi for _tid, roi, _off in person_rois]
            roi_offsets = {tid: off for tid, _roi, off in person_rois}
            roi_slot = _roi_slot_for(self.camera_id)
            roi_handle = roi_slot.write(rois, track_ids)
            embeddings_map = self.gpu_worker_client.embed(
                camera_id=self.camera_id, roi_batch_handle=roi_handle
            )
        else:
            roi_offsets = {}
            embeddings_map = {}

        events = self.camera_engine.finalize_identities(
            active_tracks, removed_tracks, embeddings_map, frame, frame_num
        )
        self.camera_engine.emit_positions(active_tracks)

        for event in events:
            self.async_logger.log_entry(event)

        return {
            "camera_id": self.camera_id,
            "frame_num": frame_num,
            "active_track_count": len(active_tracks),
            "removed_track_count": len(removed_tracks),
            "event_count": len(events),
            "recognition_ran": run_recognition,
        }


class _PersonDetectorStub:
    """See _CameraContext's comment on why this exists instead of a real
    PersonDetector. Anything beyond these two attributes being read would
    raise AttributeError — deliberately, so a future CameraEngine/
    PersonTracker change that starts calling a method on person_detector
    fails loudly here instead of silently doing nothing.
    """

    def __init__(self, confidence_threshold: float, device: str):
        self.confidence_threshold = confidence_threshold
        self.device = device


def _load_yaml_config() -> Dict[str, Any]:
    from config.startup import load_config

    return load_config() or {}


def _make_entry_logger_args(
    client_slug: str, db_names, camera_config: Dict[str, Any]
):
    """Mirrors SmartOfficeEngine._init_entry_logger's duck-typed Args object
    exactly (engine.py) — EntryLogger reads specific attributes off it, not
    kwargs, so this must match that shape rather than invent a new one.
    """
    config = _load_yaml_config()
    args = type("Args", (), {})()
    args.client_slug = client_slug
    args.logger = logger
    args.db_names = db_names
    args.production = True
    args.unrecognized_frontality_min = config.get("unrecognized_frontality_min", 0.6)
    args.unrecognized_pitch_min = config.get("unrecognized_pitch_min", 0.4)
    return args


# ── Per-process caches ───────────────────────────────────────────────────────
# One _CameraContext per camera_id this process has ever served, and one
# RoiBatchSlot per camera_id (the task is the producer for the embed RPC's
# request side — camera_worker.py's Step 6 ROI crops are computed here, not
# in the main process, so this process must own that slot, unlike the frame
# slot which the main process owns as producer).

_CONTEXTS: Dict[int, _CameraContext] = {}
_CONTEXTS_LOCK = threading.Lock()

_ROI_SLOTS: Dict[int, Any] = {}
_ROI_SLOTS_LOCK = threading.Lock()


def _context_for(camera_id: int) -> _CameraContext:
    ctx = _CONTEXTS.get(camera_id)
    if ctx is not None:
        return ctx
    with _CONTEXTS_LOCK:
        ctx = _CONTEXTS.get(camera_id)
        if ctx is None:
            ctx = _CameraContext(camera_id)
            _CONTEXTS[camera_id] = ctx
        return ctx


def _roi_slot_for(camera_id: int):
    from workers.frame_store import RoiBatchSlot

    slot = _ROI_SLOTS.get(camera_id)
    if slot is not None:
        return slot
    with _ROI_SLOTS_LOCK:
        slot = _ROI_SLOTS.get(camera_id)
        if slot is None:
            slot = RoiBatchSlot(camera_id)
            _ROI_SLOTS[camera_id] = slot
        return slot


@celery.task(name="camera.process_frame", queue="camera_frames")
def process_frame_task(
    camera_id: int, frame_handle: Dict[str, int], frame_num: int
) -> Optional[Dict[str, Any]]:
    """Entry point. Deliberately thin — all real logic lives on
    _CameraContext so it can be unit-tested without going through Celery's
    task-dispatch machinery.

    `frame_handle` arrives as a plain dict, not a FrameHandle instance:
    celery_app.py sets task_serializer='json' for every task in this app
    (embedding/detection tasks included), and FrameHandle is a frozen
    dataclass — json.dumps has no idea how to encode it. Reconstructing it
    here, rather than widening the global serializer config, keeps
    FrameHandle a real dataclass everywhere else that matters (frame_store's
    internal API, gpu_worker_rpc's pickled socket protocol) and confines the
    JSON constraint to the one hop that actually has it.
    """
    ctx = _context_for(camera_id)
    return ctx.process_frame(FrameHandle(**frame_handle), frame_num)
