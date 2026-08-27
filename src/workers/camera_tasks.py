"""Celery task for per-camera tracking, identity and logging.

The main process (CeleryCameraProducer, camera_worker.py) reads each
camera's frames, applies ROI and frame-skip, then writes the frame into a
`frame_store.CameraFrameSlot` and enqueues this task with the resulting
handle — it still owns the StreamHandler's persistent RTSP connection, which
a stateless-per-call task cannot hold open. Everything from "submit to GPU"
onward happens here, in a separate OS process.

No torch/ultralytics/insightface/onnxruntime import at module scope, and no
GPU model construction anywhere in this file — this process holds zero
models. GPU inference is reached via gpu_worker_rpc (never local), and
GlobalTrackManager via gpu_rpc/global_track_adapter (also never local): this
process touches no main-process-owned object directly, only through an RPC
client.

## What this task does not handle

  - Cross-camera global identity: RemoteGlobalTrackManager is wired in, but
    GlobalTrackIDGenerator (PersonTracker's local-track-ID counter) is left
    at its own local-fallback default (global_id_generator=None).
  - save_video / annotated debug output: tied to a local cv2.VideoWriter
    file handle, which doesn't cross a process boundary meaningfully.
  - Sticky-routing enforcement: this task assumes whatever routes it here
    keeps sending the same camera_id to the same worker process, so the
    per-process caches below stay valid. See compose.yml's camera-worker
    service for why that means exactly one replica today.
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

        # Its own PgVectorStore, unrelated to the main process's matcher.
        # Kept current by _ensure_listeners_started's EMBEDDING_RELOAD
        # subscription — without that this would go stale permanently, since
        # the context is cached for the life of the worker process.
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

        # Shared across every context in this process so the single
        # CalibrationSubscriber started below can invalidate all of them.
        self.homography_registry = _shared_homography_registry(HomographyRegistry)

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

        # Action recognition: safe to run per-worker because the recognizer
        # only talks to Ollama over HTTP -- no GPU weights are loaded here,
        # so this costs no VRAM the way YOLO/face would.
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
            global_id_generator=None,  # local-fallback default; see module docstring
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

    # ── Reload handlers, called from the listener threads ──────────────────

    def on_embedding_reload(self) -> None:
        """Mirror of engine.reload_embeddings for this worker's own copies."""
        self.face_matcher.reload_embeddings()

        new_map = {
            u["name"]: u["id"]
            for u in self.entry_logger.repository.get_user_name_to_id()
            if u.get("name") and u.get("id")
        }

        # Mutate the dict PersonStateManager holds rather than rebinding
        # camera_engine.name_to_id_map: it captured the object by reference at
        # construction, so a rebind would never reach it.
        #
        # Update-then-remove, never clear()-then-update: the task thread reads
        # this dict concurrently, and clearing first opens a window where every
        # lookup returns None (publishing a null user_id for a known person).
        # Growing then shrinking never exposes an empty map, so no lock is
        # needed -- which matters because the reader is a dict.get() deep
        # inside lum_vision that we cannot wrap.
        target = self.camera_engine.state_manager.name_to_id_map
        target.update(new_map)
        for name in [k for k in target if k not in new_map]:
            target.pop(name, None)

        self.entry_logger.current_users = self.face_matcher.db_names
        self.entry_logger.name_to_id = [
            {"name": name, "id": user_id} for name, user_id in new_map.items()
        ]
        self.entry_logger.reload_status()
        logger.info(
            f"camera_tasks[cam={self.camera_id}]: reloaded embeddings "
            f"({len(new_map)} users)"
        )

    def on_status_reload(self) -> None:
        """The backend publishes this; this worker's EntryLogger is now one of
        the loggers it is about."""
        self.entry_logger.reload_status()

    def on_camera_config_reload(self) -> None:
        """Pick up config changes (currently `application`) without a restart."""
        from config.camera_loader import load_cameras_from_db

        try:
            camera_config = self._load_camera_config(
                self.camera_id, load_cameras_from_db
            )
        except RuntimeError as e:
            # Camera removed from the org: this context is now orphaned, but
            # nothing will route frames to it either.
            logger.warning(f"camera_tasks[cam={self.camera_id}]: {e}")
            return

        self.camera_config = camera_config
        self.recognition_interval = int(
            camera_config.get("pipeline", {}).get("recognition_interval", 5)
        )
        # Only `application` is pushed onto the engine — it is the one field
        # engine.reload_camera_configs itself updates on a live camera.
        self.camera_engine.application = camera_config.get(
            "application", ["attendance"]
        )
        logger.info(
            f"camera_tasks[cam={self.camera_id}]: config reloaded, "
            f"applications={self.camera_engine.application}"
        )


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

_HOMOGRAPHY_REGISTRY = None
_LISTENERS_STARTED = False
_LISTENERS_LOCK = threading.Lock()


def _shared_homography_registry(registry_cls):
    """One registry per process, so a single CalibrationSubscriber thread can
    invalidate the entries every camera context reads."""
    global _HOMOGRAPHY_REGISTRY
    if _HOMOGRAPHY_REGISTRY is None:
        _HOMOGRAPHY_REGISTRY = registry_cls()
    return _HOMOGRAPHY_REGISTRY


def _for_each_context(method_name: str) -> None:
    """Call a reload handler on every live context.

    Iterates a snapshot: the task thread can create a context concurrently,
    and one created after this snapshot has already built itself fresh from
    the DB, so skipping it loses nothing.
    """
    for ctx in list(_CONTEXTS.values()):
        try:
            getattr(ctx, method_name)()
        except Exception as e:
            logger.exception(f"camera_tasks[cam={ctx.camera_id}]: {method_name} failed: {e}")


def _ensure_listeners_started() -> None:
    """Subscribe this process to the reload notifications the main process has
    always consumed. The workers now own the CameraEngine, FaceMatcher and
    EntryLogger those notifications are about.

    One set of listeners per process, not per camera: the handlers fan out
    over every context themselves. Called from _context_for rather than at
    import, because celery_app's `include=` also imports this module into the
    yolo and face workers, which must not open these subscriptions.
    """
    global _LISTENERS_STARTED
    if _LISTENERS_STARTED:
        return
    with _LISTENERS_LOCK:
        if _LISTENERS_STARTED:
            return

        from messaging.calibration_subscriber import CalibrationSubscriber
        from messaging.channels import INTERNAL_CHANNELS
        from messaging.subscriber import start_listener

        start_listener(
            INTERNAL_CHANNELS["EMBEDDING_RELOAD"],
            lambda _data: _for_each_context("on_embedding_reload"),
            name="reload-embeddings",
        )
        start_listener(
            INTERNAL_CHANNELS["CAMERA_CONFIG_RELOAD"],
            lambda _data: _for_each_context("on_camera_config_reload"),
            name="reload-camera-config",
        )
        start_listener(
            INTERNAL_CHANNELS["STATUS_RELOAD"],
            lambda _data: _for_each_context("on_status_reload"),
            name="reload-status",
        )
        if _HOMOGRAPHY_REGISTRY is not None:
            CalibrationSubscriber(_HOMOGRAPHY_REGISTRY).start()

        _LISTENERS_STARTED = True


def _context_for(camera_id: int) -> _CameraContext:
    ctx = _CONTEXTS.get(camera_id)
    if ctx is not None:
        return ctx
    with _CONTEXTS_LOCK:
        ctx = _CONTEXTS.get(camera_id)
        if ctx is None:
            ctx = _CameraContext(camera_id)
            _CONTEXTS[camera_id] = ctx
    # After the lock: a handler firing mid-construction would otherwise
    # block on _CONTEXTS_LOCK from the listener thread.
    _ensure_listeners_started()
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
