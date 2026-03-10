"""SmartOfficeEngine - Unified person tracking and face recognition system.

Architecture (parallel multi-camera):
- GPUInferenceWorker: single thread owns the GPU, batches YOLO + ArcFace
- CameraWorker: one thread per camera (frame read + CPU processing)
- AsyncLogger: three background I/O threads (db, gcs, redis)
- StreamManager/StreamHandler: background threads for RTSP read (existing)
"""

import time
from typing import List, Optional

from loguru import logger

# Infrastructure
from infrastructure.video import StreamManager
from infrastructure import EntryLogger
from infrastructure.storage import Repository, EmbeddingSyncService

# Domain
from domain.face_detection import ModelFactory

# Config
from config import load_cameras_from_db

# Local
from pipeline.camera_engine import CameraEngine, GlobalTrackIDGenerator

from pipeline.gpu_worker import GPUInferenceWorker
from pipeline.camera_worker import CameraWorker
from infrastructure.async_logger import AsyncLogger
from infrastructure.video.annotator import FrameAnnotator


class SmartOfficeEngine:
    """Unified engine for Smart Office person tracking and face recognition.

    Replaces the sequential FrameProcessor loop with a parallel architecture:
      - GPUInferenceWorker batches inference across all camera threads
      - CameraWorker runs one thread per camera for frame read + CPU processing
      - AsyncLogger offloads all I/O to background threads
    """

    def __init__(
        self,
        client_slug: str,
        applications: Optional[List[str]] = None,
        **kwargs
    ):
        self.client_slug = client_slug
        self.applications = applications or ["attendance"]
        self.config = kwargs

        # Lifecycle state
        self._running = False
        self._start_time = 0.0
        self.needs_reinit = False

        # Pipeline config (with defaults)
        pipeline_cfg = kwargs.get("pipeline", {})
        self._detection_interval: int = pipeline_cfg.get("detection_interval", 2)
        self._recognition_interval: int = pipeline_cfg.get("recognition_interval", 5)
        self._async_queue_size: int = pipeline_cfg.get("async_queue_size", 500)

        # Database repository
        self.repository = Repository(client_slug)

        # Load camera configurations
        self.camera_configs = load_cameras_from_db(
            client_slug=client_slug,
            applications=self.applications,
        )
        if not self.camera_configs:
            raise ValueError("No cameras configured. Check config file or database.")

        # Initialize ML models
        self.models = ModelFactory(self.config, client_slug)
        self.models.initialize_all()

        # Sync missing embeddings on startup
        self._sync_embeddings_on_startup()

        # Build name → user_id mapping
        self.name_to_id_map = self._build_name_to_id_map()

        # Initialize video streams
        self.stream_manager = StreamManager(self.camera_configs)
        self.stream_manager.init_streams()

        self.save_video = kwargs.get("save_video", False)
        if self.save_video:
            output_dir = kwargs.get("output_dir", "volumes/storage/person-tracking")
            self.stream_manager.init_video_writers(output_dir)
            self._annotator = FrameAnnotator()
        else:
            self._annotator = None

        # Initialize per-camera engines (CPU-only components)
        self.camera_engines = self._init_camera_engines()

        # Entry logger (handles status tracking + Celery dispatch)
        self.entry_logger = self._init_entry_logger()

        # GPU worker (shared across all cameras)
        n_cameras = len(self.camera_configs)
        self.gpu_worker = GPUInferenceWorker(
            detector=self.models.person_detector,
            face_detector=self.models.face_detector,
            num_cameras=n_cameras,
        )

        # Async logger (non-blocking I/O)
        self.async_logger = AsyncLogger(
            entry_logger=self.entry_logger,
            async_queue_size=self._async_queue_size,
        )

        # One CameraWorker per camera
        self.camera_workers = self._init_camera_workers()

        logger.debug(
            f"SmartOfficeEngine initialised: {n_cameras} camera(s), "
            f"detect_every={self._detection_interval} frames, "
            f"recog_every={self._recognition_interval} detection-frames"
        )

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _init_camera_engines(self) -> List[CameraEngine]:
        engines = []
        for config in self.camera_configs:
            engine = CameraEngine(
                camera_config=config,
                face_detector=self.models.face_detector,
                face_recognizer=self.models.face_recognizer,
                person_detector=self.models.person_detector,
                client_slug=self.client_slug,
                global_id_generator=self.models.global_id_generator,
                name_to_id_map=self.name_to_id_map,
                global_track_manager=self.models.global_track_manager,
                action_recognizer=self.models.action_recognizer,
            )
            engines.append(engine)
        return engines

    def _init_entry_logger(self) -> EntryLogger:
        args = type("Args", (), {})()
        args.client_slug = self.client_slug
        args.logger = logger
        args.db_names = self.models.face_recognizer.db_names
        args.production = True
        return EntryLogger(args=args)

    def _init_camera_workers(self) -> List[CameraWorker]:
        workers = []
        for idx, (config, engine) in enumerate(
            zip(self.camera_configs, self.camera_engines)
        ):
            video_writer = (
                self.stream_manager.video_writers[idx]
                if self.save_video and idx < len(self.stream_manager.video_writers)
                else None
            )
            worker = CameraWorker(
                camera_idx=idx,
                camera_config=config,
                camera_engine=engine,
                gpu_worker=self.gpu_worker,
                async_logger=self.async_logger,
                stream_handler=self.stream_manager.streams[idx],
                detection_interval=self._detection_interval,
                recognition_interval=self._recognition_interval,
                annotator=self._annotator,
                video_writer=video_writer,
            )
            workers.append(worker)
        return workers

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the engine to stop (signal-handler callback)."""
        self._running = False

    def run(self) -> None:
        """Start all threads and block until stopped."""
        self._running = True
        self._start_time = time.time()

        # Start background streams
        self.stream_manager.start_streams()

        # Start GPU worker thread
        self.gpu_worker.start()

        # Start async logger workers
        self.async_logger.start()

        # Start camera worker threads
        for worker in self.camera_workers:
            worker.start()

        logger.info("SmartOfficeEngine started (parallel pipeline)")

        last_validation_time = time.time()
        validation_interval = 30.0

        try:
            while self._running:
                time.sleep(1.0)

                # Periodic global track validation
                current_time = time.time()
                if current_time - last_validation_time >= validation_interval:
                    if self.models.global_track_manager:
                        self.models.global_track_manager.periodic_validation()
                    last_validation_time = current_time

        except Exception as e:
            logger.error(f"SmartOfficeEngine error: {e}")
            raise
        finally:
            self._cleanup()

    # ── Config reload (unchanged from original) ───────────────────────────────

    def reload_camera_configs(self) -> bool:
        """Reload camera configurations from database.

        Called when camera config changes are received via MDA.
        Returns True if reload was successful, False otherwise.
        """
        try:
            logger.info("Reloading camera configurations...")

            new_configs = load_cameras_from_db(
                client_slug=self.client_slug,
                applications=self.applications,
            )
            if not new_configs:
                logger.warning("No cameras found after reload — keeping existing config")
                return False

            old_ids = {c.get("camera_id") for c in self.camera_configs}
            new_ids = {c.get("camera_id") for c in new_configs}

            if old_ids == new_ids:
                for new_config in new_configs:
                    for i, old_config in enumerate(self.camera_configs):
                        if old_config.get("camera_id") == new_config.get("camera_id"):
                            self.camera_configs[i] = new_config
                            for engine in self.camera_engines:
                                if engine.camera_id == new_config.get("camera_id"):
                                    engine.application = new_config.get(
                                        "application", ["attendance"]
                                    )
                                    logger.info(
                                        f"Updated camera {engine.camera_id} "
                                        f"applications: {engine.application}"
                                    )
                            break
                logger.info(
                    f"Camera configurations updated (same {len(new_configs)} cameras)"
                )
            else:
                logger.info(
                    f"Camera set changed: {old_ids} → {new_ids}. Restarting engine..."
                )
                self.needs_reinit = True
                self.camera_configs = new_configs
                self.stop()

            return True

        except Exception as e:
            logger.error(f"Failed to reload camera configs: {e}")
            return False

    def reload_embeddings(self) -> bool:
        """Reload face embeddings from pgvector database.

        Called when new user embeddings are created/updated/deleted via MDA.
        Returns True if reload was successful, False otherwise.
        """
        try:
            logger.info("Reloading face embeddings...")

            self.models.face_recognizer.reload_embeddings()

            self.name_to_id_map = self._build_name_to_id_map()
            for engine in self.camera_engines:
                engine.name_to_id_map = self.name_to_id_map

            self.entry_logger.current_users = self.models.face_recognizer.db_names
            self.entry_logger.name_to_id = [
                {"name": name, "id": user_id}
                for name, user_id in self.name_to_id_map.items()
            ]
            self.entry_logger.reload_status()

            logger.info(
                f"Face embeddings reloaded: "
                f"{len(self.models.face_recognizer.db_names)} users"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to reload embeddings: {e}")
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        logger.info("Shutting down SmartOfficeEngine...")

        # Stop camera workers
        for worker in self.camera_workers:
            worker.stop(timeout=3.0)

        # Stop GPU worker
        self.gpu_worker.stop(timeout=5.0)

        # Stop async logger (let queued events drain briefly)
        self.async_logger.stop(timeout=5.0)

        # Log final global tracking metrics
        if self.models.global_track_manager and self.models.global_track_manager.enabled:
            self._log_final_metrics()

        # Stop action recognizer workers
        self.models.cleanup()

        # Stop streams
        self.stream_manager.cleanup()

        self._log_final_stats()
        logger.info("SmartOfficeEngine shutdown complete")

    def _sync_embeddings_on_startup(self) -> bool:
        try:
            logger.info("=" * 40)
            logger.info("STARTUP EMBEDDING SYNC")
            logger.info("=" * 40)

            sync_service = EmbeddingSyncService(
                client_slug=self.client_slug,
                gpu_id=0,
                config=self.config,
                detector=self.models.face_detector,
                store=self.models.face_recognizer.pgvector_store,
            )
            result = sync_service.sync_missing_embeddings()

            if result.get("success"):
                users_processed = result.get("users_processed", 0)
                embeddings_added = result.get("embeddings_added", 0)
                if users_processed > 0:
                    logger.info(
                        f"Startup sync complete: {users_processed} users, "
                        f"{embeddings_added} embeddings added"
                    )
                    self.models.face_recognizer.reload_embeddings()
                    logger.info("Face recognizer reloaded with new embeddings")
                return True
            else:
                logger.error(f"Startup sync failed: {result.get('error')}")
                logger.info("=" * 80)
                return False

        except Exception as e:
            logger.error(f"Failed to sync embeddings on startup: {e}")
            logger.warning("Continuing with existing embeddings...")
            return False

    def _build_name_to_id_map(self) -> dict:
        try:
            users = self.repository.get_user_name_to_id()
            name_map = {}
            for user in users:
                name = user.get("name")
                user_id = user.get("id")
                if name and user_id:
                    name_map[name] = user_id
            logger.debug(f"Built name-to-ID mapping for {len(name_map)} users")
            return name_map
        except Exception as e:
            logger.warning(f"Failed to build name-to-ID map: {e}")
            return {}

    def _log_final_metrics(self) -> None:
        logger.info("=" * 80)
        logger.info("PHASE 0 - FINAL BASELINE METRICS")
        logger.info("=" * 80)
        self.models.global_track_manager.log_baseline_summary()
        metrics = self.models.global_track_manager.get_baseline_metrics()
        logger.info(f"Total tracks created: {metrics['total_tracks_created']}")
        logger.info(f"Total tracks removed: {metrics['total_tracks_removed']}")
        logger.info(
            f"Average track duration: {metrics['avg_track_duration_sec']:.1f}s"
        )
        logger.info(f"Face visibility rate: {metrics['face_visibility_rate']:.1%}")
        logger.info(f"Faces detected: {metrics['total_faces_detected']}")
        logger.info(f"Faces not visible: {metrics['total_faces_not_visible']}")
        logger.info("=" * 80)

    def _log_final_stats(self) -> None:
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        logger.info(f"Total Runtime: {elapsed:.0f}s")
