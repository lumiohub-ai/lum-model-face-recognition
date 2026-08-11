"""SmartOfficeEngine - Unified person tracking and face recognition system.

Architecture (parallel multi-camera):
- GPUInferenceWorker: single thread owns the GPU, batches YOLO + ArcFace
- CameraWorker: one thread per camera (frame read + CPU processing)
- AsyncLogger: three background I/O threads (db, gcs, redis)
- StreamManager/StreamHandler: background threads for RTSP read (existing)
"""

import threading
import time
from typing import List, Optional

from loguru import logger

# Config
from config.settings import settings

# Infrastructure
from infrastructure.video import StreamManager
from infrastructure import EntryLogger
from infrastructure.storage import Repository, EmbeddingSyncService, PgVectorStore

# Models
from lum_vision import ModelFactory, VisionConfig

# Config
from config import load_cameras_from_db, build_vision_config

# Local
from pipeline.camera_engine import CameraEngine

from pipeline.gpu_worker import GPUInferenceWorker
from pipeline.camera_worker import CameraWorker
from pipeline.action_worker import ActionRecognitionWorker
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
        model_factory: Optional["ModelFactory"] = None,
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

        # Here we load the camera configurations from the database using the provided client slug and applications.
        self.camera_configs = load_cameras_from_db(
            client_slug=client_slug,
            applications=self.applications,
        )
        if not self.camera_configs:
            raise ValueError("No cameras configured. Check config file or database.")

        # Embedding store — owned here, and shared with both the face matcher
        # (via the factory) and the startup sync service.
        self.pgvector_store = PgVectorStore(client_slug)

        # Initialize ML models — reuse provided factory to avoid reloading GPU models
        if model_factory is not None:
            self.models = model_factory
            self._owns_models = False
        else:
            self.models = ModelFactory(
                build_vision_config(self.config),
                embedding_provider=self.pgvector_store,
            )
            self.models.initialize_all()
            self._owns_models = True

        # Action recognition: the model is synchronous, so the queue and worker
        # threads that drive it are owned here.
        action_cfg = kwargs.get("action_recognition", {}) or {}
        self.action_worker = ActionRecognitionWorker(
            recognizer=self.models.action_recognizer,
            client_slug=client_slug,
            max_queue_size=action_cfg.get("max_queue_size", 50),
            num_workers=action_cfg.get("async_workers", 1),
            min_crop_height=action_cfg.get("min_crop_height", 0),
            min_crop_width=action_cfg.get("min_crop_width", 0),
            min_crop_area=action_cfg.get("min_crop_area", 0),
        )
        self.action_worker.start_workers()

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

        # Homography registry + calibration subscriber 
        from domain.calibration.homography_registry import HomographyRegistry
        from messaging.calibration_subscriber import CalibrationSubscriber
        self.homography_registry = HomographyRegistry()
        self._calibration_subscriber = CalibrationSubscriber(self.homography_registry)
        self._calibration_subscriber.start()

        # Initialize per-camera engines (CPU-only components)
        self.camera_engines = self._init_camera_engines()

        # Entry logger (handles status tracking + Celery dispatch)
        self.entry_logger = self._init_entry_logger()

        # Metrics monitoring (optional — disable via SO_METRICS_ENABLED=false in .env)
        self._metrics_enabled = settings.metrics_enabled

        if self._metrics_enabled:
            import os
            from infrastructure.metrics_collector import MetricsCollector
            from infrastructure.metrics_server import MetricsDashboardServer
            from infrastructure.metrics_store import MetricsStore
            _cam_indices = list(range(len(self.camera_configs)))
            _db_path = os.path.join(
                kwargs.get("output_dir", "volumes/storage/person-tracking"),
                "metrics.db",
            )
            self.metrics = MetricsCollector()
            self._metrics_store = MetricsStore(
                self.metrics,
                db_path=_db_path,
                interval_sec=30.0,
                camera_indices=_cam_indices,
            )
            self._metrics_dashboard = MetricsDashboardServer(
                self.metrics,
                store=self._metrics_store,
                camera_indices=_cam_indices,
                port=settings.metrics_port,
            )
        else:
            self.metrics = None
            self._metrics_store = None
            self._metrics_dashboard = None

        if self.metrics is not None:
            self.stream_manager.set_metrics(self.metrics)

        # GPU worker (shared across all cameras)
        n_cameras = len(self.camera_configs)
        self.gpu_worker = GPUInferenceWorker(
            detector=self.models.person_detector,
            face_detector=self.models.face_detector,
            num_cameras=n_cameras,
            metrics_collector=self.metrics,
        )

        # Async logger (non-blocking I/O)
        self.async_logger = AsyncLogger(
            entry_logger=self.entry_logger,
            async_queue_size=self._async_queue_size,
        )

        # The action worker is constructed early (before models/metrics/logger
        # exist) so it can start its threads as soon as models are ready. Wire
        # its late-bound dependencies now that both exist — without this,
        # action-recognition metrics stay permanently zero (LSO-66) and
        # activity proof images are silently dropped rather than uploaded.
        self.action_worker.set_metrics_collector(self.metrics)
        self.action_worker.set_async_logger(self.async_logger)

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
            # Gate thresholds (LSO-7) are global config.yaml, not per-camera DB —
            # inject so CameraEngine._best_face_signals can prefer passing frames.
            config["unrecognized_frontality_min"] = self.config.get("unrecognized_frontality_min", 0.6)
            config["unrecognized_pitch_min"] = self.config.get("unrecognized_pitch_min", 0.4)
            engine = CameraEngine(
                camera_config=config,
                face_detector=self.models.face_detector,
                face_recognizer=self.models.face_matcher,
                person_detector=self.models.person_detector,
                client_slug=self.client_slug,
                global_id_generator=self.models.global_id_generator,
                name_to_id_map=self.name_to_id_map,
                global_track_manager=self.models.global_track_manager,
                action_recognizer=self.action_worker,
                homography_registry=self.homography_registry,
            )
            engines.append(engine)
        return engines

    def _init_entry_logger(self) -> EntryLogger:
        args = type("Args", (), {})()
        args.client_slug = self.client_slug
        args.logger = logger
        args.db_names = self.models.face_matcher.db_names
        args.production = True
        # Unrecognized-case orientation gate thresholds (LSO-7), from config.yaml.
        args.unrecognized_frontality_min = self.config.get("unrecognized_frontality_min", 0.6)
        args.unrecognized_pitch_min = self.config.get("unrecognized_pitch_min", 0.4)
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
                metrics_collector=self.metrics,
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

        # Start monitoring dashboard
        if self._metrics_enabled:
            self._metrics_dashboard.start()
            # Bootstrap psutil's non-blocking cpu_percent (first call returns 0.0)
            self.metrics.cpu_percent()

        logger.info("SmartOfficeEngine started (parallel pipeline)")

        pipeline_cfg = self.config.get("pipeline", {})
        metrics_interval: float = float(pipeline_cfg.get("metrics_interval", 30))

        last_validation_time = time.time()
        validation_interval = 30.0
        last_metrics_time = time.time()

        try:
            while self._running:
                time.sleep(1.0)

                current_time = time.time()

                # Periodic global track validation
                if current_time - last_validation_time >= validation_interval:
                    if self.models.global_track_manager:
                        self.models.global_track_manager.periodic_validation()
                    last_validation_time = current_time

                # Periodic metrics reporting
                if self._metrics_enabled and current_time - last_metrics_time >= metrics_interval:
                    self._report_metrics()
                    last_metrics_time = current_time
                # NOTE: per-camera stream-health heartbeats are published by the
                # Edge MediaMTX now (LSO-26), not the AI service.

        except Exception as e:
            logger.exception(f"SmartOfficeEngine error: {e}")
            raise
        finally:
            self._cleanup()

    # ── Config reload (unchanged from original) ───────────────────────────────

    def reload_camera_configs(self, allow_empty: bool = False) -> bool:
        """Reload camera configurations from database.

        Called when camera config changes are received via MDA.
        Returns True if reload was successful, False otherwise.
        allow_empty: if True, treat 0 cameras as a valid new state (e.g. StopCamera).
        """
        try:
            logger.info("Reloading camera configurations...")

            new_configs = load_cameras_from_db(
                client_slug=self.client_slug,
                applications=self.applications,
            )
            if not new_configs and not allow_empty:
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
            logger.exception(f"Failed to reload camera configs: {e}")
            return False

    def reload_embeddings(self) -> bool:
        """Reload face embeddings from pgvector database.

        Called when new user embeddings are created/updated/deleted via MDA.
        Returns True if reload was successful, False otherwise.
        """
        try:
            logger.info("Reloading face embeddings...")

            self.models.face_matcher.reload_embeddings()

            self.name_to_id_map = self._build_name_to_id_map()
            for engine in self.camera_engines:
                engine.name_to_id_map = self.name_to_id_map

            self.entry_logger.current_users = self.models.face_matcher.db_names
            self.entry_logger.name_to_id = [
                {"name": name, "id": user_id}
                for name, user_id in self.name_to_id_map.items()
            ]
            self.entry_logger.reload_status()

            logger.info(
                f"Face embeddings reloaded: "
                f"{len(self.models.face_matcher.db_names)} users"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to reload embeddings: {e}")
            return False

    def capture_frame(
        self,
        camera_id: int,
        command_id: str,
        frame_index: int = 1,
        undistort: bool = False,
        camera_matrix: list = None,
        dist_coeffs: list = None,
        calibration_model: str = 'fisheye',
    ) -> None:
        """Capture a single frame for camera calibration.

        Non-blocking — spawns a daemon thread so the main pipeline is never paused.
        Reads the latest frame already buffered by the RTSP background thread (no
        new RTSP connection), uploads to GCS, then publishes a FrameCaptured event.
        """
        threading.Thread(
            target=self._do_capture_frame,
            args=(camera_id, command_id, frame_index, undistort, camera_matrix, dist_coeffs, calibration_model),
            daemon=True,
            name=f"capture-{camera_id}",
        ).start()

    def _do_capture_frame(
        self,
        camera_id: int,
        command_id: str,
        frame_index: int = 1,
        undistort: bool = False,
        camera_matrix: list = None,
        dist_coeffs: list = None,
        calibration_model: str = 'fisheye',
    ) -> None:
        """Background: grab latest frame → optionally undistort → upload to GCS → save to DB → publish event."""
        from messaging.publisher import MDAPublisher

        publisher = MDAPublisher(self.client_slug)
        max_frame_age_sec = 5.0

        try:
            frame = self.stream_manager.get_fresh_frame(
                camera_id, max_age_sec=max_frame_age_sec
            )
            if frame is None:
                error = (
                    f"No live frame within {max_frame_age_sec:.0f}s for camera {camera_id}"
                )
                logger.warning(f"capture_frame: {error}")
                publisher.publish_frame_capture_failed(
                    command_id=command_id,
                    camera_id=camera_id,
                    error=error,
                )
                return

            was_undistorted = False
            if undistort and camera_matrix and dist_coeffs:
                try:
                    from domain.calibration.camera_calibrator import CameraCalibrator
                    calibrator = CameraCalibrator(fisheye=(calibration_model == 'fisheye'))
                    frame = calibrator.undistort(frame, camera_matrix, dist_coeffs, calibration_model)
                    was_undistorted = True
                except Exception as e:
                    # Best-effort — fall back to the raw frame rather than
                    # failing the whole capture over a bad undistort.
                    logger.exception(f"capture_frame: undistort failed for camera {camera_id}: {e}")

            h, w = frame.shape[:2]

            from infrastructure.storage.gcs import ImageFetcher
            image_url = ImageFetcher().upload_image(
                frame,
                prefix=f"calibration_frames/{self.client_slug}",
                client_slug=self.client_slug,
            )

            # Persist to calibration_frames table
            from infrastructure.storage.detection_repository import DetectionRepository
            from datetime import datetime
            record_id = DetectionRepository(self.client_slug).save_calibration_frame(
                camera_id=camera_id,
                frame_url=image_url,
                frame_index=frame_index,
                captured_at=datetime.utcnow(),
            )
            logger.info(f"Calibration frame saved to DB: id={record_id}, camera={camera_id}, frame_index={frame_index}")

            publisher.publish_frame_captured(
                command_id=command_id,
                camera_id=camera_id,
                image_url=image_url,
                metadata={'width': w, 'height': h, 'source': 'OpenCV', 'undistorted': was_undistorted},
            )
            logger.info(f"Frame captured: camera={camera_id}, command={command_id}, undistorted={was_undistorted}")

        except Exception as e:
            logger.exception(f"capture_frame failed for camera {camera_id}: {e}")
            publisher.publish_frame_capture_failed(
                command_id=command_id,
                camera_id=camera_id,
                error=str(e),
            )

    def calibrate_camera(self, camera_id: int, command_id: str) -> None:
        """Run Charuco calibration on stored frames for a camera.

        Non-blocking — spawns a daemon thread.
        Fetches calibration_frames from DB, downloads each image, runs
        CameraCalibrator.calibrate(), then publishes CalibrationComplete or
        CalibrationFailed.
        """
        threading.Thread(
            target=self._do_calibrate_camera,
            args=(camera_id, command_id),
            daemon=True,
            name=f"calibrate-{camera_id}",
        ).start()

    def _do_calibrate_camera(self, camera_id: int, command_id: str) -> None:
        """Background: fetch frames → download → calibrate → publish result."""
        from infrastructure.storage.detection_repository import DetectionRepository
        from infrastructure.storage.gcs import ImageFetcher
        from domain.calibration.camera_calibrator import CameraCalibrator
        from messaging.publisher import MDAPublisher

        publisher = MDAPublisher(self.client_slug)

        try:
            repo = DetectionRepository(self.client_slug)
            frame_records = repo.get_calibration_frames(camera_id)

            if not frame_records:
                publisher.publish_calibration_failed(
                    command_id=command_id,
                    camera_id=camera_id,
                    error="No calibration frames found in database",
                )
                return

            logger.info(
                f"Calibrating camera {camera_id} with {len(frame_records)} frames"
            )

            fetcher = ImageFetcher()
            frames = []
            for record in frame_records:
                url = record['frame_url']
                try:
                    # Strip https://storage.googleapis.com/{bucket}/ prefix → blob path
                    # then use the authenticated GCS client to download
                    gcs_prefix = f"https://storage.googleapis.com/{fetcher.gcs_bucket}/"
                    if url.startswith(gcs_prefix):
                        blob_path = url[len(gcs_prefix):].split('?')[0]
                        bucket = fetcher.gcs_client.bucket(fetcher.gcs_bucket)
                        image_bytes = bucket.blob(blob_path).download_as_bytes()
                        import numpy as np
                        import cv2
                        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    else:
                        frame = fetcher._fetch_from_gcs(url) if url.startswith('gs://') else None
                    if frame is not None:
                        frames.append(frame)
                    else:
                        logger.warning(f"Could not decode frame {url}")
                except Exception as e:
                    logger.warning(f"Could not download frame {url}: {e}")

            if not frames:
                publisher.publish_calibration_failed(
                    command_id=command_id,
                    camera_id=camera_id,
                    error="Could not download any calibration frames",
                )
                return

            calibrator = CameraCalibrator(fisheye=True)
            result = calibrator.calibrate(frames)

            if result.get('success'):
                publisher.publish_calibration_complete(
                    command_id=command_id,
                    camera_id=camera_id,
                    rms_error=result['rms_error'],
                    camera_matrix=result['camera_matrix'],
                    dist_coeffs=result['dist_coeffs'],
                    img_size=result['img_size'],
                    frames_used=result['frames_used'],
                    model=result['model'],
                )
            else:
                publisher.publish_calibration_failed(
                    command_id=command_id,
                    camera_id=camera_id,
                    error=result.get('error', 'Calibration failed'),
                )

        except Exception as e:
            logger.exception(f"calibrate_camera failed for camera {camera_id}: {e}")
            publisher.publish_calibration_failed(
                command_id=command_id,
                camera_id=camera_id,
                error=str(e),
            )

    def test_calibration(
        self,
        camera_id: int,
        command_id: str,
        camera_matrix: list,
        dist_coeffs: list,
        model: str = 'fisheye',
    ) -> None:
        """Capture a live frame and apply undistortion to test calibration.

        Non-blocking — spawns a daemon thread.
        """
        threading.Thread(
            target=self._do_test_calibration,
            args=(camera_id, command_id, camera_matrix, dist_coeffs, model),
            daemon=True,
            name=f"test-calib-{camera_id}",
        ).start()

    def _do_test_calibration(
        self,
        camera_id: int,
        command_id: str,
        camera_matrix: list,
        dist_coeffs: list,
        model: str,
    ) -> None:
        """Background: grab live frame → undistort → upload to GCS → publish event."""
        from domain.calibration.camera_calibrator import CameraCalibrator
        from infrastructure.storage.gcs import ImageFetcher
        from messaging.publisher import MDAPublisher

        publisher = MDAPublisher(self.client_slug)

        try:
            frame = self.stream_manager.get_frame(camera_id)
            if frame is None:
                logger.warning(f"test_calibration: no frame for camera {camera_id}")
                publisher.publish_calibration_failed(
                    command_id=command_id,
                    camera_id=camera_id,
                    error="Could not capture live frame for testing",
                )
                return

            calibrator = CameraCalibrator(fisheye=(model == 'fisheye'))
            undistorted = calibrator.undistort(frame, camera_matrix, dist_coeffs, model)

            image_url = ImageFetcher().upload_image(
                undistorted,
                prefix=f"calibration_test/{self.client_slug}",
                client_slug=self.client_slug,
            )

            publisher.publish_test_calibration_complete(
                command_id=command_id,
                camera_id=camera_id,
                image_url=image_url,
            )
            logger.info(
                f"TestCalibration complete: camera={camera_id}, command={command_id}"
            )

        except Exception as e:
            logger.exception(f"test_calibration failed for camera {camera_id}: {e}")
            publisher.publish_calibration_failed(
                command_id=command_id,
                camera_id=camera_id,
                error=str(e),
            )

    def compute_homography(
        self,
        camera_id: int,
        command_id: str,
        src_pts: list,
        dst_pts: list,
    ) -> None:
        """Solve a 3x3 image→floor homography from matched point pairs.

        Non-blocking — spawns a daemon thread. Publishes HomographyComputed or
        HomographyFailed when done.
        """
        threading.Thread(
            target=self._do_compute_homography,
            args=(camera_id, command_id, src_pts, dst_pts),
            daemon=True,
            name=f"homography-{camera_id}",
        ).start()

    def _do_compute_homography(
        self,
        camera_id: int,
        command_id: str,
        src_pts: list,
        dst_pts: list,
    ) -> None:
        from domain.calibration.homography import compute_homography
        from messaging.publisher import MDAPublisher

        publisher = MDAPublisher(self.client_slug)

        try:
            result = compute_homography(src_pts, dst_pts)
            publisher.publish_homography_calibrated(
                command_id=command_id,
                camera_id=camera_id,
                src_pts=src_pts,
                dst_pts=dst_pts,
                homography_matrix=result["homography_matrix"],
                reprojection_error=result["reprojection_error"],
                per_point_errors=result["per_point_errors"],
            )
            logger.info(
                f"ComputeHomography complete: camera={camera_id}, command={command_id}, "
                f"err={result['reprojection_error']:.3f}, method={result['method']}"
            )
        except ValueError as e:
            publisher.publish_homography_failed(command_id, camera_id, str(e))
        except Exception as e:
            logger.exception(f"compute_homography failed for camera {camera_id}: {e}")
            publisher.publish_homography_failed(command_id, camera_id, str(e))

    # ── Metrics reporting ─────────────────────────────────────────────────────

    def _report_metrics(self) -> None:
        """Log a metrics summary and publish alerts for critical conditions."""
        cam_indices = list(range(len(self.camera_workers)))

        # Log compact summary line
        self.metrics.log_summary(cam_indices)

        # Check for and handle critical alerts
        monitoring_cfg = self.config.get("monitoring", {})
        alerts = self.metrics.check_alerts(
            camera_indices=cam_indices,
            fps_threshold=float(monitoring_cfg.get("fps_alert_threshold", 1.0)),
            gpu_mem_threshold=float(monitoring_cfg.get("gpu_mem_threshold", 90.0)),
            ram_threshold=float(monitoring_cfg.get("ram_threshold", 90.0)),
        )

        if alerts:
            try:
                from messaging.publisher import MDAPublisher
                publisher = MDAPublisher(self.client_slug)
                for alert in alerts:
                    logger.warning(f"[Metrics] {alert['message']}")
                    publisher.publish_system_alert(
                        alert_type=alert["type"],
                        message=alert["message"],
                        details=alert,
                    )
            except Exception as e:
                logger.debug(f"Metrics alert publish failed: {e}")

        # Publish full metrics snapshot to Redis
        try:
            from messaging.publisher import MDAPublisher
            MDAPublisher(self.client_slug).publish_system_metrics(
                self.metrics.snapshot(cam_indices)
            )
        except Exception as e:
            logger.debug(f"Metrics publish failed: {e}")

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

        # Stop action recognition workers — always, since this engine owns them
        # regardless of where the models came from.
        self.action_worker.stop_workers()

        if self._owns_models:
            self.models.cleanup()

        # Stop streams
        self.stream_manager.cleanup()

        # Stop monitoring dashboard
        if self._metrics_enabled:
            self._metrics_dashboard.stop()
            self.metrics.cleanup()

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
                store=self.pgvector_store,
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
                    self.models.face_matcher.reload_embeddings()
                    logger.info("Face recognizer reloaded with new embeddings")
                return True
            else:
                logger.error(f"Startup sync failed: {result.get('error')}")
                logger.info("=" * 80)
                return False

        except Exception as e:
            logger.exception(f"Failed to sync embeddings on startup: {e}")
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
