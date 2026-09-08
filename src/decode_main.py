"""Decode-worker daemon: reads camera RTSP streams and hands detection
frames to the `yolo` queue, in its own process with its own CPU budget.

Why this exists: `person-tracking` used to decode every camera's stream
AND serve the cross-camera identity lookup (GlobalTrackManager) in the same
process, capped at a shared CPU limit. Decoding costs roughly one CPU core
per 1440p camera, so it starved the identity lookup — a blocking call with
a timeout — under load. Every timeout dropped the track to a local-only ID,
so the person was never recognised. See docs/FOLLOW_UPS.md item 5 and the
LSO-179 design for the full story, including the live fallback observed at
only 6 of 10 cameras connected.

Not a Celery worker: decoding holds a persistent RTSP connection per
camera, which a stateless-per-call task cannot do. This is a plain
long-running daemon, shaped like main.py, that SENDS Celery tasks (via
CeleryCameraProducer) but consumes none.

Camera ownership is a short-lived Redis lease
(pipeline.camera_lease.CameraLeaseManager), not a computed formula — see
that module's docstring for why `camera_id % N` was rejected. Every replica
of this process is identical and interchangeable: each claims cameras up
to SO_DECODE_WORKER_CAPACITY, renews what it holds, and releases anything
no longer eligible or on shutdown. Replica count and per-worker capacity
are config, sized N+1 for the target host by a human — not auto-scaled;
see the design doc for why that layer was deliberately left out.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_cameras_from_db
from config.settings import settings
from infrastructure.video.stream_handler import StreamHandler
from messaging.channels import INTERNAL_CHANNELS
from messaging.subscriber import start_listener
from pipeline.camera_lease import CameraLeaseManager
from pipeline.decode_metrics import StreamHealthReporter
from pipeline.frame_pump import CeleryCameraProducer
from workers.frame_store import RawFrameSlot

_CAPACITY = int(os.environ.get("SO_DECODE_WORKER_CAPACITY", "6"))
# Must comfortably exceed not just SO_DECODE_CLAIM_INTERVAL_S, but the
# worst-case time a single reconcile pass can take to claim+start every
# camera up to _CAPACITY — each StreamHandler opens a real RTSP connection
# (cv2.VideoCapture), which is not instant. Measured live on a cold start
# with 6 real cameras: ~1.7s each, ~10s total. At _CAPACITY=10 that's ~20s
# worst case. A newly claimed camera's FIRST renewal doesn't happen until
# the *next* reconcile call, so if (this pass's remaining claim time) +
# SO_DECODE_CLAIM_INTERVAL_S approaches the TTL, an early-claimed camera's
# lease can lapse before it's ever renewed — observed live: camera 39
# churned once (lost lease, stopped, reclaimed ~4.6s later) at the
# previous default of 15s. Not data loss or double-decode — the design's
# own self-correction worked exactly as intended — but it shouldn't
# depend on winning a race on every cold start. 60s gives ~2x margin over
# the measured worst case at full capacity.
_LEASE_TTL_S = int(os.environ.get("SO_DECODE_LEASE_TTL_S", "60"))
_CLAIM_INTERVAL_S = float(os.environ.get("SO_DECODE_CLAIM_INTERVAL_S", "5"))
_HEALTH_INTERVAL_S = float(os.environ.get("SO_DECODE_HEALTH_INTERVAL_S", "2"))
_UNCLAIMED_WARN_INTERVAL_S = float(os.environ.get("SO_DECODE_UNCLAIMED_WARN_INTERVAL_S", "30"))


class _OwnedCamera:
    """One camera this process currently decodes: its stream, its detection
    producer, and its raw-frame slot for calibration — kept together so a
    lost lease or a departed camera tears down all three atomically."""

    __slots__ = ("config", "stream", "producer", "raw_slot")

    def __init__(
        self,
        config: dict,
        stream: StreamHandler,
        producer: CeleryCameraProducer,
        raw_slot: RawFrameSlot,
    ):
        self.config = config
        self.stream = stream
        self.producer = producer
        self.raw_slot = raw_slot


class DecodeWorker:
    """One replica of the decode pool. See module docstring."""

    def __init__(self, client_slug: str, applications, detection_interval: int):
        self.client_slug = client_slug
        self.applications = applications
        self.detection_interval = detection_interval
        self.lease = CameraLeaseManager(ttl_seconds=_LEASE_TTL_S)
        self.health = StreamHealthReporter()
        self._owned: Dict[int, _OwnedCamera] = {}
        # camera_id -> (cumulative frames emitted, time.monotonic() at read),
        # for differencing into an fps in _sample_fps. Only ever touched from
        # the health-publish thread. Entries for departed cameras are dropped
        # in _stop_camera_locally so this can't grow across reclaims.
        self._fps_samples: Dict[int, tuple] = {}
        # Cameras claimed but not yet fully started. Held separately from
        # _owned because _start_camera's RTSP connect can take ~20s, and the
        # lease needs renewing throughout that window — otherwise it lapses
        # mid-startup and another worker legitimately takes the camera.
        self._starting: set = set()
        self._lock = threading.Lock()
        # Separate from self._lock (which only ever guards a single dict
        # read/write): serializes a full _reconcile() PASS. Two threads can
        # each call _reconcile() — run_forever()'s loop, and the
        # CAMERA_CONFIG_RELOAD pubsub listener (registered in start()),
        # which fires on every reload, not just a stream_url change. Without
        # this, both can observe the same stream_url change at once and
        # both call _restart_camera -> _start_camera for the same
        # camera_id: two live StreamHandler/CeleryCameraProducer instances
        # writing into the same camframe_<id>/camraw_<id> shared-memory
        # ring, with the first silently leaked (self._owned[camera_id] is a
        # plain dict overwrite). A plain Lock, not RLock: nothing under
        # _reconcile_locked re-enters _reconcile itself.
        self._reconcile_lock = threading.Lock()
        self._running = False
        self._periodic_thread: Optional[threading.Thread] = None
        self._last_unclaimed_warn = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        start_listener(
            INTERNAL_CHANNELS["CAMERA_CONFIG_RELOAD"],
            lambda _data: self._reconcile(),
            is_running=lambda: self._running,
            name="decode-config-reload",
        )
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, daemon=True, name="decode-periodic"
        )
        self._periodic_thread.start()
        logger.info(
            f"DecodeWorker[{self.lease.worker_id}] starting: "
            f"capacity={_CAPACITY} ttl={_LEASE_TTL_S}s "
            f"claim_interval={_CLAIM_INTERVAL_S}s"
        )

    def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                self._reconcile()
                self._sleep_interruptible(_CLAIM_INTERVAL_S)
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    def _sleep_interruptible(self, seconds: float) -> None:
        # A plain time.sleep(seconds) would delay lease release by up to a
        # full claim interval after stop() sets _running=False, since a
        # signal handler that only flips a flag does not interrupt an
        # in-progress sleep. Polling in short slices is what makes shutdown
        # (and therefore lease release, and therefore failover elsewhere)
        # actually prompt.
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    def _shutdown(self) -> None:
        logger.info(
            f"DecodeWorker[{self.lease.worker_id}] shutting down — "
            f"releasing all held leases"
        )
        with self._lock:
            owned_ids = list(self._owned.keys())
        for camera_id in owned_ids:
            self._release_camera(camera_id)
        if self._periodic_thread is not None:
            self._periodic_thread.join(timeout=3.0)
        logger.info(f"DecodeWorker[{self.lease.worker_id}] shutdown complete")

    # ── the claim/renew/reconcile loop ──────────────────────────────────

    def _reconcile(self) -> None:
        with self._reconcile_lock:
            self._reconcile_locked()

    def _reconcile_locked(self) -> None:
        try:
            eligible = load_cameras_from_db(
                client_slug=self.client_slug, applications=self.applications
            )
        except Exception as e:
            logger.warning(
                f"DecodeWorker: failed to load cameras, keeping current set: {e}"
            )
            return

        eligible_by_id = {
            c["camera_id"]: c for c in eligible if c.get("camera_id") is not None
        }

        with self._lock:
            owned_ids = list(self._owned.keys())

        # 1. Reconcile every camera already held against the DB.
        #
        # Renewal deliberately does NOT happen here — it runs on its own
        # timer in _renew_held (see _periodic_loop). This loop's own work is
        # unbounded in duration: _start_camera below opens a real RTSP
        # connection, which took ~1.7s/camera on a warm stream but ~20s
        # each right after a full-stack restart (measured live) while
        # mediamtx was still bringing paths up. Renewing here meant a slow
        # claim pass starved renewal of the cameras already held, and their
        # leases lapsed mid-pass — observed live churning 4 of 6 cameras
        # even at a 60s TTL. Renewal cadence must be independent of how
        # long claiming takes, or no TTL is ever large enough.
        for camera_id in owned_ids:
            new_config = eligible_by_id.get(camera_id)
            if new_config is None:
                # No longer eligible — removed, or its `application` no
                # longer matches. Release now rather than waiting for the
                # TTL, so another worker (or nobody, correctly) picks it up
                # immediately.
                self._release_camera(camera_id)
                continue
            self._sync_config(camera_id, new_config)

        # 2. Claim more, up to capacity, from whatever's unclaimed.
        with self._lock:
            held = len(self._owned)
        if held < _CAPACITY:
            for camera_id, config in eligible_by_id.items():
                if held >= _CAPACITY:
                    break
                with self._lock:
                    already_local = (
                        camera_id in self._owned or camera_id in self._starting
                    )
                if already_local:
                    continue
                if self.lease.claim(camera_id):
                    # Mark it claimed BEFORE the slow start, so _renew_held
                    # keeps its lease alive during startup — _start_camera
                    # can take ~20s (RTSP connect on a cold stream) and the
                    # camera isn't in _owned until that finishes.
                    with self._lock:
                        self._starting.add(camera_id)
                    try:
                        self._start_camera(camera_id, config)
                    finally:
                        with self._lock:
                            self._starting.discard(camera_id)
                    held += 1

        # 3. Visibility: cameras nobody could claim anywhere, not just here.
        with self._lock:
            owned_ids = set(self._owned.keys())
        unclaimed_locally = [cid for cid in eligible_by_id if cid not in owned_ids]
        if unclaimed_locally:
            self._warn_if_globally_unclaimed(unclaimed_locally)

    def _warn_if_globally_unclaimed(self, camera_ids) -> None:
        now = time.monotonic()
        if now - self._last_unclaimed_warn < _UNCLAIMED_WARN_INTERVAL_S:
            return
        truly_unclaimed = [cid for cid in camera_ids if not self.lease.is_claimed(cid)]
        if truly_unclaimed:
            self._last_unclaimed_warn = now
            logger.warning(
                f"DecodeWorker: {len(truly_unclaimed)} camera(s) have no "
                f"decode owner anywhere — total capacity across all decode "
                f"workers is insufficient: {sorted(truly_unclaimed)}. Add a "
                f"decode-worker replica or raise SO_DECODE_WORKER_CAPACITY."
            )

    def _sync_config(self, camera_id: int, new_config: dict) -> None:
        """Mirror engine.py's reload_camera_configs in-place-mutation
        pattern: the producer holds a direct reference to this same dict
        (frame_pump.py reads camera_config['roi'] live, every frame), so
        updating in place — not rebinding — is what makes a config change
        (e.g. a new ROI) visible without restarting anything. A changed
        stream_url still needs a real restart: the RTSP connection itself
        isn't re-read per frame the way roi is.
        """
        with self._lock:
            owned = self._owned.get(camera_id)
        if owned is None:
            return
        old_config = owned.config
        old_url = old_config.get("stream_url")
        new_url = new_config.get("stream_url")
        old_config.update(new_config)
        for key in [k for k in old_config if k not in new_config]:
            old_config.pop(key, None)
        if new_url != old_url:
            logger.info(
                f"DecodeWorker: camera {camera_id} stream_url changed — "
                f"restarting its stream"
            )
            self._restart_camera(camera_id, old_config)

    # ── starting/stopping one camera ────────────────────────────────────

    def _start_camera(self, camera_id: int, config: dict) -> None:
        stream = StreamHandler(src=config["stream_url"], logger=logger)
        stream.start()
        producer = CeleryCameraProducer(
            camera_id=camera_id,
            camera_config=config,
            stream_handler=stream,
            detection_interval=self.detection_interval,
            metrics_collector=None,
        )
        producer.start()
        raw_slot = RawFrameSlot(camera_id)
        with self._lock:
            self._owned[camera_id] = _OwnedCamera(config, stream, producer, raw_slot)
        logger.info(
            f"DecodeWorker[{self.lease.worker_id}]: claimed and started "
            f"camera {camera_id}"
        )

    def _restart_camera(self, camera_id: int, config: dict) -> None:
        """Reconnect one camera's stream after its stream_url changed,
        without releasing the lease — this worker still owns the camera,
        it just needs a fresh StreamHandler pointed at the new URL.

        Adds camera_id to _starting for the same reason the initial-claim
        path does: _start_camera's RTSP connect can take seconds, and the
        camera is in neither _owned nor _starting between the
        _stop_camera_locally pop and _start_camera's re-add — _renew_held
        would skip it during that window otherwise.
        """
        self._stop_camera_locally(camera_id)
        with self._lock:
            self._starting.add(camera_id)
        try:
            self._start_camera(camera_id, config)
        finally:
            with self._lock:
                self._starting.discard(camera_id)

    def _stop_camera_locally(self, camera_id: int) -> None:
        """Stop decoding a camera WITHOUT releasing its lease — used when
        the lease is already gone (lost the renew race), so there is
        nothing left to release, and as the first half of a restart."""
        with self._lock:
            owned = self._owned.pop(camera_id, None)
        # Dropped unconditionally: a restart builds a fresh producer whose
        # counter starts at 0, and a stale baseline would make the first
        # sample after it look like a negative delta.
        self._fps_samples.pop(camera_id, None)
        if owned is None:
            return
        owned.producer.stop()
        owned.stream.stop()
        owned.raw_slot.close()

    def _release_camera(self, camera_id: int) -> None:
        """Stop decoding AND give up the lease immediately — used when a
        camera leaves this worker's eligible set, and on shutdown."""
        self._stop_camera_locally(camera_id)
        self.lease.release(camera_id)
        logger.info(
            f"DecodeWorker[{self.lease.worker_id}]: released camera {camera_id}"
        )

    # ── periodic: lease renewal + stream health + raw-frame publishing ──

    def _periodic_loop(self) -> None:
        while self._running:
            # Renew FIRST, before the (slower, best-effort) health/raw-frame
            # publishing below — keeping ownership is correctness, telemetry
            # is not, so a slow publish must never delay a renewal.
            self._periodic_tick_renew()
            with self._lock:
                items = list(self._owned.items())
            for camera_id, owned in items:
                self._publish_camera_state(camera_id, owned)
            self._sleep_interruptible(_HEALTH_INTERVAL_S)

    def _periodic_tick_renew(self) -> None:
        """One renewal pass over everything this worker currently holds."""
        with self._lock:
            # Cameras mid-startup hold a lease but aren't in _owned yet —
            # they need renewing too, or a slow RTSP connect outlives the
            # lease it was claimed under.
            renew_ids = list(self._owned.keys()) + list(self._starting)
        self._renew_held(renew_ids)

    def _renew_held(self, camera_ids) -> None:
        """Extend the lease on every camera this worker holds.

        Runs on this thread's own fixed timer (_HEALTH_INTERVAL_S, default
        2s), deliberately independent of _reconcile: claiming a camera opens
        an RTSP connection, whose duration is unbounded (~1.7s warm, ~20s
        during a cold start with mediamtx still coming up), and renewal must
        not be starved behind that. A failed renew means another worker now
        owns the camera, so this one stops decoding it immediately rather
        than running a second decoder against the same shared-memory slot.
        """
        for camera_id in camera_ids:
            if not self._running:
                return
            if not self.lease.renew(camera_id):
                logger.warning(
                    f"DecodeWorker[{self.lease.worker_id}]: lost lease for "
                    f"camera {camera_id} — stopping it locally"
                )
                self._stop_camera_locally(camera_id)

    def _sample_fps(self, camera_id: int, owned: _OwnedCamera) -> Optional[float]:
        """Detection-frame rate since this camera's previous publish.

        Derived by differencing the producer's own cumulative counter rather
        than by timestamping frames: the engine's MetricsCollector (which does
        keep per-frame timestamps) is in another process now, and shipping a
        deque of timestamps across Redis every tick to re-derive a number the
        producer can compute from two ints would be a poor trade.

        Returns None — "no data", not 0.0 — for the first sample after a
        camera starts or is reclaimed. A camera that is genuinely stalled
        reports a real 0.0, and the dashboard must be able to tell those
        apart.
        """
        total = getattr(owned.producer, "_frames_emitted", None)
        if total is None:
            return None
        now = time.monotonic()
        prev = self._fps_samples.get(camera_id)
        self._fps_samples[camera_id] = (total, now)
        if prev is None:
            return None
        prev_total, prev_at = prev
        elapsed = now - prev_at
        # A restarted producer resets the counter, so a negative delta means
        # "different producer", not negative work — skip one interval.
        if elapsed <= 0 or total < prev_total:
            return None
        return round((total - prev_total) / elapsed, 2)

    def _publish_camera_state(self, camera_id: int, owned: _OwnedCamera) -> None:
        """One publish per tick, carrying both the health snapshot and the
        latest raw-frame handle — the frame is written to shared memory
        first so the handle in this same message always points at real
        pixels, never a segment that hasn't been written yet.

        Calibration tolerates a frame up to 5s old (engine.py's
        `capture_frame`); publishing every `_HEALTH_INTERVAL_S` (default
        2s) comfortably clears that with no separate timer needed.
        """
        try:
            handle = None
            ret, frame = owned.stream.read()
            if ret and frame is not None:
                handle = owned.raw_slot.write(frame)

            health = owned.stream.get_health()
            self.health.publish(
                camera_id,
                read_ms=owned.stream.get_read_avg_ms(),
                decode_ms=owned.stream.get_decode_avg_ms(),
                state=health.get("state"),
                frame_handle=handle,
                fps=self._sample_fps(camera_id, owned),
            )
        except Exception as e:
            logger.debug(f"periodic publish failed for camera {camera_id}: {e}")


def _signal_handler(worker: DecodeWorker):
    def handler(signum, _frame):
        logger.info(f"Received signal {signum}, shutting down decode worker...")
        worker.stop()

    return handler


def main() -> None:
    client_slug = settings.client_slug
    if not client_slug:
        logger.error("SO_CLIENT_SLUG environment variable is required")
        sys.exit(1)

    detection_interval = int(os.environ.get("SO_DETECTION_INTERVAL", "2"))
    worker = DecodeWorker(
        client_slug=client_slug,
        applications=["attendance"],
        detection_interval=detection_interval,
    )

    handler = _signal_handler(worker)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    worker.run_forever()


if __name__ == "__main__":
    main()
