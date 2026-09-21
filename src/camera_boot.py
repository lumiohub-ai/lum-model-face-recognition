"""Camera-worker entrypoint (LSO-218).

Each replica is identical and interchangeable. It:

  1. claims individual cameras from Redis, up to its fair share of the fleet
     (pipeline/camera_lease.CameraLeaseManager, key prefix `track:cam:lease:`),
  2. adds a Celery consumer on ITSELF for each claimed camera's own queue
     (`cam.<id>`), and cancels it when the camera leaves,
  3. renews the leases it holds on a timer independent of the claim loop,
  4. releases every lease on graceful shutdown.

Why per-camera instead of the LSO-186 hash-to-slot scheme (`crc32(id) % N`
onto `cam-slot-<n>`): the hash balances by camera *count* and by luck, not by
what each camera costs. On Incheon that put 10 of 14 cameras on one worker,
which then OOM-crash-looped and took all 10 offline at once (LSO-218). Per
camera leases do three things the slots could not:

  - balance by construction — every worker stops claiming at its fair share,
    so no worker is handed a disproportionate load;
  - shrink the failure blast radius — when a worker dies, its cameras are
    re-claimed one at a time by the survivors, instead of a whole slot going
    dark until that one container comes back;
  - make rebalancing a `replicas` / capacity change, not an image rebuild
    (no `camera_slot_count` constant baked into the image any more).

Fair share and failover
-----------------------
`fair_share = ceil(active_cameras / SO_CAMERA_WORKER_REPLICAS)`. Normally a
worker claims only up to fair_share, which keeps the fleet balanced. If a
camera has been owned by nobody for longer than SO_CAMERA_FAILOVER_GRACE_S,
that is a signal a replica is missing, and any worker may claim it past its
fair share, up to the hard cap SO_CAMERA_WORKER_CAPACITY. With R replicas and
capacity >= ceil(C / (R - 1)), the fleet survives the loss of one worker
without any single worker exceeding its memory budget. `REPLICAS` must match
the compose `replicas` count (it is the fleet size used to compute the fair
share, which a lone booting replica cannot infer from Redis).

Consumers are dynamic because the camera set is: a camera enabled in the app
is claimed on the next reconcile pass and starts being consumed; one that is
disabled has its consumer cancelled and lease released. The worker boots
consuming an idle placeholder queue, then adds/cancels real `cam.<id>`
consumers on itself once Celery is ready (worker_ready hands us the Consumer
to schedule onto — see CeleryConsumerRegistry for why not remote control).
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Callable, Dict, Iterable, List

from loguru import logger

from config.settings import settings
from pipeline.camera_lease import CameraLeaseManager
from workers.celery_app import camera_queue_name, celery

# Fleet size the fair share is computed against. Must match the camera-worker
# `replicas` in the deploy. Not discoverable at runtime: a replica booting
# alone into an otherwise-empty Redis cannot tell "I am the only worker" from
# "the others are still starting".
_REPLICAS = int(os.getenv("SO_CAMERA_WORKER_REPLICAS", "3"))
# Hard cap on cameras one worker will ever hold. Sized so a surviving worker
# stays inside its memory limit while covering a dead peer's share:
# capacity >= ceil(C / (R - 1)) is what makes single-worker failover possible.
_CAPACITY = int(os.getenv("SO_CAMERA_WORKER_CAPACITY", "7"))
# TTL long enough to ride out a GC pause / slow renew; renewed at TTL/3.
_TTL = int(os.getenv("SO_CAMERA_LEASE_TTL_S", "30"))
_CLAIM_INTERVAL_S = float(os.getenv("SO_CAMERA_CLAIM_INTERVAL_S", "5"))
# How long a camera must be owned by nobody before a worker may claim it past
# its fair share. Long enough that a normal fleet boot (workers racing for
# their share over a few seconds) settles first, short enough that a crashed
# worker's cameras are picked up promptly.
_FAILOVER_GRACE_S = float(os.getenv("SO_CAMERA_FAILOVER_GRACE_S", "30"))
_UNCLAIMED_WARN_INTERVAL_S = float(
    os.getenv("SO_CAMERA_UNCLAIMED_WARN_INTERVAL_S", "30")
)
# Distinct from decode's prefix so the two ownership decisions never collide.
_KEY_PREFIX = os.getenv("SO_CAMERA_LEASE_KEY_PREFIX", "track:cam:lease:")
# A queue no producer publishes to: the worker must boot consuming something,
# and the real per-camera consumers are added once it is ready.
_IDLE_QUEUE = "cam-worker-idle"
# Consecutive failed renew CALLS (Redis unreachable, not ownership loss) to
# tolerate before exiting. TTL/interval is ~3 renews per lease lifetime, so 2
# rides out a brief blip while still bailing before the lease truly lapses.
_RENEW_MAX_TRANSIENT_FAILS = 2


class CeleryConsumerRegistry:
    """Adds/removes Celery consumers on THIS worker, in-process.

    Deliberately NOT `app.control.add_consumer`: that is a pidbox broadcast
    addressed to a node, and — verified in a spike — one sent from the
    worker's own `worker_ready` moment is silently dropped (the pidbox has
    not finished subscribing that early; `reply=[]` for the first seconds).
    It is also a pointless broker round trip to talk to ourselves.

    `Consumer.call_soon` schedules onto this worker's own event loop — exactly
    what the remote-control command does internally — and is safe to call
    from the reconcile thread. The Consumer is handed to us by `worker_ready`,
    which also means the worker is far enough along that its loop is running.
    """

    def __init__(self, consumer):
        self._consumer = consumer

    def add(self, queue: str) -> None:
        self._consumer.call_soon(self._consumer.add_task_queue, queue)

    def remove(self, queue: str) -> None:
        self._consumer.call_soon(self._consumer.cancel_task_queue, queue)


def _load_eligible() -> List[dict]:
    """Every attendance camera this AI should be tracking right now."""
    from config import load_cameras_from_db

    slug = settings.client_slug
    if not slug:
        raise RuntimeError(
            "SO_CLIENT_SLUG is unset — cannot resolve this site's cameras"
        )
    return load_cameras_from_db(client_slug=slug, applications=["attendance"])


class CameraWorker:
    """One replica's claim/renew/reconcile loop. See module docstring."""

    def __init__(
        self,
        *,
        registry: CeleryConsumerRegistry,
        lease: CameraLeaseManager,
        load_eligible: Callable[[], Iterable[dict]],
        replicas: int = _REPLICAS,
        capacity: int = _CAPACITY,
        failover_grace_s: float = _FAILOVER_GRACE_S,
        claim_interval_s: float = _CLAIM_INTERVAL_S,
        warn_interval_s: float = _UNCLAIMED_WARN_INTERVAL_S,
        ttl_seconds: int = _TTL,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._registry = registry
        self._lease = lease
        self._load_eligible = load_eligible
        self._replicas = max(1, int(replicas))
        self._capacity = max(1, int(capacity))
        self._failover_grace_s = failover_grace_s
        self._claim_interval_s = claim_interval_s
        self._warn_interval_s = warn_interval_s
        self._ttl = max(1, int(ttl_seconds))
        self._clock = clock

        self._held: set[int] = set()
        # camera_id -> monotonic time we first saw it owned by nobody, so a
        # genuinely orphaned camera (a replica is missing) can be told apart
        # from one another worker is about to claim in the normal boot race.
        self._unowned_since: Dict[int, float] = {}
        self._lock = threading.Lock()
        # Serializes a full reconcile PASS so the loop and any future
        # config-reload trigger cannot run one concurrently.
        self._reconcile_lock = threading.Lock()
        self._stop = threading.Event()
        self._running = False
        self._last_unclaimed_warn = 0.0
        self._transient_renew_fails = 0

    @property
    def held(self) -> set:
        with self._lock:
            return set(self._held)

    @property
    def worker_id(self) -> str:
        return self._lease.worker_id

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin reconciling. Call once, from `worker_ready`."""
        self._running = True
        threading.Thread(
            target=self._reconcile_loop, daemon=True, name="camera-reconcile"
        ).start()
        threading.Thread(
            target=self._renew_loop, daemon=True, name="camera-renew"
        ).start()

    def stop(self) -> None:
        self._stop.set()
        self._running = False

    def release_all(self) -> None:
        """Stop and hand back every camera — prompt failover on restart,
        rather than making the survivors wait out each lease's TTL."""
        self.stop()
        for camera_id in self.held:
            self._release(camera_id)

    def _reconcile_loop(self) -> None:
        while self._running:
            try:
                self.reconcile()
            except Exception as e:
                # A transient camera_loader / DB failure must not kill the
                # loop — keep the current set and try again next tick.
                logger.warning(f"camera-worker[{self.worker_id}]: reconcile failed: {e}")
            if self._stop.wait(self._claim_interval_s):
                break

    def _renew_loop(self) -> None:
        interval = max(1, self._ttl // 3)
        while not self._stop.wait(interval):
            self.renew_once()

    def renew_once(self) -> None:
        """One renewal pass over everything this worker holds.

        Runs on its own timer, deliberately independent of `reconcile`:
        claiming is cheap here (no RTSP connect, unlike decode), but keeping
        ownership must never be starved behind a slow pass either.
        """
        with self._lock:
            ids = list(self._held)
        for camera_id in ids:
            status = self._lease.renew_status(camera_id)
            if status is True:
                self._transient_renew_fails = 0
            elif status is False:
                # Definitive loss: another worker now owns this camera. Stop
                # consuming immediately so two trackers never run over the
                # same camera's frames.
                logger.warning(
                    f"camera-worker[{self.worker_id}]: lost lease for "
                    f"camera {camera_id} — releasing it"
                )
                self._release(camera_id)
            else:
                # None: the renew CALL failed (Redis blip), not proof of loss.
                # Tolerate a couple, then exit to re-claim cleanly rather than
                # run past the TTL.
                self._transient_renew_fails += 1
                if self._transient_renew_fails >= _RENEW_MAX_TRANSIENT_FAILS:
                    logger.critical(
                        f"camera-worker[{self.worker_id}]: "
                        f"{self._transient_renew_fails} consecutive renew "
                        f"failures (Redis unreachable) — exiting to reclaim"
                    )
                    os._exit(1)
                logger.warning(
                    f"camera-worker[{self.worker_id}]: renew failed "
                    f"transiently ({self._transient_renew_fails}/"
                    f"{_RENEW_MAX_TRANSIENT_FAILS}) — retrying"
                )

    # ── the claim/release reconcile pass ─────────────────────────────────

    def reconcile(self) -> None:
        with self._reconcile_lock:
            self._reconcile_locked()

    def _reconcile_locked(self) -> None:
        eligible = self._load_eligible()
        eligible_ids = [
            c["camera_id"] for c in eligible if c.get("camera_id") is not None
        ]
        eligible_set = set(eligible_ids)

        # 1. Give up anything no longer eligible (camera disabled/removed).
        for camera_id in self.held - eligible_set:
            self._release(camera_id)

        # 2. Find cameras owned by nobody, remembering since when. is_claimed
        #    assumes claimed on a Redis failure, so an outage reads as "no
        #    orphans" — it cannot itself trigger a claim storm.
        now = self._clock()
        unowned = set()
        for camera_id in eligible_ids:
            if camera_id in self.held:
                continue
            if self._lease.is_claimed(camera_id):
                self._unowned_since.pop(camera_id, None)
            else:
                self._unowned_since.setdefault(camera_id, now)
                unowned.add(camera_id)
        # Forget timers for cameras that are gone.
        for camera_id in list(self._unowned_since):
            if camera_id not in eligible_set:
                self._unowned_since.pop(camera_id, None)

        fair_share = max(1, math.ceil(len(eligible_ids) / self._replicas))
        stale = {
            cid
            for cid in unowned
            if now - self._unowned_since.get(cid, now) >= self._failover_grace_s
        }

        # 3. Claim, lowest id first, while under fair share — or past it to
        #    cover a camera nobody has held for the failover grace, up to cap.
        for camera_id in eligible_ids:
            if camera_id not in unowned:
                continue
            held = self.held
            if len(held) >= self._capacity:
                break
            if len(held) < fair_share or camera_id in stale:
                if self._lease.claim(camera_id):
                    self._adopt(camera_id)
                    self._unowned_since.pop(camera_id, None)

        # 4. Visibility: cameras nobody owns anywhere, once the grace has
        #    passed (before that, the boot race is still settling).
        remaining = [cid for cid in unowned if cid not in self.held]
        if remaining and stale and now - self._last_unclaimed_warn >= self._warn_interval_s:
            self._last_unclaimed_warn = now
            logger.warning(
                f"camera-worker[{self.worker_id}]: {len(remaining)} camera(s) "
                f"have no owner after {self._failover_grace_s:.0f}s — the "
                f"fleet is short a replica (replicas={self._replicas}, "
                f"capacity={self._capacity}). Uncovered: {sorted(remaining)}"
            )

    def _adopt(self, camera_id: int) -> None:
        """Start consuming a camera we just won the lease for."""
        queue = camera_queue_name(camera_id)
        try:
            self._registry.add(queue)
        except Exception as e:
            # Hand the lease back rather than hold a camera we cannot consume
            # — better another worker tries than this one silently drops it.
            logger.warning(
                f"camera-worker[{self.worker_id}]: failed to add consumer for "
                f"camera {camera_id} ({queue}): {e} — releasing it"
            )
            self._lease.release(camera_id)
            return
        with self._lock:
            self._held.add(camera_id)
        logger.info(
            f"camera-worker[{self.worker_id}]: claimed camera {camera_id}, "
            f"consuming {queue}"
        )

    def _release(self, camera_id: int) -> None:
        """Stop consuming a camera and give its lease up immediately."""
        with self._lock:
            self._held.discard(camera_id)
        try:
            self._registry.remove(camera_queue_name(camera_id))
        except Exception as e:
            logger.debug(
                f"camera-worker[{self.worker_id}]: cancel consumer for "
                f"camera {camera_id} failed: {e}"
            )
        self._lease.release(camera_id)


def main() -> None:
    from celery.signals import worker_ready, worker_shutdown, worker_shutting_down

    lease = CameraLeaseManager(ttl_seconds=_TTL, key_prefix=_KEY_PREFIX)
    # The registry needs the node name, which Celery only reveals once the
    # worker is up — so it is created in the worker_ready handler below.
    worker_holder: Dict[str, CameraWorker] = {}

    @worker_ready.connect
    def _on_ready(sender=None, **_kwargs):  # noqa: ANN001
        # `sender` is this worker's Consumer; it is what can add per-camera
        # queues to our own consumer group (see CeleryConsumerRegistry).
        node = getattr(sender, "hostname", None) or "camera-worker"
        worker = CameraWorker(
            registry=CeleryConsumerRegistry(sender),
            lease=lease,
            load_eligible=_load_eligible,
        )
        worker_holder["worker"] = worker
        logger.info(
            f"camera-worker[{lease.worker_id}]: ready as {node}; claiming "
            f"cameras (replicas={_REPLICAS} capacity={_CAPACITY})"
        )
        worker.start()

    @worker_shutting_down.connect
    def _log_reason(sig=None, how=None, exitcode=None, **_kwargs):  # noqa: ANN001
        # A silent restart (exit 0, nothing logged) is unattributable after
        # the fact; Celery hands us the trigger here (LSO-218/0.8.5).
        logger.warning(
            f"camera-worker[{lease.worker_id}]: shutting down — "
            f"sig={sig} how={how} exitcode={exitcode}"
        )

    @worker_shutdown.connect
    def _release_all(**_kwargs):  # noqa: ANN001
        worker = worker_holder.get("worker")
        if worker is not None:
            worker.release_all()

    celery.worker_main(
        argv=[
            "worker",
            "-n", "camera-worker@%h",
            "-Q", _IDLE_QUEUE,
            "--pool=solo",
            "-l", "warning",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
