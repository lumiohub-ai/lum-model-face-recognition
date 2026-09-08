"""Camera-worker entrypoint (LSO-186).

Replaces the old static `celery worker -Q cam.2,cam.3,…` command. Each replica:

  1. leases one free slot from Redis (pipeline/slot_lease.py),
  2. renews it on a background timer,
  3. runs Celery consuming only that slot's queue (`cam-slot-<n>`).

So `deploy.replicas: N` alone divides the slots across identical containers —
no hand-maintained per-camera `-Q` list, and a camera added in the app hashes
to a slot a worker already consumes.

Fails loudly rather than idling: a camera-worker with no slot consumes no
queue and tracks nothing, which is the exact silent under-coverage this ticket
removes. If it can't claim a slot (replicas > N, or Redis down past the
timeout) it exits non-zero so the failure is visible, not a quietly useless
container.
"""

import os
import sys
import threading
import time

from loguru import logger

from config.settings import settings
from workers.celery_app import celery, slot_queue_name
from pipeline.slot_lease import SlotLeaseManager

# TTL long enough to ride out a GC pause / slow renew; renewed at TTL/3.
_TTL = int(os.getenv("SO_SLOT_LEASE_TTL", "30"))
# At cold boot all replicas race for slots via SET NX — that resolves in one
# pass. This timeout only covers Redis not being ready yet; a persistent None
# past it means a real misconfig (replicas > N), which should crash-loop
# visibly rather than spin forever.
_CLAIM_TIMEOUT = int(os.getenv("SO_SLOT_CLAIM_TIMEOUT", "60"))


def _claim_with_retry(mgr: SlotLeaseManager) -> int | None:
    deadline = time.monotonic() + _CLAIM_TIMEOUT
    while True:
        slot = mgr.claim_free_slot()
        if slot is not None:
            return slot
        if time.monotonic() >= deadline:
            return None
        time.sleep(2)


def _renew_loop(mgr: SlotLeaseManager, slot: int, stop: threading.Event) -> None:
    interval = max(1, _TTL // 3)
    while not stop.wait(interval):
        if not mgr.renew(slot):
            # Another worker owns this slot now (our TTL lapsed) — continuing
            # would run two trackers over the same cameras. Exit hard so the
            # container restarts and re-claims a slot cleanly.
            logger.critical(
                f"camera_boot: lost slot {slot} lease — exiting to reclaim"
            )
            os._exit(1)


def main() -> None:
    n = settings.camera_slot_count
    mgr = SlotLeaseManager(n_slots=n, ttl_seconds=_TTL)

    slot = _claim_with_retry(mgr)
    if slot is None:
        logger.critical(
            f"camera_boot: could not lease any of {n} slots within "
            f"{_CLAIM_TIMEOUT}s — replicas exceed SO_CAMERA_SLOT_COUNT, or "
            f"Redis is unreachable. A camera-worker with no slot tracks "
            f"nothing; exiting so this is visible."
        )
        sys.exit(1)

    queue = slot_queue_name(slot)
    stop = threading.Event()
    threading.Thread(
        target=_renew_loop, args=(mgr, slot, stop), daemon=True, name="slot-renew"
    ).start()

    # Release the slot on graceful shutdown so a deliberate restart doesn't
    # cost the full TTL before another replica can take the slot over.
    from celery.signals import worker_shutdown

    @worker_shutdown.connect
    def _release(**_kwargs):  # noqa: ANN001
        stop.set()
        mgr.release(slot)

    logger.info(
        f"camera_boot[{mgr.worker_id}]: leased slot {slot}/{n}, consuming {queue}"
    )
    # Same flags the old static command used; --pool=solo keeps one camera's
    # frames strictly ordered on this worker.
    celery.worker_main(
        argv=["worker", "-Q", queue, "--pool=solo", "-l", "warning"]
    )


if __name__ == "__main__":
    main()
