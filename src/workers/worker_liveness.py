"""Worker-era AI liveness beats for Uptime Kuma (LSO-214).

After the Celery split (LSO-187) the old single `person-tracking:8765` TCP
monitor only proves one of seven services has a port open — it stops meaning
"the pipeline is alive". This sidecar pushes a per-component liveness beat to a
Kuma **push** monitor, plus the **slot-lease invariant** monitor that would have
caught the camera-worker crash-loop (LSO-218) instead of it hiding for days.

How liveness is derived per component:

  celery-consumed queues (via `celery inspect active_queues`):
    camera-worker     -> every `cam-slot-<i>` for i in range(camera_slot_count)
    yolo-worker       -> `yolo`
    face-worker       -> `face`
    reid-worker       -> `reid`
    globaltrack-worker-> `globaltrack`
    celery-worker     -> `embeddings`, `detections`
  redis leases:
    decode-worker     -> at least one `decode:lease:cam:*` key (decode is NOT a
                         celery worker, so it can't be seen via inspect)
    slot-lease        -> exactly `camera_slot_count` `track:slot:lease:*` keys,
                         held by that many DISTINCT workers (collision/starvation
                         is exactly the LSO-218 bug class)

Each component is pushed ONLY if a Kuma push URL is configured for it (opt-in),
so you add monitors in the Kuma UI and wire their push URLs into env at your own
pace. A component with no URL is silently skipped. Missing a beat is what makes
Kuma flip the monitor DOWN, so on any check failure we simply don't push "up"
(and, when we can, push an explicit "down" with a reason for a faster flip).

Env:
  SO_LIVENESS_INTERVAL           seconds between checks (default 30)
  SO_LIVENESS_CELERY_TIMEOUT     celery inspect timeout seconds (default 5)
  SO_KUMA_PUSH_CAMERA_WORKER     Kuma push URL (…/api/push/<token>)
  SO_KUMA_PUSH_YOLO_WORKER
  SO_KUMA_PUSH_FACE_WORKER
  SO_KUMA_PUSH_REID_WORKER
  SO_KUMA_PUSH_GLOBALTRACK_WORKER
  SO_KUMA_PUSH_CELERY_WORKER
  SO_KUMA_PUSH_DECODE_WORKER
  SO_KUMA_PUSH_SLOT_LEASE

Run:  python3 -m workers.worker_liveness
"""

import os
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from loguru import logger

from config.settings import settings

INTERVAL = float(os.getenv("SO_LIVENESS_INTERVAL", "30"))
CELERY_TIMEOUT = float(os.getenv("SO_LIVENESS_CELERY_TIMEOUT", "5"))
PUSH_TIMEOUT = float(os.getenv("SO_LIVENESS_PUSH_TIMEOUT", "10"))

# component -> env var holding its Kuma push URL
_PUSH_ENV = {
    "camera-worker": "SO_KUMA_PUSH_CAMERA_WORKER",
    "yolo-worker": "SO_KUMA_PUSH_YOLO_WORKER",
    "face-worker": "SO_KUMA_PUSH_FACE_WORKER",
    "reid-worker": "SO_KUMA_PUSH_REID_WORKER",
    "globaltrack-worker": "SO_KUMA_PUSH_GLOBALTRACK_WORKER",
    "celery-worker": "SO_KUMA_PUSH_CELERY_WORKER",
    "decode-worker": "SO_KUMA_PUSH_DECODE_WORKER",
    "slot-lease": "SO_KUMA_PUSH_SLOT_LEASE",
}


def _celery_worker_queues() -> "dict[str, list[str]]":
    """component -> required celery queues. camera-worker's slot queues follow
    camera_slot_count so this stays correct if N is ever rebalanced."""
    return {
        "camera-worker": [f"cam-slot-{i}" for i in range(settings.camera_slot_count)],
        "yolo-worker": ["yolo"],
        "face-worker": ["face"],
        "reid-worker": ["reid"],
        "globaltrack-worker": ["globaltrack"],
        "celery-worker": ["embeddings", "detections"],
    }


def _consumed_queues() -> "set[str] | None":
    """Set of queue names with at least one live celery consumer, or None if the
    broker/cluster can't be inspected at all (→ every celery component is down)."""
    try:
        from workers.celery_app import celery

        replies = celery.control.inspect(timeout=CELERY_TIMEOUT).active_queues()
    except Exception as e:
        logger.warning(f"celery inspect failed: {type(e).__name__}: {e}")
        return None
    if not replies:
        return set()  # broker reachable but no workers answered
    consumed: "set[str]" = set()
    for queues in replies.values():
        for q in queues or []:
            name = q.get("name") if isinstance(q, dict) else None
            if name:
                consumed.add(name)
    return consumed


def _redis():
    from messaging.redis_client import RedisClient

    return RedisClient.get_instance().client


def check_celery_components() -> "dict[str, tuple[bool, str]]":
    """component -> (ok, message) for every celery-backed worker type."""
    consumed = _consumed_queues()
    out: "dict[str, tuple[bool, str]]" = {}
    for comp, required in _celery_worker_queues().items():
        if consumed is None:
            out[comp] = (False, "celery inspect unreachable")
            continue
        missing = [q for q in required if q not in consumed]
        out[comp] = (not missing, "all queues consumed" if not missing
                     else f"no consumer for {','.join(missing)}")
    return out


def check_decode() -> "tuple[bool, str]":
    """decode-worker isn't celery — infer liveness from its camera leases."""
    try:
        keys = list(_redis().scan_iter(match="decode:lease:cam:*", count=1000))
    except Exception as e:
        return False, f"redis error: {type(e).__name__}"
    return (len(keys) > 0, f"{len(keys)} camera(s) being decoded" if keys
            else "no decode leases held")


def check_slot_lease() -> "tuple[bool, str]":
    """The LSO-187 runbook invariant: exactly camera_slot_count slot leases,
    held by that many DISTINCT workers. Fewer keys = a slot went unclaimed
    (starvation); duplicate holder = a collision. Either drops cameras."""
    n = settings.camera_slot_count
    try:
        r = _redis()
        keys = list(r.scan_iter(match="track:slot:lease:*", count=1000))
        holders = {r.get(k) for k in keys}
        holders.discard(None)
    except Exception as e:
        return False, f"redis error: {type(e).__name__}"
    if len(keys) == n and len(holders) == n:
        return True, f"{n}/{n} slots leased by {n} distinct workers"
    return False, f"expected {n} slots/{n} workers, got {len(keys)} keys / {len(holders)} holders"


def _push(url: str, ok: bool, msg: str) -> None:
    params = urlencode({"status": "up" if ok else "down", "msg": msg[:255], "ping": ""})
    try:
        with urlopen(f"{url}?{params}", timeout=PUSH_TIMEOUT) as resp:
            resp.read()
    except Exception as e:
        logger.warning(f"Kuma push failed ({url.rsplit('/', 1)[-1]}): {type(e).__name__}: {e}")


def run_once() -> None:
    results: "dict[str, tuple[bool, str]]" = {}
    results.update(check_celery_components())
    results["decode-worker"] = check_decode()
    results["slot-lease"] = check_slot_lease()

    for comp, (ok, msg) in results.items():
        url = os.getenv(_PUSH_ENV[comp], "").strip()
        level = logger.info if ok else logger.warning
        level(f"[liveness] {comp}: {'UP' if ok else 'DOWN'} — {msg}"
              + ("" if url else " (no push URL; not reported)"))
        if url:
            _push(url, ok, msg)


def main() -> None:
    configured = [c for c, e in _PUSH_ENV.items() if os.getenv(e, "").strip()]
    logger.info(
        f"[liveness] starting: interval={INTERVAL}s slots={settings.camera_slot_count} "
        f"reporting={configured or 'NONE (set SO_KUMA_PUSH_* to enable)'}"
    )
    while True:
        try:
            run_once()
        except Exception as e:
            logger.exception(f"[liveness] cycle error: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
