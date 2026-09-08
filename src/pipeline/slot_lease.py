"""Camera-worker slot ownership, backed by a short-lived Redis lease (LSO-186).

The problem this solves: `deploy.replicas: N` launches N *identical*
camera-worker containers — same command, same env — so Compose gives us no way
to tell replica #0 to consume `cam-slot-0` and replica #1 to consume
`cam-slot-1`. The replicas have to divide the slots among themselves at
runtime. They do it through Redis, with the exact primitive
`pipeline/camera_lease.py` already uses to divide *cameras* among decode
replicas: `SET key value NX EX ttl` to claim, a compare-and-extend Lua to
renew, a compare-and-delete Lua to release. See that module for why the Lua
scripts must be atomic (a plain GET-then-EXPIRE has a race window).

Difference from CameraLeaseManager: a decode worker claims a *specific* key
(the camera it wants); a camera-worker claims *any free slot* in `0..N-1`, so
`claim_free_slot` walks the range and takes the first one it wins.

Why a lease and not `camera_id % replica_index`: identical replicas have no
stable index in Compose, and even with one, a crashed replica's cameras would
have nowhere to go until it came back. A lease is self-healing — a dead
worker stops renewing, its slot's key expires, and a restarted (or spare)
replica reclaims it on its next attempt. The tracking cost of a slot changing
hands (per-camera tracker state re-inits on the new worker) is identical to
what already happens whenever a camera-worker restarts today.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

# Reuse decode's worker-id scheme (hostname + random suffix) so `redis-cli GET
# track:slot:lease:0` shows which container holds a slot during debugging.
from pipeline.camera_lease import default_worker_id

_KEY_PREFIX = "track:slot:lease:"

# Only renew if this worker still owns the slot — see camera_lease.py.
_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
else
    return 0
end
"""

# Only release if this worker still owns the slot.
_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


def _key(slot: int) -> str:
    return f"{_KEY_PREFIX}{slot}"


class SlotLeaseManager:
    """Atomic claim/renew/release of the right to consume one camera-worker slot.

    Modelled directly on `CameraLeaseManager`: same `SET NX EX` claim, same
    atomic Lua renew/release, same decline-on-Redis-outage posture (better to
    hold no slot than to have two workers both believe they own one, which
    would split a camera's frames across two trackers).
    """

    def __init__(
        self,
        n_slots: int,
        ttl_seconds: int,
        worker_id: Optional[str] = None,
        redis_client=None,
    ):
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self._n_slots = int(n_slots)
        # EX takes whole seconds and rejects 0, so guard a sub-second TTL here
        # rather than letting it raise at claim time.
        self._ttl = max(1, int(ttl_seconds))
        self._worker_id = worker_id or default_worker_id()
        self._redis = redis_client
        self._owns_client = redis_client is None
        self._renew_sha: Optional[str] = None
        self._release_sha: Optional[str] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def n_slots(self) -> int:
        return self._n_slots

    def _client(self):
        if self._redis is not None:
            return self._redis
        from messaging.redis_client import RedisClient

        self._redis = RedisClient.get_instance().client
        return self._redis

    def _drop_client(self) -> None:
        if self._owns_client:
            self._redis = None

    def claim_free_slot(self) -> Optional[int]:
        """Take the lowest-numbered slot no other worker currently holds.

        Returns the slot number, or None if every slot in `0..n_slots-1` is
        already leased (this replica is surplus — replicas > N) or Redis is
        unreachable. The caller must treat None as fatal for a camera-worker:
        a worker with no slot consumes no queue and tracks nothing, which is
        exactly the silent under-coverage LSO-186 exists to eliminate, so it
        must fail loudly rather than idle.
        """
        try:
            client = self._client()
        except Exception as e:
            logger.warning(f"SlotLeaseManager: Redis unavailable at claim: {e}")
            self._drop_client()
            return None

        for slot in range(self._n_slots):
            try:
                won = client.set(_key(slot), self._worker_id, nx=True, ex=self._ttl)
            except Exception as e:
                logger.warning(f"SlotLeaseManager: claim of slot {slot} failed: {e}")
                self._drop_client()
                return None
            if won:
                logger.info(
                    f"SlotLeaseManager[{self._worker_id}]: claimed slot {slot} "
                    f"of {self._n_slots}"
                )
                return slot
        return None

    def renew(self, slot: int) -> bool:
        """Extend a slot lease this worker still owns.

        False means either Redis failed or — the case that matters — another
        worker now owns this slot because this worker's TTL lapsed before it
        renewed. Either way the caller must stop consuming that slot's queue
        immediately (restart to re-claim), so two workers don't track the same
        cameras against divergent state.
        """
        try:
            return bool(self._eval(self._renew_sha_ref(), slot, with_ttl=True))
        except Exception as e:
            logger.warning(f"SlotLeaseManager: renew of slot {slot} failed: {e}")
            self._drop_client()
            return False

    def release(self, slot: int) -> None:
        """Give up a slot immediately on shutdown, rather than waiting out the
        TTL — so a deliberate restart doesn't cost the TTL before another
        replica can pick the slot up."""
        try:
            self._eval(self._release_sha_ref(), slot, with_ttl=False)
        except Exception as e:
            # An unreleased lease still expires within the TTL, so worst case
            # is one TTL of delay before another replica reclaims it.
            logger.debug(f"SlotLeaseManager: release of slot {slot} failed: {e}")

    # ── internals ──────────────────────────────────────────────────────────

    def _renew_sha_ref(self):
        return ("renew", _RENEW_SCRIPT)

    def _release_sha_ref(self):
        return ("release", _RELEASE_SCRIPT)

    def _eval(self, script_ref, slot: int, with_ttl: bool):
        """Run a cached Lua script, reloading once on NOSCRIPT (server restart
        flushed its script cache)."""
        which, source = script_ref
        client = self._client()
        args = [self._worker_id]
        if with_ttl:
            args.append(self._ttl)

        sha = self._renew_sha if which == "renew" else self._release_sha
        if sha is None:
            sha = client.script_load(source)
            self._set_sha(which, sha)
        try:
            return client.evalsha(sha, 1, _key(slot), *args)
        except Exception:
            sha = client.script_load(source)
            self._set_sha(which, sha)
            return client.evalsha(sha, 1, _key(slot), *args)

    def _set_sha(self, which: str, sha: str) -> None:
        if which == "renew":
            self._renew_sha = sha
        else:
            self._release_sha = sha
