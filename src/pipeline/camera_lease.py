"""Per-camera decode ownership, backed by a short-lived Redis lease.

Rejected first: `camera_id % N` (compute ownership from a replica index and
count). It scales the camera count fine, but changing `N` — adding or
removing a decode worker — reshuffles almost every camera's assignment at
once, not just the delta, since every camera's modulus changes when the
divisor does. For decoding that's not corruption (unlike tracking, decoding
has no memory to lose — see below), but it means every camera reconnects
for no reason related to the actual camera count changing. Not genuinely
scalable.

Instead: each decode worker claims cameras up to its own capacity using the
same primitive already proven in this codebase —
`workers/identity_throttle.py`'s `RedisIdentityThrottle`, which claims
per-identity work via `SET key value NX EX <interval>`, a single atomic
operation regardless of how many processes race for it. This class is that
same pattern plus a renew step, so a worker can keep a camera across many
claim cycles instead of re-claiming it from scratch each time:

    claim(camera_id)   -> SET key worker_id NX EX ttl   (True = won it)
    renew(camera_id)   -> Lua: if GET key == worker_id: EXPIRE key ttl
    release(camera_id) -> Lua: if GET key == worker_id: DEL key

`renew` and `release` both compare-and-act atomically on the server: a
plain GET-then-EXPIRE (or GET-then-DEL) has a race window where another
worker's claim could land between the two calls, letting this worker
extend or delete a lease it no longer owns.

Why self-assignment is safe for decoding but was rejected for tracking:
tracking has memory (frame 100 needs frame 99's conclusion, so moving a
camera mid-flight splits one person into two identities — measured: global
track count climbing 33 -> 45 and never settling). Decoding reads a frame
and hands it on; reassigning it costs a few dropped frames, nothing else.
That is exactly why Redis's native TTL is enough here, with no hysteresis
state machine or control-plane polling needed — a crashed decode worker's
leases simply expire, and whichever worker has spare capacity picks them
up on its next claim cycle.
"""

from __future__ import annotations

import os
import socket
from typing import Optional

from loguru import logger

_KEY_PREFIX = "decode:lease:cam:"

# Compare-and-extend: only renew if this worker still owns the key. A plain
# GET then EXPIRE has a race — another worker's claim could land between
# the two calls — so the check and the extend must happen as one atomic
# operation on the server.
_RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
else
    return 0
end
"""

# Compare-and-delete: only release if this worker still owns the key —
# otherwise a slow release racing a new owner's claim could delete someone
# else's lease.
_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


def _key(camera_id: int) -> str:
    return f"{_KEY_PREFIX}{camera_id}"


def default_worker_id() -> str:
    """A stable-enough identity for one decode worker process: its container
    hostname (Docker sets this to the container id) plus a short random
    suffix, so two processes sharing a hostname (host networking, or two
    local dev runs) still cannot be mistaken for each other. Visible via
    `redis-cli GET decode:lease:cam:<id>` — useful for seeing which
    container owns what during debugging.
    """
    return f"{socket.gethostname()}:{os.urandom(4).hex()}"


class CameraLeaseManager:
    """Atomic claim/renew/release of the right to decode one camera.

    Modelled directly on `RedisIdentityThrottle`
    (`workers/identity_throttle.py`) — same `SET NX EX` claim call, same
    degrade-on-Redis-outage posture (decline rather than risk two workers
    both believing they own a camera), same "drop and re-resolve the
    client on failure" pattern.
    """

    def __init__(
        self,
        ttl_seconds: int,
        worker_id: Optional[str] = None,
        redis_client=None,
    ):
        # EX takes whole seconds and rejects 0, so a sub-second TTL would
        # silently raise at claim time rather than at construction.
        self._ttl = max(1, int(ttl_seconds))
        self._worker_id = worker_id or default_worker_id()
        self._redis = redis_client
        # An injected client is the caller's to manage; only a client this
        # instance resolved itself may be dropped and re-resolved on failure.
        self._owns_client = redis_client is None
        self._degraded_logged = False
        self._renew_sha: Optional[str] = None
        self._release_sha: Optional[str] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def degraded(self) -> bool:
        """True once a Redis call has failed and not yet succeeded again."""
        return self._degraded_logged

    def _client(self):
        if self._redis is not None:
            return self._redis
        from messaging.redis_client import RedisClient

        self._redis = RedisClient.get_instance().client
        return self._redis

    def _on_success(self) -> None:
        if self._degraded_logged:
            logger.info("Camera lease manager recovered")
            self._degraded_logged = False

    def _on_failure(self, e: Exception, action: str, camera_id: int) -> None:
        if not self._degraded_logged:
            # Once, not per call: a Redis outage would otherwise log on
            # every camera on every claim cycle.
            logger.warning(
                f"Camera lease {action} unavailable ({type(e).__name__}: {e}) "
                f"— declining to {action} camera {camera_id} until Redis "
                f"recovers"
            )
            self._degraded_logged = True
        if self._owns_client:
            self._redis = None

    def claim(self, camera_id: int) -> bool:
        """Attempt to become this camera's decode owner.

        Returns True for at most one caller at a time, across every
        process. On a Redis failure this returns **False** — declining to
        claim rather than risking two workers both believing they own the
        camera, since that means two decoders writing the same
        shared-memory slot.
        """
        try:
            claimed = self._client().set(
                _key(camera_id), self._worker_id, nx=True, ex=self._ttl
            )
            self._on_success()
            return bool(claimed)
        except Exception as e:
            self._on_failure(e, "claim", camera_id)
            return False

    def renew(self, camera_id: int) -> bool:
        """Extend a lease this worker still owns.

        False means either a Redis failure, or — the case that matters —
        that another worker now owns this key, because this worker's TTL
        lapsed before it renewed. Either way the caller must stop decoding
        that camera immediately: continuing risks two decoders writing the
        same shared-memory slot for longer than the brief, self-correcting
        race this design tolerates (see the module docstring).
        """
        try:
            result = self._eval_renew(camera_id)
            self._on_success()
            return bool(result)
        except Exception as e:
            self._on_failure(e, "renew", camera_id)
            return False

    def _eval_renew(self, camera_id: int) -> int:
        client = self._client()
        if self._renew_sha is None:
            self._renew_sha = client.script_load(_RENEW_SCRIPT)
        try:
            return client.evalsha(
                self._renew_sha, 1, _key(camera_id), self._worker_id, self._ttl
            )
        except Exception:
            # The server restarted and flushed its script cache (NOSCRIPT)
            # — reload once rather than caching a dead sha forever.
            self._renew_sha = client.script_load(_RENEW_SCRIPT)
            return client.evalsha(
                self._renew_sha, 1, _key(camera_id), self._worker_id, self._ttl
            )

    def release(self, camera_id: int) -> None:
        """Give up a camera this worker owns, immediately rather than
        waiting for the TTL.

        Used both when a camera leaves this worker's eligible set and on
        shutdown, so a deliberate restart doesn't cost the TTL wait before
        another worker can pick the camera up.
        """
        try:
            self._eval_release(camera_id)
        except Exception as e:
            # Not worth degrading over: an unreleased lease still expires
            # within the TTL on its own, so the worst case is one TTL's
            # worth of delay before failover, not a permanently stuck
            # camera.
            logger.debug(
                f"Camera lease release failed for camera {camera_id}: {e}"
            )

    def _eval_release(self, camera_id: int) -> int:
        client = self._client()
        if self._release_sha is None:
            self._release_sha = client.script_load(_RELEASE_SCRIPT)
        try:
            return client.evalsha(
                self._release_sha, 1, _key(camera_id), self._worker_id
            )
        except Exception:
            self._release_sha = client.script_load(_RELEASE_SCRIPT)
            return client.evalsha(
                self._release_sha, 1, _key(camera_id), self._worker_id
            )

    def is_claimed(self, camera_id: int) -> bool:
        """Whether ANY worker currently holds this camera's lease.

        Only for the "is this genuinely unclaimed anywhere, or does some
        other replica already have it" visibility check — never used in
        claim/renew/release logic itself. On a Redis failure this assumes
        claimed, so a transient outage cannot itself trigger a false
        "nobody owns this camera" alert.
        """
        try:
            return bool(self._client().exists(_key(camera_id)))
        except Exception:
            return True
