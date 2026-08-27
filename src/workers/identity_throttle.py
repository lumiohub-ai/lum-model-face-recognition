"""Cross-process per-identity throttle, backed by Redis (LSO-67).

`ActionRecognitionWorker` throttles action recognition per *identity*, not
per camera: a person visible on three cameras should be classified once per
interval, not three times. In-process that was a dict guarded by a
`threading.Lock` — correct while every camera was a thread in one process.

Once cameras run as Celery tasks in separate processes that lock protects
nothing: each worker holds its own dict, each independently believes it has
claimed the right to classify, and the same person gets N concurrent VLM
inferences, N Ollama round-trips (~28s on a cold model), N GCS proof
uploads and N activity records. The visible symptom would be duplicate
activity rows, which reads as a data bug rather than a concurrency one.

`SET key value NX EX <interval>` is a single atomic operation on the Redis
server, so exactly one caller wins regardless of how many processes or hosts
race — the same property the `threading.Lock` gave within one process.

Expiry does the cleanup the old code did by hand (`MAX_TRACKED` pruning):
a key simply vanishes when its interval elapses, so there is no unbounded
dict and no eviction pass.
"""

from __future__ import annotations

from loguru import logger

# Namespaced so these keys are obvious in redis-cli and cannot collide with
# the command/event channels or the Celery broker's own keyspace.
_KEY_PREFIX = "throttle:action:"


def _key(identity: str) -> str:
    return f"{_KEY_PREFIX}{identity}"


class RedisIdentityThrottle:
    """Atomic claim/release of the right to act on an identity.

    Deliberately mirrors ActionRecognitionWorker's existing
    `reserve_check`/`cancel_check` surface so it can be swapped in without
    touching the call sites in camera_engine.py.
    """

    def __init__(self, interval_seconds: int, redis_client=None):
        # EX takes whole seconds and rejects 0, so a sub-second interval
        # would silently raise at claim time rather than at construction.
        self._interval = max(1, int(interval_seconds))
        self._redis = redis_client
        # An injected client is the caller's to manage; only a client we
        # resolved ourselves may be dropped and re-resolved on failure.
        self._owns_client = redis_client is None
        self._degraded_logged = False

    @property
    def degraded(self) -> bool:
        """True once a Redis call has failed and not yet succeeded again.

        Surfaced in ActionRecognitionWorker.get_stats() because while this is
        True every action check is being declined -- action recognition is
        effectively off, and nothing else in the pipeline would show it.
        """
        return self._degraded_logged

    def _client(self):
        if self._redis is not None:
            return self._redis
        from messaging.redis_client import RedisClient

        self._redis = RedisClient.get_instance().client
        return self._redis

    def reserve(self, identity: str) -> bool:
        """Claim the right to classify `identity` now.

        Returns True for at most one caller per interval, across every
        process. On a Redis failure this returns **False** — declining to act
        rather than letting every worker act at once, since a duplicate
        inference costs a VLM round-trip and a spurious activity record,
        while a skipped one costs one interval of latency for that person.
        """
        if not identity:
            return False
        try:
            claimed = self._client().set(_key(identity), "1", nx=True, ex=self._interval)
            if self._degraded_logged:
                # Clear on recovery, so `degraded` reports the current state
                # rather than latching forever after one blip -- and so a
                # later outage is logged again instead of passing silently.
                logger.info("Action throttle recovered")
                self._degraded_logged = False
            return bool(claimed)
        except Exception as e:
            if not self._degraded_logged:
                # Once, not per call: a Redis outage would otherwise log on
                # every identity on every frame across every camera.
                logger.warning(
                    f"Action throttle unavailable ({type(e).__name__}: {e}) — "
                    f"declining action checks until Redis recovers"
                )
                self._degraded_logged = True
            # Drop the cached client: if the failure was the connection (or a
            # RedisClient singleton built before the server came up), reusing
            # it would fail forever. Re-resolved on the next call.
            if self._owns_client:
                self._redis = None
            return False

    def cancel(self, identity: str) -> None:
        """Release a claim whose work never got queued, so the person is not
        skipped for a whole interval."""
        if not identity:
            return
        try:
            self._client().delete(_key(identity))
        except Exception as e:
            # Not worth degrading the caller over: the key expires on its own
            # within the interval, so the worst case is one skipped check.
            logger.debug(f"Action throttle cancel failed for {identity!r}: {e}")
