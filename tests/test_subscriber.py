"""Unit tests for messaging/subscriber.py::start_listener.

Run: PYTHONPATH=src python tests/test_subscriber.py
"""

import json
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))


class FakePubSub:
    """Enough of redis-py's PubSub for start_listener: subscribe() + a
    listen() generator fed by test-pushed messages, plus close()."""

    def __init__(self, owner):
        self._owner = owner
        self._closed = False

    def subscribe(self, channel):
        self._owner.subscribed_channels.append(channel)

    def listen(self):
        while not self._closed:
            msg = self._owner._next_message()
            if msg is not None:
                yield msg
            else:
                time.sleep(0.01)

    def close(self):
        self._closed = True


class FakeRedisClient:
    """Stands in for RedisClient/CentralRedisClient's `.client.pubsub()`."""

    def __init__(self):
        self.subscribed_channels = []
        self._queue = []
        self._lock = threading.Lock()
        self.client = self
        self.pubsub_calls = 0

    def pubsub(self):
        self.pubsub_calls += 1
        return FakePubSub(self)

    def push(self, data):
        with self._lock:
            self._queue.append({"type": "message", "data": json.dumps(data)})

    def _next_message(self):
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None


class StartListenerClientFactoryTests(unittest.TestCase):
    """LSO-189: start_listener must use whichever client_factory it's given,
    not always the default (local) RedisClient — the reload channels need to
    listen on a central Redis instead."""

    def test_defaults_to_local_redis_client_when_no_factory_given(self):
        import messaging.redis_client as redis_client_module
        from messaging.subscriber import start_listener

        fake = FakeRedisClient()
        original_get_instance = redis_client_module.RedisClient.get_instance
        redis_client_module.RedisClient.get_instance = staticmethod(lambda: fake)
        thread = None
        try:
            handler_calls = []
            thread = start_listener(
                "some-channel", handler_calls.append, name="t-default",
            )
            fake.push({"hello": "world"})
            self._wait_for(lambda: len(handler_calls) == 1)
        finally:
            redis_client_module.RedisClient.get_instance = original_get_instance
            if thread is not None:
                self._stop(thread, fake)

        self.assertEqual(fake.subscribed_channels, ["some-channel"])
        self.assertEqual(handler_calls, [{"hello": "world"}])

    def test_uses_the_given_client_factory_instead_of_the_default(self):
        from messaging.subscriber import start_listener

        central_fake = FakeRedisClient()
        handler_calls = []
        thread = start_listener(
            "reload-channel",
            handler_calls.append,
            name="t-central",
            client_factory=lambda: central_fake,
        )
        try:
            central_fake.push({"reload": True})
            self._wait_for(lambda: len(handler_calls) == 1)
        finally:
            self._stop(thread, central_fake)

        self.assertEqual(central_fake.subscribed_channels, ["reload-channel"])
        self.assertEqual(handler_calls, [{"reload": True}])

    @staticmethod
    def _wait_for(predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        raise AssertionError("condition not met within timeout")

    @staticmethod
    def _stop(thread, fake):
        # is_running defaults to "forever" in start_listener, so tests signal
        # shutdown by closing the fake pubsub directly rather than waiting on
        # a real is_running flag.
        for value in list(vars(fake).values()):
            if isinstance(value, FakePubSub):
                value.close()
        thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
