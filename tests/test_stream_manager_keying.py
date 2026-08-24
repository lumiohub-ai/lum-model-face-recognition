"""LSO-130: StreamManager keys streams by DB camera id, not list position.

With a list, removing a camera from the middle shifted every later camera's
index — silently handing a CameraWorker a different camera's stream.
"""

import unittest
from unittest import mock


from infrastructure.video import stream_manager as sm


def cfg(camera_id, name=None):
    return {
        "camera_id": camera_id,
        "camera_name": name or f"cam{camera_id}",
        "stream_url": f"rtsp://example/{camera_id}",
        "cam_type": "IN",
    }


class FakeStream:
    def __init__(self, src, logger=None):
        self.src = src
        self.connected = True
        self.cap = None
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class TestStreamKeying(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(sm, "StreamHandler", FakeStream)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_streams_are_keyed_by_camera_id(self):
        m = sm.StreamManager([cfg(7), cfg(22), cfg(5)])
        m.init_streams()
        self.assertEqual(sorted(m.streams), [5, 7, 22])

    def test_removing_a_middle_camera_leaves_the_others_pointing_at_their_own_stream(self):
        m = sm.StreamManager([cfg(7), cfg(22), cfg(5)])
        m.init_streams()
        s7, s5 = m.streams[7], m.streams[5]

        m.remove_stream(22)

        # Identity, not just presence — a positional store would have shifted
        # camera 5's handler into camera 22's old slot.
        self.assertIs(m.streams[7], s7)
        self.assertIs(m.streams[5], s5)
        self.assertNotIn(22, m.streams)

    def test_remove_stops_the_stream(self):
        m = sm.StreamManager([cfg(7)])
        m.init_streams()
        s = m.streams[7]
        m.remove_stream(7)
        self.assertTrue(s.stopped)

    def test_add_stream_registers_by_id(self):
        m = sm.StreamManager([cfg(7)])
        m.init_streams()
        m.add_stream(cfg(31))
        self.assertIn(31, m.streams)
        self.assertIsNot(m.streams[31], m.streams[7])

    def test_add_and_remove_are_idempotent(self):
        m = sm.StreamManager([cfg(7)])
        m.init_streams()
        m.add_stream(cfg(7))
        self.assertEqual(list(m.streams), [7])
        m.remove_stream(99)
        self.assertEqual(list(m.streams), [7])

    def test_get_frame_for_unknown_camera_returns_none(self):
        m = sm.StreamManager([cfg(7)])
        m.init_streams()
        self.assertIsNone(m.get_frame(999))


if __name__ == "__main__":
    unittest.main()
