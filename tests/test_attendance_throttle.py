"""Per-track attendance send-throttle in EntryLogger.log_person_entry (LSO-193).

The authoritative IN/OUT dedup lives in the DB (record_attendance, covered by
test_attendance_dedup.py). This suite locks in the *edge* throttle that keeps
one appearance from emitting an attendance event on every recognition frame,
while still always sending on a genuinely new (track, identity).

EntryLogger is built via __new__ to skip __init__'s DB wiring — these tests
exercise only log_person_entry's throttle, with _send_attendance mocked.
"""

import unittest
from collections import OrderedDict, deque
from datetime import datetime, timezone
from unittest.mock import MagicMock

from infrastructure.attendance_state import EntryLogger, _SENT_ATTENDANCE_CAP


class _Args:
    production = True


def _make_logger() -> EntryLogger:
    lg = EntryLogger.__new__(EntryLogger)
    lg.args = _Args()
    lg.person_last_camera = {}
    lg._sent_attendance = OrderedDict()
    lg.recent_entries = deque(maxlen=3)
    lg._send_attendance = MagicMock()
    lg._send_location_data = MagicMock()
    return lg


def _log(lg, name, status, track_id, camera="cam1"):
    return lg.log_person_entry(
        name=name,
        status=status,
        appear_time=datetime.now(timezone.utc),
        camera_name=camera,
        camera_id=1,
        track_id=track_id,
    )


class PerTrackThrottleTests(unittest.TestCase):
    def test_same_track_and_name_sends_once(self):
        lg = _make_logger()
        self.assertTrue(_log(lg, "alice", "IN", track_id=5))
        self.assertFalse(_log(lg, "alice", "IN", track_id=5))
        self.assertEqual(lg._send_attendance.call_count, 1)

    def test_same_track_different_name_sends_again(self):
        # Identity correction mid-track: the send must NOT be swallowed; the DB
        # dedups if it turns out to be a no-op.
        lg = _make_logger()
        _log(lg, "alice", "IN", track_id=5)
        _log(lg, "bob", "IN", track_id=5)
        self.assertEqual(lg._send_attendance.call_count, 2)

    def test_new_track_same_name_sends_again(self):
        # A fresh appearance (new track) always re-sends; the DB decides IN/OUT.
        lg = _make_logger()
        _log(lg, "alice", "IN", track_id=5)
        _log(lg, "alice", "OUT", track_id=9)
        self.assertEqual(lg._send_attendance.call_count, 2)

    def test_none_track_id_bypasses_throttle_and_always_sends(self):
        lg = _make_logger()
        _log(lg, "alice", "IN", track_id=None)
        _log(lg, "alice", "IN", track_id=None)
        self.assertEqual(lg._send_attendance.call_count, 2)
        self.assertEqual(len(lg._sent_attendance), 0)

    def test_cap_evicts_oldest_track(self):
        lg = _make_logger()
        for t in range(_SENT_ATTENDANCE_CAP + 1):
            _log(lg, "u", "IN", track_id=t)
        # Set stays bounded, and track 0 (the oldest) was evicted...
        self.assertLessEqual(len(lg._sent_attendance), _SENT_ATTENDANCE_CAP)
        self.assertNotIn((0, "u"), lg._sent_attendance)
        # ...so re-logging track 0 is treated as new and sends again.
        before = lg._send_attendance.call_count
        _log(lg, "u", "IN", track_id=0)
        self.assertEqual(lg._send_attendance.call_count, before + 1)


if __name__ == "__main__":
    unittest.main()
