"""LSO-110: every DetectionRepository write must carry branch_id.

This repository's INSERT/UPSERT calls bypass FastAPI's `attendance/
service.py::async_create` (and its siblings) entirely — the AI service
writes straight to Postgres via raw SQL, so the branch_id derivation
logic added there never touched these code paths. That gap left every
production attendance/activity/location/unrecognized-face record with a
NULL branch_id, even months after cameras and users were fully branch-
assigned, because this repository never looked their branch up at all.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from infrastructure.storage.detection_repository import DetectionRepository


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """Routes each execute() to a canned result based on which table the
    SQL text mentions — cameras/users lookups for `_derive_branch_id`,
    then whatever the actual INSERT/UPSERT returns."""

    def __init__(self, camera_branch_id=None, user_branch_id=None, insert_id=1):
        self.camera_branch_id = camera_branch_id
        self.user_branch_id = user_branch_id
        self.insert_id = insert_id
        self.executed = []

    def execute(self, clause, params=None):
        sql = str(clause)
        self.executed.append((sql, params))
        if ".cameras" in sql and "SELECT branch_id" in sql:
            return _FakeResult((self.camera_branch_id,) if self.camera_branch_id is not None else None)
        if ".users" in sql and "SELECT branch_id" in sql:
            return _FakeResult((self.user_branch_id,) if self.user_branch_id is not None else None)
        return _FakeResult((self.insert_id,))

    def commit(self):
        pass


def _repo(conn):
    with patch.object(DetectionRepository, "__init__", lambda self, slug: None):
        r = DetectionRepository("humblebee")
    r.schema = "org_humblebee"
    db = MagicMock()
    db.get_connection.return_value.__enter__.return_value = conn
    db.get_connection.return_value.__exit__.return_value = False
    r._db = db
    return r


class TestRecordAttendanceBranchId(unittest.TestCase):
    def test_uses_camera_branch_when_present(self):
        conn = _FakeConn(camera_branch_id=7, user_branch_id=9)
        _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=datetime(2026, 1, 1), status="in"
        )
        _insert_sql, insert_params = conn.executed[-1]
        self.assertIn("attendance_records", _insert_sql)
        self.assertEqual(insert_params["branch_id"], 7)

    def test_falls_back_to_user_branch_when_camera_has_none(self):
        conn = _FakeConn(camera_branch_id=None, user_branch_id=9)
        _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=datetime(2026, 1, 1), status="in"
        )
        _insert_sql, insert_params = conn.executed[-1]
        self.assertEqual(insert_params["branch_id"], 9)

    def test_null_when_neither_camera_nor_user_has_a_branch(self):
        conn = _FakeConn(camera_branch_id=None, user_branch_id=None)
        _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=datetime(2026, 1, 1), status="in"
        )
        _insert_sql, insert_params = conn.executed[-1]
        self.assertIsNone(insert_params["branch_id"])

    def test_no_camera_id_skips_straight_to_user_lookup(self):
        conn = _FakeConn(user_branch_id=9)
        _repo(conn).record_attendance(
            user_id=1, camera_id=None, timestamp=datetime(2026, 1, 1), status="in"
        )
        _insert_sql, insert_params = conn.executed[-1]
        self.assertEqual(insert_params["branch_id"], 9)
        self.assertFalse(
            any(".cameras" in sql for sql, _ in conn.executed),
            "no camera_id means no camera lookup should run at all",
        )


class TestSaveUnrecognizedFaceBranchId(unittest.TestCase):
    def test_derives_from_camera_only(self):
        conn = _FakeConn(camera_branch_id=3)
        _repo(conn).save_unrecognized_face(
            camera_id=5, camera_name="Lobby Cam", timestamp=datetime(2026, 1, 1), status=None
        )
        _insert_sql, insert_params = conn.executed[-1]
        self.assertIn("unrecognized_faces", _insert_sql)
        self.assertEqual(insert_params["branch_id"], 3)


class TestRecordActivityBranchId(unittest.TestCase):
    def test_both_activity_and_current_activity_get_branch_id(self):
        conn = _FakeConn(camera_branch_id=4)
        _repo(conn).record_activity(
            user_id=1,
            camera_id=5,
            activity_type="working",
            timestamp=datetime(2026, 1, 1),
        )
        _activity_sql, activity_params = conn.executed[-2]
        _current_sql, current_params = conn.executed[-1]
        self.assertIn("activity_records", _activity_sql)
        self.assertEqual(activity_params["branch_id"], 4)
        self.assertIn("user_current_activities", _current_sql)
        self.assertEqual(current_params["branch_id"], 4)


class TestUpdateUserLocationBranchId(unittest.TestCase):
    def test_derives_branch_id(self):
        conn = _FakeConn(camera_branch_id=2)
        _repo(conn).update_user_location(
            user_id=1, camera_id=5, timestamp=datetime(2026, 1, 1), status="in"
        )
        _insert_sql, insert_params = conn.executed[-1]
        self.assertIn("user_locations", _insert_sql)
        self.assertEqual(insert_params["branch_id"], 2)


if __name__ == "__main__":
    unittest.main()
