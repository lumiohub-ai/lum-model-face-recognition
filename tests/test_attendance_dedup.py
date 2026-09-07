"""LSO-182: cross-process attendance dedup guard in `record_attendance`.

After LSO-67 split tracking across camera-worker processes, each holds its own
in-memory IN/OUT map, so two cameras seeing the same person both queue a record.
`DetectionRepository.record_attendance` is the authoritative guard: under a
`FOR UPDATE` lock on the users row it reads the user's latest status and
suppresses a write that repeats it (returning 0), while genuine transitions
insert normally. These tests exercise that guard directly with a fake
connection returning different prior statuses -- the path the pre-existing
`_FakeConn` never reached.
"""

import unittest
from unittest.mock import MagicMock, patch

from infrastructure.storage.detection_repository import DetectionRepository


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _DedupFakeConn:
    """Routes execute() by SQL shape: the FOR UPDATE lock, the prior-status
    lookup (canned `prior_status`), branch_id lookups, and the INSERT. Records
    commit()/rollback() so tests can assert lock-hold behaviour."""

    def __init__(self, prior_status=None, insert_id=42, insert_returns_row=True):
        self.prior_status = prior_status          # None => no prior row
        self.insert_id = insert_id
        self.insert_returns_row = insert_returns_row
        self.executed = []
        self.committed = False
        self.rolled_back = False

    def execute(self, clause, params=None):
        sql = str(clause)
        self.executed.append((sql, params))
        if "SELECT 1 FROM" in sql:      # the users-row lock (SELECT 1 ... FOR UPDATE)
            return _FakeResult((1,))
        if "attendance_records" in sql and "ORDER BY" in sql:   # the prior-status SELECT
            return _FakeResult((self.prior_status,) if self.prior_status is not None else None)
        if ".cameras" in sql and "SELECT branch_id" in sql:
            return _FakeResult((7,))
        if ".users" in sql and "SELECT branch_id" in sql:
            return _FakeResult((9,))
        if "INSERT INTO" in sql:
            return _FakeResult((self.insert_id,) if self.insert_returns_row else None)
        return _FakeResult((self.insert_id,))

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _repo(conn):
    with patch.object(DetectionRepository, "__init__", lambda self, slug: None):
        r = DetectionRepository("humblebee")
    r.schema = "org_humblebee"
    db = MagicMock()
    db.get_connection.return_value.__enter__.return_value = conn
    db.get_connection.return_value.__exit__.return_value = False
    r._db = db
    return r


def _did_insert(conn):
    return any("INSERT INTO" in sql for sql, _ in conn.executed)


class TestAttendanceDedupGuard(unittest.TestCase):
    def test_duplicate_is_suppressed_and_returns_zero(self):
        # Already IN; another IN is a duplicate observation, not a transition.
        conn = _DedupFakeConn(prior_status="in")
        result = _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=None, status="IN"
        )
        self.assertEqual(result, 0)
        self.assertTrue(conn.rolled_back)
        self.assertFalse(conn.committed)
        self.assertFalse(_did_insert(conn))          # no row written

    def test_genuine_transition_inserts_and_returns_id(self):
        # Currently OUT; an IN is a real transition and must be recorded.
        conn = _DedupFakeConn(prior_status="out", insert_id=99)
        result = _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=None, status="IN"
        )
        self.assertEqual(result, 99)
        self.assertTrue(conn.committed)
        self.assertTrue(_did_insert(conn))

    def test_first_ever_record_inserts(self):
        # No prior row at all; the first sighting must record.
        conn = _DedupFakeConn(prior_status=None, insert_id=1)
        result = _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=None, status="IN"
        )
        self.assertEqual(result, 1)
        self.assertTrue(_did_insert(conn))

    def test_status_casing_is_normalised(self):
        # Prior stored lowercase 'in'; an uppercase 'IN' must still match.
        conn = _DedupFakeConn(prior_status="in")
        result = _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=None, status="IN"
        )
        self.assertEqual(result, 0)

    def test_missing_returning_row_raises_not_zero(self):
        # A failed INSERT (no RETURNING id) must fail loudly, not masquerade
        # as a suppressed duplicate (which also returns 0).
        conn = _DedupFakeConn(prior_status="out", insert_returns_row=False)
        with self.assertRaises(RuntimeError):
            _repo(conn).record_attendance(
                user_id=1, camera_id=5, timestamp=None, status="IN"
            )
        self.assertTrue(conn.rolled_back)

    def test_prior_status_query_orders_by_id_not_timestamp(self):
        # Regression guard (PR #100 review): `timestamp` is caller-supplied and
        # unreliable across processes; the latest status must be picked by the
        # server-assigned id, which the FOR UPDATE lock keeps in commit order.
        conn = _DedupFakeConn(prior_status="out")
        _repo(conn).record_attendance(
            user_id=1, camera_id=5, timestamp=None, status="IN"
        )
        prior_sql = next(
            sql for sql, _ in conn.executed
            if "attendance_records" in sql and "SELECT status" in sql
        )
        self.assertIn("ORDER BY id DESC", prior_sql)
        self.assertNotIn("timestamp DESC", prior_sql)


class TestTaskSkipsPublishOnDuplicate(unittest.TestCase):
    def test_task_returns_skipped_and_does_not_publish(self):
        # When the guard suppresses (record_id == 0), the Celery task must
        # short-circuit before constructing MDAPublisher, so no duplicate event
        # reaches the backend.
        from workers import detection_tasks

        # The task does a local `from infrastructure.storage.detection_repository
        # import DetectionRepository`, so patch it at its source module.
        with patch("infrastructure.storage.detection_repository.DetectionRepository") as MockRepo, \
             patch("messaging.publisher.MDAPublisher") as mock_pub:
            MockRepo.return_value.record_attendance.return_value = 0
            result = detection_tasks.task_record_attendance.apply(kwargs=dict(
                client_slug="humblebee",
                user_id=1,
                user_name="Test User",
                status="IN",
                camera_id=5,
                recorded_at="2026-01-01T00:00:00",
            )).get()

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "duplicate")
        mock_pub.assert_not_called()


if __name__ == "__main__":
    unittest.main()
