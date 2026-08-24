"""LSO-133: the AI must only claim its own branch's cameras.

The AI resolves every camera against its LOCAL edge (`SO_EDGE_RTSP_BASE`), so
an unscoped AI in a multi-site org loads other branches' cameras and then tries
to read them from an edge that has never heard of those paths. Scoping is what
makes a second site's AI possible at all.
"""

import unittest
from unittest.mock import MagicMock, patch

from infrastructure.storage.repository import Repository


class _FakeConn:
    """Captures the SQL and params `get_cameras` actually issues."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.sql = None
        self.params = None

    def execute(self, clause, params=None):
        self.sql = str(clause)
        self.params = params
        result = MagicMock()
        result.fetchall.return_value = self.rows
        result.fetchone.return_value = self.rows[0] if self.rows else None
        return result

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _repo(conn):
    with patch.object(Repository, "__init__", lambda self, slug: None):
        r = Repository("humblebee")
    r.client_slug = "humblebee"
    r.schema = "org_humblebee"
    db = MagicMock()
    db.get_connection.return_value = conn
    r._db = db
    return r


class TestBranchScopedQuery(unittest.TestCase):
    def test_branch_code_filters_in_sql(self):
        conn = _FakeConn()
        _repo(conn).get_cameras(branch_code="namangan")
        self.assertIn("b.code = :branch_code", conn.sql)
        self.assertEqual(conn.params, {"branch_code": "namangan"})

    def test_no_branch_code_means_no_filter(self):
        conn = _FakeConn()
        _repo(conn).get_cameras()
        self.assertNotIn("b.code", conn.sql)
        self.assertEqual(conn.params, {})

    def test_left_join_so_unbranched_cameras_survive(self):
        # An INNER JOIN would drop every camera on a deployment that has not
        # adopted branches — i.e. all of them.
        conn = _FakeConn()
        _repo(conn).get_cameras()
        self.assertIn("LEFT JOIN", conn.sql.upper())

    def test_branch_code_exists_checks_branches_table(self):
        conn = _FakeConn(rows=[(1,)])
        self.assertTrue(_repo(conn).branch_code_exists("incheon"))
        self.assertIn("org_humblebee.branches", conn.sql)
        self.assertEqual(conn.params, {"code": "incheon"})

    def test_branch_code_exists_false_when_absent(self):
        self.assertFalse(_repo(_FakeConn(rows=[])).branch_code_exists("nope"))


class TestLoaderGuard(unittest.TestCase):
    """A typo'd branch code loads zero cameras, which looks exactly like every
    camera being down. It must fail instead."""

    def _load(self, branch_code, exists):
        from config import camera_loader

        repo = MagicMock()
        repo.branch_code_exists.return_value = exists
        repo.get_cameras.return_value = []
        with patch("infrastructure.storage.Repository", return_value=repo), \
             patch.object(camera_loader.settings, "edge_branch_code", branch_code):
            camera_loader.load_cameras_from_db("humblebee", ["attendance"])
        return repo

    def test_unmatched_branch_code_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._load("typo", exists=False)
        self.assertIn("matches no branch", str(ctx.exception))

    def test_matched_branch_code_is_passed_through(self):
        repo = self._load("namangan", exists=True)
        repo.get_cameras.assert_called_with(
            application="attendance", branch_code="namangan"
        )

    def test_unset_branch_code_passes_none_and_does_not_check(self):
        repo = self._load("", exists=True)
        repo.get_cameras.assert_called_with(
            application="attendance", branch_code=None
        )
        repo.branch_code_exists.assert_not_called()


if __name__ == "__main__":
    unittest.main()
