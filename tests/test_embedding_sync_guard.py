"""LSO-219: the embedding sync must never delete another branch's face
embeddings, and must fail closed when the backend user list is empty.

face_embeddings is an ORG-WIDE table; on a branch box the backend user list is
branch-scoped, so a "missing" user is usually just another branch's — deleting
it wiped Incheon's whole gallery once. These pin the two guards.

EmbeddingSyncService.__init__ loads a FaceDetector (heavy), so we build the
instance via __new__ and set only what the tested methods touch — same pattern
as test_engine_reload.py. Import needs lum_vision (module-level import in
embedding_sync), so skip when it's unavailable, like test_gpu_rpc.py.

Run: PYTHONPATH=src python -m pytest tests/test_embedding_sync_guard.py
"""

import importlib.util
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

_LUM_VISION_AVAILABLE = importlib.util.find_spec("lum_vision") is not None


def _svc(store):
    from infrastructure.storage.embedding_sync import EmbeddingSyncService

    svc = EmbeddingSyncService.__new__(EmbeddingSyncService)
    svc.store = store
    svc.client_slug = "humblebee"
    return svc


@unittest.skipUnless(_LUM_VISION_AVAILABLE, "lum_vision not importable")
class RemoveStaleGuardTests(unittest.TestCase):
    def test_branch_scoped_never_deletes_out_of_scope_user(self):
        # pgvector has user 99, the branch-scoped backend list doesn't -> must NOT
        # delete (99 belongs to another branch in the org-wide table).
        store = mock.MagicMock()
        store.delete_all_for_user.return_value = 3
        svc = _svc(store)
        with mock.patch("infrastructure.storage.embedding_sync.settings") as s:
            s.edge_branch_code = "incheon"
            deleted, imgs = svc._remove_stale_embeddings({}, {"99": {"url"}})
        store.delete_all_for_user.assert_not_called()
        self.assertEqual((deleted, imgs), (0, 0))

    def test_unscoped_still_deletes_genuinely_stale_user(self):
        # whole-org sync (no branch code): absence really means deleted.
        store = mock.MagicMock()
        store.delete_all_for_user.return_value = 3
        svc = _svc(store)
        with mock.patch("infrastructure.storage.embedding_sync.settings") as s:
            s.edge_branch_code = ""
            deleted, imgs = svc._remove_stale_embeddings({}, {"99": {"url"}})
        store.delete_all_for_user.assert_called_once_with("99")
        self.assertEqual(deleted, 1)


@unittest.skipUnless(_LUM_VISION_AVAILABLE, "lum_vision not importable")
class SyncFailClosedTests(unittest.TestCase):
    def test_empty_user_list_aborts_without_deleting(self):
        # get_all_users() returns [] on a fetch error too — must abort, not wipe.
        store = mock.MagicMock()
        svc = _svc(store)
        with mock.patch("infrastructure.storage.embedding_sync.Repository") as Repo:
            Repo.return_value.get_all_users.return_value = []
            res = svc.sync_missing_embeddings()
        self.assertFalse(res["success"])
        self.assertTrue(res.get("aborted"))
        self.assertIn("empty_user_list", res.get("error", ""))  # caller logs 'error'
        store.delete_all_for_user.assert_not_called()


if __name__ == "__main__":
    unittest.main()
