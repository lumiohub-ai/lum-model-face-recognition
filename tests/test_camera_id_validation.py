"""LSO-130: camera ids must be present and unique before they key anything.

With positional keys two cameras could never collide. With dict keys they can:
a duplicate or missing id silently overwrites another camera's queues and
streams, and that camera just stops getting frames with no error. Since the
whole point of this change is removing silent misrouting, the ids get checked
loudly rather than trusted.
"""

import unittest

from pipeline.engine import SmartOfficeEngine


def cfg(camera_id, name="cam"):
    return {"camera_id": camera_id, "camera_name": name}


class _Ids:
    """Borrow the real implementation without constructing an engine.

    SmartOfficeEngine.__init__ loads models and opens streams, so it can't be
    instantiated in a unit test — but _camera_ids only reads camera_configs.
    """

    def __init__(self, configs):
        self.camera_configs = configs

    _camera_ids = SmartOfficeEngine._camera_ids


class TestCameraIdValidation(unittest.TestCase):
    def test_returns_ids_in_config_order(self):
        self.assertEqual(_Ids([cfg(7), cfg(22), cfg(5)])._camera_ids(), [7, 22, 5])

    def test_duplicate_ids_raise(self):
        with self.assertRaises(ValueError) as ctx:
            _Ids([cfg(7), cfg(7, "other")])._camera_ids()
        self.assertIn("Duplicate", str(ctx.exception))

    def test_missing_id_raises_and_names_the_camera(self):
        with self.assertRaises(ValueError) as ctx:
            _Ids([cfg(7), cfg(None, "Reception")])._camera_ids()
        self.assertIn("Reception", str(ctx.exception))

    def test_two_missing_ids_do_not_silently_collide_on_none(self):
        with self.assertRaises(ValueError):
            _Ids([cfg(None, "a"), cfg(None, "b")])._camera_ids()


if __name__ == "__main__":
    unittest.main()
