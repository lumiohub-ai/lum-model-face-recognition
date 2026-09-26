"""LSO-224: roles that run no model must not load lum_vision at import time.

`import lum_vision` pulls torch, ultralytics, insightface and onnxruntime
(~0.6 GB RSS per process). decode-worker, celery-worker (detections) and the
startup path of every role import `config` and `infrastructure.storage`, so
those must stay free of it; the model is loaded only where it is used.

Each check runs in a fresh interpreter, since this test process may already
have lum_vision loaded by another test.

Run: PYTHONPATH=src python -m pytest tests/test_lazy_lum_vision.py
"""

import os
import subprocess
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
HEAVY = ("lum_vision", "torch", "ultralytics", "insightface", "onnxruntime")


def _loaded_after_import(*modules: str) -> list:
    code = (
        "import importlib, sys\n"
        f"for m in {list(modules)!r}: importlib.import_module(m)\n"
        f"print(','.join(h for h in {HEAVY!r} if h in sys.modules))\n"
    )
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()
    return [h for h in (out[-1] if out else "").split(",") if h]


class LazyLumVisionTests(unittest.TestCase):
    def test_config_package_does_not_load_models(self):
        self.assertEqual(_loaded_after_import("config", "config.vision"), [])

    def test_storage_package_does_not_load_models(self):
        self.assertEqual(_loaded_after_import("infrastructure.storage"), [])

    def test_decode_worker_entrypoint_does_not_load_models(self):
        self.assertEqual(_loaded_after_import("decode_main"), [])

    def test_celery_app_does_not_load_models(self):
        self.assertEqual(
            _loaded_after_import("workers", "workers.detection_tasks", "workers.embedding_tasks"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
