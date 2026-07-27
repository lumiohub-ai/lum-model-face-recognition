"""The package must work outside its source repo.

These are the properties that broke when this code lived in the application:
config found via the CWD, weights written next to the process, and threads
started behind the caller's back.
"""

import os
import threading
from pathlib import Path

import pytest

from lum_vision import GlobalTrackManager, ModelFactory, VisionConfig


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """Run from a directory that is not the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_config_loads_from_packaged_resource_not_the_cwd(elsewhere):
    manager = GlobalTrackManager()

    # 0.70 comes from the packaged global_tracking.yaml; {} would give 0.70 too
    # via the code default, so assert on a key only the file provides.
    assert manager.config != {}
    assert manager.similarity_threshold == pytest.approx(0.70)


def test_enabled_defaults_instead_of_raising_when_config_lacks_the_key(elsewhere):
    """Regression: an if/elif with no else left `enabled` unassigned."""
    manager = GlobalTrackManager(config_path="/nonexistent/global_tracking.yaml")

    assert isinstance(manager.enabled, bool)


def test_explicit_enabled_argument_overrides_the_yaml(elsewhere):
    assert GlobalTrackManager(enabled=False).enabled is False
    assert GlobalTrackManager(enabled=True).enabled is True


def test_reid_weights_resolve_under_the_given_directory(elsewhere):
    weights = elsewhere / "models" / "weights"
    manager = GlobalTrackManager(enabled=False, weights_dir=weights)

    assert Path(manager.reid_weights_path).parent == weights
    assert "volumes/models" not in manager.reid_weights_path


def test_reid_device_does_not_assume_cuda(elsewhere):
    """Must construct on a CPU-only machine."""
    import torch

    expected = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert GlobalTrackManager(enabled=False).reid_device == expected


def test_config_derives_every_weight_path_from_one_root(tmp_path):
    config = VisionConfig(model_cache_dir=tmp_path)

    assert config.weights_dir == tmp_path / "weights"
    assert config.insightface_dir == tmp_path / "insightface"


def test_model_cache_dir_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LUM_VISION_MODEL_DIR", str(tmp_path / "custom"))
    assert VisionConfig().model_cache_dir == tmp_path / "custom"


def test_factory_starts_no_threads(elsewhere):
    """The package owns no concurrency; the caller supplies it."""
    before = set(threading.enumerate())

    factory = ModelFactory(VisionConfig(model_cache_dir=elsewhere))
    _ = factory.action_recognizer
    _ = factory.global_id_generator
    _ = factory.global_track_manager

    assert set(threading.enumerate()) == before


def test_face_matcher_without_a_provider_fails_loudly(elsewhere):
    factory = ModelFactory(VisionConfig(model_cache_dir=elsewhere))

    with pytest.raises(ValueError, match="embedding_provider"):
        _ = factory.face_matcher


def test_boxmot_resolves_from_site_packages_not_a_vendored_path():
    """The old sys.path hack pointed at modules/yolo_tracking in the repo."""
    import lum_vision.person_tracking.tracker  # noqa: F401  (triggers the import)
    import boxmot

    assert "modules/yolo_tracking" not in os.path.abspath(boxmot.__file__)
