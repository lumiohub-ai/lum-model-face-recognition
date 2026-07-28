"""FaceDetector reports coordinates in input-image space, not padded space.

The detector pads the frame before inference so faces near an edge still get
detected. That padding is an implementation detail: everything it hands back has
to be shifted out of padded space first, or callers draw boxes offset by
(pad_w, pad_h) — down and to the right of the actual face.
"""

import numpy as np
import pytest

from insightface.app.common import Face

from lum_vision import FaceDetector


IMAGE_H, IMAGE_W = 500, 1000
PAD_PCT = 20.0
PAD_W, PAD_H = 200, 100  # 20% of 1000 and 500


class StubFaceModel:
    """Stands in for InsightFace's FaceAnalysis, returning padded-space faces."""

    def __init__(self, *faces):
        self.faces = list(faces)
        self.seen_shape = None

    def get(self, image):
        self.seen_shape = image.shape
        return self.faces


def make_detector(monkeypatch, *faces, padding_percent=PAD_PCT):
    """A FaceDetector wired to a stub model — no weights, no ONNX runtime."""
    monkeypatch.setattr(FaceDetector, "_initialize_model", lambda self: None)
    detector = FaceDetector(padding_percent=padding_percent)
    detector.model = StubFaceModel(*faces)
    return detector


@pytest.fixture
def image():
    return np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)


def test_detect_shifts_bbox_out_of_padded_space(monkeypatch, image):
    # A face at (10, 20)-(60, 90) in the real frame is found here once padded.
    face = Face(
        bbox=np.array([10 + PAD_W, 20 + PAD_H, 60 + PAD_W, 90 + PAD_H], dtype=np.float32),
        kps=None,
        det_score=0.9,
    )
    detector = make_detector(monkeypatch, face)

    (detected,) = detector.detect(image)

    assert detected.bbox.tolist() == [10.0, 20.0, 60.0, 90.0]


def test_detect_pads_the_image_it_runs_inference_on(monkeypatch, image):
    detector = make_detector(monkeypatch)

    detector.detect(image)

    assert detector.model.seen_shape == (IMAGE_H + 2 * PAD_H, IMAGE_W + 2 * PAD_W, 3)


def test_detect_shifts_every_landmark_set(monkeypatch, image):
    face = Face(
        bbox=np.zeros(4, dtype=np.float32),
        kps=np.full((5, 2), [PAD_W, PAD_H], dtype=np.float32),
        landmark_2d_106=np.full((106, 2), [PAD_W + 5, PAD_H + 7], dtype=np.float32),
        # Third column is depth, not an image coordinate — it must survive intact.
        landmark_3d_68=np.full((68, 3), [PAD_W, PAD_H, 42.0], dtype=np.float32),
        det_score=0.9,
    )
    detector = make_detector(monkeypatch, face)

    (detected,) = detector.detect(image)

    assert np.allclose(detected.kps, 0.0)
    assert np.allclose(detected.landmark_2d_106, [5.0, 7.0])
    assert np.allclose(detected.landmark_3d_68, [0.0, 0.0, 42.0])


def test_detect_leaves_translation_invariant_fields_alone(monkeypatch, image):
    embedding = np.arange(512, dtype=np.float32)
    pose = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    face = Face(
        bbox=np.zeros(4, dtype=np.float32),
        kps=None,
        embedding=embedding,
        pose=pose,
        det_score=0.9,
    )
    detector = make_detector(monkeypatch, face)

    (detected,) = detector.detect(image)

    assert np.array_equal(detected.embedding, embedding)
    assert np.array_equal(detected.pose, pose)


def test_detect_is_a_no_op_when_padding_is_disabled(monkeypatch, image):
    bbox = np.array([10.0, 20.0, 60.0, 90.0], dtype=np.float32)
    face = Face(bbox=bbox.copy(), kps=None, det_score=0.9)
    detector = make_detector(monkeypatch, face, padding_percent=0.0)

    (detected,) = detector.detect(image)

    assert detected.bbox.tolist() == bbox.tolist()
    assert detector.model.seen_shape == (IMAGE_H, IMAGE_W, 3)


def test_detect_keeps_boxes_that_fall_outside_the_frame(monkeypatch, image):
    """A face found against the replicated border legitimately lands off-frame.

    Clamping here would distort the box, so callers clamp when they crop.
    """
    face = Face(bbox=np.zeros(4, dtype=np.float32), kps=None, det_score=0.9)
    detector = make_detector(monkeypatch, face)

    (detected,) = detector.detect(image)

    assert detected.bbox.tolist() == [-PAD_W, -PAD_H, -PAD_W, -PAD_H]


def test_extract_face_features_reports_input_space_coordinates(monkeypatch, image):
    face = Face(
        bbox=np.array([10 + PAD_W, 20 + PAD_H, 60 + PAD_W, 90 + PAD_H], dtype=np.float32),
        kps=np.full((5, 2), [PAD_W, PAD_H], dtype=np.float32),
        embedding=np.full(512, 3.0, dtype=np.float32),
        det_score=0.75,
    )
    detector = make_detector(monkeypatch, face)

    (features,) = detector.extract_face_features(image)

    assert features["bbox"][:4] == [10, 20, 60, 90]
    assert features["bbox"][4] == pytest.approx(0.75)
    assert np.array_equal(features["landmarks"], np.zeros((5, 2), dtype=int))
    assert np.linalg.norm(features["embedding"]) == pytest.approx(1.0)
