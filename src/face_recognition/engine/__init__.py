# This file makes the directory a package.
from .model import FaceRecognitionModel
from .tracking import TrackManager

"__all__" == ["FaceRecognitionModel", "TrackManager"]