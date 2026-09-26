"""LSO-251: a signed storage.googleapis.com URL must map to the plain blob name.

Run: PYTHONPATH=src python -m pytest tests/test_gcs_url.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from infrastructure.storage.gcs import ImageFetcher  # noqa: E402

BASE = "https://storage.googleapis.com/hbai-general-data/2025/cv.face-recognition/humblebee/client_images/"


class GsUrlFromHttpsTests(unittest.TestCase):
    def test_signed_url_drops_query(self):
        url = BASE + "Asadbek_Kiyomov_48_0_4658eceb.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Expires=7200&X-Goog-Signature=abc"
        self.assertEqual(
            ImageFetcher._gs_url_from_https(url),
            "gs://hbai-general-data/2025/cv.face-recognition/humblebee/client_images/Asadbek_Kiyomov_48_0_4658eceb.jpg",
        )

    def test_plain_url_unchanged_apart_from_scheme(self):
        self.assertEqual(
            ImageFetcher._gs_url_from_https(BASE + "a.jpg"),
            "gs://hbai-general-data/2025/cv.face-recognition/humblebee/client_images/a.jpg",
        )

    def test_percent_encoded_name_is_decoded(self):
        self.assertEqual(
            ImageFetcher._gs_url_from_https(BASE + "Ali%20Valiyev_1.jpg#x"),
            "gs://hbai-general-data/2025/cv.face-recognition/humblebee/client_images/Ali Valiyev_1.jpg",
        )

    def test_calibration_blob_path_from_signed_url(self):
        # engine.py's calibration fetch strips "gs://<bucket>/" off the same helper.
        url = BASE + "frame%201.jpg?X-Goog-Signature=abc#f"
        self.assertEqual(
            ImageFetcher._gs_url_from_https(url).removeprefix("gs://hbai-general-data/"),
            "2025/cv.face-recognition/humblebee/client_images/frame 1.jpg",
        )


if __name__ == "__main__":
    unittest.main()
