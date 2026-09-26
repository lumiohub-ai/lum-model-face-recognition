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


class MediumVariantTests(unittest.TestCase):
    def test_original_maps_to_medium(self):
        self.assertEqual(
            ImageFetcher._medium_variant("a/b/Batkhuu_Byambajav_58_0_99e1.jpg"),
            "a/b/Batkhuu_Byambajav_58_0_99e1_medium.jpg",
        )

    def test_resized_copies_have_no_further_fallback(self):
        self.assertIsNone(ImageFetcher._medium_variant("a/x_medium.jpg"))
        self.assertIsNone(ImageFetcher._medium_variant("a/x_thumb.jpg"))

    def test_no_extension(self):
        self.assertIsNone(ImageFetcher._medium_variant("a/noext"))


class FetchFallbackTests(unittest.TestCase):
    def test_missing_original_falls_back_to_medium(self):
        from unittest import mock
        import numpy as np
        from infrastructure.storage import gcs

        fetched = []

        def blob(name):
            b = mock.Mock()
            def dl():
                fetched.append(name)
                if not name.endswith("_medium.jpg"):
                    raise gcs.GCSNotFound("gone")
                return b"img"
            b.download_as_bytes.side_effect = dl
            return b

        f = ImageFetcher.__new__(ImageFetcher)
        f.gcs_client = mock.Mock()
        f.gcs_client.bucket.return_value.blob.side_effect = blob
        with mock.patch.object(ImageFetcher, "_decode_image_bytes", return_value=np.zeros((2, 2, 3))):
            img = f._fetch_from_gcs("gs://bkt/p/x.jpg")
        self.assertIsNotNone(img)
        self.assertEqual(fetched, ["p/x.jpg", "p/x_medium.jpg"])


if __name__ == "__main__":
    unittest.main()
