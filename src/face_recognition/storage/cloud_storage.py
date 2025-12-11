"""Cloud storage management for Google Cloud Storage."""

import os
from typing import List, Optional

import gcsfs
from loguru import logger


class CloudStorageManager:
    """Manager for Google Cloud Storage operations.

    This class provides methods for reading and writing files to Google Cloud Storage.
    """

    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize the cloud storage manager.

        Args:
            credentials_path: Path to Google Cloud credentials JSON file.
                            If None, uses GCS_CREDENTIALS_PATH environment variable.
        """
        if credentials_path is None:
            credentials_path = os.getenv('GCS_CREDENTIALS_PATH')

        self.fs = gcsfs.GCSFileSystem(token=credentials_path)
        logger.info("Initialized Google Cloud Storage filesystem")

    def read_file(self, gcs_path: str) -> bytes:
        """Read a file from Google Cloud Storage.

        Args:
            gcs_path: Path to the file in GCS (without gs:// prefix)

        Returns:
            File contents as bytes

        Raises:
            Exception: If file read fails
        """
        try:
            with self.fs.open(gcs_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file from GCS: {gcs_path}. Error: {e}")
            raise

    def write_file(self, gcs_path: str, data: bytes) -> bool:
        """Write a file to Google Cloud Storage.

        Args:
            gcs_path: Path to the file in GCS (without gs:// prefix)
            data: Data to write as bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.fs.open(gcs_path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error(f"Failed to write file to GCS: {gcs_path}. Error: {e}")
            return False

    def exists(self, gcs_path: str) -> bool:
        """Check if a file exists in Google Cloud Storage.

        Args:
            gcs_path: Path to the file in GCS (without gs:// prefix)

        Returns:
            True if file exists, False otherwise
        """
        try:
            return self.fs.exists(gcs_path)
        except Exception as e:
            logger.error(f"Failed to check file existence in GCS: {gcs_path}. Error: {e}")
            return False

    def list_files(self, gcs_path: str) -> List[str]:
        """List files in a Google Cloud Storage directory.

        Args:
            gcs_path: Path to the directory in GCS (without gs:// prefix)

        Returns:
            List of file paths
        """
        try:
            return self.fs.ls(gcs_path)
        except Exception as e:
            logger.error(f"Failed to list files in GCS: {gcs_path}. Error: {e}")
            return []
