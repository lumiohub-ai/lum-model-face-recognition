"""Cloud storage management for Google Cloud Storage."""

import os
import uuid
from datetime import timedelta
from typing import Dict, List, Optional

import cv2
import gcsfs
import numpy as np
from google.cloud import storage
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

    def upload_frame(
        self, 
        frame: np.ndarray, 
        org_slug: str, 
        camera_id: int,
        frame_index: int,
        quality: int = 85,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Upload a camera frame to GCS for calibration.

        Args:
            frame: Frame as numpy array (BGR format from OpenCV)
            org_slug: Organization slug
            camera_id: Camera ID
            frame_index: Frame index (1-30)
            quality: JPEG quality (1-100)
            metadata: Additional metadata to attach to the file

        Returns:
            Dictionary with 'gcs_path' and 'signed_url'

        Raises:
            Exception: If upload fails
        """
        try:
            # Generate unique filename
            frame_uuid = str(uuid.uuid4())
            filename = f"{frame_index}_{frame_uuid}.jpg"
            
            # Construct GCS path
            from datetime import datetime
            year = datetime.now().year
            bucket_name = os.getenv('GCS_BUCKET', 'hbai-general-data')
            calibration_path = os.getenv('CALIBRATION_FRAMES_PATH', 'cv.calibration')
            
            gcs_path = f"{bucket_name}/{year}/{calibration_path}/{org_slug}/cameras/{camera_id}/frames/{filename}"
            
            # Encode frame as JPEG
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                raise Exception("Failed to encode frame as JPEG")
            
            frame_bytes = buffer.tobytes()
            
            # Upload using gcsfs
            with self.fs.open(gcs_path, 'wb') as f:
                f.write(frame_bytes)
            
            # Set metadata using google-cloud-storage client
            try:
                credentials_path = os.getenv('GCS_CREDENTIALS_PATH')
                storage_client = storage.Client.from_service_account_json(credentials_path)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(f"{year}/{calibration_path}/{org_slug}/cameras/{camera_id}/frames/{filename}")
                
                # Set custom metadata
                if metadata:
                    blob.metadata = metadata
                    blob.patch()
            except Exception as e:
                logger.warning(f"Failed to set metadata for {gcs_path}: {e}")
            
            # Generate signed URL
            signed_url = self.generate_signed_url(gcs_path)
            
            logger.info(f"Uploaded frame to GCS: {gcs_path}")
            
            return {
                'gcs_path': f"gs://{gcs_path}",
                'signed_url': signed_url,
                'size_bytes': len(frame_bytes)
            }
            
        except Exception as e:
            logger.error(f"Failed to upload frame to GCS: {e}")
            raise

    def generate_signed_url(
        self, 
        gcs_path: str, 
        expiration_seconds: int = 3600
    ) -> str:
        """Generate a signed URL for a GCS file.

        Args:
            gcs_path: Path to the file in GCS (without gs:// prefix)
            expiration_seconds: URL expiration time in seconds (default: 1 hour)

        Returns:
            Signed URL for accessing the file

        Raises:
            Exception: If signed URL generation fails
        """
        try:
            # Parse bucket and blob path
            parts = gcs_path.split('/', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid GCS path format: {gcs_path}")
            
            bucket_name, blob_path = parts
            
            # Initialize storage client
            credentials_path = os.getenv('GCS_CREDENTIALS_PATH')
            storage_client = storage.Client.from_service_account_json(credentials_path)
            
            # Get bucket and blob
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Generate signed URL
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration_seconds),
                method="GET"
            )
            
            return signed_url
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {gcs_path}: {e}")
            raise
