"""Database repository for reading shared data.

Direct database access replaces HTTP calls to Backend.
All read operations for users, cameras, and attendance status.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy import text
from loguru import logger

from .db_config import DatabaseConfig
from .validators import validate_client_slug


class Repository:
    """Repository for reading data from PostgreSQL.

    Replaces HTTP calls to Backend API with direct database queries.
    """

    def __init__(self, client_slug: str):
        self.client_slug = validate_client_slug(client_slug)
        self.schema = f"org_{self.client_slug}"
        self._db: Optional[DatabaseConfig] = None

    @property
    def db(self) -> DatabaseConfig:
        """Lazy initialization of database connection."""
        if self._db is None:
            self._db = DatabaseConfig()
        return self._db

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all active users with their images.

        Returns:
            List of users with id, full_name, image_urls
        """
        try:
            with self.db.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT id, full_name, image_urls
                    FROM {self.schema}.users
                    ORDER BY full_name
                """))
                users = []
                for row in result.fetchall():
                    image_urls = row[2] or []
                    user = {
                        'id': row[0],
                        'name': row[1],
                        'full_name': row[1],
                        'image_urls': image_urls
                    }
                    users.append(user)
                logger.info(f"Fetched {len(users)} active users from database")
                return users
        except Exception as e:
            logger.error(f"Failed to fetch users: {e}")
            return []

    def get_users_by_status(self, status: str) -> List[str]:
        """Get user names by their current attendance status.

        Args:
            status: 'in' or 'out'

        Returns:
            List of full names
        """
        status = status.lower()
        if status not in ('in', 'out'):
            logger.warning(f"Invalid status: {status}")
            return []

        try:
            with self.db.get_connection() as conn:
                # Get latest attendance status for each user
                result = conn.execute(text(f"""
                    WITH latest AS (
                        SELECT DISTINCT ON (user_id)
                            user_id, status
                        FROM {self.schema}.attendance_records
                        ORDER BY user_id, timestamp DESC
                    )
                    SELECT u.full_name
                    FROM {self.schema}.users u
                    JOIN latest l ON u.id = l.user_id
                    WHERE l.status = :status
                """), {'status': status})

                names = [row[0] for row in result.fetchall()]
                logger.debug(f"Found {len(names)} users with status '{status}'")
                return names
        except Exception as e:
            logger.error(f"Failed to fetch users by status: {e}")
            return []

    def get_user_name_to_id(self) -> List[Dict[str, Any]]:
        """Get mapping of user names to IDs.

        Returns:
            List of dicts with 'name' and 'id' keys
        """
        try:
            with self.db.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT id, full_name
                    FROM {self.schema}.users
                """))
                return [
                    {'id': row[0], 'name': row[1]}
                    for row in result.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to fetch user name mapping: {e}")
            return []

    def get_cameras(self, application: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get camera configurations.

        Args:
            application: Filter by application type (e.g., 'attendance')

        Returns:
            List of camera configs
        """
        try:
            with self.db.get_connection() as conn:
                # Base query
                query = f"""
                    SELECT id, name, stream_url, camera_type, application,
                           matching_threshold, virtual_line_points, roi_points, status
                    FROM {self.schema}.cameras
                """

                result = conn.execute(text(query))
                cameras = []

                for row in result.fetchall():
                    camera = {
                        'id': row[0],
                        'name': row[1],
                        'stream_url': row[2],
                        'camera_type': row[3],
                        'application': row[4],
                        'matching_threshold': row[5],
                        'virtual_line_points': row[6],
                        'roi_points': row[7],
                        'status': row[8],
                    }

                    # Filter by application if specified
                    if application:
                        cam_apps = camera.get('application', [])
                        if isinstance(cam_apps, list) and application in cam_apps:
                            cameras.append(camera)
                        elif isinstance(cam_apps, str) and cam_apps == application:
                            cameras.append(camera)
                    else:
                        cameras.append(camera)

                logger.info(f"Fetched {len(cameras)} cameras from database")
                return cameras
        except Exception as e:
            logger.error(f"Failed to fetch cameras: {e}")
            return []

    def check_new_and_deleted_users(
        self,
        current_users: List[str]
    ) -> tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """Compare current users with database to find new and deleted.

        Args:
            current_users: List of current user names in the system

        Returns:
            Tuple of (new_users, deleted_users, name_to_id)
        """
        db_users = self.get_all_users()
        name_to_id = [{'name': u['name'], 'id': u['id']} for u in db_users]
        db_names = {u['name'] for u in db_users}
        current_set = set(current_users)

        # Find new users (in DB but not in current)
        new_users = []
        for user in db_users:
            if user['name'] not in current_set:
                if user.get('image_url'):
                    new_users.append({
                        'name': user['name'],
                        'image_path': user['image_url']
                    })
                else:
                    logger.warning(f"Skipping user '{user['name']}' - no image_url")

        # Find deleted users (in current but not in DB)
        deleted_users = [name for name in current_users if name not in db_names]

        return new_users, deleted_users, name_to_id
