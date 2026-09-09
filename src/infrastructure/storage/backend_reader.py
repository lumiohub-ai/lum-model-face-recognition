"""Database repository for reading shared data.

Direct database access replaces HTTP calls to Backend.
All read operations for users, cameras, and attendance status.
"""

import json
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from loguru import logger

from config.settings import settings
from .db_config import DatabaseConfig
from .validators import validate_client_slug, schema_name_for


class Repository:
    """Repository for reading data from PostgreSQL.

    Replaces HTTP calls to Backend API with direct database queries.
    """

    def __init__(self, client_slug: str):
        self.client_slug = validate_client_slug(client_slug)
        self.schema = schema_name_for(self.client_slug)
        # Use singleton database config (shared connection pool)
        self._db = DatabaseConfig.get_instance()

    @property
    def db(self) -> DatabaseConfig:
        """Get the shared database connection."""
        return self._db

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all active users with their images.

        Returns:
            List of users with id, full_name, image_urls
        """
        try:
            with self.db.get_connection() as conn:
                # Branch scoping (mirrors get_all_embeddings / get_user_name_to_id):
                # when SO_EDGE_BRANCH_CODE is set, only this branch's users (plus
                # org-wide null-branch users). This is the list embedding_sync walks
                # to enrol faces and that check_new_and_deleted_users diffs — a branch
                # box must not fetch/enrol the whole org, only its own people.
                branch_code = settings.edge_branch_code
                if branch_code:
                    result = conn.execute(text(f"""
                        SELECT u.id, u.full_name, u.image_urls
                        FROM {self.schema}.users AS u
                        LEFT JOIN {self.schema}.branches AS b ON b.id = u.branch_id
                        WHERE LOWER(b.code) = :branch_code OR u.branch_id IS NULL
                        ORDER BY u.full_name
                    """), {"branch_code": branch_code})
                else:
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
                return users
        except Exception as e:
            logger.exception(f"Failed to fetch users: {e}")
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
                # Get latest attendance status for each user.
                # Branch scoping (mirrors get_all_users / get_user_name_to_id): a
                # branch box only primes its own users' last status, so the status
                # map stays in step with the branch-scoped register instead of
                # warning about every other branch's users on each reload.
                branch_code = settings.edge_branch_code
                if branch_code:
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
                        LEFT JOIN {self.schema}.branches b ON b.id = u.branch_id
                        WHERE l.status = :status
                          AND (LOWER(b.code) = :branch_code OR u.branch_id IS NULL)
                    """), {'status': status, 'branch_code': branch_code})
                else:
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
            logger.exception(f"Failed to fetch users by status: {e}")
            return []

    def get_user_name_to_id(self) -> List[Dict[str, Any]]:
        """Get mapping of user names to IDs.

        Returns:
            List of dicts with 'name' and 'id' keys
        """
        try:
            with self.db.get_connection() as conn:
                # Branch scoping: match get_all_embeddings — when SO_EDGE_BRANCH_CODE
                # is set, only this branch's users, so the name→id map stays in step
                # with the branch-scoped register (a name absent from the register
                # must not resolve to a user_id here either).
                branch_code = settings.edge_branch_code
                if branch_code:
                    result = conn.execute(text(f"""
                        SELECT u.id, u.full_name
                        FROM {self.schema}.users AS u
                        LEFT JOIN {self.schema}.branches AS b ON b.id = u.branch_id
                        WHERE LOWER(b.code) = :branch_code OR u.branch_id IS NULL
                    """), {"branch_code": branch_code})
                else:
                    result = conn.execute(text(f"""
                        SELECT id, full_name
                        FROM {self.schema}.users
                    """))
                return [
                    {'id': row[0], 'name': row[1]}
                    for row in result.fetchall()
                ]
        except Exception as e:
            logger.exception(f"Failed to fetch user name mapping: {e}")
            return []

    def branch_code_exists(self, branch_code: str) -> bool:
        """True if `branch_code` names a real branch in this tenant schema.

        Used to fail loudly at startup rather than silently loading zero
        cameras: a typo'd SO_EDGE_BRANCH_CODE matches nothing, and an AI that
        starts with no cameras looks identical to one whose cameras are down.

        Compared case-insensitively. `^[a-z0-9]+$` on branch codes is enforced
        by the backend's Pydantic schema only — there is no DB CHECK — and the
        dev/prod branches were seeded by direct SQL, which bypasses it. A
        mixed-case row would otherwise make a correct env var refuse to start.
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    text(
                        f"SELECT 1 FROM {self.schema}.branches "
                        f"WHERE LOWER(code) = :code LIMIT 1"
                    ),
                    {"code": branch_code.lower()},
                ).fetchone()
                return row is not None
        except Exception as e:
            # Unlike every other method here, this re-raises instead of
            # returning a safe default. Swallowing it would hand back False,
            # which the caller reads as "no such branch" and turns into a
            # refuse-to-start — blaming a typo for what is really a DB outage.
            logger.exception(f"Failed to check branch code {branch_code!r}: {e}")
            raise

    def get_cameras(
        self,
        application: Optional[str] = None,
        branch_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get camera configurations.

        Args:
            application: Filter by application type (e.g., 'attendance').
                Applied in Python after the fetch, unlike `branch_code` which
                filters in SQL — the `application` column is JSONB and may
                arrive as a raw string depending on the driver.
            branch_code: Restrict to one branch's cameras (LSO-133). None/empty
                returns every camera, which is only correct for a single-site
                org — see `Settings.edge_branch_code`.

        Returns:
            List of camera configs
        """
        try:
            with self.db.get_connection() as conn:
                # LEFT JOIN, matching the edge sidecar: `cameras.branch_id` is
                # nullable and nothing backfills it, so an INNER JOIN would drop
                # every un-branched camera. When `branch_code` is set, cameras
                # with no branch are excluded on purpose — they belong to no
                # site, so a site-scoped AI must not claim them.
                query = f"""
                    SELECT c.id, c.name, c.stream_url, c.camera_type, c.application,
                           c.matching_threshold, c.virtual_line_points, c.roi_points,
                           c.status
                    FROM {self.schema}.cameras c
                    LEFT JOIN {self.schema}.branches b ON b.id = c.branch_id
                """
                params: Dict[str, Any] = {}
                if branch_code:
                    # LOWER() for the same reason as branch_code_exists: the
                    # lowercase invariant is API-level, not enforced by the DB.
                    query += " WHERE LOWER(b.code) = :branch_code"
                    params["branch_code"] = branch_code.lower()

                result = conn.execute(text(query), params)
                cameras = []

                for row in result.fetchall():
                    # JSONB may arrive as a raw JSON string depending on the driver version
                    cam_apps = row[4]
                    if isinstance(cam_apps, str):
                        try:
                            cam_apps = json.loads(cam_apps)
                        except (json.JSONDecodeError, ValueError):
                            pass

                    camera = {
                        'id': row[0],
                        'name': row[1],
                        'stream_url': row[2],
                        'camera_type': row[3],
                        'application': cam_apps,
                        'matching_threshold': row[5],
                        'virtual_line_points': row[6],
                        'roi_points': row[7],
                        'status': row[8],
                    }

                    # Filter by application if specified
                    if application:
                        if isinstance(cam_apps, list) and application in cam_apps:
                            cameras.append(camera)
                        elif isinstance(cam_apps, str) and cam_apps == application:
                            cameras.append(camera)
                    else:
                        cameras.append(camera)

                scope = f"branch={branch_code}" if branch_code else "branch=<all>"
                logger.info(
                    f"Fetched {len(cameras)} cameras from database ({scope})"
                )
                return cameras
        except Exception as e:
            logger.exception(f"Failed to fetch cameras: {e}")
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
                if user.get('image_urls'):
                    new_users.append({
                        'name': user['name'],
                        'image_urls': user['image_urls']
                    })
                else:
                    logger.warning(f"Skipping user '{user['name']}' - no images")

        # Find deleted users (in current but not in DB)
        deleted_users = [name for name in current_users if name not in db_names]

        return new_users, deleted_users, name_to_id
