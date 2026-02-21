#!/bin/bash
set -euo pipefail

echo "INFO: Running '${FR_SLUG}' docker-entrypoint.sh..."

# Fix permissions on mounted volumes (runs as root)
echo "INFO: Fixing permissions on mounted volumes..."
chown -R "${USER}:${GROUP}" "${FR_HOME_DIR}" 2>/dev/null || true
chown -R "${USER}:${GROUP}" "${FR_DATA_DIR}" 2>/dev/null || true
chown -R "${USER}:${GROUP}" "${FR_LOGS_DIR}" 2>/dev/null || true
chown -R "${USER}:${GROUP}" "/app/volumes/storage" 2>/dev/null || true
chown -R "${USER}:${GROUP}" "/home/${USER}/.insightface" 2>/dev/null || true
chown -R "${USER}:${GROUP}" "/home/${USER}/.cache" 2>/dev/null || true
chmod -R 775 "${FR_HOME_DIR}" "${FR_DATA_DIR}" "${FR_LOGS_DIR}" "/app/volumes/storage" "/home/${USER}/.insightface" "/home/${USER}/.cache" 2>/dev/null || true

echo "INFO: Starting SmartOfficeEngine as ${USER}..."

# Switch to non-root user and run the application
exec gosu "${USER}" python3 -m src.main
