#!/bin/bash
set -euo pipefail

echo "INFO: Running '${FR_SLUG}' docker-entrypoint.sh..."

# Set permissions
umask 0002
find "${FR_HOME_DIR}" "${FR_DATA_DIR}" "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -path "*/modules" -prune -o -name ".env" -o -print0 2>/dev/null | xargs -0 chown -c "${USER}:${GROUP}" 2>/dev/null || true
find "${FR_DIR}" "${FR_DATA_DIR}" -type d -not -path "*/modules/*" -not -path "*/scripts/*" -exec chmod 770 {} + 2>/dev/null || true
find "${FR_DIR}" "${FR_DATA_DIR}" -type f -not -path "*/modules/*" -not -path "*/scripts/*" -exec chmod 660 {} + 2>/dev/null || true
find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type d -exec chmod 775 {} + 2>/dev/null || true
find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type f -exec chmod 664 {} + 2>/dev/null || true

echo "INFO: Starting SmartOfficeEngine..."
sleep 2

exec python3 -m examples.clients.smart_office_main
