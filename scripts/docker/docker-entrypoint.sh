#!/bin/bash
set -euo pipefail

echo "INFO: Starting Person Tracking Service..."

# If arguments are passed (from docker-compose command), run them
# Otherwise run the default main.py
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python3 -m src.main
fi
