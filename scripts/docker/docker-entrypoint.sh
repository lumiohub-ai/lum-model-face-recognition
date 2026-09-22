#!/bin/bash
set -euo pipefail

echo "INFO: Starting Person Tracking Service..."

# LSO-185: reconcile bind-mounted runtime dirs to the in-image app user, then
# drop root. A bind mount hides whatever ownership the Dockerfile baked into
# the image at that path, so this has to happen here, at container start,
# not at build time. Not every service mounts every one of these (e.g.
# decode-worker has neither models nor person-tracking's storage dir), so
# each is guarded rather than assumed present.
for d in /app/logs /app/volumes/models /app/volumes/storage/person-tracking /app/.cache/huggingface; do
    [ -d "$d" ] && chown -R appuser:appgroup "$d" || true
done

# "appuser" alone, not "appuser:appgroup" — gosu only calls initgroups()
# (picking up every supplementary group from /etc/group, e.g. the "devs"
# group local dev's src bind-mount needs) when no explicit group is given.
# Naming the group here would silently drop every supplementary group and
# leave the process with just its primary one.
#
# If arguments are passed (from docker-compose command), run them
# Otherwise run the default main.py
if [ $# -gt 0 ]; then
    exec gosu appuser "$@"
else
    exec gosu appuser python3 -m src.main
fi
