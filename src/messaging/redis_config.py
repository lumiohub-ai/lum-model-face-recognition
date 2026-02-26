"""
Redis Configuration (Single Source of Truth)

All Redis connection settings in one place.
"""

import os

REDIS_HOST = os.getenv('SO_REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('SO_REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('SO_REDIS_DB', 0))
REDIS_URL = os.getenv('SO_REDIS_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}')
