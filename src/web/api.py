"""REST API for face recognition model management using FastAPI."""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
import os
import json
import sys
import argparse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directory to path to import face_recognition modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from face_recognition.storage import Database
from face_recognition.core import FaceEngine
from face_recognition.services.redis_pubsub import RedisPublisher

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Face Recognition API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store active engine instances per client
active_engines = {}

# Initialize Redis publisher
redis_publisher = RedisPublisher()


# Pydantic models for request/response validation
class UserData(BaseModel):
    id: int
    full_name: str
    external_id: Optional[str] = None
    image_urls: List[str]


class EmbeddingUpdateRequest(BaseModel):
    action: str  # 'add_user', 'update_user', 'delete_user'
    client_slug: str
    user_data: UserData


class RebuildRequest(BaseModel):
    client_slug: str


class HealthResponse(BaseModel):
    status: str
    active_clients: List[str]
    total_embeddings: Dict[str, int]


@app.post("/api/v1/embeddings/update")
@limiter.limit("30/minute")  # Limit to 30 updates per minute per IP
async def update_embeddings(request: Request, data: EmbeddingUpdateRequest):
    """Handle incremental embedding updates for individual users."""
    try:
        logger.info(f"📥 Received {data.action} request for client {data.client_slug}, user: {data.user_data.full_name}")

        # Get or create engine for this client
        engine = _get_or_create_engine(data.client_slug)

        if data.action == 'add_user':
            new_users = [{
                'name': data.user_data.full_name,
                'image_path': json.dumps(data.user_data.image_urls),  # JSON array of GCS URLs
            }]
            engine.update_database(new_users=new_users, deleted_users=[])
            logger.info(f"✅ Added user {data.user_data.full_name} to database")

            # Publish Redis event
            redis_publisher.publish_embedding_update(
                data.client_slug, 'add_user', data.user_data.full_name
            )

        elif data.action == 'update_user':
            # Delete old embeddings, add new ones
            deleted_users = [data.user_data.full_name]
            new_users = [{
                'name': data.user_data.full_name,
                'image_path': json.dumps(data.user_data.image_urls),
            }]
            engine.update_database(new_users=new_users, deleted_users=deleted_users)
            logger.info(f"✅ Updated user {data.user_data.full_name} embeddings")

            # Publish Redis event
            redis_publisher.publish_embedding_update(
                data.client_slug, 'update_user', data.user_data.full_name
            )

        elif data.action == 'delete_user':
            deleted_users = [data.user_data.full_name]
            engine.update_database(new_users=[], deleted_users=deleted_users)
            logger.info(f"✅ Deleted user {data.user_data.full_name} from database")

            # Publish Redis event
            redis_publisher.publish_embedding_update(
                data.client_slug, 'delete_user', data.user_data.full_name
            )
        else:
            raise HTTPException(status_code=400, detail=f'Invalid action: {data.action}')

        return {
            'success': True,
            'message': f'Successfully processed {data.action} for {data.user_data.full_name}',
            'embedding_count': len(engine.face_recognition.db_embs),
            'client_slug': data.client_slug
        }

    except Exception as e:
        logger.error(f"❌ Error updating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embeddings/rebuild")
@limiter.limit("5/hour")  # Limit rebuild operations to 5 per hour per IP
async def rebuild_database(http_request: Request, request: RebuildRequest):
    """Rebuild entire face database for a client."""
    try:
        logger.info(f"🔨 Rebuilding database for {request.client_slug}")

        # For now, just reload existing pickle file
        # In production, you would fetch all users from backend API here
        engine = _get_or_create_engine(request.client_slug)

        # Reload embeddings from pickle file
        engine.face_recognition.db_names, engine.face_recognition.db_embs = \
            engine.face_recognition.load_embeddings()

        logger.info(f"✅ Reloaded database with {len(engine.face_recognition.db_embs)} embeddings")

        return {
            'success': True,
            'message': f'Rebuilt database for {request.client_slug}',
            'embedding_count': len(engine.face_recognition.db_embs),
            'database_path': engine.face_recognition.args.db_path
        }

    except Exception as e:
        logger.error(f"❌ Error rebuilding database: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health", response_model=HealthResponse)
@limiter.limit("30/minute")  # Limit health checks to 30 per minute per IP
async def health_check(request: Request):
    """Health check endpoint."""
    return HealthResponse(
        status='healthy',
        active_clients=list(active_engines.keys()),
        total_embeddings={
            client: len(engine.face_recognition.db_embs)
            for client, engine in active_engines.items()
        }
    )


# ============================================================================
# pgvector Sync Endpoints (NEW)
# ============================================================================

class UserSyncRequest(BaseModel):
    """Request model for user sync operations."""
    client_slug: str
    user_data: Dict  # Contains: id, full_name, external_id, image_urls


class UserDeleteRequest(BaseModel):
    """Request model for user deletion."""
    client_slug: str
    user_id: str


class RebuildPgVectorRequest(BaseModel):
    """Request model for pgvector rebuild."""
    client_slug: str
    users: Optional[List[Dict]] = None  # If None, fetch from backend


@app.post("/api/v1/sync/add_user")
@limiter.limit("60/minute")  # Limit to 60 user additions per minute per IP
async def sync_add_user(http_request: Request, request: UserSyncRequest):
    """Add user embeddings to pgvector database.

    Called by backend when a new user is created.
    """
    try:
        logger.info(f"📥 Sync add_user for {request.client_slug}: {request.user_data.get('full_name')}")

        from face_recognition.services.embedding_sync import EmbeddingSyncService

        sync_service = EmbeddingSyncService(
            client_slug=request.client_slug,
            gpu_id=int(os.getenv('GPU_ID', '0'))
        )

        result = sync_service.handle_user_created(request.user_data)

        return {
            'success': True,
            'result': result,
            'message': f"Added {result['embeddings_added']} embeddings for user {result['user_name']}"
        }

    except Exception as e:
        logger.error(f"❌ Error in sync_add_user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/update_user")
@limiter.limit("60/minute")  # Limit to 60 user updates per minute per IP
async def sync_update_user(http_request: Request, request: UserSyncRequest):
    """Update user embeddings in pgvector database.

    Called by backend when a user is updated (name change, images added/removed).
    """
    try:
        logger.info(f"📥 Sync update_user for {request.client_slug}: {request.user_data.get('full_name')}")

        from face_recognition.services.embedding_sync import EmbeddingSyncService

        sync_service = EmbeddingSyncService(
            client_slug=request.client_slug,
            gpu_id=int(os.getenv('GPU_ID', '0'))
        )

        result = sync_service.handle_user_updated(request.user_data)

        return {
            'success': True,
            'result': result,
            'message': f"Updated {result['embeddings_added']} embeddings for user {result['user_name']}"
        }

    except Exception as e:
        logger.error(f"❌ Error in sync_update_user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/delete_user")
@limiter.limit("60/minute")  # Limit to 60 user deletions per minute per IP
async def sync_delete_user(http_request: Request, request: UserDeleteRequest):
    """Delete user embeddings from pgvector database.

    Called by backend when a user is deleted.
    """
    try:
        logger.info(f"📥 Sync delete_user for {request.client_slug}: {request.user_id}")

        from face_recognition.services.embedding_sync import EmbeddingSyncService

        sync_service = EmbeddingSyncService(
            client_slug=request.client_slug,
            gpu_id=int(os.getenv('GPU_ID', '0'))
        )

        result = sync_service.handle_user_deleted(request.user_id)

        return {
            'success': True,
            'result': result,
            'message': f"Deleted {result['embeddings_deleted']} embeddings for user {request.user_id}"
        }

    except Exception as e:
        logger.error(f"❌ Error in sync_delete_user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync/rebuild")
@limiter.limit("2/hour")  # Limit rebuild operations to 2 per hour per IP (very resource intensive)
async def sync_rebuild(http_request: Request, request: RebuildPgVectorRequest):
    """Rebuild all embeddings for a client from backend API.

    This fetches all users from backend and rebuilds the pgvector database.
    """
    try:
        logger.info(f"🔨 Sync rebuild for {request.client_slug}")

        from face_recognition.services.embedding_sync import EmbeddingSyncService

        sync_service = EmbeddingSyncService(
            client_slug=request.client_slug,
            gpu_id=int(os.getenv('GPU_ID', '0'))
        )

        # If users provided, use them. Otherwise fetch from backend
        if request.users:
            result = sync_service.rebuild_all(request.users)
        else:
            result = sync_service.rebuild_all_from_backend()

        return {
            'success': True,
            'result': result,
            'message': f"Rebuilt {result['total_embeddings_added']} embeddings for {result['successful']} users"
        }

    except Exception as e:
        logger.error(f"❌ Error in sync_rebuild: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _get_or_create_engine(client_slug: str) -> FaceEngine:
    """Get existing engine or create new one for client."""
    if client_slug not in active_engines:
        logger.info(f"🔧 Creating new FaceEngine for client: {client_slug}")
        args = _create_args_for_client(client_slug)
        active_engines[client_slug] = FaceEngine(args)
    return active_engines[client_slug]


def _create_args_for_client(client_slug: str) -> argparse.Namespace:
    """Create configuration args for a client's face engine."""
    args = argparse.Namespace()
    args.client_slug = client_slug

    # pgvector or pickle mode
    args.use_pgvector = os.getenv('USE_PGVECTOR', 'false').lower() == 'true'

    if not args.use_pgvector:
        # Legacy pickle mode
        fr_slug = os.getenv('FR_SLUG', 'face-recognition')
        base_path = f'/app/volumes/storage/{fr_slug}/data'
        args.db_path = f'{base_path}/{client_slug}/main.pkl'

        # Ensure directory exists
        os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

        # Create empty pickle if doesn't exist
        if not os.path.exists(args.db_path):
            logger.warning(f"⚠️ Database not found at {args.db_path}, creating empty one")
            import pickle
            import numpy as np
            with open(args.db_path, 'wb') as f:
                pickle.dump({'embeddings': np.array([]), 'names': []}, f)
    else:
        # pgvector mode - no pickle file needed
        args.db_path = None
        logger.info(f"✅ Using pgvector mode for {client_slug}")

    # GPU configuration
    args.gpu_id = int(os.getenv('GPU_ID', '0'))
    args.device = 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu'

    # Recognition configuration
    args.match_threshold = float(os.getenv('MATCH_THRESHOLD', '0.3'))

    # Other configurations
    args.timezone = os.getenv('TIMEZONE', 'UTC')
    args.minimum_face_size = int(os.getenv('MIN_FACE_SIZE', '50'))
    args.max_track_lifetime_seconds = int(os.getenv('MAX_TRACK_LIFETIME', '120'))

    # Camera/stream configurations (optional for API mode)
    args.camera_name = f'api-{client_slug}'
    args.cam_type = 'api'
    args.line_points = None
    args.eval = False

    # Logging
    args.logger = logger

    return args


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('FR_PORT', '8000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    logger.info(f"🚀 Starting Face Recognition API on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')
