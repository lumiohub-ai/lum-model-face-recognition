"""REST API for face recognition model management."""

from flask import Flask, request, jsonify
import logging
import os
import json
import sys
import argparse

# Add parent directory to path to import face_recognition modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from face_recognition.database import Database
from face_recognition.engine import FaceEngine

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store active engine instances per client
active_engines = {}


@app.route('/api/v1/embeddings/update', methods=['POST'])
def update_embeddings():
    """Handle incremental embedding updates for individual users."""
    try:
        data = request.json
        action = data.get('action')  # 'add_user', 'update_user', 'delete_user'
        client_slug = data.get('client_slug')
        user_data = data.get('user_data')

        if not all([action, client_slug, user_data]):
            return jsonify({'error': 'Missing required fields: action, client_slug, user_data'}), 400

        logger.info(f"📥 Received {action} request for client {client_slug}, user: {user_data.get('full_name')}")

        # Get or create engine for this client
        engine = _get_or_create_engine(client_slug)

        if action == 'add_user':
            new_users = [{
                'name': user_data['full_name'],
                'image_path': json.dumps(user_data['image_urls']),  # JSON array of GCS URLs
            }]
            engine.update_database(new_users=new_users, deleted_users=[])
            logger.info(f"✅ Added user {user_data['full_name']} to database")

        elif action == 'update_user':
            # Delete old embeddings, add new ones
            deleted_users = [user_data['full_name']]
            new_users = [{
                'name': user_data['full_name'],
                'image_path': json.dumps(user_data['image_urls']),
            }]
            engine.update_database(new_users=new_users, deleted_users=deleted_users)
            logger.info(f"✅ Updated user {user_data['full_name']} embeddings")

        elif action == 'delete_user':
            deleted_users = [user_data['full_name']]
            engine.update_database(new_users=[], deleted_users=deleted_users)
            logger.info(f"✅ Deleted user {user_data['full_name']} from database")
        else:
            return jsonify({'error': f'Invalid action: {action}'}), 400

        return jsonify({
            'success': True,
            'message': f'Successfully processed {action} for {user_data["full_name"]}',
            'embedding_count': len(engine.face_recognition.db_embs),
            'client_slug': client_slug
        })

    except Exception as e:
        logger.error(f"❌ Error updating embeddings: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/embeddings/rebuild', methods=['POST'])
def rebuild_database():
    """Rebuild entire face database for a client."""
    try:
        data = request.json
        client_slug = data.get('client_slug')

        if not client_slug:
            return jsonify({'error': 'client_slug required'}), 400

        logger.info(f"🔨 Rebuilding database for {client_slug}")

        # For now, just reload existing pickle file
        # In production, you would fetch all users from backend API here
        engine = _get_or_create_engine(client_slug)

        # Reload embeddings from pickle file
        engine.face_recognition.db_names, engine.face_recognition.db_embs = \
            engine.face_recognition.load_embeddings()

        logger.info(f"✅ Reloaded database with {len(engine.face_recognition.db_embs)} embeddings")

        return jsonify({
            'success': True,
            'message': f'Rebuilt database for {client_slug}',
            'embedding_count': len(engine.face_recognition.db_embs),
            'database_path': engine.face_recognition.args.db_path
        })

    except Exception as e:
        logger.error(f"❌ Error rebuilding database: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'active_clients': list(active_engines.keys()),
        'total_embeddings': {
            client: len(engine.face_recognition.db_embs)
            for client, engine in active_engines.items()
        }
    })


def _get_or_create_engine(client_slug):
    """Get existing engine or create new one for client."""
    if client_slug not in active_engines:
        logger.info(f"🔧 Creating new FaceEngine for client: {client_slug}")
        args = _create_args_for_client(client_slug)
        active_engines[client_slug] = FaceEngine(args)
    return active_engines[client_slug]


def _create_args_for_client(client_slug):
    """Create configuration args for a client's face engine."""
    args = argparse.Namespace()
    args.client_slug = client_slug

    # Determine database path
    fr_slug = os.getenv('FR_SLUG', 'face-recognition')
    base_path = f'/app/volumes/storage/{fr_slug}/data'
    args.db_path = f'{base_path}/{client_slug}/face_db.pkl'

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    # Create empty pickle if doesn't exist
    if not os.path.exists(args.db_path):
        logger.warning(f"⚠️ Database not found at {args.db_path}, creating empty one")
        import pickle
        import numpy as np
        with open(args.db_path, 'wb') as f:
            pickle.dump({'embeddings': np.array([]), 'names': []}, f)

    # GPU configuration
    args.gpu_id = int(os.getenv('GPU_ID', '0'))
    args.device = 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu'

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
    port = int(os.getenv('FR_PORT', '8000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'

    logger.info(f"🚀 Starting Face Recognition API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
