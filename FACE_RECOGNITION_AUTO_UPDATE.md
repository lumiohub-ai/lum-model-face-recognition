# Face Recognition Model Auto-Update Integration

## Overview

This document describes the integration between the Smart Office backend and the face recognition service that enables **automatic model updates** when users are created, updated, or deleted.

## Architecture

### System Components

```
┌─────────────────┐      pg_notify()       ┌──────────────────┐
│   PostgreSQL    │─────────────────────────▶│  Backend Node.js │
│   (Database)    │                          │   (database.js)  │
└─────────────────┘                          └────────┬─────────┘
                                                      │
                                                      │ HTTP POST
                                                      ▼
                                             ┌──────────────────┐
                                             │  Face Recognition│
                                             │   FastAPI Server │
                                             │   (web/api.py)   │
                                             └────────┬─────────┘
                                                      │
                                                      │ update_database()
                                                      ▼
                                             ┌──────────────────┐
                                             │   Face Engine    │
                                             │  (core/engine.py)│
                                             └──────────────────┘
```

### Data Flow

1. **User Operation** (Create/Update/Delete) → Backend UserService
2. **Database Trigger** → `pg_notify('face_model_update', payload)`
3. **Notification Listener** → Backend `database.js` receives notification
4. **HTTP Request** → `FaceRecognitionService` calls FR API
5. **Model Update** → FR service updates embeddings in memory and pickle file

## Backend Integration

### 1. Database Notification System (`src/config/database.js`)

The backend listens to PostgreSQL notifications on the `face_model_update` channel:

```javascript
function setupNotificationListener(io) {
  pool.connect((err, client, done) => {
    const channels = ['user_status_change', 'unrecognized_user_detected', 'face_model_update'];
    channels.forEach(channel => client.query(`LISTEN ${channel}`));

    client.on('notification', async (msg) => {
      if (msg.channel === 'face_model_update') {
        const FaceRecognitionService = require('../services/FaceRecognitionService');
        const frService = new FaceRecognitionService();

        // Non-blocking async call
        frService.updateUserEmbeddings({
          action: payload.action,
          client_slug: payload.client_slug,
          user_data: payload.user_data,
        }).catch(err => {
          logger.warn('Face recognition update failed (non-blocking):', err.message);
        });
      }
    });
  });
}
```

### 2. User Service Notifications (`src/services/UserService.js`)

When users are modified, the backend sends notifications:

```javascript
async notifyFaceModelUpdate(client, payload) {
  try {
    await client.query(`SELECT pg_notify('face_model_update', $1)`, [
      JSON.stringify(payload),
    ]);
    console.log(`✅ Face model update notification sent: ${payload.action}`);
  } catch (e) {
    console.warn("UserService.notifyFaceModelUpdate warn:", e?.message || e);
  }
}
```

#### Payload Structure

```javascript
{
  action: 'add_user' | 'update_user' | 'delete_user',
  client_slug: 'humblebee',
  schema_name: 'humblebee',
  user_data: {
    id: 12345,
    external_id: 'uuid-here',
    full_name: 'John Doe',
    image_urls: [
      'https://storage.googleapis.com/.../image1.jpg',
      'https://storage.googleapis.com/.../image2.jpg'
    ]
  }
}
```

### 3. Face Recognition Service (`src/services/FaceRecognitionService.js`)

Makes HTTP requests to the FR API with retry logic:

```javascript
async updateUserEmbeddings({ action, client_slug, user_data }, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/api/v1/embeddings/update`,
        { action, client_slug, user_data },
        { timeout: this.timeout }
      );
      return response.data;
    } catch (error) {
      // Exponential backoff: 2s, 4s, 8s
      const delay = Math.pow(2, attempt) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}
```

## Face Recognition Service

### API Endpoints (`src/web/api.py`)

#### 1. Update Embeddings
```
POST /api/v1/embeddings/update
```

**Request Body:**
```json
{
  "action": "add_user",
  "client_slug": "humblebee",
  "user_data": {
    "id": 12345,
    "full_name": "John Doe",
    "external_id": "uuid",
    "image_urls": [
      "https://storage.googleapis.com/.../image1.jpg"
    ]
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed add_user for John Doe",
  "embedding_count": 150,
  "client_slug": "humblebee"
}
```

**Actions:**
- `add_user`: Download images, compute embeddings, add to database
- `update_user`: Delete old embeddings, compute new ones, add to database
- `delete_user`: Remove all embeddings for the user

#### 2. Rebuild Database
```
POST /api/v1/embeddings/rebuild
```

**Request Body:**
```json
{
  "client_slug": "humblebee"
}
```

Reloads the entire face database from the pickle file.

#### 3. Health Check
```
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "active_clients": ["humblebee", "dev"],
  "total_embeddings": {
    "humblebee": 150,
    "dev": 45
  }
}
```

### Engine Integration (`src/face_recognition/core/engine.py`)

The refactored `FaceEngine` provides the `update_database()` method:

```python
def update_database(self, new_users: List[dict], deleted_users: List[str]) -> None:
    """Update the face recognition database with new users and delete old ones.

    Args:
        new_users: List of new users to add to the database
            [{'name': 'John Doe', 'image_path': '["url1", "url2"]'}]
        deleted_users: List of user names to remove from the database
            ['John Doe', 'Jane Smith']
    """
    # Delete old embeddings
    for user_name in deleted_users:
        indices_to_delete = [i for i, name in enumerate(self.face_recognition.db_names)
                           if name.split('_')[0] == user_name]
        self.face_recognition.db_names = [name for i, name in enumerate(self.face_recognition.db_names)
                                         if i not in indices_to_delete]
        self.face_recognition.db_embs = np.delete(self.face_recognition.db_embs, indices_to_delete, axis=0)

    # Add new embeddings
    for user in new_users:
        embedding = self.get_emb(user['image_path'])
        if embedding is None:
            continue
        for emb in embedding:
            self.face_recognition.db_names.append(user['name'])
            self.face_recognition.db_embs = np.append(
                self.face_recognition.db_embs, [emb], axis=0
            )

    # Save to pickle file
    self.face_recognition.save_embeddings()
```

## Configuration

### Environment Variables

**Backend (.env):**
```bash
FACE_RECOGNITION_API_URL=http://face-recognition:8000
```

**Face Recognition (.env):**
```bash
FR_PORT=8000
FR_SLUG=face-recognition
GOOGLE_APPLICATION_CREDENTIALS=/app/keys/gcs-service-account.json
USE_GPU=false
GPU_ID=0
MIN_FACE_SIZE=50
MAX_TRACK_LIFETIME=120
```

### Docker Compose

The face-recognition service should be:
- Connected to the same network as backend
- Able to access PostgreSQL for direct queries (optional)
- Have access to GCS credentials for image downloads

```yaml
face-recognition:
  build:
    context: ./volumes/src/face-recognition
    dockerfile: Dockerfile
  ports:
    - "${FACE_RECOGNITION_PORT:-8000}:8000"
  depends_on:
    - backend
    - database
  volumes:
    - ./volumes/storage/face-recognition/data:/app/volumes/storage/face-recognition/data
    - ./volumes/src/backend/src/keys:/app/keys:ro
  environment:
    BACKEND_URL: http://backend:${SO_BACKEND_PORT}
    FR_PORT: 8000
  command: uvicorn src.web.api:app --host 0.0.0.0 --port 8000
```

## Testing

### 1. Test API with Jupyter Notebook

Use `notebooks/api.ipynb` for comprehensive testing:

**Face Recognition Model Update Tests (Cells 31-44):**
- **Cell 31-32**: Setup and Health Check
- **Cell 33-34**: Add User to FR Model
- **Cell 35-36**: Update User Embeddings
- **Cell 37-38**: Delete User from FR Model
- **Cell 39-40**: Rebuild Entire Database
- **Cell 41-42**: End-to-End Integration Test

**Unrecognized Faces Tests (Cells 45-49):**
- **Cell 45-46**: Setup for Unrecognized Faces
- **Cell 47**: Upload Unrecognized Face
- **Cell 48**: List and Verify Signed URLs
- **Cell 49**: Retention Cleanup Testing

Quick example - Health Check:
```python
import requests

FR_API_BASE = "http://localhost:8000"

# Check health
response = requests.get(f"{FR_API_BASE}/api/v1/health")
print(response.json())
# Output: {"status": "healthy", "active_clients": ["humblebee"], "total_embeddings": {"humblebee": 150}}

# Add user
response = requests.post(
    f'{FR_API_BASE}/api/v1/embeddings/update',
    json={
        'action': 'add_user',
        'client_slug': 'humblebee',
        'user_data': {
            'id': 123,
            'full_name': 'Test User',
            'image_urls': ['https://storage.googleapis.com/.../face.jpg']
        }
    }
)
print(response.json())
```

### 2. Test End-to-End Integration

1. **Create a user** via backend API:
   ```bash
   POST /api/org/{slug}/users
   ```

2. **Check logs** in backend:
   ```
   ✅ Face model update notification sent: add_user
   🔄 [FaceRecognition] Attempt 1/3: add_user for Test User
   ✅ [FaceRecognition] Successfully processed add_user for Test User
   ```

3. **Verify embeddings** via health endpoint:
   ```bash
   GET http://localhost:8000/api/v1/health
   ```

### 3. Test Unrecognized Faces

Use `notebooks/api.ipynb` - **Cells 45-49 "Unrecognized Faces API Tests"**:

```python
# Cell 45-46: Setup
UNREC_API_BASE = "http://localhost:7091"
UNREC_SLUG = "humblebee"

# Cell 47: Upload unrecognized face
upload_unrecognized_face(image_path="/path/to/face.jpg")

# Cell 48: List and verify signed URLs
list_and_verify_unrecognized_faces(limit=5)

# Cell 49: Test retention cleanup (create faces with old dates)
create_test_unrecognized_face_with_date(days_old=95, label="TEST")
```

This tests the complete unrecognized faces workflow including signed URL generation and retention cleanup.

## Key Features

### ✅ Automatic Updates
- User creation → embeddings added automatically
- User update (image change) → embeddings refreshed
- User deletion → embeddings removed

### ✅ Non-Blocking Operation
- User operations succeed even if FR service is down
- Retry logic with exponential backoff (2s, 4s, 8s)
- Graceful degradation

### ✅ Multi-Tenancy Support
- Each client has its own face database
- Databases stored at `/app/volumes/storage/face-recognition/data/{client_slug}/face_db.pkl`
- Engines cached in memory per client

### ✅ Image Handling
- Supports multiple images per user
- Downloads from GCS with authentication
- Computes embeddings from frontal faces

## Integration Status

### ✅ Completed
- [x] Backend notification system (pg_notify)
- [x] FaceRecognitionService with retry logic
- [x] FastAPI endpoints for model updates
- [x] Engine update_database() method
- [x] Refactored codebase integration (feat/refactor merged)
- [x] Multi-client support
- [x] Unrecognized faces API preserved
- [x] Comprehensive testing notebook with 50 cells
  - Backend API tests (cells 0-30)
  - FR model update tests (cells 31-44)
  - Unrecognized faces tests (cells 45-49)

### ⏳ Future Enhancements
- [ ] PostgreSQL LISTEN directly in FR service (alternative to HTTP)
- [ ] Batch update support for multiple users
- [ ] Metrics and monitoring dashboard
- [ ] Webhook notifications on update completion
- [ ] Model versioning and rollback support

## For AI Team

### Getting Started

1. **Clone the repository** on `integration/so-124` branch
2. **Review the code**:
   - Backend: `src/services/FaceRecognitionService.js`
   - FR API: `src/web/api.py`
   - Engine: `src/face_recognition/core/engine.py`
3. **Test locally** using the Jupyter notebooks
4. **Extend** the API with additional endpoints as needed

### Important Files

```
face-recognition/
├── src/
│   ├── web/
│   │   └── api.py                 # FastAPI server
│   └── face_recognition/
│       ├── core/
│       │   ├── engine.py          # FaceEngine with update_database()
│       │   ├── recognizer.py      # Face recognition logic
│       │   └── detector.py        # Face detection
│       └── storage/
│           └── database.py        # Pickle file management
└── notebooks/
    ├── api.ipynb                  # API testing with unrecognized faces
    └── test_fr_api.ipynb          # FR API endpoint testing
```

### Integration Points

1. **Receive user updates**: Already handled via HTTP POST
2. **Download images**: Use GCS credentials at `/app/keys/gcs-service-account.json`
3. **Compute embeddings**: Use `engine.compute_embeddings(image)`
4. **Update database**: Use `engine.update_database(new_users, deleted_users)`
5. **Save state**: Automatically saved to pickle after updates

### Next Steps for AI Team

1. ✅ **Review** this document and the code structure
2. ✅ **Test** the integration with sample data
3. 🔄 **Optimize** embedding computation for your use case
4. 🔄 **Add** any additional features needed (e.g., face quality checks)
5. 🔄 **Monitor** performance and adjust as needed

## Support

For questions about this integration:
- Backend issues: Check `src/services/FaceRecognitionService.js` logs
- FR API issues: Check FastAPI logs in face-recognition container
- Model issues: Review `src/face_recognition/core/engine.py`

## Changelog

### 2025-11-10 (Latest Update)
- **Added comprehensive testing notebook** (`notebooks/api.ipynb` with 50 cells)
  - Face Recognition Model Update Tests (cells 31-44)
  - Unrecognized Faces Tests (cells 45-49) - RESTORED
- Updated documentation with detailed notebook cell references
- Ready for AI team testing and development

### 2025-11-10 (Earlier)
- Merged `feat/refactor` branch with refactored codebase structure
- Updated `src/web/api.py` to use new imports from refactored modules
- Documented complete integration for AI team handoff

### 2025-11-09
- Initial integration with pg_notify system
- Created FaceRecognitionService with retry logic
- Implemented FastAPI endpoints for model updates
- Added multi-tenancy support
