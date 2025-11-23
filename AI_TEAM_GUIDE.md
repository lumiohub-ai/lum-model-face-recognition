# Face Recognition Auto-Update: AI Team Guide

## 🎯 Overview

The face recognition model now **automatically updates** when users are created, updated, or deleted in the backend. No manual intervention needed!

## 🔄 How It Works (Automatic)

```
User Created/Updated/Deleted in Backend
         ↓
PostgreSQL sends notification (pg_notify)
         ↓
Backend LISTEN receives it
         ↓
Backend calls your FastAPI endpoint
         ↓
Your FaceEngine updates in-memory model
         ↓
✅ Model is ready for next recognition request!
```

## 📡 The FastAPI Endpoint (Already Implemented)

You already have this endpoint in `src/web/api.py`:

```python
@app.post("/api/v1/embeddings/update")
async def update_embeddings(request: EmbeddingUpdateRequest):
    """
    Automatically called by backend when users change.

    Request payload:
    {
        "action": "add_user" | "update_user" | "delete_user",
        "client_slug": "humblebee",
        "user_data": {
            "id": 74,
            "full_name": "John Doe",
            "external_id": "EMP001",
            "image_urls": [
                "https://storage.googleapis.com/.../john_doe_original.jpg",
                "https://storage.googleapis.com/.../john_doe_thumb.jpg",
                "https://storage.googleapis.com/.../john_doe_medium.jpg"
            ]
        }
    }
    """
```

## 🚀 What Happens for Each Action

### 1️⃣ Add User (`action: "add_user"`)

**When:** New employee enrolled in system
**What happens:**
```python
# Backend automatically calls:
POST /api/v1/embeddings/update
{
    "action": "add_user",
    "client_slug": "humblebee",
    "user_data": {
        "id": 74,
        "full_name": "Jane Smith",
        "image_urls": ["https://...image1.jpg", "https://...image2.jpg"]
    }
}

# Your code does:
engine.update_database(
    new_users=[{
        'name': 'Jane Smith',
        'image_path': '["https://...image1.jpg", "https://...image2.jpg"]'
    }],
    deleted_users=[]
)
```

**Result:** Jane's face embeddings added to in-memory database. Next camera frame will recognize her!

---

### 2️⃣ Update User (`action: "update_user"`)

**When:** Employee's photos updated (new profile picture, better quality images)
**What happens:**
```python
# Backend automatically calls:
POST /api/v1/embeddings/update
{
    "action": "update_user",
    "client_slug": "humblebee",
    "user_data": {
        "id": 74,
        "full_name": "Jane Smith",
        "image_urls": ["https://...new_image1.jpg", "https://...new_image2.jpg"]
    }
}

# Your code does:
# 1. Delete old embeddings
engine.update_database(
    new_users=[{
        'name': 'Jane Smith',
        'image_path': '["https://...new_image1.jpg", ...]'
    }],
    deleted_users=['Jane Smith']  # Remove old embeddings first
)
```

**Result:** Jane's face embeddings refreshed with new images. Better recognition accuracy!

---

### 3️⃣ Delete User (`action: "delete_user"`)

**When:** Employee leaves company or is terminated
**What happens:**
```python
# Backend automatically calls:
POST /api/v1/embeddings/update
{
    "action": "delete_user",
    "client_slug": "humblebee",
    "user_data": {
        "id": 74,
        "full_name": "Jane Smith",
        "image_urls": []
    }
}

# Your code does:
engine.update_database(
    new_users=[],
    deleted_users=['Jane Smith']
)
```

**Result:** Jane removed from model. Camera will mark her as "unrecognized" if detected.

---

## 🧪 Testing the Integration

### Option 1: Via Jupyter Notebook (Recommended)

Use `notebooks/test_fr_api.ipynb`:

```python
# Cell 4: Install dependencies
!pip install psycopg2-binary python-dotenv requests pillow

# Cell 5: Setup
import requests, json, time
# ... loads config ...

# Cell 9: Login
auth_token = login_to_backend(
    email='your-admin@example.com',
    password='your-password',
    client_slug='humblebee'
)

# Cell 11: Create test user (triggers auto-update!)
created_user = create_test_user(auth_token)

# Cell 13: Check logs (verify it worked)
check_backend_logs()
```

### Option 2: Direct API Call

```bash
# Manually trigger an update (for testing)
curl -X POST http://localhost:8000/api/v1/embeddings/update \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add_user",
    "client_slug": "humblebee",
    "user_data": {
      "id": 999,
      "full_name": "Test User",
      "image_urls": ["https://storage.googleapis.com/...test.jpg"]
    }
  }'
```

---

## 📊 Monitoring & Logs

### Backend Logs (PostgreSQL Notifications)

```bash
docker compose logs -f backend | grep -i face
```

**What to look for:**
```
✅ Face model update notification sent: add_user
🔄 [FaceRecognition] Attempt 1/3: add_user for user Jane Smith (client: humblebee)
✅ [FaceRecognition] Success: add_user for Jane Smith
```

### Face Recognition Logs (Your Service)

```bash
docker compose logs -f face-recognition
```

**What to look for:**
```
📥 Received add_user request for client humblebee, user: Jane Smith
✅ Added user Jane Smith to database
```

---

## 🎓 For AI Team: What You Need to Know

### ✅ Already Done (No Action Needed)

1. **FastAPI endpoint** - Already implemented in `api.py:53-96`
2. **Request validation** - Pydantic models handle it
3. **Engine management** - `_get_or_create_engine()` handles multi-tenant
4. **Error handling** - Catches exceptions, returns 500 on failure
5. **Logging** - Info and error logs already in place

### 🔧 What You CAN Customize

#### 1. Embedding Update Logic

Currently in `api.py:62-83`, you can modify:

```python
# Example: Add custom preprocessing
elif request.action == 'update_user':
    # Your custom logic here:
    # - Validate image quality
    # - Pre-process images
    # - Update metadata
    # - Custom embedding parameters

    deleted_users = [request.user_data.full_name]
    new_users = [{
        'name': request.user_data.full_name,
        'image_path': json.dumps(request.user_data.image_urls),
        # Add custom fields here
    }]
    engine.update_database(new_users=new_users, deleted_users=deleted_users)
```

#### 2. Response Data

Customize what you return to backend:

```python
return {
    'success': True,
    'message': f'Successfully processed {request.action}',
    'embedding_count': len(engine.face_recognition.db_embs),
    'client_slug': request.client_slug,
    # Add custom metrics:
    'processing_time_ms': processing_time,
    'image_quality_score': quality_score,
    'confidence_threshold': threshold
}
```

#### 3. Batch Operations

If you want to optimize for multiple users:

```python
@app.post("/api/v1/embeddings/batch-update")
async def batch_update(requests: List[EmbeddingUpdateRequest]):
    """Handle multiple user updates in one request"""
    results = []
    for req in requests:
        result = await update_embeddings(req)
        results.append(result)
    return {'results': results, 'total': len(results)}
```

---

## 🚨 Error Handling

### Backend Retry Logic

If your service is down or slow:

```python
# Backend automatically retries 3 times with exponential backoff:
# Attempt 1: immediate
# Attempt 2: wait 2s
# Attempt 3: wait 4s
# After 3 failures: logs error, continues running (non-blocking)
```

**Your service MUST:**
- Return HTTP 200 on success
- Return HTTP 4xx/5xx on error
- Respond within 30 seconds (backend timeout)

### Graceful Degradation

```python
try:
    engine.update_database(new_users=users, deleted_users=deleted)
    return {'success': True, 'message': 'Updated successfully'}
except Exception as e:
    logger.error(f"Failed to update: {e}")
    # Return error - backend will retry
    raise HTTPException(status_code=500, detail=str(e))
```

---

## 📝 Common Questions

### Q: Do I need to manually rebuild the database?

**A:** No! The model updates automatically. You only need `/api/v1/embeddings/rebuild` for:
- Initial setup (first deployment)
- Recovery from errors
- Manual sync if notifications were missed

### Q: What if the face-recognition service is offline?

**A:** Backend will:
1. Retry 3 times with exponential backoff
2. Log the failure
3. Continue running (non-blocking)
4. Next time service is online, new updates will work

**Note:** Missed updates during downtime won't be replayed. Use `/rebuild` endpoint to resync.

### Q: How do I test without affecting production?

**A:** Use the Jupyter notebook:
1. Create test users with synthetic images
2. Monitor logs to verify updates
3. Delete test users when done

### Q: Can I see the notification payload?

**A:** Yes! Run this in the notebook:

```python
# Cell 7: Listen to PostgreSQL notifications
listen_to_notifications(duration_seconds=30)

# In another tab:
# Cell 11: Create user
created_user = create_test_user(auth_token)

# You'll see the exact JSON payload sent to your API!
```

---

## 🎯 Quick Reference

| Event | Trigger | Payload Action | Your Code |
|-------|---------|---------------|-----------|
| User enrolled | Admin adds employee | `add_user` | `engine.update_database(new_users=[...])` |
| Photos updated | Admin changes pictures | `update_user` | Delete old + add new |
| Employee leaves | Admin deletes user | `delete_user` | `engine.update_database(deleted_users=[...])` |

---

## 📚 Related Files

- **API Implementation:** `src/web/api.py`
- **Test Notebook:** `notebooks/test_fr_api.ipynb`
- **Integration Docs:** `FACE_RECOGNITION_AUTO_UPDATE.md`
- **Backend Service:** `../backend/src/services/FaceRecognitionService.js`
- **Database Listener:** `../backend/src/config/database.js`

---

## 🆘 Need Help?

1. Check logs: `docker compose logs face-recognition backend`
2. Run health check: `GET /api/v1/health`
3. Test with notebook: `notebooks/test_fr_api.ipynb`
4. Review integration docs: `FACE_RECOGNITION_AUTO_UPDATE.md`

---

**Last Updated:** 2025-11-10
**Integration Status:** ✅ Active
**Tested:** Local dev environment (backend notifications working, FR service graceful degradation confirmed)
