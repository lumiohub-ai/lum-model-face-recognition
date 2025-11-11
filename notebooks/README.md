# Face Recognition Notebooks

This directory contains Jupyter notebooks for testing and monitoring the Smart Office face recognition integration.

## 📓 Available Notebooks

### 1. `test_fr_api.ipynb` - Face Recognition API Testing
**Purpose:** Test the PostgreSQL NOTIFY integration for automatic model updates

**Use Cases:**
- Test user creation → pg_notify → FR API update flow
- Monitor notifications in real-time
- Verify integration without running FR service locally
- Debug notification payload

**Quick Start:**
```python
# Run these cells in order:
1. Cell 4: Install dependencies (psycopg2-binary, pillow, etc.)
2. Cell 5: Setup imports and configuration
3. Cell 9-10: Login to backend
4. Cell 11: Create test user (triggers notification)
5. Cell 13: Check backend logs (verify it worked)
```

**Key Tests:**
- **Test 1 (Cell 7):** Listen to PostgreSQL notifications in real-time
- **Test 2 (Cell 11):** Create user and trigger notification
- **Test 3 (Cell 13):** Check backend logs for notification flow
- **Test 4 (Cell 15):** Complete E2E integration test

---

### 2. `check_user_status.ipynb` - User Status Checker
**Purpose:** Query database for user statuses and attendance records

**Use Cases:**
- Get all users with current status (in/out)
- Find when users were last tracked by cameras
- Lookup specific user details and history
- Export user data to CSV for analysis
- Find users never tracked by cameras

**Quick Start:**
```python
# Run these cells in order:
1. Cell 2: Install dependencies (psycopg2-binary, pandas, tabulate)
2. Cell 3: Setup database connection
3. Cell 9: Get all users with status
4. Cell 11: Lookup specific user by ID (edit USER_ID variable)
```

**Key Functions:**
- `get_all_users_with_status()` - Returns DataFrame with all users, statuses, last seen timestamps
- `get_user_by_id(user_id)` - Returns detailed user info + recent attendance records
- `get_status_summary()` - Returns statistics (total users, IN count, OUT count)

**Example Output:**
```
📊 STATUS SUMMARY
   Total Users: 47
   Status IN:   12 (25.5%)
   Status OUT:  35 (74.5%)
   Never Tracked: 3

👥 ALL USERS
╔════╦═══════════════════════╦═════════════╦════════╦═══════════════════════╗
║ ID ║ Name                  ║ External ID ║ Status ║ Last Seen             ║
╠════╬═══════════════════════╬═════════════╬════════╬═══════════════════════╣
║ 74 ║ Jane Smith            ║ EMP001      ║ in     ║ 2025-11-10 17:18:25   ║
║ 73 ║ John Doe              ║ EMP002      ║ out    ║ 2025-11-09 14:32:10   ║
╚════╩═══════════════════════╩═════════════╩════════╩═══════════════════════╝
```

---

### 3. `api.ipynb` - Original Backend API Tests
**Purpose:** Comprehensive backend API testing (50 cells)

**Use Cases:**
- Test all backend endpoints (users, cameras, attendance, etc.)
- Face recognition model update tests
- Unrecognized faces tests

**Note:** This is a large notebook with many test cases. Use `test_fr_api.ipynb` for focused FR testing.

---

## 🔧 Setup

### Prerequisites

1. **Backend running:**
   ```bash
   docker compose up -d backend database
   ```

2. **Environment variables:**
   Create `.env` file or ensure these are set:
   ```bash
   SO_DB_EXTERNAL_PORT=5432
   SO_DB_NAME=smart-office
   SO_DB_USER=user
   SO_DB_PASSWORD=passW0rd
   SO_BACKEND_API_URL=http://localhost:7091
   ```

3. **Python dependencies:**
   All notebooks include installation cells, but you can pre-install:
   ```bash
   pip install psycopg2-binary python-dotenv requests pillow pandas tabulate
   ```

---

## 📊 Common Workflows

### Workflow 1: Test Face Recognition Integration
```python
# Notebook: test_fr_api.ipynb

# 1. Setup
Run Cell 4 (install deps) + Cell 5 (setup)

# 2. Start listener in one tab
Run Cell 7 (listen_to_notifications)

# 3. Create user in another tab
Run Cell 9 (login) + Cell 11 (create_test_user)

# 4. See notification appear in Cell 7!
# 5. Check logs in Cell 13
```

### Workflow 2: Check Who's Currently in the Building
```python
# Notebook: check_user_status.ipynb

# 1. Setup
Run Cell 2 (install deps) + Cell 3 (setup)

# 2. Get all users
users_df = get_all_users_with_status()

# 3. Filter for users currently IN
users_in = users_df[users_df['status'] == 'in']
print(f"Currently in building: {len(users_in)} users")
```

### Workflow 3: Debug Why User Not Recognized
```python
# Notebook: check_user_status.ipynb

# 1. Lookup user
user = get_user_by_id(74)

# 2. Check their data
print(f"Status: {user['status']}")
print(f"Images: {user['image_count']}")
print(f"Last seen: {user['recent_attendance'][0][0]}")

# 3. Verify images accessible
for img in user['image_urls']:
    print(img['original'])
```

---

## 🎯 For AI Team

### Use Case 1: Know Which Users to Expect
```python
# check_user_status.ipynb
users = get_all_users_with_status()
active_users = users[users['status'] == 'in']

# These are users your model should prioritize
print(f"Expecting {len(active_users)} users in building")
```

### Use Case 2: Verify Your Embeddings Match Database
```python
# check_user_status.ipynb
db_users = get_all_users_with_status()

# Compare with your in-memory model
your_engine_users = engine.face_recognition.db_embs.keys()

# Find mismatches
db_names = set(db_users['full_name'])
missing_in_model = db_names - your_engine_users
print(f"Users in DB but not in model: {missing_in_model}")
```

### Use Case 3: Monitor Auto-Updates
```python
# test_fr_api.ipynb - Cell 7
# Listen for real-time updates
listen_to_notifications(duration_seconds=60)

# In another terminal: Admin creates/updates user
# You'll see the notification payload here instantly!
```

---

## 🆘 Troubleshooting

### Database Connection Errors
```python
# Check database is running
!docker compose ps database

# Test connection
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='smart-office',
    user='user',
    password='passW0rd'
)
print("✅ Connected!")
conn.close()
```

### ModuleNotFoundError
```python
# Install missing packages
!pip install psycopg2-binary python-dotenv requests pillow pandas tabulate
```

### No Users Found
```python
# Check you're using the right schema
summary = get_status_summary(schema_name='org_humblebee')

# List all schemas
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'org_%'")
print("Available schemas:", cur.fetchall())
conn.close()
```

---

## 📚 Related Documentation

- **Integration Guide:** `../FACE_RECOGNITION_AUTO_UPDATE.md`
- **AI Team Guide:** `../AI_TEAM_GUIDE.md`
- **Backend API Docs:** `http://localhost:7091/` (when backend running)

---

**Last Updated:** 2025-11-10
