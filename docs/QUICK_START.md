# Quick Start Guide - Face Recognition Dashboard

## 🚀 5-Minute Setup

### Development (Localhost)

```bash
# 1. Install dependencies
pip install fastapi uvicorn[standard]

# 2. Start face recognition (Terminal 1)
python main.py

# 3. Start dashboard (Terminal 2)
python backend.py

# 4. Open browser
http://localhost:5000
```

**Done!** 🎉

---

### Docker (Recommended)

```bash
# 1. Build and start
docker-compose up -d

# 2. Check logs
docker-compose logs -f dashboard

# 3. Open browser
http://localhost:5000
```

**Done!** 🎉

---

## 🌍 Global Access (Production)

### Option 1: Direct IP Access (LAN)

```bash
# Find your server IP
hostname -I

# Start dashboard
python backend.py

# Access from any device on network
http://192.168.1.100:5000
```

### Option 2: Internet with HTTPS (Recommended)

**Using Nginx:**

```bash
# 1. Install Nginx
sudo apt install nginx certbot python3-certbot-nginx

# 2. Create config: /etc/nginx/sites-available/dashboard
server {
    listen 80;
    server_name dashboard.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name dashboard.yourdomain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_buffering off;
        proxy_cache off;
    }
}

# 3. Enable and get SSL
sudo ln -s /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/
sudo certbot --nginx -d dashboard.yourdomain.com

# 4. Start services
python backend.py
sudo systemctl restart nginx

# 5. Access globally
https://dashboard.yourdomain.com
```

**Using Caddy (Easier):**

```bash
# 1. Install Caddy
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# 2. Create Caddyfile
dashboard.yourdomain.com {
    reverse_proxy localhost:5000 {
        flush_interval -1
    }
}

# 3. Start services
python backend.py
sudo caddy run

# 4. Access globally (HTTPS automatic!)
https://dashboard.yourdomain.com
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
DASHBOARD_PORT=5000
DASHBOARD_HOST=0.0.0.0
CORS_ORIGINS=*
JPEG_QUALITY=85
TARGET_FPS=30
```

### Common Configurations

**Low Bandwidth:**
```bash
export JPEG_QUALITY=70
export TARGET_FPS=15
python backend.py
```

**High Quality:**
```bash
export JPEG_QUALITY=95
export TARGET_FPS=60
python backend.py
```

**Production:**
```bash
export CORS_ORIGINS=https://yourdomain.com
export WORKERS=4
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:5000
```

---

## 🔍 Verify Setup

### Check Services

```bash
# Health check
curl http://localhost:5000/health

# List cameras
curl http://localhost:5000/cameras

# API documentation
http://localhost:5000/docs
```

### Test Streaming

```bash
# Test with curl
curl http://localhost:5000/video/camera_0

# Test with VLC
vlc http://localhost:5000/video/camera_0

# Test in browser
http://localhost:5000
```

---

## 📊 Monitoring

### View Logs

```bash
# Docker
docker-compose logs -f dashboard

# Direct
python backend.py  # Shows logs in terminal
```

### Check Stats

```bash
curl http://localhost:5000/api/stats
```

**Response:**
```json
{
  "total_cameras": 4,
  "camera_ids": ["camera_0", "camera_1", "camera_2", "camera_3"],
  "cleanup_interval": 5.0,
  "max_frame_age": 2.0
}
```

---

## 🐛 Troubleshooting

### No video showing

```bash
# Check if camera processor is initialized
# Look for: "Camera processor initialized for dashboard streaming"

# Check if frames are being sent
docker-compose logs -f face-recognition | grep "camera_processor"

# Restart services
docker-compose restart
```

### Can't access from other devices

```bash
# Check firewall
sudo ufw allow 5000/tcp

# Check if binding to 0.0.0.0 (not 127.0.0.1)
export DASHBOARD_HOST=0.0.0.0
python backend.py
```

### High CPU usage

```bash
# Reduce quality and FPS
export JPEG_QUALITY=70
export TARGET_FPS=20
python backend.py
```

---

## 📚 Full Documentation

- **[DASHBOARD_SETUP.md](DASHBOARD_SETUP.md)** - Complete setup guide
- **[PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)** - Production deployment with HTTPS, auth, CDN
- **API Docs** - http://localhost:5000/docs (when running)

---

## ✅ Summary

| Deployment | Access | Setup Time | Use Case |
|------------|--------|------------|----------|
| **Localhost** | `http://localhost:5000` | 2 min | Development |
| **LAN** | `http://192.168.1.x:5000` | 3 min | Internal network |
| **Internet (HTTP)** | `http://YOUR_IP:5000` | 5 min | Testing (not secure) |
| **Production (HTTPS)** | `https://yourdomain.com` | 15 min | Global deployment |

**Recommended:** Start with localhost → Test on LAN → Deploy to production with HTTPS

**Questions?** Check the full documentation or raise an issue on GitHub.
