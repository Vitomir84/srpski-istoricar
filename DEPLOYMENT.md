# Deployment Guide - www.srpski-istoricar.rs

This guide explains how to deploy the Serbian History RAG application to your production web server.

## Overview

The application consists of two parts:
- **Backend**: FastAPI server (Python)
- **Frontend**: React application (Node.js build)

## Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Web server (nginx, Apache, or similar)
- Domain configured: www.srpski-istoricar.rs
- SSL certificate for HTTPS

## Step 1: Configure Backend for Production

### 1.1 Create Production Environment File

Copy the production example and configure it:

```powershell
# In the project root
Copy-Item .env.production.example .env
```

### 1.2 Edit .env File

Update the following values in `.env`:

```env
# Your OpenAI API Key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Server Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=5000

# CORS Configuration - IMPORTANT!
ALLOWED_ORIGINS=https://www.srpski-istoricar.rs,https://srpski-istoricar.rs

# Frontend URL
FRONTEND_URL=https://www.srpski-istoricar.rs
```

**Important:** The `ALLOWED_ORIGINS` must include your production domain(s) with the `https://` protocol.

### 1.3 Install Backend Dependencies

```powershell
pip install -r requirements.txt
```

### 1.4 Populate the Database

Before deploying, make sure your FAISS database is populated:

```powershell
python populate_database.py
```

This will process all documents from the `docs/` folder and create the vector database.

## Step 2: Configure Frontend for Production

### 2.1 Update Frontend Production Environment

The file `frontend/.env.production` is already configured with:

```env
REACT_APP_API_URL=https://www.srpski-istoricar.rs
```

**Update this URL** if your backend API is hosted at a different URL (e.g., `https://api.srpski-istoricar.rs` if using a subdomain).

### 2.2 Build Frontend for Production

```powershell
cd frontend
npm install
npm run build
```

This creates an optimized production build in `frontend/build/`.

## Step 3: Deploy Backend

### Option A: Run Directly (Simple, for testing)

```powershell
python app.py
```

The backend will start on the configured host and port (default: 0.0.0.0:5000).

### Option B: Run with Gunicorn (Recommended for Production)

Install gunicorn:

```powershell
pip install gunicorn uvicorn[standard]
```

Run the backend:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 app:app
```

- `-w 4`: 4 worker processes (adjust based on CPU cores)
- `-k uvicorn.workers.UvicornWorker`: Use Uvicorn workers for async support
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000

### Option C: Run as a System Service (Best for Production)

Create a systemd service file `/etc/systemd/system/srpski-istoricar.service`:

```ini
[Unit]
Description=Srpski Istoricar Backend
After=network.target

[Service]
Type=notify
User=your-username
WorkingDirectory=/path/to/srpski-istoricar
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable srpski-istoricar
sudo systemctl start srpski-istoricar
sudo systemctl status srpski-istoricar
```

## Step 4: Deploy Frontend

### Option A: Serve with Web Server (Recommended)

#### Using nginx:

1. Copy build files to web server:

```bash
cp -r frontend/build/* /var/www/srpski-istoricar.rs/
```

2. Configure nginx (`/etc/nginx/sites-available/srpski-istoricar.rs`):

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name srpski-istoricar.rs www.srpski-istoricar.rs;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name srpski-istoricar.rs www.srpski-istoricar.rs;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/srpski-istoricar.rs/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/srpski-istoricar.rs/privkey.pem;

    # Frontend - React build
    root /var/www/srpski-istoricar.rs;
    index index.html;

    # Serve React app
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # API docs
    location /docs {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

3. Enable the site and reload nginx:

```bash
sudo ln -s /etc/nginx/sites-available/srpski-istoricar.rs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Option B: Same Server As Backend

If frontend and backend are on the same URL, update `.env.production`:

```env
# Use relative URLs (empty API_URL means same origin)
REACT_APP_API_URL=
```

Then rebuild the frontend:

```powershell
cd frontend
npm run build
```

## Step 5: SSL Certificate (HTTPS)

Using Let's Encrypt (free):

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d srpski-istoricar.rs -d www.srpski-istoricar.rs
```

Follow the prompts to configure automatic renewal.

## Step 6: Verify Deployment

1. **Check Backend Health:**
   ```bash
   curl https://www.srpski-istoricar.rs/api/health
   ```

2. **Check Frontend:**
   Open browser: https://www.srpski-istoricar.rs

3. **Test Chat:**
   - Select a period
   - Send a test question
   - Verify you receive a response

## Troubleshooting

### CORS Errors

If you see CORS errors in browser console:

1. Verify `ALLOWED_ORIGINS` in backend `.env` includes your domain with `https://`
2. Restart the backend service
3. Clear browser cache

### API Not Reachable

1. Check backend is running: `curl http://localhost:5000/api/health`
2. Check firewall rules allow port 5000
3. Verify nginx proxy configuration
4. Check backend logs

### Frontend Shows Blank Page

1. Check browser console for errors
2. Verify `REACT_APP_API_URL` in `.env.production`
3. Rebuild frontend: `npm run build`
4. Clear browser cache

## Configuration Summary

### Development (localhost)
- Backend: `http://localhost:5000`
- Frontend: `http://localhost:3000`
- CORS: `http://localhost:3000,http://127.0.0.1:3000`

### Production (www.srpski-istoricar.rs)
- Backend: `https://www.srpski-istoricar.rs` (proxied via nginx)
- Frontend: `https://www.srpski-istoricar.rs`
- CORS: `https://www.srpski-istoricar.rs,https://srpski-istoricar.rs`

## Monitoring

### View Backend Logs

```bash
# If using systemd
sudo journalctl -u srpski-istoricar -f

# If running directly
# Logs appear in terminal where app.py is running
```

### Check System Status

```bash
# Backend service
sudo systemctl status srpski-istoricar

# Nginx
sudo systemctl status nginx

# Disk space (FAISS database can be large)
df -h
```

## Updating the Application

### Update Backend Code

```bash
cd /path/to/srpski-istoricar
git pull  # or copy new files
pip install -r requirements.txt
sudo systemctl restart srpski-istoricar
```

### Update Frontend

```bash
cd /path/to/srpski-istoricar/frontend
npm install
npm run build
sudo cp -r build/* /var/www/srpski-istoricar.rs/
```

### Update Database

```bash
python populate_database.py
# or add individual documents:
python add_txt_to_faiss.py path/to/document.txt
```

## Security Checklist

- [ ] HTTPS enabled with valid SSL certificate
- [ ] CORS properly configured (only allow your domain)
- [ ] OpenAI API key stored securely in `.env` (not in git)
- [ ] `.env` file has restricted permissions: `chmod 600 .env`
- [ ] Firewall configured (allow only 80, 443, SSH)
- [ ] Backend not directly exposed (proxied through nginx)
- [ ] Regular backups of FAISS database and bibliography
- [ ] System and dependencies kept up to date
- [ ] Rate limiting configured on nginx (optional)

## Support

For issues or questions:
- Check logs first
- Review this deployment guide
- Verify all configuration files match production settings
- Test backend and frontend separately to isolate issues

---

**Congratulations!** Your Serbian History RAG application is now deployed at www.srpski-istoricar.rs 🎉
