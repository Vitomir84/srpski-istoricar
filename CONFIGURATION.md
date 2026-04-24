# Configuration Guide - Quick Reference

## Overview

Your application is now fully configurable for deployment to **www.srpski-istoricar.rs** or any other domain.

## What Was Changed

### Backend Configuration (`app.py`)
- ✅ CORS origins now configurable via `ALLOWED_ORIGINS` environment variable
- ✅ Server host configurable via `BACKEND_HOST` environment variable  
- ✅ Server port configurable via `BACKEND_PORT` environment variable
- ✅ Frontend URL configurable via `FRONTEND_URL` environment variable

### Frontend Configuration (`frontend/src/App.js`)
- ✅ API URL now uses `REACT_APP_API_URL` environment variable
- ✅ Automatically switches between development and production configurations

## Configuration Files

### Backend
- `.env.example` - Template for all configurations
- `.env.production.example` - Production-specific template
- Create `.env` from template and customize for your environment

### Frontend
- `frontend/.env.example` - Template for frontend config
- `frontend/.env.development` - Used automatically during `npm start`
- `frontend/.env.production` - Used automatically during `npm run build`

## Quick Start

### For Local Development (Current Setup)

**Backend** (`.env`):
```env
BACKEND_HOST=127.0.0.1
BACKEND_PORT=5000
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
FRONTEND_URL=http://localhost:3000
```

**Frontend** is already configured in `frontend/.env.development`:
```env
REACT_APP_API_URL=http://localhost:5000
```

Run as usual:
```powershell
.\start.ps1
```

### For Production (www.srpski-istoricar.rs)

**Backend** (`.env`):
```env
BACKEND_HOST=0.0.0.0
BACKEND_PORT=5000
ALLOWED_ORIGINS=https://www.srpski-istoricar.rs,https://srpski-istoricar.rs
FRONTEND_URL=https://www.srpski-istoricar.rs
```

**Frontend** is already configured in `frontend/.env.production`:
```env
REACT_APP_API_URL=https://www.srpski-istoricar.rs
```

Build and deploy:
```powershell
# Build frontend
cd frontend
npm run build

# Deploy (see DEPLOYMENT.md for full instructions)
```

## Environment Variables Reference

### Backend Environment Variables

| Variable | Description | Development | Production |
|----------|-------------|-------------|------------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required | Required |
| `OPENAI_BASE_URL` | OpenAI API endpoint | `https://api.openai.com/v1` | `https://api.openai.com/v1` |
| `MODEL_NAME` | OpenAI model to use | `gpt-4o-mini` | `gpt-4o-mini` |
| `BACKEND_HOST` | Server bind address | `127.0.0.1` | `0.0.0.0` |
| `BACKEND_PORT` | Server port | `5000` | `5000` |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `http://localhost:3000,http://127.0.0.1:3000` | `https://www.srpski-istoricar.rs,https://srpski-istoricar.rs` |
| `FRONTEND_URL` | Frontend URL for links | `http://localhost:3000` | `https://www.srpski-istoricar.rs` |
| `QDRANT_URL` | Qdrant database URL (if used) | `http://localhost:6333` | Your Qdrant cloud URL |

### Frontend Environment Variables

| Variable | Description | Development | Production |
|----------|-------------|-------------|------------|
| `REACT_APP_API_URL` | Backend API base URL | `http://localhost:5000` | `https://www.srpski-istoricar.rs` |

**Note:** React environment variables must start with `REACT_APP_` to be accessible in the app.

## Testing Configuration

### Test Backend Configuration
```powershell
# Check what config is loaded
python app.py
# Look for log messages showing:
# - Backend host and port
# - Allowed CORS origins
```

### Test Frontend Configuration
```powershell
cd frontend

# For development
npm start
# Backend API should be http://localhost:5000

# For production build
npm run build
# Check frontend/build/static/js/*.js for REACT_APP_API_URL
```

### Test CORS
```powershell
# From browser console on your frontend:
fetch('http://localhost:5000/api/health')
  .then(r => r.json())
  .then(console.log)
```

Should succeed if CORS is configured correctly.

## Migration Checklist

When ready to migrate to www.srpski-istoricar.rs:

- [ ] Update backend `.env` with production values
- [ ] Update `ALLOWED_ORIGINS` to include `https://www.srpski-istoricar.rs`
- [ ] Verify `frontend/.env.production` has correct `REACT_APP_API_URL`
- [ ] Build frontend: `cd frontend && npm run build`
- [ ] Deploy backend to server (see DEPLOYMENT.md)
- [ ] Deploy frontend build to web server
- [ ] Configure SSL certificate (HTTPS)
- [ ] Test health endpoint: `curl https://www.srpski-istoricar.rs/api/health`
- [ ] Test frontend in browser
- [ ] Verify chat functionality works

## Common Issues

### CORS Errors
**Problem:** Browser shows CORS error when frontend tries to access backend.

**Solution:** 
1. Verify `ALLOWED_ORIGINS` in backend `.env` includes your frontend URL
2. Must include the protocol (`http://` or `https://`)
3. Restart backend after changing `.env`

### API Not Found (404)
**Problem:** Frontend can't reach backend API.

**Solution:**
1. Check `REACT_APP_API_URL` in frontend `.env.production` or `.env.development`
2. Verify backend is running on the expected URL
3. Rebuild frontend after changing environment variables

### Environment Variables Not Loading
**Problem:** Changes to `.env` don't take effect.

**Solution:**
1. Backend: Restart `app.py`
2. Frontend: Stop and restart `npm start`
3. Frontend production: Rebuild with `npm run build`

## Getting Help

1. **Read full deployment guide:** See [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Check logs:** Backend logs show configuration on startup
3. **Verify environment:** Print configuration to verify it's loaded correctly

---

**Summary:** Your application is now fully configurable! Just update the environment variables in `.env` files and you're ready to deploy to www.srpski-istoricar.rs or any other server.
