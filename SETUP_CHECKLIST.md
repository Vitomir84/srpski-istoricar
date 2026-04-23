# ✅ Setup Checklist - Српски историчар React Version

Use this checklist to ensure everything is properly set up.

## Prerequisites

- [ ] Python 3.8+ installed
- [ ] Node.js 14+ installed (check: `node --version`)
- [ ] npm installed (check: `npm --version`)
- [ ] Git installed (optional)

## Backend Setup

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file exists in root directory
- [ ] `.env` contains valid `OPENAI_API_KEY`
- [ ] `.env` has other required variables (see below)
- [ ] FAISS database populated (optional but recommended)
- [ ] Tesseract OCR installed (optional, for PDF OCR)

### Required .env Variables

```env
OPENAI_API_KEY=sk-...your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
FAISS_INDEX_PATH=./faiss_data/serbian_history.index
FAISS_METADATA_PATH=./faiss_data/metadata.pkl
```

## Frontend Setup

- [ ] `frontend/` directory exists
- [ ] Node modules installed (`npm install` in frontend directory)
- [ ] `frontend/node_modules/` directory exists
- [ ] Images copied to `frontend/public/`:
  - [ ] `hero-bg.jpg`
  - [ ] `SvetiSavaMileseva.jpg`

## Quick Setup Commands

```powershell
# Setup frontend (from root directory)
.\setup-frontend.ps1

# Copy images manually if needed
Copy-Item pictures\hero-bg.jpg frontend\public\
Copy-Item pictures\SvetiSavaMileseva.jpg frontend\public\
```

## Testing the Setup

### 1. Test Backend

```powershell
# Start backend
python app.py
```

Expected output:
```
✓ OpenAI API ключ учитан
✓ FAISS индекс учитан: X вектора
🚀 Покретам Flask сервер на http://0.0.0.0:5000
```

Test API:
```powershell
# In another terminal
curl http://localhost:5000/api/health
```

Expected: JSON response with status "ok"

### 2. Test Frontend

```powershell
# Start frontend (in another terminal)
cd frontend
npm start
```

Expected:
- Browser opens automatically at http://localhost:3000
- Period selection screen appears
- No console errors (press F12)

### 3. Test Full Application

- [ ] Backend running on port 5000
- [ ] Frontend running on port 3000
- [ ] Period selection screen loads
- [ ] Select a period (e.g., "Сви периоди")
- [ ] Welcome message appears
- [ ] Input field is enabled
- [ ] Documents panel loads on the right
- [ ] Send a test message: "Ко је био Стефан Немања?"
- [ ] Response appears without errors

## Common Issues & Solutions

### Issue: "npm: command not found"
**Solution:** Install Node.js from https://nodejs.org/

### Issue: "Module not found" errors in backend
**Solution:** 
```powershell
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution:** 
```powershell
# Find and kill the process using port 5000
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force
```

### Issue: Port 3000 already in use
**Solution:** React will prompt to use a different port (type 'y')

### Issue: Images not loading
**Solution:** 
```powershell
Copy-Item pictures\*.jpg frontend\public\ -Force
```

### Issue: CORS errors in browser console
**Solution:** 
1. Make sure backend is running
2. Check Flask CORS configuration in `app.py`
3. Backend should allow http://localhost:3000

### Issue: "Cannot GET /" error in React
**Solution:** Make sure React dev server is running, not just Flask

### Issue: API calls return 404
**Solution:** 
1. Verify backend is running
2. Check `package.json` proxy setting: `"proxy": "http://localhost:5000"`
3. API endpoints should be `/api/*` not just `/*`

## Verification Commands

```powershell
# Check Python version
python --version

# Check Node.js version
node --version

# Check npm version
npm --version

# Check if backend is running
curl http://localhost:5000/api/health

# Check if frontend is built (after setup)
Test-Path frontend\node_modules

# Check if images are copied
Test-Path frontend\public\hero-bg.jpg
Test-Path frontend\public\SvetiSavaMileseva.jpg
```

## Startup Checklist (Daily Use)

### Quick Start (Recommended)
```powershell
.\start.ps1
```

### Manual Start
Terminal 1:
```powershell
python app.py
```

Terminal 2:
```powershell
cd frontend
npm start
```

## Database Population (First Time)

If FAISS database is empty:

```powershell
# Option 1: Populate from sample documents
python populate_database.py

# Option 2: Parse all PDFs in docs/ folder
python pdf_parser.py

# Option 3: Add specific PDF
python add_pdf_to_faiss.py "path/to/document.pdf" --period novi_vek
```

## Success Indicators

✅ Backend terminal shows:
```
✓ OpenAI клијент иницијализован
✓ FAISS индекс учитан
✓ Библиографија учитана
🚀 Покретам Flask сервер
```

✅ Frontend browser shows:
- No errors in console (F12)
- Period selection screen
- Documents panel on right
- Chat interface loads

✅ Test chat works:
- Can select period
- Can type message
- Can send message
- Receives response
- Response includes citations

## Next Steps After Setup

1. [ ] Read `QUICKSTART.md` for usage guide
2. [ ] Test chat with various questions
3. [ ] Explore document filtering
4. [ ] Add more documents if needed
5. [ ] Review `REACT_MIGRATION_GUIDE.md` for architecture details

## Getting Help

If stuck:
1. Check this checklist again
2. Review `QUICKSTART.md`
3. Check browser console (F12) for frontend errors
4. Check backend terminal for server errors
5. Review `REACT_MIGRATION_GUIDE.md` for detailed info

---

**Setup Complete?** Run `.\start.ps1` and enjoy! 🎉
