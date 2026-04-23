# 🚀 Quick Start Guide - Српски историчар (React + FastAPI Version)

This guide will help you get the modernized React + FastAPI version of Српски историчар up and running quickly.

## Prerequisites

- **Python 3.8+** with pip
- **Node.js 14+** with npm
- **OpenAI API Key** (in `.env` file)

## Quick Setup (Windows PowerShell)

### Option 1: Automated Setup

Run the setup script to install everything:

```powershell
.\setup-frontend.ps1
```

### Option 2: Manual Setup

1. **Install Frontend Dependencies:**
```powershell
cd frontend
npm install
cd ..
```

2. **Copy Images:**
```powershell
Copy-Item pictures\hero-bg.jpg frontend\public\
Copy-Item pictures\SvetiSavaMileseva.jpg frontend\public\
```

## Running the Application

### Option A: Automated Start (Both Servers)

```powershell
.\start.ps1
```

This will start both the FastAPI backend and React frontend automatically.

### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```powershell
python app.py
```
Backend runs on: http://localhost:5000
API docs: http://localhost:5000/docs

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm start
```
Frontend opens at: http://localhost:3000

### Option C: Individual Servers

**Backend only:**
```powershell
.\start-backend.ps1
```

**Frontend only:**
```powershell
.\start-frontend.ps1
```

## First Time Setup Checklist

- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with `OPENAI_API_KEY`
- [ ] Node.js installed (check with `node --version`)
- [ ] Frontend dependencies installed (`npm install` in frontend folder)
- [ ] Images copied to `frontend/public/`
- [ ] FAISS database populated (optional but recommended)

## Populating the Database

To add documents to the knowledge base:

```powershell
# For PDF documents
python populate_database.py

# For text files
python add_txt_to_faiss.py
```

See `PDF_PARSER_README.md` for details.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 5000 already in use | Kill the process or change Flask port |
| Port 3000 already in use | React will prompt to use different port |
| Images not loading | Run `Copy-Item pictures\*.jpg frontend\public\` |
| CORS errors | Make sure backend is running first |
| API key errors | Check `.env` file has valid `OPENAI_API_KEY` |
| Missing Node modules | Run `npm install` in frontend directory |

## What's Different from the Old Version?
FastAPI + React) |
|-----------------------|-------------|
| Single HTML file | Component-based architecture |
| Vanilla JavaScript | Modern React with hooks |
| Flask web server | FastAPI async API + React SPA |
| Server-side rendering | Client-side SPA |
| Page reloads | Asynchronous updates |
| Static templates | Dynamic state management |
| Synchronous endpoints | Native async/await support |
| No auto-generated docs | Interactive API docs at /docs
| Static templates | Dynamic state management |

## Development

### Hot Reloading

- **React**: Edit files in `frontend/src/`, browser updates automatically
- **Flask**: Restart required for code changes (use debug mode for auto-reload)

### API Testing

Test endpoints directly:

```powershell
# Health check
curl http://localhost:5000/api/health

# Get documents
curl http://localhost:5000/api/documents

# Chat (requires JSON)
Invoke-WebRequest -Uri http://localhost:5000/api/chat `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"message":"Ко је био Стефан Немања?"}'
```

## Building for Production

```powershell
cd frontend
npm run build
```

The optimized build will be in `frontend/build/`

## Next Steps

1. ✅ Setup complete - Application running
2. 📚 Populate database with documents
3. 💬 Test chat functionality
4. 🎨 Customize styles in component CSS files
5. 🚀 Deploy to production (see deployment guide)

## Getting Help

- Check `REACT_MIGRATION_GUIDE.md` for detailed information
- Review `frontend/README.md` for React-specific details
- Check browser console (F12) for frontend errors
- Check terminal for backend errors

## File Structure Overview
astAPI backend ⚡
```
srpski-istoricar/
├── app.py                 # Flask API backend ⚙️
├── frontend/              # React application ⚛️
│   ├── src/
│   │   ├── components/   # React components
│   │   └── App.js        # Main app
│   └── public/           # Static files & images
├── start.ps1             # Start both servers 🚀
├── setup-frontend.ps1    # Setup script 🔧
└── REACT_MIGRATION_GUIDE.md  # Full docs 📖
```

---

**Ready to go!** Run `.\start.ps1` and visit http://localhost:3000 🎉
