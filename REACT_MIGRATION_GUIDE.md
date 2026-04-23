# Српски историчар - React Migration

## Overview

This project has been migrated from a Flask-with-templates application to a modern React frontend with Flask REST API backend.

## Architecture

### Backend (Flask)
- **Location**: Root directory (`app.py`)
- **Port**: 5000
- **Purpose**: REST API server providing chat, documents, and health endpoints
- **Key Changes**:
  - Removed `render_template` - no longer serves HTML
  - Updated CORS configuration to allow React dev server (port 3000)
  - Added `/static/<path>` endpoint to serve images
  - All responses are JSON (API-only)

### Frontend (React)
- **Location**: `frontend/` directory
- **Dev Port**: 3000
- **Purpose**: Single-page application with modern component architecture
- **Features**:
  - Asynchronous API calls with Axios
  - Component-based UI (Header, Chat, Documents, etc.)
  - State management with React hooks
  - Proxy configuration for API calls to backend

## Setup Instructions

### 1. Backend Setup (Flask)

No changes needed to your existing Flask setup:

```powershell
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the Flask backend
python app.py
```

The backend will run on http://localhost:5000

### 2. Frontend Setup (React)

```powershell
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Copy image files to public directory
Copy-Item ..\pictures\hero-bg.jpg public\ -ErrorAction SilentlyContinue
Copy-Item ..\pictures\SvetiSavaMileseva.jpg public\ -ErrorAction SilentlyContinue

# Start React development server
npm start
```

The frontend will open automatically at http://localhost:3000

## Running the Application

### Development Mode

You need **two terminal windows**:

**Terminal 1 - Backend:**
```powershell
python app.py
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm start
```

Access the application at: http://localhost:3000

### Production Mode

For production, build the React app and serve it:

```powershell
# Build React app
cd frontend
npm run build

# The build files will be in frontend/build/
# You can serve these with any static file server
# or integrate them with Flask
```

## Project Structure

```
srpski-istoricar/
├── app.py                      # Flask API backend
├── requirements.txt            # Python dependencies
├── pictures/                   # Static images
│   ├── hero-bg.jpg
│   └── SvetiSavaMileseva.jpg
├── frontend/                   # React application
│   ├── package.json           # Node dependencies
│   ├── public/                # Static files
│   │   ├── index.html
│   │   ├── hero-bg.jpg        # Copy from ../pictures/
│   │   └── SvetiSavaMileseva.jpg  # Copy from ../pictures/
│   └── src/                   # React source code
│       ├── App.js             # Main component
│       ├── components/        # React components
│       │   ├── Header.js
│       │   ├── ChatContainer.js
│       │   ├── InputContainer.js
│       │   ├── Message.js
│       │   ├── LoadingIndicator.js
│       │   ├── PeriodSelection.js
│       │   ├── WelcomeMessage.js
│       │   ├── DocumentsPanel.js
│       │   └── DocumentItem.js
│       └── index.js           # Entry point
├── docs/                      # Document folders
├── faiss_data/               # FAISS index
└── templates/                # Old Flask templates (no longer used)
```

## Key Differences from Old Version

### Old (Flask + Templates)
- Single HTML file in `templates/index.html`
- Vanilla JavaScript embedded in HTML
- Server-side rendering with Jinja2
- Synchronous page loads

### New (React + Flask API)
- Component-based React architecture
- Separate frontend and backend
- Asynchronous API calls
- Modern JavaScript with ES6+
- Hot module reloading in development
- Better code organization and maintainability

## API Endpoints

The Flask backend provides these endpoints:

- `GET /` - API root (returns JSON with endpoint info)
- `GET /api/health` - Health check
- `GET /api/documents` - List all documents
- `POST /api/chat` - Send message and get response
- `GET /static/<filename>` - Serve static images

## Troubleshooting

### Backend won't start
- Check if port 5000 is available
- Verify `.env` file has correct OpenAI API key
- Check Python dependencies are installed

### Frontend won't start
- Ensure Node.js is installed (v14+)
- Run `npm install` in frontend directory
- Check if port 3000 is available

### CORS errors
- Make sure backend is running before frontend
- Flask `app.py` has been updated with CORS configuration for localhost:3000

### Images not loading
- Copy image files from `pictures/` to `frontend/public/`:
  ```powershell
  cd frontend
  Copy-Item ..\pictures\hero-bg.jpg public\
  Copy-Item ..\pictures\SvetiSavaMileseva.jpg public\
  ```

### API calls failing
- Verify backend is running on port 5000
- Check browser console for errors
- Verify `proxy` setting in `frontend/package.json` points to http://localhost:5000

## Development Tips

### React Hot Reloading
When you edit React components, the browser will automatically update without losing state.

### API Testing
You can test the API directly:
```powershell
# Test health endpoint
curl http://localhost:5000/api/health

# Test documents endpoint
curl http://localhost:5000/api/documents

# Test chat endpoint
curl -X POST http://localhost:5000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"message": "Ко је био Стефан Немања?"}'
```

## Next Steps

Consider these enhancements:
- Add React Router for better URL management
- Implement state management (Redux/Context)
- Add TypeScript for type safety
- Implement server-side rendering (Next.js)
- Add automated tests (Jest, React Testing Library)
- Set up CI/CD pipeline
- Containerize with Docker

## Support

For issues or questions:
1. Check console logs in browser (F12)
2. Check Flask logs in backend terminal
3. Verify both frontend and backend are running
4. Review CORS configuration if API calls fail
