# Српски историчар - React Frontend Setup

This is the React frontend for the Српски историчар application.

## Prerequisites

- Node.js (version 14 or higher)
- npm (comes with Node.js)

## Installation

1. Navigate to the frontend directory:
```powershell
cd frontend
```

2. Install dependencies:
```powershell
npm install
```

3. Copy the image files to the public directory:
```powershell
# Copy background images from the pictures folder to frontend/public/
Copy-Item ..\pictures\hero-bg.jpg public\ -ErrorAction SilentlyContinue
Copy-Item ..\pictures\SvetiSavaMileseva.jpg public\ -ErrorAction SilentlyContinue
```

## Running the Application

### Development Mode

1. First, start the Flask backend (in a separate terminal):
```powershell
# From the root directory
python app.py
```

The backend will run on http://localhost:5000

2. Then, start the React development server:
```powershell
# From the frontend directory
npm start
```

The frontend will automatically open in your browser at http://localhost:3000

The React app is configured to proxy API requests to the Flask backend running on port 5000.

## Building for Production

To create a production build:

```powershell
npm run build
```

This will create a `build` folder with optimized production files.

### Serving the Production Build

After building, you can serve the production build with Flask:

1. Move the build folder contents to a location Flask can serve
2. Update Flask to serve the React build files

Or use a static file server:

```powershell
npx serve -s build
```

## Project Structure

```
frontend/
├── public/           # Static files
│   ├── index.html   # HTML template
│   ├── hero-bg.jpg  # Background image
│   └── SvetiSavaMileseva.jpg  # Chat background
├── src/
│   ├── components/  # React components
│   │   ├── Header.js
│   │   ├── ChatContainer.js
│   │   ├── InputContainer.js
│   │   ├── DocumentsPanel.js
│   │   ├── PeriodSelection.js
│   │   └── ...
│   ├── App.js      # Main application component
│   ├── App.css     # Main styles
│   └── index.js    # Entry point
└── package.json    # Dependencies and scripts
```

## Features

- **Asynchronous API calls** using Axios
- **Component-based architecture** with React
- **Real-time chat** with loading indicators
- **Document filtering** with checkbox selection
- **Period-based search** with visual selection
- **Responsive design** with modern CSS

## API Endpoints

The frontend communicates with these backend endpoints:

- `GET /api/health` - Check server status
- `GET /api/documents` - Get list of available documents
- `POST /api/chat` - Send chat message and get response

## Troubleshooting

### Port already in use
If port 3000 is already in use, React will ask if you want to use a different port.

### CORS errors
Make sure the Flask backend is running with the updated CORS configuration that allows requests from http://localhost:3000.

### Images not loading
Make sure you've copied the image files from the `pictures` folder to `frontend/public/`:
- hero-bg.jpg
- SvetiSavaMileseva.jpg
