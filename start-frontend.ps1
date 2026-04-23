# Start Frontend Only
# This script starts only the React development server

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Starting React Frontend Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path "frontend")) {
    Write-Host "ERROR: frontend directory not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "ERROR: Node modules not installed!" -ForegroundColor Red
    Write-Host "Please run: .\setup-frontend.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Frontend will run on: http://localhost:3000" -ForegroundColor Green
Write-Host "Make sure the backend is running on: http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Push-Location frontend
npm start
Pop-Location
